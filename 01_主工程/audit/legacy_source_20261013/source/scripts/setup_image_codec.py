"""安装锁定的 CompressAI 源码；Windows 使用本机 MinGW 编译。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from prepare_image_codec import ROOT, SOURCE_REVISION


def run(command, directory=ROOT, environment=None):
    subprocess.run([str(part) for part in command], cwd=directory, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-dir", type=Path, default=Path(r"C:\mingw64\bin"))
    args = parser.parse_args()
    source = ROOT / "tools/CompressAI"
    if not source.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/InterDigitalInc/CompressAI.git", source])
        run(["git", "fetch", "--depth", "1", "origin", SOURCE_REVISION], source)
        run(["git", "checkout", "--detach", SOURCE_REVISION], source)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != SOURCE_REVISION:
        raise ValueError("本地 CompressAI 版本与锁定提交不一致")
    run([sys.executable, "-m", "pip", "install", "--index-url", "https://pypi.org/simple",
         "-r", ROOT / "requirements-image-codec.txt"])
    environment = os.environ.copy()
    if os.name == "nt":
        if not (args.compiler_dir / "g++.exe").is_file():
            raise FileNotFoundError("找不到 MinGW g++，用 --compiler-dir 指定已有编译器目录")
        environment["PATH"] = str(args.compiler_dir) + os.pathsep + environment.get("PATH", "")
        path = source / "setup.py"
        text = path.read_text(encoding="utf-8")
        # Pybind11 默认把 Windows 当成 MSVC；这里只调整编译选项，不修改模型或熵编码算法。
        text = text.replace("from pybind11.setup_helpers import Pybind11Extension, build_ext",
                            "from pybind11 import get_include\nfrom setuptools import Extension as Pybind11Extension\nfrom setuptools.command.build_ext import build_ext")
        text = text.replace("include_dirs=[rans_lib_dir, rans_ext_dir],", "include_dirs=[str(rans_lib_dir), str(rans_ext_dir), get_include()],")
        text = text.replace('name=f"{package_name}._CXX",', 'name=f"{package_name}._CXX",\n            include_dirs=[get_include()],') if "include_dirs=[get_include()]" not in text else text
        text = text.replace("extra_compile_args=extra_compile_args,", "extra_compile_args=extra_compile_args,\n            extra_link_args=[\"-static-libgcc\", \"-static-libstdc++\"],") if "-static-libgcc" not in text else text
        path.write_text(text, encoding="utf-8")
    wheels = ROOT / "tools/codec-wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "setup.py", "build_ext"]
    if os.name == "nt":
        command += ["--compiler=mingw32"]
    run(command + ["bdist_wheel", "--dist-dir", wheels], source, environment)
    matches = sorted(wheels.glob("compressai-1.2.9.dev0-*.whl"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError("没有生成 CompressAI 安装包")
    run([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", matches[-1]])
    record = {"repository": "https://github.com/InterDigitalInc/CompressAI", "revision": revision,
              "version": "1.2.9.dev0", "windows_build": "MinGW, static C++ runtime; build flags only",
              "compiler_directory": str(args.compiler_dir), "wheel": matches[-1].name}
    (ROOT / "tools/image-codec-build.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    run([sys.executable, ROOT / "scripts/prepare_image_codec.py"])


if __name__ == "__main__":
    main()
