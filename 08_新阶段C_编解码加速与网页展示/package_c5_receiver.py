"""Build a self-contained asset folder; Python environment is installed on target."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from c4_config import HERE, REPO, load_settings
from c2_watch import write


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "bundle_manifest.json").read_text("utf-8"))
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Missing or unsafe bundle path: " + name)
        if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError("Bundle checksum differs: " + name)
    upstream = root / "semantic_video_test/runtime/GLC"
    revision = git("-C", upstream, "rev-parse", "HEAD")
    if revision != manifest["glc_revision"] or git("-C", upstream, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Bundled GLC checkout differs from its fixed revision")
    return {"status": "passed", "files": len(manifest["files"]), "glc_revision": revision}


def git(*arguments):
    return subprocess.run(["git", *map(str, arguments)], check=True, capture_output=True,
                          text=True, encoding="utf-8", timeout=120,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0).stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_bundle(args.verify)), flush=True)
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = load_settings()["runtime"]
    project = output / "semantic_video_test"
    stage = project / HERE.name
    stage.mkdir(parents=True)
    files = [p for p in HERE.iterdir() if p.is_file() and p.suffix in (".py", ".json", ".md", ".ps1", ".cmd", ".txt")
             and ".local." not in p.name]
    files += list((HERE / "c5_console").glob("*"))
    files += list((HERE / "native").glob("*"))
    files += [REPO / "04_外部依赖快照/semantic-video-training" / name
              for name in ("demo_server.py", "requirements-glc.txt")]
    files += list((REPO / "04_外部依赖快照/semantic-video-training/scripts").glob("glc_*"))
    files += [REPO / "01_主工程/semantic_mvp" / name for name in ("__init__.py", "transport.py")]
    for path in files:
        if path.is_file():
            target = project / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    bundle = project / "runtime"
    bundle.mkdir()
    # Clone tracked upstream files + history, not the local environment/cache.
    revision = git("-C", runtime["repo"], "rev-parse", "HEAD")
    if git("-C", runtime["repo"], "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("GLC upstream has uncommitted changes; refusing an ambiguous bundle")
    git("clone", "--no-hardlinks", "--no-checkout", runtime["repo"], bundle / "GLC")
    git("-C", bundle / "GLC", "checkout", "--detach", revision)
    git("-C", bundle / "GLC", "remote", "set-url", "origin", "https://github.com/jzyustc/GLC.git")
    weights = bundle / "checkpoints/glc"
    weights.mkdir(parents=True)
    for name in ("GLC_image.pth.tar", "GLC_Video.pth.tar"):
        shutil.copy2(runtime["weights"] / name, weights / name)
    ffmpeg = bundle / "ffmpeg/bin"
    ffmpeg.mkdir(parents=True)
    for name in ("ffmpeg.exe", "ffprobe.exe"):
        shutil.copy2(runtime["ffmpeg"].with_name(name), ffmpeg / name)
    # Keep FFmpeg distribution's licensing material with its binaries when available.
    distribution = runtime["ffmpeg"].parent.parent
    for pattern in ("LICENSE*", "COPYING*", "README*"):
        for path in distribution.glob(pattern):
            if path.is_file():
                shutil.copy2(path, bundle / "ffmpeg" / path.name)
    if (distribution / "doc").is_dir():
        shutil.copytree(distribution / "doc", bundle / "ffmpeg/doc")
    shutil.copy2(runtime["native_library"], bundle / "glc_rans.dll")
    shutil.copy2(HERE / "runtime.example.json", stage / "runtime.local.json")
    shutil.copy2(HERE / "c5_receiver.example.json", stage / "c5_receiver.local.json")
    (output / "先读我.txt").write_text(
        "C5 接收端迁移文件夹（Windows）\n\n"
        "包含代码、固定 GLC 源码、两份权重、FFmpeg/FFprobe、native DLL。\n"
        "不含 Python 环境、连接口令、发送视频或历史任务。\n"
        "1. 复制整个文件夹到接收电脑，安装 Python 3.12 与 Git。\n"
        "2. 在 semantic_video_test\\08_新阶段C_编解码加速与网页展示 中运行 安装C5接收环境.cmd。\n"
        "   该步骤从官方 PyTorch 和 PyPI 安装固定版本，需要联网，不改变系统 Python 包。\n"
        "3. 双击 启动C5接收端.cmd，在网页检查环境；首次配对按 C5双机操作说明.md 执行。\n"
        "4. 先最大速度；4070 实机尚未验收，不把本机测试当双机通过。\n"
        "安装前自动核验 bundle_manifest.json 的所有资产摘要；不要修改包内文件再安装。\n", encoding="utf-8-sig")
    manifest = {"schema_version": 1, "role": "receiver", "python_environment_included": False,
                "target_gpu_verified": False, "connection_token_included": False, "glc_revision": revision,
                "producer_commit": git("-C", REPO, "rev-parse", "HEAD"),
                "producer_worktree_dirty": bool(git("-C", REPO, "status", "--porcelain")),
                "files": {p.relative_to(output).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)}
                          for p in sorted(output.rglob("*")) if p.is_file() and ".git" not in p.parts}}
    write(output / "bundle_manifest.json", manifest)
    verify_bundle(output)
    print(json.dumps({"output": str(output), "files": len(manifest["files"]),
                      "bytes": sum(item["bytes"] for item in manifest["files"].values())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
