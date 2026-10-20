"""Explicit environment check. Importing this module never imports torch."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from c4_config import load_settings, REPO
from c2_watch import utc, write

BASELINE = "a6de3270e762c861d71e4cc2fec25b28a0cc4cba"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def command(argv):
    result = subprocess.run(list(map(str, argv)), capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"Command failed: {argv[0]}")
    return result.stdout.strip()


def collect(runtime, device="cuda"):
    missing = [key for key, value in runtime.items() if not Path(value).exists()]
    if set(missing) - {"native_library"}:
        raise ValueError("Missing runtime paths: " + ", ".join(missing))
    sys.path.insert(0, str(REPO / "04_外部依赖快照/semantic-video-training/scripts"))
    import glc_video_codec as codec
    codec.check_source(runtime["repo"])
    codec.check_weights(runtime["weights"])
    from glc_entropy import GaussianRans
    entropy = GaussianRans()
    backend = {"python": True, "native": False, "native_sha256": None}
    if "native_library" not in missing:
        from c2_native import NativeGaussianRans
        native = NativeGaussianRans(runtime["native_library"], entropy)
        if native.table_sha256 != entropy.table_sha256:
            raise ValueError("Native CDF differs")
        backend.update(native=True, native_sha256=digest(runtime["native_library"]))
    packages = {}
    for name in ("torch", "torchvision", "numpy", "opencv-python", "Pillow", "einops", "timm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    arithmetic = codec.configure(device)
    result = {"schema_version": 1, "observed_utc": utc(), "python": sys.version,
              "os": platform.platform(), "packages": packages, "arithmetic": arithmetic,
              "model_revision": codec.REVISION, "entropy": codec.ENTROPY_ID,
              "weights": {key: value["sha256"] for key, value in codec.ASSETS.items()},
              "table_sha256": entropy.table_sha256, "backends": backend,
              "ffmpeg": command([runtime["ffmpeg"], "-version"]).splitlines()[0],
              "frozen_c4_commit": BASELINE, "neural_inference_performed": False,
              "cross_device_verified": False}
    if device == "cuda":
        result["gpu_snapshot"] = command(["nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--runtime-json", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserve earlier evidence")
    if args.runtime_json:
        runtime = {k: Path(v) for k, v in json.loads(args.runtime_json.read_text("utf-8"))["runtime"].items()}
    else:
        runtime = load_settings(args.config)["runtime"]
    result = collect(runtime, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
