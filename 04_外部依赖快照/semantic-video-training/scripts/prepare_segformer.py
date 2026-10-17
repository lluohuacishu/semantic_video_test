"""下载 NVIDIA 发布的 SegFormer-B0 权重和作者仓库的演示图片。"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
REVISION = "489d5cd81a0b59fab9b7ea758d3548ebe99677da"
SOURCE_REVISION = "65fa8cfa9b52b6ee7e8897a98705abf8570f9e32"
MODEL_DIR = ROOT / "checkpoints/pretrained/segformer-b0-ade"
SAMPLE_PATH = ROOT / "data/raw/segformer_demo.png"
FILES = ("config.json", "preprocessor_config.json", "model.safetensors", "README.md")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download(url: str, path: Path, previous: dict | None = None) -> dict:
    if path.is_file():
        if previous and previous.get("requested_url") == url and sha256(path) == previous.get("sha256"):
            print(f"已缓存：{path.name}", flush=True)
            return previous
        raise ValueError(f"已有文件没有匹配的来源记录，不覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "semantic-video-training/segformer-baseline"})
    print(f"下载：{url}", flush=True)
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as handle:
        final_url = response.url
        expected = response.headers.get("Content-Length")
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    if expected is not None and partial.stat().st_size != int(expected):
        raise OSError(f"下载长度不完整：{path.name}")
    if partial.stat().st_size == 0:
        raise OSError(f"下载文件为空：{path.name}")
    partial.replace(path)
    return {"requested_url": url, "resolved_host": urllib.parse.urlparse(final_url).hostname,
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = args.model_dir / "provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("revision") != REVISION:
            parser.error("目录中的模型版本不同，请指定一个新目录")
    else:
        provenance = {"model_id": MODEL_ID, "revision": REVISION,
                      "official_code": "https://github.com/NVlabs/SegFormer",
                      "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "files": {}}
    def save_provenance():
        temporary = provenance_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(provenance_path)
    for name in FILES:
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
        provenance["files"][name] = download(url, args.model_dir / name, provenance["files"].get(name))
        save_provenance()
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "segformer" or len(config.get("id2label", {})) != 150:
        raise ValueError("下载的配置不是预期的 ADE20K 150 类 SegFormer")
    from safetensors import safe_open
    with safe_open(args.model_dir / "model.safetensors", framework="pt", device="cpu") as weights:
        if not list(weights.keys()):
            raise ValueError("权重文件没有张量")
    url = f"https://raw.githubusercontent.com/NVlabs/SegFormer/{SOURCE_REVISION}/demo/demo.png"
    provenance["demo"] = download(url, SAMPLE_PATH, provenance.get("demo"))
    save_provenance()
    print(f"模型已准备好：{args.model_dir}\n演示图片：{SAMPLE_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
