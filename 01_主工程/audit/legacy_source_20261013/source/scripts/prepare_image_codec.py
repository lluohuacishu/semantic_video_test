"""从 CompressAI 官方权重地址准备图片编码器；演示时仅加载本地文件。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_segformer import download, sha256

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "checkpoints/pretrained/image-codec"
SOURCE_REVISION = "1c9e3eee83a8f4c8162d38b52f78f5b0c40de175"
ARCHITECTURE = "bmshj2018-hyperprior"
URLS = {
    1: "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-hyperprior-1-7eb97409.pth.tar",
    3: "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-hyperprior-3-6d87be32.pth.tar",
    5: "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-hyperprior-5-f8b614e1.pth.tar",
}


def weight_path(directory: Path, quality: int) -> Path:
    if quality not in URLS:
        raise ValueError("当前图片编码器支持质量档位 1、3、5")
    return directory / f"hyperprior-q{quality}.pth.tar"


def verify_weight(directory: Path, quality: int) -> tuple[Path, str]:
    path = weight_path(directory, quality)
    provenance_path = directory / "provenance.json"
    if not path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError("图片压缩权重未准备好，请运行 setup_image_codec.cmd")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    row = provenance.get("weights", {}).get(str(quality), {})
    digest = sha256(path)
    if (provenance.get("architecture") != ARCHITECTURE
            or row.get("requested_url") != URLS[quality] or row.get("sha256") != digest):
        raise ValueError("图片编码器权重与官方来源记录不一致")
    return path, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    record = args.model_dir / "provenance.json"
    provenance = json.loads(record.read_text(encoding="utf-8")) if record.is_file() else {}
    provenance.update(architecture=ARCHITECTURE, metric="mse", source_revision=SOURCE_REVISION,
                      source_url=f"https://github.com/InterDigitalInc/CompressAI/blob/{SOURCE_REVISION}/compressai/zoo/image.py")
    provenance.setdefault("weights", {})
    for quality, url in URLS.items():
        path = weight_path(args.model_dir, quality)
        provenance["weights"][str(quality)] = download(url, path, provenance["weights"].get(str(quality)))
        record.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        verify_weight(args.model_dir, quality)
    print(f"图片压缩权重已准备好：{args.model_dir}", flush=True)


if __name__ == "__main__":
    main()
