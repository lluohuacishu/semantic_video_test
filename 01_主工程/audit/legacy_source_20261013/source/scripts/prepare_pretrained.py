"""从 torchvision 指定的 PyTorch 官方地址获取迁移学习权重。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import torch
from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights

PROJECT = Path(__file__).resolve().parents[1]
OFFICIAL_FILENAME = "lraspp_mobilenet_v3_large-d234d4ea.pth"


def prepare_pretrained() -> Path:
    weights = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
    url = weights.url
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "download.pytorch.org":
        raise ValueError("仅从 PyTorch 官方 HTTPS 下载域名获取权重")
    if Path(parsed.path).name != OFFICIAL_FILENAME:
        raise ValueError("torchvision 权重地址与已记录版本不同，请先核对")
    destination = PROJECT / "checkpoints" / "pretrained" / OFFICIAL_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        print(f"下载官方权重：{url}", flush=True)
        pending = destination.with_suffix(".download")
        torch.hub.download_url_to_file(url, str(pending), hash_prefix="d234d4ea", progress=False)
        pending.replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if not digest.startswith("d234d4ea"):
        raise ValueError("本地预训练权重与 torchvision 官方版本指纹不符")
    state = torch.load(destination, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("预训练文件不是纯张量权重字典")
    record = {
        "source": url, "provider": "torchvision official weights enum",
        "weight_name": "LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1",
        "bytes": destination.stat().st_size, "sha256": digest, "path": str(destination),
        "pretraining_data": "COCO subset with 20 Pascal VOC foreground labels and background",
        "use": "保留已训练的特征提取与共享分割层，重新初始化八类 SUIM 输出层",
        "note": "HTTPS 官方来源为首要依据，指纹用于完整性核对；以 weights_only=True 读取。",
    }
    destination.with_suffix(".json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"权重已准备：{destination} ({record['bytes']/1024**2:.2f}MiB)", flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    prepare_pretrained()
