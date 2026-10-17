"""读取固定的数据划分；训练和评估共用同一份名单。"""
from __future__ import annotations

import json
from pathlib import Path


def load_manifest(path: Path) -> tuple[Path, dict, dict[str, list[Path]]]:
    path = path.resolve()
    metadata = json.loads(path.read_text(encoding="utf-8"))
    root = (path.parent / metadata.get("root", ".")).resolve()
    splits = metadata["splits"]
    files, seen_paths, seen_hashes = {}, {}, {}
    for name in ("train", "validation", "test"):
        files[name] = []
        for row in splits.get(name, []):
            image = (root / row["image"]).resolve()
            if not image.is_relative_to(root) or not image.is_file():
                raise ValueError(f"图片路径越界或不存在：{image}")
            if image in seen_paths:
                raise ValueError(f"图片路径重复：{row['image']}")
            seen_paths[image] = name
            digest = row.get("rgb_sha256")
            if digest and digest in seen_hashes and seen_hashes[digest] != name:
                raise ValueError(f"同一图片像素跨越不同划分：{row['image']}")
            if digest:
                seen_hashes[digest] = name
            files[name].append(image)
    if not files["train"] or not files["validation"]:
        raise ValueError("manifest 必须包含非空的训练和验证集")
    return root, metadata, files
