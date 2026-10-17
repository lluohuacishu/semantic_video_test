"""提取 SUIM 原图与合并标签，检查配对并固定训练、验证、测试名单。"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import stat
import zlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
CLASSES = [
    ("BW", "背景水体"), ("HD", "潜水员"), ("PF", "水生植物"), ("WR", "沉船与遗迹"),
    ("RO", "机器人与仪器"), ("RI", "珊瑚礁与无脊椎动物"), ("FV", "鱼与脊椎动物"), ("SR", "海床与岩石"),
]


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mask_to_indices(rgb: np.ndarray) -> np.ndarray:
    # 最近的 RGB 立方体顶点。源 BMP 有抗锯齿/颜色噪声，不能仅匹配八种纯色。
    bits = (rgb >= 128).astype(np.uint8)
    return bits[..., 0] * 4 + bits[..., 1] * 2 + bits[..., 2]


def extract_core(archive: Path, root: Path) -> dict:
    with ZipFile(archive) as bundle:
        selected = []
        seen = set()
        for item in bundle.infolist():
            if item.is_dir():
                continue
            match = re.fullmatch(r"(train_val|TEST)/(images|masks)/([^/\\:]+)\.(jpg|bmp)", item.filename)
            if item.filename != "INFO.txt" and not match:
                continue
            if match and (match[2], match[4]) not in {("images", "jpg"), ("masks", "bmp")}:
                raise ValueError(f"图片/标签扩展名错误：{item.filename}")
            target = (root / item.filename).resolve()
            if not target.is_relative_to(root) or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError(f"不支持的压缩包路径：{item.filename}")
            if item.filename in seen:
                raise ValueError(f"压缩包内路径重复：{item.filename}")
            seen.add(item.filename)
            selected.append(item)
        for group, expected in (("train_val", 1525), ("TEST", 110)):
            images = {Path(n).stem for n in seen if n.startswith(group + "/images/")}
            masks = {Path(n).stem for n in seen if n.startswith(group + "/masks/")}
            if images != masks or len(images) != expected:
                raise ValueError(f"{group} 配对或数量异常：images={len(images)}, masks={len(masks)}")
        total_bytes = sum(item.file_size for item in selected)
        if shutil.disk_usage(root.parent).free < total_bytes + 512 * 1024**2:
            raise OSError("解压所需磁盘空间不足。")
        print("检查整个 ZIP 的 CRC（检查文件损坏，不代表来源认证或病毒扫描）……", flush=True)
        damaged = bundle.testzip()
        if damaged:
            raise ValueError(f"ZIP 文件损坏：{damaged}")
        root.mkdir(parents=True, exist_ok=True)
        for i, item in enumerate(selected, 1):
            target = root / item.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                checksum = 0
                with target.open("rb") as existing:
                    for block in iter(lambda: existing.read(1024**2), b""):
                        checksum = zlib.crc32(block, checksum)
                if target.stat().st_size != item.file_size or checksum != item.CRC:
                    raise ValueError(f"已解压的源文件与 ZIP 不一致：{target}")
                continue
            with bundle.open(item) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=1024**2)
            if i % 500 == 0:
                print(f"提取 {i}/{len(selected)} 个文件", flush=True)
        return {"archive": str(archive), "archive_bytes": archive.stat().st_size,
                "full_archive_crc": "passed", "extracted_files": len(selected), "extracted_bytes": total_bytes,
                "excluded": ["Checkpoint_Data", "Benchmark_Evaluation", "TEST/masks 下的二值掩码副本"]}


def inspect_pairs(root: Path) -> list[dict]:
    records = []
    for group in ("train_val", "TEST"):
        images = sorted((root / group / "images").glob("*.jpg"))
        mask_stems = {p.stem for p in (root / group / "masks").glob("*.bmp")}
        expected = 1525 if group == "train_val" else 110
        if len(images) != expected or {p.stem for p in images} != mask_stems:
            raise ValueError(f"{group} 解压目录的数量/配对不符合官方结构，请检查是否混入其他文件")
        index_dir = root / "indexed_masks" / group
        index_dir.mkdir(parents=True, exist_ok=True)
        for i, path in enumerate(images, 1):
            mask_path = root / group / "masks" / (path.stem + ".bmp")
            with Image.open(path) as source:
                source.load()
                image = source.convert("RGB")
                size = image.size
                digest = hashlib.sha256(str(size).encode() + image.tobytes()).hexdigest()
            with Image.open(mask_path) as mask_source:
                mask_source.load()
                mask_size = mask_source.size
                rgb = np.asarray(mask_source.convert("RGB"))
            indices = mask_to_indices(rgb)
            index_path = index_dir / (path.stem + ".png")
            Image.fromarray(indices).save(index_path)
            counts = np.bincount(indices.reshape(-1), minlength=8).tolist()
            off_palette = np.any((rgb != 0) & (rgb != 255), axis=-1)
            ambiguous = np.any((rgb > 32) & (rgb < 223), axis=-1)
            records.append({
                "id": group + "/" + path.stem, "official_split": group,
                "image": path.relative_to(root).as_posix(), "mask_rgb": mask_path.relative_to(root).as_posix(),
                "mask_index": index_path.relative_to(root).as_posix(), "width": size[0], "height": size[1],
                "mask_width": mask_size[0], "mask_height": mask_size[1], "label_aligned": mask_size == size,
                "rgb_sha256": digest, "class_pixels": counts,
                "off_palette_pixels": int(off_palette.sum()), "ambiguous_color_pixels": int(ambiguous.sum()),
            })
            if i % 200 == 0 or i == len(images):
                print(f"检查图片和标签 {group}: {i}/{len(images)}", flush=True)
    return records


def make_splits(records: list[dict], seed: int, validation_fraction: float) -> tuple[dict, dict]:
    test = [r for r in records if r["official_split"] == "TEST"]
    test_hashes = {r["rgb_sha256"] for r in test}
    # 完全相同的解码像素作为一组，不跨训练/验证；测试重叠样本不参与训练和选模型。
    excluded = [r for r in records if r["official_split"] == "train_val" and r["rgb_sha256"] in test_hashes]
    unaligned = [r for r in records if not r["label_aligned"]]
    groups = defaultdict(list)
    for record in records:
        if record["official_split"] == "train_val" and record["rgb_sha256"] not in test_hashes and record["label_aligned"]:
            groups[record["rgb_sha256"]].append(record)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("去除测试重叠后独立训练图片不足。")
    random.Random(seed).shuffle(keys)
    count = max(1, min(len(keys) - 1, round(len(keys) * validation_fraction)))
    validation_keys = set(keys[:count])
    splits = {
        "train": sorted([r for key in keys if key not in validation_keys for r in groups[key]], key=lambda r: r["id"]),
        "validation": sorted([r for key in keys if key in validation_keys for r in groups[key]], key=lambda r: r["id"]),
        "test": test,
    }
    duplicate_groups = defaultdict(list)
    for record in records:
        duplicate_groups[record["rgb_sha256"]].append(record["id"])
    report = {
        "excluded_train_val_overlapping_test": [r["id"] for r in excluded],
        "label_dimension_mismatches": [{k: r[k] for k in ("id", "width", "height", "mask_width", "mask_height")} for r in unaligned],
        "excluded_train_val_unaligned_labels": [r["id"] for r in unaligned if r["official_split"] == "train_val"],
        "exact_duplicate_groups": [ids for ids in duplicate_groups.values() if len(ids) > 1],
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "test_unique_images": len(test_hashes),
        "split_method": "seeded image groups by identical decoded RGB; no video/scene IDs available",
        "limitation": "精确去重不能识别近似图片或相邻视频帧；此验证划分不保证场景独立。",
    }
    return splits, report


def save_preview(root: Path, splits: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = np.asarray([[(i >> 2 & 1) * 255, (i >> 1 & 1) * 255, (i & 1) * 255] for i in range(8)], dtype=np.uint8)
    rows = splits["train"][::max(1, len(splits["train"]) // 4)][:4]
    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 2.5 * len(rows)), squeeze=False, layout="constrained")
    for row, record in zip(axes, rows):
        for ax, key in zip(row, ("image", "mask_rgb", "mask_index")):
            with Image.open(root / record[key]) as im:
                pixels = np.asarray(im)
                ax.imshow(palette[pixels] if key == "mask_index" else pixels)
            ax.axis("off")
        row[0].set_title(record["id"], fontsize=9)
        row[1].set_title("Original RGB annotation", fontsize=9)
        row[2].set_title("8-class label (display colors)", fontsize=9)
    fig.savefig(root / "dataset_preview.png", dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=PROJECT / "data" / "SUIM.zip")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "data" / "SUIM")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--reuse-extracted", action="store_true", help="重新核对源文件 CRC，接着处理已解压的数据")
    args = parser.parse_args()
    archive, root = args.archive.resolve(), args.output_dir.resolve()
    if not archive.is_file():
        parser.error(f"压缩包不存在：{archive}")
    if not 0 < args.validation_fraction < 1:
        parser.error("validation-fraction 必须在 0 和 1 之间")
    if root.exists() and (not root.is_dir() or (any(root.iterdir()) and not args.reuse_extracted)):
        parser.error(f"输出目录已存在且非空，不覆盖数据：{root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    extraction = extract_core(archive, root)
    records = inspect_pairs(root)
    splits, duplicate_report = make_splits(records, args.seed, args.validation_fraction)
    classes = [{"index": i, "symbol": symbol, "name_zh": name,
                "rgb": [(i >> 2 & 1) * 255, (i >> 1 & 1) * 255, (i & 1) * 255]}
               for i, (symbol, name) in enumerate(CLASSES)]
    manifest = {
        "version": 1, "dataset": "SUIM", "root": ".", "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "source": "https://irvlab.cs.umn.edu/resources/suim-dataset",
        "validation_mode": "held_out_images_exact_duplicates_grouped", "classes": classes,
        "mask_conversion": "RGB channel >=128 gives a bit; class_id=4*R_bit+2*G_bit+B_bit",
        "splits": splits,
    }
    report = {
        "created": datetime.now().isoformat(timespec="seconds"), **extraction, **duplicate_report,
        "images_checked": len(records), "pairs_checked": len(records),
        "official_counts": {"train_val": 1525, "TEST": 110},
        "pixels": sum(sum(r["class_pixels"]) for r in records),
        "off_palette_pixels": sum(r["off_palette_pixels"] for r in records),
        "ambiguous_color_pixels": sum(r["ambiguous_color_pixels"] for r in records),
        "images_with_off_palette_labels": sum(r["off_palette_pixels"] > 0 for r in records),
        "class_pixels_by_split": {s: np.asarray([r["class_pixels"] for r in rows]).sum(axis=0).tolist() for s, rows in splits.items()},
        "label_note": "原始 BMP 原样保留；另生成 0~7 索引 PNG。非纯色像素按最近的八色顶点归类，边界标签存在噪声。缩放标签应使用最近邻插值。",
        "source_note": "数量、目录及颜色表与 SUIM 官方页面相符；包内 INFO.txt 的 Source 指向了其他数据集，使用官方 SUIM 页面为准。CRC 只验证损坏，不认证来源。",
    }
    write_json(root / "manifest.json", manifest)
    write_json(root / "preparation_report.json", report)
    save_preview(root, splits)
    print(json.dumps({"dataset": str(root), "split_counts": duplicate_report["split_counts"],
                      "excluded_test_overlap": len(duplicate_report["excluded_train_val_overlapping_test"]),
                      "excluded_unaligned_train_val": len(duplicate_report["excluded_train_val_unaligned_labels"]),
                      "duplicate_groups": len(duplicate_report["exact_duplicate_groups"]),
                      "off_palette_pixels": report["off_palette_pixels"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
