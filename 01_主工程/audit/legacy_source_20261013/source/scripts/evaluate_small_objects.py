"""按标注连通区域统计小目标召回，辅助判断分辨率和下采样影响。"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from segmentation import SegmentationDataset, load_segmentation_model, manifest_info, write_json


def area_bin(area_fraction: float) -> str:
    # 用面积占有效图像的比例分桶，换评估尺寸时仍然可比。
    if area_fraction <= 0.001:
        return "tiny_<=0.1%"
    if area_fraction <= 0.01:
        return "small_0.1-1%"
    if area_fraction <= 0.05:
        return "medium_1-5%"
    return "large_>5%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/SUIM/manifest.json"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--size", type=int, help="覆盖模型默认输入尺寸，用于分辨率消融")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.size is not None and (args.size < 32 or args.size % 8):
        parser.error("size 必须是至少 32 的 8 倍数")
    if args.batch_size < 1:
        parser.error("batch-size 必须为正")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"输出目录非空，不覆盖：{args.output_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    torch.set_num_threads(4)
    root, metadata, digest = manifest_info(args.manifest)
    model, state = load_segmentation_model(args.checkpoint, device)
    if state["manifest_sha256"] != digest:
        parser.error("模型与数据清单指纹不同，停止评估")
    eval_size = args.size or state["size"]
    dataset = SegmentationDataset(root, metadata["splits"][args.split], eval_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    aggregate = defaultdict(lambda: {"objects": 0, "gt_pixels": 0, "correct_pixels": 0, "detected_objects": 0, "recall_sum": 0.0})
    class_aggregate = defaultdict(lambda: {"objects": 0, "gt_pixels": 0, "correct_pixels": 0, "detected_objects": 0, "recall_sum": 0.0})
    output_rows = []
    processed_images = 0
    with torch.inference_mode():
        for images, targets in loader:
            predictions = model(images.to(device)).argmax(1).cpu().numpy()
            for image_idx, (target, prediction) in enumerate(zip(targets.numpy(), predictions)):
                valid_pixels = int((target != 255).sum())
                if valid_pixels == 0:
                    continue
                for class_idx in range(1, 8):
                    binary = (target == class_idx).astype(np.uint8)
                    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                    for component_idx in range(1, count):
                        area = int(stats[component_idx, cv2.CC_STAT_AREA])
                        if area == 0:
                            continue
                        fraction = area / valid_pixels
                        bucket = area_bin(fraction)
                        correct = int((prediction[labels == component_idx] == class_idx).sum())
                        recall = correct / area
                        key = (bucket, class_idx)
                        for table in (aggregate[bucket], class_aggregate[key]):
                            table["objects"] += 1
                            table["gt_pixels"] += area
                            table["correct_pixels"] += correct
                            table["detected_objects"] += int(recall >= 0.5)
                            table["recall_sum"] += recall
                        output_rows.append({
                            "image": dataset.records[processed_images + image_idx]["image"], "class_index": class_idx,
                            "class_symbol": metadata["classes"][class_idx]["symbol"],
                            "area_pixels": area, "area_fraction": fraction, "size_bin": bucket,
                            "correct_pixels": correct, "pixel_recall": recall,
                        })
            processed_images += len(images)

    def finish(table):
        objects = table["objects"]
        return {
            "objects": objects, "gt_pixels": table["gt_pixels"], "correct_pixels": table["correct_pixels"],
            "pixel_recall": table["correct_pixels"] / table["gt_pixels"] if table["gt_pixels"] else None,
            "detected_at_50pct_recall": table["detected_objects"] / objects if objects else None,
            "mean_object_recall": table["recall_sum"] / objects if objects else None,
        }

    report = {
        "checkpoint": str(args.checkpoint.resolve()), "model_name": state["model_name"],
        "split": args.split, "size": eval_size, "model_training_size": state["size"],
        "device": str(device), "bins": {bucket: finish(table) for bucket, table in sorted(aggregate.items())},
        "per_class": {f"{bucket}/{class_idx}": finish(table) for (bucket, class_idx), table in sorted(class_aggregate.items())},
        "definition": "每个前景类别的 8 邻域标注连通区域为一个目标；pixel_recall 是该区域被预测为同类的比例；detected_at_50pct_recall 统计召回至少 50% 的目标比例。面积分桶按目标面积/该图有效像素比例。",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "small_object_report.json", report)
    with (args.output_dir / "components.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]) if output_rows else ["image"])
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"完成：{args.output_dir.resolve()}")
    for bucket, values in report["bins"].items():
        print(f"{bucket}: objects={values['objects']}, pixel_recall={values['pixel_recall']:.3f}, detected@50%={values['detected_at_50pct_recall']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
