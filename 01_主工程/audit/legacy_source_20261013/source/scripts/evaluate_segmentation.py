"""在固定 SUIM 验证/测试集上评价分割，保存标签、可视化和可复核的混淆矩阵。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from segmentation import (
    IGNORE_INDEX, SegmentationDataset, color_mask, confusion_matrix, load_segmentation_model,
    manifest_info, metrics_from_confusion, palette_from_classes, save_comparison, write_json,
)


def save_class_plot(metrics, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = metrics["per_class"]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    values = [100*r["iou"] if r["iou"] is not None else 0 for r in rows]
    bars = ax.bar([r["symbol"] for r in rows], values, color="#25808a")
    ax.bar_label(bars, fmt="%.1f")
    ax.set(ylim=(0, 105), ylabel="IoU (%)", title="SUIM: per-class IoU")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/SUIM/manifest.json"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--size", type=int, help="覆盖模型默认输入尺寸，用于分辨率消融；必须为 8 的倍数")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size 必须为正")
    if args.size is not None and (args.size < 32 or args.size % 8):
        parser.error("size 必须是至少 32 的 8 倍数")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用")
    torch.set_num_threads(4)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    root, metadata, digest = manifest_info(args.manifest)
    model, state = load_segmentation_model(args.checkpoint, device)
    if state["manifest_sha256"] != digest:
        parser.error("模型与数据清单指纹不同，停止评估")
    records = metadata["splits"][args.split]
    eval_size = args.size or state["size"]
    dataset = SegmentationDataset(root, records, eval_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    output = args.output_dir or args.checkpoint.parent / ("evaluation_" + args.split)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"输出目录非空，不覆盖：{output}")
    for folder in ("predictions", "ground_truth", "inputs", "color_predictions", "overlays"):
        (output / folder).mkdir(parents=True)
    palette = palette_from_classes(metadata["classes"])
    total_confusion = np.zeros((8, 8), dtype=np.int64)
    preview_indices = set(np.linspace(0, len(dataset)-1, min(6, len(dataset)), dtype=int).tolist())
    rows, samples = [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for images, targets in loader:
            predictions = model(images.to(device)).argmax(1).cpu()
            for image, target, prediction in zip(images, targets, predictions):
                index = len(rows)
                matrix = confusion_matrix(prediction, target).numpy()
                total_confusion += matrix
                item_metrics = metrics_from_confusion(matrix, metadata["classes"])
                name = f"{index:04d}_{Path(records[index]['image']).stem}.png"
                pred_array = prediction.numpy().astype(np.uint8)
                target_array = target.numpy().astype(np.uint8)
                pred_array[target_array == IGNORE_INDEX] = IGNORE_INDEX
                ref = (image.permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)
                colors = color_mask(pred_array, palette)
                overlay = (ref*0.55 + colors*0.45).round().astype(np.uint8)
                overlay[target_array == IGNORE_INDEX] = 80
                Image.fromarray(pred_array).save(output / "predictions" / name)
                Image.fromarray(target_array).save(output / "ground_truth" / name)
                Image.fromarray(ref).save(output / "inputs" / name)
                Image.fromarray(colors).save(output / "color_predictions" / name)
                Image.fromarray(overlay).save(output / "overlays" / name)
                rows.append({
                    "image": records[index]["image"], "miou_present_union": item_metrics["miou"],
                    "pixel_accuracy": item_metrics["pixel_accuracy"], "valid_pixels": item_metrics["valid_pixels"],
                    "prediction": "predictions/" + name, "ground_truth": "ground_truth/" + name,
                })
                if index in preview_indices:
                    samples.append((Path(records[index]["image"]).stem, image, target_array, pred_array))
    metrics = metrics_from_confusion(total_confusion, metadata["classes"])
    for filename, records_to_write in (
        ("per_image.csv", rows),
        ("per_class.csv", [{k: r[k] for k in ("index", "symbol", "name_zh", "iou", "dice", "recall", "true_pixels", "predicted_pixels")} for r in metrics["per_class"]]),
    ):
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records_to_write[0]))
            writer.writeheader()
            writer.writerows(records_to_write)
    with (output / "confusion_matrix.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/predicted"] + [r["symbol"] for r in metadata["classes"]])
        for row, values in zip(metadata["classes"], total_confusion.tolist()):
            writer.writerow([row["symbol"]] + values)
    save_comparison(samples, metadata["classes"], output / "comparison.png")
    save_class_plot(metrics, output / "class_iou.png")
    majority = np.zeros_like(total_confusion)
    # 全部预测成背景的简单参照，不用于选模型。
    majority[:, 0] = total_confusion.sum(axis=1)
    summary = {
        "split": args.split, "images": len(dataset), "size": eval_size, "model_training_size": state["size"], "precision": "float32",
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": state["epoch"],
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "manifest_sha256": digest, "preprocessing": state["preprocessing"],
        "device": str(device), **metrics, "confusion_matrix": total_confusion.tolist(),
        "always_background_baseline": metrics_from_confusion(majority, metadata["classes"]),
        "elapsed_s": round(time.perf_counter()-start, 3),
        "preview_selection": "6 images equally spaced through the fixed sorted split; not selected by model quality",
        "note": "在等比缩放、居中补边后的评估分辨率计分，忽略补边 255；不是原始分辨率或官方论文协议。测试集未用于训练或轮次选择。逐图 mIoU 的平均不等于这里全局累计的 mIoU。size 覆盖仅用于分辨率消融，未重新训练模型。",
    }
    write_json(output / "summary.json", summary)
    print(f"{args.split}: images={len(dataset)}, mIoU={metrics['miou']*100:.2f}%, pixel_accuracy={metrics['pixel_accuracy']*100:.2f}%", flush=True)
    print(f"结果：{output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
