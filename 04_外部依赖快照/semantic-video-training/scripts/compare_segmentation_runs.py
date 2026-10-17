"""比较两次相同测试协议的分割结果，并生成固定样本的前后对照。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from segmentation import color_mask, palette_from_classes, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="旧实验目录")
    parser.add_argument("--candidate", type=Path, required=True, help="新实验目录")
    parser.add_argument("--baseline-label", default="Baseline", help="对照图中的旧模型名称")
    parser.add_argument("--candidate-label", default="Candidate", help="对照图中的新模型名称")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    before_dir, after_dir = args.baseline / "evaluation_test", args.candidate / "evaluation_test"
    before = json.loads((before_dir / "summary.json").read_text(encoding="utf-8"))
    after = json.loads((after_dir / "summary.json").read_text(encoding="utf-8"))
    for key in ("split", "size", "manifest_sha256", "preprocessing", "precision", "images"):
        if before[key] != after[key]:
            raise ValueError(f"评价条件不同：{key}，不能直接比较")
    if before["split"] != "test":
        raise ValueError("此对照入口要求已经完成独立测试")
    old_rows = list(csv.DictReader((before_dir / "per_image.csv").open(encoding="utf-8-sig")))
    new_rows = list(csv.DictReader((after_dir / "per_image.csv").open(encoding="utf-8-sig")))
    if len(old_rows) != before["images"] or len(new_rows) != after["images"]:
        raise ValueError("逐图记录数量与评价汇总不一致")
    if [r["image"] for r in old_rows] != [r["image"] for r in new_rows]:
        raise ValueError("两次评价的图片顺序不一致")
    for old, new in zip(old_rows, new_rows):
        with Image.open(before_dir / old["ground_truth"]) as a, Image.open(after_dir / new["ground_truth"]) as b:
            if not np.array_equal(np.array(a), np.array(b)):
                raise ValueError(f"真值像素不一致：{new['image']}")
        old_input = before_dir / "inputs" / Path(old["prediction"]).name
        new_input = after_dir / "inputs" / Path(new["prediction"]).name
        with Image.open(old_input) as a, Image.open(new_input) as b:
            if not np.array_equal(np.array(a), np.array(b)):
                raise ValueError(f"输入像素不一致：{new['image']}")
    output = args.output_dir or args.candidate / "comparison_with_baseline"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"输出目录非空：{output}")
    output.mkdir(parents=True, exist_ok=True)
    classes = after["per_class"]
    table = []
    for old, new in zip(before["per_class"], classes):
        if old["index"] != new["index"]:
            raise ValueError("类别编号不一致")
        table.append({"class": new["name_zh"], "symbol": new["symbol"],
                      "before_iou_percent": 100*old["iou"] if old["iou"] is not None else None,
                      "after_iou_percent": 100*new["iou"] if new["iou"] is not None else None,
                      "change_percentage_points": 100*(new["iou"]-old["iou"]) if new["iou"] is not None and old["iou"] is not None else None})
    with (output / "per_class_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    summary = {
        "baseline": str(args.baseline.resolve()), "candidate": str(args.candidate.resolve()),
        "same_inputs_and_ground_truth_verified": True, "images": after["images"], "size": after["size"],
        "before_miou": before["miou"], "after_miou": after["miou"],
        "miou_change_percentage_points": 100*(after["miou"]-before["miou"]),
        "before_pixel_accuracy": before["pixel_accuracy"], "after_pixel_accuracy": after["pixel_accuracy"],
        "per_class": table, "manifest_sha256": after["manifest_sha256"],
        "baseline_label": args.baseline_label, "candidate_label": args.candidate_label,
        "note": "同图同标签同预处理同分辨率评价；训练条件另见各实验配置，不能自动视为单因素消融。测试集此前已被查看，当前仍以验证集选择模型；不能称为全新未知外部测试集。",
    }
    write_json(output / "summary.json", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    indices = np.linspace(0, len(new_rows)-1, min(6, len(new_rows)), dtype=int)
    palette = palette_from_classes(classes)
    fig, axes = plt.subplots(len(indices), 5, figsize=(14, 2.8*len(indices)+.5), squeeze=False, layout="constrained")
    for axes_row, index in zip(axes, indices):
        old, new = old_rows[index], new_rows[index]
        with Image.open(after_dir / "inputs" / Path(new["prediction"]).name) as im:
            pixels = np.array(im)
        with Image.open(after_dir / new["ground_truth"]) as im:
            truth = np.array(im)
        with Image.open(before_dir / old["prediction"]) as im:
            old_prediction = np.array(im)
        with Image.open(after_dir / new["prediction"]) as im:
            new_prediction = np.array(im)
        new_colors = color_mask(new_prediction, palette)
        overlay = (0.55*pixels + 0.45*new_colors).round().astype(np.uint8)
        overlay[truth == 255] = 80
        for ax, image in zip(axes_row, (pixels, color_mask(truth, palette), color_mask(old_prediction, palette), new_colors, overlay)):
            ax.imshow(image)
            ax.axis("off")
        axes_row[0].set_title(Path(new["image"]).stem, fontsize=9)
    for ax, title in zip(axes[0], ("Input", "Ground truth", args.baseline_label, args.candidate_label, "Candidate overlay")):
        ax.set_title(title, fontsize=11)
    fig.legend(handles=[Patch(facecolor=np.array(row["rgb"])/255, edgecolor="gray", label=row["symbol"]) for row in classes],
               loc="outside lower center", ncol=8, fontsize=10)
    fig.savefig(output / "before_after.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    positions = np.arange(8)
    ax.bar(positions-.2, [row["before_iou_percent"] or 0 for row in table], width=.4, label="Before")
    ax.bar(positions+.2, [row["after_iou_percent"] or 0 for row in table], width=.4, label="After")
    ax.set(xticks=positions, xticklabels=[row["symbol"] for row in table], ylim=(0, 100), ylabel="IoU (%)")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.savefig(output / "class_iou_comparison.png", dpi=150)
    plt.close(fig)
    print(f"mIoU {100*before['miou']:.2f}% -> {100*after['miou']:.2f}% ({summary['miou_change_percentage_points']:+.2f} percentage points)")
    print(f"对照：{output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
