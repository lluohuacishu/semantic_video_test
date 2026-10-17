"""对单张新图片做语义分割，保存与原图同尺寸的类别图和叠加图。"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from segmentation import color_mask, load_segmentation_model, palette_from_classes, predict_original, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用")
    torch.set_num_threads(4)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model, state = load_segmentation_model(args.checkpoint, device)
    with Image.open(args.input) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    output = args.output_dir or Path("runs") / datetime.now().strftime("segmentation_prediction_%Y%m%d-%H%M%S-%f")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"输出目录非空：{output}")
    prediction = predict_original(model, image, state["size"], device)
    colors = color_mask(prediction, palette_from_classes(state["classes"]))
    overlay = (0.55*np.asarray(image) + 0.45*colors).round().astype(np.uint8)
    output.mkdir(parents=True, exist_ok=True)
    image.save(output / "input.png")
    Image.fromarray(prediction).save(output / "prediction_index.png")
    Image.fromarray(colors).save(output / "prediction_color.png")
    Image.fromarray(overlay).save(output / "overlay.png")
    counts = np.bincount(prediction.ravel(), minlength=8)
    write_json(output / "summary.json", {
        "input": str(args.input.resolve()), "checkpoint": str(args.checkpoint.resolve()),
        "input_width": image.width, "input_height": image.height, "model_input_size": state["size"],
        "classes": [{**row, "predicted_pixels": int(counts[i]), "predicted_fraction": float(counts[i]/counts.sum())}
                    for i, row in enumerate(state["classes"])],
        "note": "模型对低分辨率等比缩放图推理，去除补边后将 logits 双线性上采样到原尺寸，再取类别。输出虽与原图同尺寸，并不是原生高清推理。无真值标签，不报告准确率。",
    })
    print(f"预测结果：{output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
