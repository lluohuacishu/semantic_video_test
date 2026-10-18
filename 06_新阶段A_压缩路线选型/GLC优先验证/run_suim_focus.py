"""Run existing CompressAI image coding on three fixed SUIM validation categories."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--codec-scripts", type=Path, required=True,
                        help="semantic-video-training/scripts")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="existing SUIM SegFormer checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    sys.path.insert(0, str(args.codec_scripts.resolve()))
    from image_codec import ImageCodec, quality_metrics, read_rgb
    from segment_video import choose_device
    from segmentation import color_mask, load_segmentation_model, palette_from_classes, predict_original
    import torch

    root = args.manifest.resolve().parent
    metadata = json.loads(args.manifest.read_text(encoding="utf-8"))
    validation = metadata["splits"]["validation"]
    rules = [
        ("fish_only", lambda r: r["class_pixels"][6] > 0 and r["class_pixels"][1] == 0,
         lambda r: r["class_pixels"][6]),
        ("diver_only", lambda r: r["class_pixels"][1] > 0 and r["class_pixels"][6] == 0,
         lambda r: r["class_pixels"][1]),
        ("fish_and_diver", lambda r: r["class_pixels"][1] > 0 and r["class_pixels"][6] > 0,
         lambda r: r["class_pixels"][1] + r["class_pixels"][6]),
    ]
    selected = []
    for tag, matches, score in rules:
        candidates = [row for row in validation if matches(row)]
        record = max(candidates, key=score)
        selected.append((tag, record))

    args.output.mkdir(parents=True)
    codec = ImageCodec("cpu")
    device = choose_device("cuda")
    segmenter, state = load_segmentation_model(args.checkpoint, device)
    palette = palette_from_classes(state["classes"])
    model_size = int(state["size"])
    rows, panels = [], []
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    for tag, record in selected:
        image = read_rgb(root / record["image"])
        mask = np.asarray(Image.open(root / record["mask_index"]), dtype=np.uint8)
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Unaligned label for {record['id']}")
        target_class = 6 if record["class_pixels"][6] >= record["class_pixels"][1] else 1
        target = (mask == target_class).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(target, connectivity=8)
        if count < 2:
            raise ValueError(f"No target component for {record['id']}")
        x, y, w, h, _ = map(int, max(stats[1:], key=lambda item: item[4]))
        pad_x, pad_y = max(16, w // 4), max(16, h // 4)
        crop = (max(0, x - pad_x), max(0, y - pad_y),
                min(image.shape[1], x + w + pad_x), min(image.shape[0], y + h + pad_y))
        folder = args.output / record["id"].replace("/", "_")
        folder.mkdir()
        Image.fromarray(image).save(folder / "original.png")
        Image.fromarray(mask).save(folder / "ground_truth_index.png")
        prediction_before = predict_original(segmenter, Image.fromarray(image), model_size, device)
        Image.fromarray(prediction_before).save(folder / "prediction_original.png")
        reconstructions = {"original": image}
        byte_counts = {"original": None}
        for quality in (1, 3, 5):
            encoded = codec.encode(image, folder / f"q{quality}.svic", quality)
            reconstructed, _ = codec.decode(folder / f"q{quality}.svic", folder / f"q{quality}.png")
            metrics = quality_metrics(image, reconstructed)
            prediction_after = predict_original(segmenter, Image.fromarray(reconstructed), model_size, device)
            Image.fromarray(prediction_after).save(folder / f"prediction_q{quality}.png")
            Image.fromarray(color_mask(prediction_after, palette)).save(folder / f"prediction_q{quality}_color.png")
            error = np.square(reconstructed.astype(np.float64) - image.astype(np.float64))
            row = {"sample_rule": tag, "id": record["id"], "primary_target_class": target_class,
                   "fish_gt_pixels": int(record["class_pixels"][6]),
                   "diver_gt_pixels": int(record["class_pixels"][1]), "quality": quality,
                   "file_bytes": encoded["file_bytes"], "full_psnr_db": metrics["psnr_db"],
                   "full_ssim": metrics["ssim"],
                   "segmentation_pixel_agreement": float(np.mean(prediction_before == prediction_after)),
                   "fish_iou_before": None, "fish_iou_after": None,
                   "diver_iou_before": None, "diver_iou_after": None}
            for class_index, class_name in ((6, "fish"), (1, "diver")):
                pred_before_class = prediction_before == class_index
                pred_after_class = prediction_after == class_index
                truth_class = mask == class_index
                for suffix, pred in (("before", pred_before_class), ("after", pred_after_class)):
                    union_pred = int(np.logical_or(pred, truth_class).sum())
                    intersection_pred = int(np.logical_and(pred, truth_class).sum())
                    row[f"{class_name}_iou_{suffix}"] = (intersection_pred / union_pred
                                                          if union_pred else None)
            for class_index, class_name in ((6, "fish"), (1, "diver")):
                roi = mask == class_index
                mse = float(error[roi].mean()) if roi.any() else None
                row[f"{class_name}_psnr_db"] = (10 * np.log10(255**2 / mse)
                                                 if mse is not None and mse > 0 else
                                                 99.0 if mse == 0 else None)
            row["minimum_file_seconds_20kbps"] = encoded["file_bytes"] * 8 / 20000
            row["minimum_file_seconds_40kbps"] = encoded["file_bytes"] * 8 / 40000
            rows.append(row)
            reconstructions[f"q{quality}"] = reconstructed
            byte_counts[f"q{quality}"] = encoded["file_bytes"]
            print(f"{record['id']} q{quality}: {encoded['file_bytes']} bytes, PSNR {metrics['psnr_db']:.2f} dB",
                  flush=True)
        panels.append({"tag": tag, "id": record["id"], "target_class": target_class,
                       "crop_xyxy": crop, "images": reconstructions, "bytes": byte_counts})

    with (args.output / "metrics.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    preview = Image.new("RGB", (4 * 400, 3 * 300), "#f4f6f8")
    draw = ImageDraw.Draw(preview)
    for row_index, sample in enumerate(panels):
        for column, key in enumerate(("original", "q1", "q3", "q5")):
            image = Image.fromarray(sample["images"][key]).crop(sample["crop_xyxy"])
            image.thumbnail((390, 250), Image.Resampling.LANCZOS)
            x, y = column * 400, row_index * 300
            preview.paste(image, (x + (400 - image.width) // 2,
                                  y + 32 + (260 - image.height) // 2))
            title = f"{sample['tag']} · {sample['id']} · {key}"
            if key != "original":
                title += f" · {sample['bytes'][key] / 1024:.1f} KiB"
            draw.text((x + 5, y + 5), title, font=font, fill="#132b42")
    preview.save(args.output / "comparison_target_crops.png")
    summary = {"dataset": "SUIM", "split": "validation",
               "selection": "one largest-labelled sample for fish-only, diver-only, and jointly labelled fish+diver cases",
               "manifest": str(args.manifest.resolve()), "manifest_sha256": __import__("hashlib").sha256(
                   args.manifest.read_bytes()).hexdigest(),
               "codec": "existing CompressAI bmshj2018-hyperprior; CPU entropy coding; no fine-tuning",
               "semantic_evaluator": {"model": state["model_name"], "checkpoint": str(args.checkpoint.resolve()),
                                      "input_size": model_size, "device": str(device)},
               "quality_levels": [1, 3, 5], "rows": rows,
               "limitations": ["Three targeted validation images; not a dataset-level score.",
                               "Label IoU uses the same SegFormer before and after compression; this does not guarantee true recognition ability.",
                               "PSNR within class labels measures pixel error, not recognition or stripe accuracy.",
                               "No human viewing score was generated."]}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "samples": [x[1]["id"] for x in selected]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
