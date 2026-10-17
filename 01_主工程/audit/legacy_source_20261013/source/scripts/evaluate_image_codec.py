"""在固定 SUIM 验证样本上检查实际图片码量、还原质量与分割变化。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from dataset_manifest import load_manifest
from image_codec import ImageCodec, quality_metrics, read_rgb
from segment_video import choose_device
from segmentation import (confusion_matrix, load_segmentation_model,
                          metrics_from_confusion, predict_original, write_json)

ROOT = Path(__file__).resolve().parents[1]


def save_preview(samples, qualities, destination):
    cell_width, cell_height = 300, 250
    preview = Image.new("RGB", (cell_width * (len(qualities) + 1), cell_height * len(samples)), "#f5f6f7")
    draw = ImageDraw.Draw(preview)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = ImageFont.truetype(str(font_path), 15) if font_path.exists() else ImageFont.load_default()
    for row, sample in enumerate(samples):
        entries = [(sample["original"], f"原图 · {sample['id']}")]
        entries.extend((sample["reconstructed"][quality],
                        f"q{quality} · {sample['bytes'][quality] / 1024:.1f} KiB") for quality in qualities)
        for column, (array, label) in enumerate(entries):
            image = Image.fromarray(array)
            image.thumbnail((cell_width - 12, cell_height - 35), Image.Resampling.LANCZOS)
            x, y = column * cell_width, row * cell_height
            preview.paste(image, (x + (cell_width - image.width) // 2, y + 28))
            draw.text((x + 8, y + 5), label, fill="#1b2832", font=font)
    preview.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/SUIM/manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/suim_segformer_b0_20260915/segmentation_best.pt")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--qualities", type=int, nargs="+", choices=(1, 3, 5), default=[1, 3, 5])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="只控制语义分割；压缩编解码使用 CPU")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root, metadata, _ = load_manifest(args.manifest)
    records = metadata["splits"]["validation"]
    if not 1 <= args.samples <= len(records):
        parser.error("样本数应介于 1 与验证集图片数之间")
    qualities = sorted(set(args.qualities))
    selected = [records[index] for index in np.linspace(0, len(records) - 1, args.samples, dtype=int)]
    output = args.output_dir or ROOT / "runs" / f"image_codec_validation_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=False)
    device = choose_device(args.device)
    model, state = load_segmentation_model(args.checkpoint, device)
    codec = ImageCodec("cpu")
    classes, size = state["classes"], int(state["size"])
    original_confusion = torch.zeros((8, 8), dtype=torch.int64)
    after_confusion = {quality: original_confusion.clone() for quality in qualities}
    agreement_confusion = {quality: original_confusion.clone() for quality in qualities}
    rows, previews = [], []
    for number, record in enumerate(selected):
        original = read_rgb(root / record["image"])
        with Image.open(root / record["mask_index"]) as image:
            target = np.array(image)
        if target.shape != original.shape[:2]:
            raise ValueError(f"图片与真值尺寸不一致：{record['id']}")
        before = predict_original(model, Image.fromarray(original), size, device)
        original_confusion += confusion_matrix(torch.from_numpy(before), torch.from_numpy(target))
        folder = output / f"sample_{number:02d}"
        folder.mkdir()
        Image.fromarray(original).save(folder / "input.png")
        preview = {"id": record["id"], "original": original, "reconstructed": {}, "bytes": {}}
        for quality in qualities:
            encoded = codec.encode(original, folder / f"q{quality}.svic", quality)
            reconstructed, decoded = codec.decode(folder / f"q{quality}.svic", folder / f"q{quality}.png")
            metrics = quality_metrics(original, reconstructed)
            after = predict_original(model, Image.fromarray(reconstructed), size, device)
            after_confusion[quality] += confusion_matrix(torch.from_numpy(after), torch.from_numpy(target))
            agreement_confusion[quality] += confusion_matrix(torch.from_numpy(after), torch.from_numpy(before))
            rows.append({"id": record["id"], "image": record["image"], "quality": quality,
                         "width": encoded["width"], "height": encoded["height"],
                         "file_bytes": encoded["file_bytes"], "bpp": encoded["bpp"],
                         "psnr_db": metrics["psnr_db"], "ssim": metrics["ssim"],
                         "encode_compute_seconds": encoded["encode_compute_seconds"],
                         "decode_compute_seconds": decoded["decode_compute_seconds"],
                         "semantic_pixel_agreement": float(np.mean(before == after)),
                         "minimum_file_seconds_20kbps": encoded["file_bytes"] * 8 / 20000,
                         "minimum_file_seconds_40kbps": encoded["file_bytes"] * 8 / 40000})
            preview["reconstructed"][quality] = reconstructed
            preview["bytes"][quality] = encoded["file_bytes"]
            print(f"{number + 1}/{len(selected)} · q{quality} · {encoded['file_bytes']} bytes · PSNR {metrics['psnr_db']:.2f}", flush=True)
        if len(previews) < 3:
            previews.append(preview)
    with (output / "per_image.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    results = []
    for quality in qualities:
        group = [row for row in rows if row["quality"] == quality]
        means = {key: float(np.mean([row[key] for row in group if row[key] is not None]))
                 for key in ("file_bytes", "bpp", "psnr_db", "ssim", "encode_compute_seconds", "decode_compute_seconds")}
        results.append({"quality": quality, "mean": means,
                        "segmentation_after": metrics_from_confusion(after_confusion[quality], classes),
                        "prediction_consistency": metrics_from_confusion(agreement_confusion[quality], classes)})
    summary = {"sample_count": len(selected), "split": "validation", "selection": "固定名单上的等间隔样本，无按效果挑选",
               "records": selected, "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
               "checkpoint": str(args.checkpoint.resolve()), "segmentation_input_size": size,
               "codec_device": "cpu", "segmentation_device": str(device), "results": results,
               "segmentation_before": metrics_from_confusion(original_confusion, classes),
               "quality_definition": metrics["definition"],
               "notes": ["小样本流程检查，不是完整验证集或测试集结论。",
                         "GT 指标在原图尺寸累计；不能与原先 256 补边图上的 mIoU 直接对比。",
                         "20/40 kbps 列是仅按文件大小估算的最短发送时间，不是真实 TCP 实测。",
                         "预览图缩小排版，所有实际编码、重建与质量计算保留原尺寸。"]}
    write_json(output / "summary.json", summary)
    save_preview(previews, qualities, output / "comparison.png")
    print(json.dumps({"report": str(output.resolve()), "before_miou": summary["segmentation_before"]["miou"],
                      "qualities": [{"quality": row["quality"], **row["mean"],
                                     "after_miou": row["segmentation_after"]["miou"]} for row in results]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
