"""Diagnose three fixed SUIM images with the existing GLC real-stream adapter.

This is a single-I-frame diagnostic, not a video or an equal-budget ranking.
Run in the existing SegFormer/CompressAI environment; GLC uses a separate Python.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def invoke(command, log):
    print(f"Running {log.name}", flush=True)
    with log.open("x", encoding="utf-8") as handle:
        subprocess.run(list(map(str, command)), stdout=handle, stderr=subprocess.STDOUT,
                       check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def make_stripe_detail(previous, output):
    """Show a geometrically fixed central crop; no outcome-based crop search."""
    sid = "train_val_f_r_1036_"
    original = Image.open(previous / sid / "original.png").convert("RGB")
    w, h = original.size
    crop = (round(w * .3), round(h * .3), round(w * .8), round(h * .8))
    items = [("原图", original),
             ("GLC I 帧 q63 · 仅静态诊断", Image.open(output / sid / "q63/receiver/frames/im00001.png").convert("RGB")),
             ("CompressAI q5 · 文件预算不同", Image.open(previous / sid / "q5.png").convert("RGB"))]
    canvas = Image.new("RGB", (1920, 540), "#f3f5f7")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 23)
    for i, (title, image) in enumerate(items):
        draw.text((i * 640 + 12, 12), title, font=font, fill="#152d45")
        canvas.paste(image.crop(crop).resize((640, 480), Image.Resampling.NEAREST), (i * 640, 60))
    canvas.save(output / "fish_stripes_detail.png")
    metadata = {"id": "train_val/f_r_1036_", "crop_xyxy": crop,
                "rule": "central 30%-80% of width and height; nearest-neighbor 2x display",
                "equal_budget_comparison": False}
    (output / "fish_stripes_detail.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--glc-python", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    previous = json.loads((args.previous / "summary.json").read_text(encoding="utf-8"))
    manifest = Path(previous["manifest"])
    if sha(manifest) != previous["manifest_sha256"]:
        raise ValueError("SUIM manifest changed since sample selection")
    records = {row["id"]: row for row in json.loads(manifest.read_text(encoding="utf-8"))["splits"]["validation"]}
    selected = [row for row in previous["rows"] if row["quality"] == 5]
    args.output.mkdir(parents=True)
    adapter = args.training_root / "scripts/glc_video_codec.py"
    common = ["--repo", args.training_root / "tools/GLC", "--weights",
              args.training_root / "checkpoints/glc", "--ffmpeg", args.ffmpeg]
    checks, commands = [], []
    for sample in selected:
        sid = sample["id"].replace("/", "_")
        prior_dir = args.previous / sid
        folder = args.output / sid
        folder.mkdir()
        original = rgb(prior_dir / "original.png")
        dataset_image = rgb(manifest.parent / records[sample["id"]]["image"])
        if not np.array_equal(original, dataset_image):
            raise ValueError(f"Original differs from dataset for {sid}")
        input_file = folder / "single_frame_lossless.mkv"
        command = [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                   "-framerate", "1", "-i", prior_dir / "original.png", "-frames:v", "1",
                   "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", input_file]
        invoke(command, folder / "input.log")
        commands.append(list(map(str, command)))
        for q in (0, 63):
            run = folder / f"q{q}"
            run.mkdir()
            stream = run / "stream.glcv"
            encode = [args.glc_python, adapter, "encode", "--input", input_file,
                      "--output", stream, "--workdir", run / "sender", "--frames", "1",
                      "--width", "0", "--q", str(q), "--intra-period", "64",
                      "--reset-interval", "32", *common]
            decode = [args.glc_python, adapter, "decode", "--input", stream,
                      "--output", run / "receiver", *common]
            invoke(encode, run / "encode.log")
            invoke(decode, run / "decode.log")
            commands.extend([list(map(str, encode)), list(map(str, decode))])
            enc = json.loads((run / "sender/encode.json").read_text(encoding="utf-8"))
            dec = json.loads((run / "receiver/decode.json").read_text(encoding="utf-8"))
            source = rgb(run / "sender/source/im00001.png")
            sender = rgb(run / "sender/frames/im00001.png")
            receiver = rgb(run / "receiver/frames/im00001.png")
            if not np.array_equal(original, source) or not np.array_equal(sender, receiver):
                raise ValueError(f"Pixel verification failed for {sid} q{q}")
            if (not dec["all_frame_checksums_match"] or not dec["independent_decoder"]
                    or dec["source_media_reads"] or len(dec["frames"]) != 1):
                raise ValueError(f"Independent decoder verification failed for {sid} q{q}")
            if enc["header"]["frames"] != 1 or enc["frames"][0]["type"] != "I":
                raise ValueError("Expected exactly one I frame")
            checks.append({"id": sample["id"], "quality": q, "file_bytes": stream.stat().st_size,
                           "stream_sha256": sha(stream), "width": original.shape[1],
                           "height": original.shape[0], "source_rgb_exact": True,
                           "receiver_rgb_exact": True, "source_media_reads": [],
                           "container_overhead_bytes": enc["container_overhead_bytes"]})
            print(f"Verified {sid} q{q}: {stream.stat().st_size} B", flush=True)
    (args.output / "verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    (args.output / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2), encoding="utf-8")

    sys.path.insert(0, str(args.training_root / "scripts"))
    from image_codec import quality_metrics
    from segmentation import load_segmentation_model, predict_original
    import torch
    device = torch.device("cuda")
    checkpoint = Path(previous["semantic_evaluator"]["checkpoint"])
    segmenter, state = load_segmentation_model(checkpoint, device)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
    caption = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
    panel = Image.new("RGB", (1600, 1120), "#f3f5f7")
    draw = ImageDraw.Draw(panel)
    draw.text((14, 8), "SUIM 静态 I 帧诊断：同一原图与裁剪；文件预算不同，不能作为等预算排名", font=font, fill="#152d45")
    rows, crops = [], []
    for index, sample in enumerate(selected):
        sid = sample["id"].replace("/", "_")
        prior_dir = args.previous / sid
        original = rgb(prior_dir / "original.png")
        mask = np.asarray(Image.open(prior_dir / "ground_truth_index.png"))
        before = np.asarray(Image.open(prior_dir / "prediction_original.png"))
        target = (mask == sample["primary_target_class"]).astype(np.uint8)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(target, connectivity=8)
        x, y, w, h, area = map(int, max(stats[1:], key=lambda s: s[4]))
        px, py = max(16, w // 4), max(16, h // 4)
        crop = (max(0, x-px), max(0, y-py), min(original.shape[1], x+w+px), min(original.shape[0], y+h+py))
        crops.append({"id": sample["id"], "crop_xyxy": crop})
        images = [("原图", original, None)]
        for q in (0, 63):
            run = args.output / sid / f"q{q}"
            recon = rgb(run / "receiver/frames/im00001.png")
            prediction = predict_original(segmenter, Image.fromarray(recon), int(state["size"]), device)
            Image.fromarray(prediction).save(run / "prediction.png")
            quality = quality_metrics(original, recon)
            error = (recon.astype(np.float64) - original.astype(np.float64)) ** 2
            check = next(c for c in checks if c["id"] == sample["id"] and c["quality"] == q)
            row = {**check, "sample_rule": sample["sample_rule"], "full_psnr_db": quality["psnr_db"],
                   "full_ssim": quality["ssim"], "segmentation_pixel_agreement": float((before == prediction).mean())}
            for cls, name in ((6, "fish"), (1, "diver")):
                truth = mask == cls
                for suffix, pred in (("before", before), ("after", prediction)):
                    union = int(np.logical_or(pred == cls, truth).sum())
                    intersection = int(np.logical_and(pred == cls, truth).sum())
                    row[f"{name}_iou_{suffix}"] = intersection / union if union else None
                mse = float(error[truth].mean()) if truth.any() else None
                row[f"{name}_psnr_db"] = 10*np.log10(255**2/mse) if mse else None
            for kbps in (20, 30, 40):
                row[f"minimum_file_seconds_{kbps}kbps"] = row["file_bytes"] * 8 / (kbps * 1000)
            rows.append(row)
            images.append((f"GLC I 帧 q{q}", recon, row["file_bytes"]))
        images.append(("CompressAI q5 参考", rgb(prior_dir / "q5.png"), sample["file_bytes"]))
        for col, (title, data, size) in enumerate(images):
            image = Image.fromarray(data).crop(crop)
            image.thumbnail((388, 278), Image.Resampling.LANCZOS)
            left, top = col * 400, 40 + index * 360
            draw.text((left + 8, top), title + (f" · {size/1000:.1f} KB" if size else ""), font=font, fill="#152d45")
            draw.text((left + 8, top + 27), sample["id"], font=caption, fill="#4e6376")
            panel.paste(image, (left + (400-image.width)//2, top + 66 + (280-image.height)//2))
    panel.save(args.output / "comparison_glc_target_crops.png")
    make_stripe_detail(args.previous, args.output)
    with (args.output / "metrics.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"date": "2026-10-18", "dataset": "SUIM validation", "selection": previous["selection"],
               "manifest_sha256": previous["manifest_sha256"], "quality_levels": [0, 63],
               "codec": "Official GLC pretrained I-frame network; interpolated 64 q levels; existing local glcv/rANS adapter",
               "adapter_sha256": sha(adapter), "semantic_evaluator": previous["semantic_evaluator"],
               "semantic_checkpoint_sha256": sha(checkpoint), "rows": rows, "crops": crops,
               "limitations": ["Single-I-frame static diagnostic; no video temporal claim.",
                               "GLC q and CompressAI q are unrelated scales; file budgets are not matched.",
                               "All stream bytes include local research container overhead; not an official GLC bitstream.",
                               "Three targeted large-object validation images; not representative dataset scores.",
                               "Same-model segmentation is auxiliary; no human viewing score.",
                               "No training or new codec implementation."]}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "all_pixel_checks_passed": True}), flush=True)


if __name__ == "__main__":
    main()
