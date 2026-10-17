"""以真实 CPU 码流验证联合类别解码；普通压缩仅作为初步参考。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from image_codec import ImageCodec, quality_metrics, read_package
from segment_video import choose_device
from segmentation import (SegmentationDataset, color_mask, confusion_matrix, load_segmentation_model,
                          manifest_info, metrics_from_confusion, palette_from_classes, write_json)
from semantic_image_codec import SemanticImageCodec
from train_semantic_codec import TEACHER, subset

ROOT = Path(__file__).resolve().parents[1]


def preview(samples, palette, path):
    canvas = Image.new("RGB", (280 * 5, 255 * len(samples)), "#f4f5f6")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
    titles = ("输入（训练尺寸）", "真值", "发送端语义预测", "接收端直接解码类别", "联合码流还原图片")
    for row, sample in enumerate(samples):
        for col, (title, pixels) in enumerate(zip(titles, sample)):
            draw.text((col * 280 + 10, row * 255 + 8), title, fill="#243238", font=font)
            image = Image.fromarray(pixels)
            image.thumbnail((264, 220))
            canvas.paste(image, (col * 280 + 8 + (264-image.width)//2, row * 255 + 32))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, default=TEACHER)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/SUIM/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=0, help="0 为全部验证集；其余等间隔固定抽样")
    args = parser.parse_args()
    if args.samples < 0:
        parser.error("samples 不能为负数")
    args.output.mkdir(parents=True, exist_ok=False)
    root, metadata, manifest_sha = manifest_info(args.manifest)
    codec = SemanticImageCodec(args.checkpoint)
    if (codec.state["manifest_sha256"] != manifest_sha
            or hashlib.sha256(args.teacher.read_bytes()).hexdigest() != codec.state["sender_teacher_sha256"]):
        raise ValueError("验证数据划分或发送端教师与训练记录不一致")
    records = subset(metadata["splits"]["validation"], args.samples)
    data = SegmentationDataset(root, records, codec.state["size"], cache=True)
    device = choose_device("auto")
    teacher, _ = load_segmentation_model(args.teacher, device)
    baseline = ImageCodec()
    baseline.model(3)
    matrices = {key: torch.zeros(8, 8, dtype=torch.int64) for key in ("sender", "joint_direct", "baseline_after_segmentation")}
    errors = {key: 0.0 for key in ("joint", "baseline")}
    total_valid, rows, samples = 0, [], []
    audit = None
    palette = palette_from_classes(metadata["classes"])
    for i in range(len(data)):
        image, target = data[i]
        rgb = image.mul(255).round().byte().permute(1, 2, 0).numpy()
        with torch.inference_mode():
            sender = teacher(image.unsqueeze(0).to(device)).argmax(1)[0].cpu().byte()
        matrices["sender"] += confusion_matrix(sender, target)
        guidance = sender.numpy().copy()
        guidance[target.numpy() == 255] = 0
        folder = args.output / f"sample_{i:03d}"
        folder.mkdir()
        encoded = codec.encode(rgb, guidance, folder / "joint.svic")
        reconstructed, direct, decoded = codec.decode(folder / "joint.svic", folder / "joint.png", folder / "direct_classes.png")
        baseline_encoded = baseline.encode(rgb, folder / "baseline.svic", 3)
        ordinary, _ = baseline.decode(folder / "baseline.svic", folder / "baseline.png")
        with torch.inference_mode():
            baseline_tensor = torch.from_numpy(ordinary.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
            baseline_classes = teacher(baseline_tensor).argmax(1)[0].cpu()
        matrices["joint_direct"] += confusion_matrix(torch.from_numpy(direct), target)
        matrices["baseline_after_segmentation"] += confusion_matrix(baseline_classes, target)
        valid = target.numpy() != 255
        pixels = int(valid.sum())
        total_valid += pixels
        for key, output in (("joint", reconstructed), ("baseline", ordinary)):
            errors[key] += float(np.square(rgb.astype(np.float64)[valid] - output.astype(np.float64)[valid]).sum())
        metrics = quality_metrics(rgb, reconstructed)
        rows.append({"id": records[i]["id"], "joint_file_bytes": encoded["file_bytes"],
                     "baseline_file_bytes": baseline_encoded["file_bytes"], "joint_file_bpp_256_square": encoded["bpp"],
                     "baseline_file_bpp_256_square": baseline_encoded["bpp"], "joint_payload_bytes": encoded["payload_bytes"],
                     "joint_psnr_db_entire_square": metrics["psnr_db"], "joint_ssim_entire_square": metrics["ssim"],
                     "joint_encode_seconds": encoded["encode_compute_seconds"], "joint_decode_seconds": decoded["decode_compute_seconds"]})
        if i in (0, len(data)//2, len(data)-1):
            visual_sender, visual_direct = sender.numpy().copy(), direct.copy()
            visual_sender[~valid] = visual_direct[~valid] = 255
            samples.append((rgb, color_mask(target.numpy(), palette), color_mask(visual_sender, palette),
                            color_mask(visual_direct, palette), reconstructed))
        if audit is None:
            altered = ((guidance.astype(np.int16) + 1) % 8).astype(np.uint8)
            codec.encode(rgb, altered, folder / "altered_guidance.svic")
            _, altered_classes, _ = codec.decode(folder / "altered_guidance.svic", folder / "altered.png", folder / "altered_classes.png")
            audit = {"same_rgb_input": True, "guidance_change": "class index cyclic shift by 1; control only",
                     "actual_entropy_payload_changed": read_package(folder / "joint.svic")[1] != read_package(folder / "altered_guidance.svic")[1],
                     "receiver_class_changed_fraction": float(np.mean(direct[valid] != altered_classes[valid])),
                     "original_payload_sha256": read_package(folder / "joint.svic")[0]["payload_sha256"],
                     "altered_payload_sha256": read_package(folder / "altered_guidance.svic")[0]["payload_sha256"],
                     "note": "证明发送语义影响码流与类别解码，不证明比普通压缩更省码量"}
        if i == 0 or (i+1) % 50 == 0 or i+1 == len(data):
            print(f"{i+1}/{len(data)}: joint={encoded['file_bytes']} bytes", flush=True)
    with (args.output / "per_image.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"sample_count": len(data), "split": "validation", "selection": "all" if len(data) == len(metadata["splits"]["validation"]) else "fixed evenly spaced subset",
               "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": codec.digest,
               "training_epoch": codec.state["epoch"], "manifest_sha256": manifest_sha,
               "protocol": "256 long-edge center pad; GT metrics and pixel PSNR ignore padding 255; bpp counts complete 256-square file including header",
               "semantics": {key: metrics_from_confusion(matrix.numpy(), metadata["classes"]) for key, matrix in matrices.items()},
               "rgb_psnr_valid_pixels": {key: 10*math.log10(255**2/(value/(3*total_valid))) for key, value in errors.items()},
               "mean": {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "id"},
               "semantic_causality_audit": audit,
               "notes": ["真实 rANS CPU 编解码；接收端语义直接由 latent 解码头输出。",
                         "验证集也用于挑选训练轮次；这不是独立测试集结论。",
                         "普通 q3 为通用预训练参考，未做同数据同预算微调，不能作为严格消融。",
                         "本评估不发送 TCP 文件；限速传输另行实测。", "两端预先部署相同权重，模型大小不计入单张文件。",
                         "语义预览灰色区域是忽略的补边；码流与原始类别索引文件保留完整输出。"]}
    write_json(args.output / "summary.json", summary)
    preview(samples, palette, args.output / "comparison.png")
    print({"direct_miou": summary["semantics"]["joint_direct"]["miou"],
           "sender_miou": summary["semantics"]["sender"]["miou"], "mean": summary["mean"], "audit": audit}, flush=True)


if __name__ == "__main__":
    main()
