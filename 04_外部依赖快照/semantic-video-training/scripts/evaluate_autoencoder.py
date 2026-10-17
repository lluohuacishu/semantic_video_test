"""在固定的验证集或测试集上评估模型，保存 RGB 重建图及逐图指标。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset_manifest import load_manifest
from train_autoencoder import ConvAutoencoder, FrameDataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size 必须为正")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前环境无法使用 CUDA")
    torch.set_num_threads(4)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    root, metadata, splits = load_manifest(args.manifest)
    files = splits[args.split]
    if not files:
        parser.error(f"{args.split} 没有图片")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    manifest_digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if state.get("manifest_sha256") != manifest_digest:
        parser.error("此模型训练时的数据清单与当前清单不一致，停止评估以免混淆划分")
    size = state["size"]
    model = ConvAutoencoder().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    output = args.output_dir or args.checkpoint.parent / ("evaluation_" + args.split)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"输出目录已存在且非空：{output}")
    decoded_dir = output / "reconstructed"
    reference_dir = output / "reference_crops"
    decoded_dir.mkdir(parents=True)
    reference_dir.mkdir()
    loader = DataLoader(FrameDataset(files, size), batch_size=args.batch_size, shuffle=False, num_workers=0)
    preview_indices = set(np.linspace(0, len(files) - 1, min(6, len(files)), dtype=int).tolist())
    preview, preview_sources, rows = [], [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            predictions = model(batch.to(device)).clamp(0, 1).cpu()
            for reference, prediction in zip(batch, predictions):
                index = len(rows)
                relative = files[index].relative_to(root).as_posix()
                ref = reference.permute(1, 2, 0).numpy()
                # 指标使用实际保存的 8-bit PNG 像素，使重建文件可以复核结果。
                decoded = (prediction.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
                reconstructed = decoded.astype(np.float32) / 255
                mse = float(np.mean((ref - reconstructed) ** 2, dtype=np.float64))
                psnr = -10 * math.log10(max(mse, 1e-12))
                ssim = float(structural_similarity(ref, reconstructed, data_range=1.0, channel_axis=-1))
                if not all(math.isfinite(value) for value in (mse, psnr, ssim)):
                    raise RuntimeError(f"指标出现 NaN/Inf：{relative}")
                filename = f"{index:04d}_{files[index].stem}.png"
                Image.fromarray(decoded).save(decoded_dir / filename)
                Image.fromarray((ref * 255).round().astype(np.uint8)).save(reference_dir / filename)
                rows.append({"image": relative, "mse": mse, "psnr_db": psnr, "ssim": ssim,
                             "reconstruction": "reconstructed/" + filename, "reference": "reference_crops/" + filename})
                if index in preview_indices:
                    preview.extend([reference, torch.from_numpy(reconstructed).permute(2, 0, 1)])
                    preview_sources.append(relative)
    with (output / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_image(torch.stack(preview), output / "comparison.png", nrow=2)
    summary = {
        "split": args.split, "images": len(rows), "size": size, "checkpoint_epoch": state["epoch"],
        "checkpoint": str(args.checkpoint.resolve()), "manifest_sha256": manifest_digest,
        "device": str(device), "precision": "float32", "preprocessing": "resize_short_edge_then_center_crop",
        "mean_mse": float(np.mean([r["mse"] for r in rows])),
        "mean_psnr_db": float(np.mean([r["psnr_db"] for r in rows])),
        "mean_ssim": float(np.mean([r["ssim"] for r in rows])),
        "psnr_db_min": min(r["psnr_db"] for r in rows), "psnr_db_max": max(r["psnr_db"] for r in rows),
        "elapsed_s": round(time.perf_counter() - start, 3), "preview_sources": preview_sources,
        "metric_definition": "逐图计算后等权平均。RGB [0,1]；PSNR 最高截断为 120dB；SSIM 使用 scikit-image 默认 7x7 窗和 channel_axis=-1。按实际保存的 8-bit PNG 计算。",
        "note": f"测试名单未参与训练或选择轮次。只评价 {size}×{size} 中心裁剪图的重建；没有码率、视频时序或语义分割指标。",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("split", "images", "checkpoint_epoch", "mean_psnr_db", "mean_ssim", "elapsed_s")}, ensure_ascii=False, indent=2), flush=True)
    print(f"评估结果：{output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
