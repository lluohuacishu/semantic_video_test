"""训练 SUIM 八类语义分割，支持从头训练或官方权重迁移学习。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from segmentation import (
    MODEL_NAME, TRANSFER_MODEL_NAME, HR_TRANSFER_MODEL_NAME, FPN_TRANSFER_MODEL_NAME, SEGFORMER_MODEL_NAME,
    PREPROCESSING, SegmentationDataset, confusion_matrix,
    balanced_sampling_probabilities, create_segmentation_model,
    manifest_info, metrics_from_confusion, save_comparison, segmentation_loss, write_json,
)


def run_epoch(model, loader, device, weights, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total, count = 0., 0
    confusion = torch.zeros((8, 8), dtype=torch.int64, device=device)
    memory_format = torch.channels_last if device.type == "cuda" and hasattr(model, "network") else torch.contiguous_format
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images = images.to(device, non_blocking=True, memory_format=memory_format)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = segmentation_loss(logits, targets, weights)
            if not torch.isfinite(loss):
                raise RuntimeError("分割损失出现 NaN/Inf，已停止")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                confusion += confusion_matrix(logits.argmax(1), targets)
            total += loss.detach().item() * len(images)
            count += len(images)
    return total / count, confusion.cpu().numpy()


def save_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="Train")
    axes[0].plot(epochs, [r["validation_loss"] for r in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="0.7 weighted CE + 0.3 Dice loss")
    axes[0].legend()
    axes[1].plot(epochs, [100*r["validation_miou"] for r in history], label="Validation mIoU")
    axes[1].set(xlabel="Epoch", ylabel="mIoU (%)")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/SUIM/manifest.json"))
    parser.add_argument("--model", choices=("tiny-unet", "lraspp", "lraspp-hr4", "lraspp-fpn", "segformer-b0"), default="tiny-unet",
                        help="lraspp-hr4 使用 1/4 低层特征；lraspp-fpn 融合 1/4、1/8、1/16 三个尺度")
    parser.add_argument("--pretrained", type=Path, help="LR-ASPP 预训练文件，或 SegFormer-B0 本地预训练目录；未提供则随机初始化")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--backbone-lr-factor", type=float, default=0.1)
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--augmentation", choices=("basic", "strong"), default="basic")
    parser.add_argument("--patience", type=int, default=12, help="验证 mIoU 连续多少轮未改善后停止；0 表示禁用")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--cache-images", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 0 or args.cpu_threads < 1:
        parser.error("轮数、批量、线程数必须为正；patience 不能为负")
    if args.size < 32 or args.size % 8 or args.base_channels < 8 or args.base_channels % 8:
        parser.error("size 必须是至少 32 的 8 倍数；base-channels 必须是至少 8 的 8 倍数")
    if not 0 < args.learning_rate < 1:
        parser.error("learning-rate 必须在 0 和 1 之间")
    if not 0 < args.backbone_lr_factor <= 1:
        parser.error("backbone-lr-factor 必须在 (0,1] 内")
    if args.model == "segformer-b0" and (args.size < 64 or args.size % 32):
        parser.error("SegFormer 的 size 必须是至少 64 的 32 倍数")
    if args.pretrained:
        if args.model == "segformer-b0":
            if not args.pretrained.is_dir() or not (args.pretrained / "model.safetensors").is_file():
                parser.error("SegFormer 的 pretrained 必须指向已准备的本地权重目录")
        elif args.model not in ("lraspp", "lraspp-hr4", "lraspp-fpn") or not args.pretrained.is_file():
            parser.error("LR-ASPP 的 pretrained 必须是存在的权重文件")
    model_name = {"lraspp": TRANSFER_MODEL_NAME, "lraspp-hr4": HR_TRANSFER_MODEL_NAME,
                  "lraspp-fpn": FPN_TRANSFER_MODEL_NAME, "segformer-b0": SEGFORMER_MODEL_NAME}.get(args.model, MODEL_NAME)
    is_transfer = args.model in ("lraspp", "lraspp-hr4", "lraspp-fpn", "segformer-b0")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("无法使用 CUDA，请先运行 run_check.cmd")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    root, metadata, digest = manifest_info(args.manifest)
    prefix = "suim_transfer_fpn_" if args.model == "lraspp-fpn" else ("suim_transfer_hr4_" if args.model == "lraspp-hr4" else ("suim_transfer_" if is_transfer else "suim_segmentation_"))
    if args.model == "segformer-b0":
        prefix = "suim_segformer_b0_"
    output = args.output_dir or Path("runs") / datetime.now().strftime(prefix + "%Y%m%d-%H%M%S-%f")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"输出目录非空，不覆盖：{output}")
    total_start = time.perf_counter()
    print(f"设备={device}；训练={len(metadata['splits']['train'])}；验证={len(metadata['splits']['validation'])}；size={args.size}", flush=True)
    print("读取图片、配对标签，统计训练集类别分布……", flush=True)
    train_data = SegmentationDataset(root, metadata["splits"]["train"], args.size, augment=True, cache=args.cache_images, augmentation=args.augmentation)
    validation_data = SegmentationDataset(root, metadata["splits"]["validation"], args.size, cache=args.cache_images)
    histogram = train_data.class_histogram()
    if (histogram == 0).any():
        parser.error("当前训练预处理使某一类别没有有效像素，请检查数据和输入尺寸")
    frequency = histogram.float() / histogram.sum()
    weights = frequency.rsqrt()
    weights = (weights / weights.mean()).clamp(0.25, 4.0)
    probabilities = balanced_sampling_probabilities(metadata["splits"]["train"]) if args.balanced_sampling else None
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "supervised_semantic_segmentation", "model_name": model_name,
        "size": args.size, "base_channels": args.base_channels, "num_classes": 8,
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "patience": args.patience, "seed": args.seed, "device": str(device), "torch": str(torch.__version__),
        "manifest": str(args.manifest.resolve()), "manifest_sha256": digest, "classes": metadata["classes"],
        "preprocessing": PREPROCESSING, "validation_mode": metadata["validation_mode"],
        "train_files": [r["image"] for r in metadata["splits"]["train"]],
        "validation_files": [r["image"] for r in metadata["splits"]["validation"]],
        "training_class_pixels": histogram.tolist(), "class_weights": weights.tolist(),
        "class_weights_method": "训练集缩放后的有效像素频率倒数平方根，归一为均值 1，再限制到 [0.25,4]",
        "loss": "0.7 weighted cross entropy + 0.3 soft Dice over batch-present target classes; ignore_index=255",
        "augmentation": args.augmentation,
        "augmentation_definition": "basic: paired horizontal flip, brightness/saturation U[0.9,1.1]; strong: also rotation +/-8 degrees, zoom U[1,1.2], translation +/-2%, brightness/saturation U[0.8,1.2]",
        "balanced_sampling": args.balanced_sampling,
        "sampling_probabilities": probabilities.tolist() if probabilities is not None else None,
        "sampling_definition": "50% uniform + 50% class-balanced by training-image presence at >=0.5% labeled area; replacement, samples per epoch equal training set size",
        "optimizer": "AdamW, weight_decay=0.0001", "scheduler": "cosine multiplier from 1 to 0.05, preserves parameter-group LR ratios",
        "gradient_clip_norm": 1.0, "cache_images": args.cache_images, "cpu_threads": args.cpu_threads,
        "backbone_lr_factor": args.backbone_lr_factor if is_transfer else None,
        "pretrained_weights": str(args.pretrained.resolve()) if args.pretrained else None,
        "pretrained_sha256": hashlib.sha256((args.pretrained / "model.safetensors" if args.model == "segformer-b0" else args.pretrained).read_bytes()).hexdigest() if args.pretrained else None,
    }
    write_json(output / "config.json", config)
    sampler = WeightedRandomSampler(probabilities, len(train_data), replacement=True) if probabilities is not None else None
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=0, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = create_segmentation_model(model_name, args.base_channels, args.pretrained).to(device)
    if is_transfer and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    weights = weights.to(device)
    parameters = model.parameters()
    if args.model == "segformer-b0":
        parameters = [{"params": model.network.segformer.parameters(), "lr": args.learning_rate*args.backbone_lr_factor},
                      {"params": model.network.decode_head.parameters(), "lr": args.learning_rate}]
        import transformers
        config["transformers"] = transformers.__version__
        config["model_config"] = model.network.config.to_dict()
        config["label_handling"] = "SUIM 标签保持 0..7；背景 0 参与训练，255 只用于补边；不做 ADE20K reduce_labels"
        config["transfer_initialization"] = "ADE20K 预训练 B0；仅最终 1x1 分类层随机初始化，主干和解码器均微调"
        write_json(output / "config.json", config)
    elif is_transfer:
        parameters = [{"params": model.network.backbone.parameters(), "lr": args.learning_rate*args.backbone_lr_factor},
                      {"params": model.network.classifier.parameters(), "lr": args.learning_rate}]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    # 按比例下降，保持特征层与新分类层之间的学习率比例。
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.05 + 0.95 * (1 + math.cos(math.pi * epoch / args.epochs)) / 2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_miou, best_epoch, stale = -1., 0, 0
    history = []
    fit_start = time.perf_counter()
    with (output / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "validation_loss", "validation_miou", "validation_pixel_accuracy", "learning_rate", "epoch_s"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            lr = optimizer.param_groups[-1]["lr"]
            train_loss, _ = run_epoch(model, train_loader, device, weights, optimizer, scaler)
            validation_loss, matrix = run_epoch(model, validation_loader, device, weights)
            metrics = metrics_from_confusion(matrix, metadata["classes"])
            row = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
                   "validation_miou": metrics["miou"], "validation_pixel_accuracy": metrics["pixel_accuracy"],
                   "learning_rate": lr, "epoch_s": round(time.perf_counter()-epoch_start, 3)}
            history.append(row)
            writer.writerow(row)
            handle.flush()
            scheduler.step()
            state = {
                "model": model.state_dict(), "model_name": model_name, "base_channels": args.base_channels,
                "size": args.size, "epoch": epoch, "manifest_sha256": digest, "preprocessing": PREPROCESSING,
                "classes": metadata["classes"], "seed": args.seed, "validation_miou": metrics["miou"],
            }
            pending = output / "segmentation_last.tmp"
            torch.save(state, pending)
            pending.replace(output / "segmentation_last.pt")
            if metrics["miou"] > best_miou:
                best_miou, best_epoch, stale = metrics["miou"], epoch, 0
                pending = output / "segmentation_best.tmp"
                torch.save(state, pending)
                pending.replace(output / "segmentation_best.pt")
                write_json(output / "best_validation.json", {"epoch": epoch, **metrics, "confusion_matrix": matrix.tolist()})
            else:
                stale += 1
            print(f"epoch {epoch:03d}/{args.epochs}: loss={train_loss:.4f}, val_loss={validation_loss:.4f}, val_mIoU={100*metrics['miou']:.2f}%, {row['epoch_s']:.1f}s", flush=True)
            if args.patience and stale >= args.patience:
                print(f"验证 mIoU 连续 {args.patience} 轮未改善，提前停止。", flush=True)
                break
    fit_elapsed = time.perf_counter() - fit_start
    model.load_state_dict(torch.load(output / "segmentation_best.pt", map_location=device, weights_only=True)["model"])
    model.eval()
    samples = []
    indices = [round(i*(len(validation_data)-1)/3) for i in range(4)] if len(validation_data) >= 4 else list(range(len(validation_data)))
    with torch.inference_mode():
        for index in indices:
            image, target = validation_data[index]
            prediction = model(image.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy().astype("uint8")
            prediction[target.numpy() == 255] = 255
            samples.append((Path(validation_data.records[index]["image"]).stem, image, target.numpy(), prediction))
    save_comparison(samples, metadata["classes"], output / "validation_comparison.png")
    save_curves(history, output / "learning_curve.png")
    summary = {
        "device": str(device), "model_name": model_name, "epochs_completed": len(history), "epochs_requested": args.epochs,
        "best_epoch": best_epoch, "best_validation_miou": best_miou,
        "parameters": sum(p.numel() for p in model.parameters()), "size": args.size, "batch_size": args.batch_size,
        "train_images": len(train_data), "validation_images": len(validation_data), "test_images_used_for_training": 0,
        "fit_elapsed_s": round(fit_elapsed, 3), "total_elapsed_s": round(time.perf_counter()-total_start, 3),
        "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated()/1024**2, 1) if device.type == "cuda" else None,
        "peak_cuda_reserved_mib": round(torch.cuda.max_memory_reserved()/1024**2, 1) if device.type == "cuda" else None,
        "best_checkpoint": str((output / "segmentation_best.pt").resolve()),
        "note": "这是八类图片语义分割；没有实现视频时序模型、码率控制或语义通信。最佳模型只根据验证 mIoU 选择。",
    }
    write_json(output / "summary.json", summary)
    print(f"完成：{output.resolve()}；最佳 epoch={best_epoch}, val_mIoU={100*best_miou:.2f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
