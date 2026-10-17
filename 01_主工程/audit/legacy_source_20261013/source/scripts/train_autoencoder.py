"""视频帧重建入门实验，尚未包含量化、熵编码和语义监督。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from tqdm import tqdm

from dataset_manifest import load_manifest


class FrameDataset(Dataset):
    def __init__(self, files: list[Path], size: int, cache: bool = False) -> None:
        self.files = files
        self.size = size
        self.cached = [self.read_image(i) for i in range(len(files))] if cache else None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = self.cached[index] if self.cached is not None else self.read_image(index)
        return image.to(torch.float32).div_(255)

    def read_image(self, index: int) -> torch.Tensor:
        with Image.open(self.files[index]) as image:
            image = image.convert("RGB")
            # 保持比例后取中心区域，不把宽屏画面拉伸成正方形。
            image = TF.resize(image, self.size, InterpolationMode.BILINEAR)
            return TF.pil_to_tensor(TF.center_crop(image, [self.size, self.size]))


class ConvAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, 3, 1, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def split_files(files: list[Path], root: Path, seed: int) -> tuple[list[Path], list[Path]]:
    # 一级目录代表独立场景；同一视频的不同版本必须放在同一组内。
    groups: dict[str, list[Path]] = {}
    for file in files:
        parts = file.relative_to(root).parts
        group = parts[0] if len(parts) > 1 else "."
        groups.setdefault(group, []).append(file)
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    count = max(1, round(len(names) * 0.2)) if len(names) > 1 else 0
    validation_names = set(names[:count])
    train = [file for name in names if name not in validation_names for file in groups[name]]
    validation = [file for name in names if name in validation_names for file in groups[name]]
    return train, validation


def run_epoch(model, loader, device, optimizer, scaler) -> float:
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, desc="train" if training else "validation", leave=False, disable=not sys.stderr.isatty()):
            batch = batch.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = nn.functional.mse_loss(model(batch), batch)
            if not torch.isfinite(loss):
                raise RuntimeError("损失出现 NaN/Inf，训练已停止。")
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total += loss.detach().item() * batch.size(0)
            count += batch.size(0)
    return total / count


def save_plot(history: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot([row["epoch"] for row in history], [row["train_mse"] for row in history], marker="o", label="Train")
    if history[0]["validation_mse"] is not None:
        ax.plot([row["epoch"] for row in history], [row["validation_mse"] for row in history], marker="o", label="Validation")
    ax.set(xlabel="Epoch", ylabel="MSE", title="Frame reconstruction")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frames-dir", type=Path)
    source.add_argument("--manifest", type=Path, help="prepare_suim.py 生成的固定划分")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--cache-images", action="store_true", help="在内存缓存裁剪后的 uint8 图片，减少反复解码")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--allow-single-video", action="store_true", help="允许单场景调试，不报告验证指标")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.cpu_threads < 1 or args.size < 32 or args.size % 8:
        parser.error("epochs、batch-size、cpu-threads 必须为正；size 必须至少为 32 且是 8 的倍数")
    torch.set_num_threads(args.cpu_threads)
    manifest_metadata = None
    if args.manifest:
        try:
            root, manifest_metadata, splits = load_manifest(args.manifest)
        except (ValueError, KeyError, OSError) as error:
            parser.error(str(error))
        train_files, validation_files = splits["train"], splits["validation"]
    else:
        root = args.frames_dir.resolve()
        files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".png")
        if len(files) < 2:
            parser.error("至少需要 2 张 PNG 帧，请先运行 extract_frames.py")
        train_files, validation_files = split_files(files, root, args.seed)
    if not validation_files and not args.allow_single_video:
        parser.error("至少需要两个独立场景目录；仅调试一段视频时请加 --allow-single-video")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前环境无法使用 CUDA，请运行 environment_check.py 检查")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    output = args.output_dir or Path("runs") / datetime.now().strftime("autoencoder_%Y%m%d-%H%M%S-%f")
    checkpoint_dir = args.checkpoint_dir or output
    checkpoint = checkpoint_dir / "autoencoder_last.pt"
    best_checkpoint = checkpoint_dir / "autoencoder_best.pt"
    if checkpoint.exists() or best_checkpoint.exists():
        parser.error(f"模型文件已存在，请换一个输出目录：{checkpoint_dir}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"输出目录非空，请换一个目录：{output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "frame_reconstruction", "size": args.size, "batch_size": args.batch_size,
        "epochs": args.epochs, "seed": args.seed, "device": str(device),
        "validation_mode": manifest_metadata["validation_mode"] if manifest_metadata else ("held_out_scene" if validation_files else "none_single_video_debug"),
        "train_files": [str(path.relative_to(root)) for path in train_files],
        "validation_files": [str(path.relative_to(root)) for path in validation_files],
        "frames_root": str(root), "torch": torch.__version__,
        "preprocessing": "resize_short_edge_then_center_crop",
        "manifest": str(args.manifest.resolve()) if args.manifest else None,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest() if args.manifest else None,
        "cache_images": args.cache_images, "cpu_threads": args.cpu_threads, "learning_rate": 2e-4,
    }
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"设备={device}；训练帧={len(train_files)}；验证帧={len(validation_files)}", flush=True)
    if args.cache_images:
        print("缓存等比缩放和中心裁剪后的图片到内存……", flush=True)
    train_loader = DataLoader(FrameDataset(train_files, args.size, args.cache_images), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(FrameDataset(validation_files, args.size, args.cache_images), batch_size=args.batch_size, num_workers=0, pin_memory=device.type == "cuda") if validation_files else None
    if not validation_files:
        print("单场景调试：无独立验证集，不能据此判断泛化能力。", flush=True)
    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start = time.perf_counter()
    history = []
    best_loss, best_epoch = float("inf"), None
    with (output / "loss.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_mse", "validation_mse"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, device, optimizer, scaler)
            validation_loss = run_epoch(model, validation_loader, device, None, scaler) if validation_loader else None
            row = {"epoch": epoch, "train_mse": train_loss, "validation_mse": validation_loss}
            history.append(row)
            writer.writerow(row)
            handle.flush()
            message = f"epoch {epoch:03d}/{args.epochs}: train={train_loss:.6f}"
            print(message + (f", validation={validation_loss:.6f}" if validation_loss is not None else ""), flush=True)
            pending = checkpoint.with_suffix(".tmp")
            state = {"model": model.state_dict(), "size": args.size, "epoch": epoch, "seed": args.seed,
                     "train_images": len(train_files), "validation_images": len(validation_files),
                     "manifest_sha256": config["manifest_sha256"], "preprocessing": config["preprocessing"]}
            torch.save(state, pending)
            pending.replace(checkpoint)
            if validation_loss is not None and validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch
                pending = best_checkpoint.with_suffix(".tmp")
                torch.save(state, pending)
                pending.replace(best_checkpoint)

    if best_epoch is not None:
        model.load_state_dict(torch.load(best_checkpoint, map_location=device, weights_only=True)["model"])
    model.eval()
    preview_files = (validation_files or train_files)[:4]
    preview_dataset = FrameDataset(preview_files, args.size)
    batch = torch.stack([preview_dataset[i] for i in range(len(preview_files))]).to(device)
    with torch.inference_mode():
        predictions = model(batch)
        # 每行左侧为参考裁剪帧，右侧为模型重建。
        save_image(torch.stack([batch, predictions], dim=1).flatten(0, 1), output / "reconstruction.png", nrow=2)
    save_plot(history, output / "loss.png")
    if device.type == "cuda":
        torch.cuda.synchronize()
    summary = {
        "device": str(device), "epochs": args.epochs, "train_images": len(train_files), "validation_images": len(validation_files),
        "final_train_mse": history[-1]["train_mse"], "final_validation_mse": history[-1]["validation_mse"],
        "best_epoch": best_epoch, "best_validation_mse": best_loss if best_epoch is not None else None,
        "best_checkpoint": str(best_checkpoint.resolve()) if best_epoch is not None else None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_s": round(time.perf_counter() - start, 3),
        "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device.type == "cuda" else None,
        "checkpoint": str(checkpoint.resolve()), "preview_source": "validation" if validation_files else "training",
        "preview_checkpoint": "autoencoder_best.pt" if best_epoch is not None else "autoencoder_last.pt",
        "note": "帧重建实验，未实现语义分割、量化、熵编码或实际压缩码流。",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"完成，结果目录：{output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
