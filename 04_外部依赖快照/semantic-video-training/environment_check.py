"""检查依赖、CUDA 前后向传播、优化器更新、权重保存和图像读写。"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torch import nn

from scripts.train_autoencoder import ConvAutoencoder


def check(output: Path, video: Path | None, allow_cpu: bool) -> dict:
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("CUDA 不可用，GPU 自检失败。若只测试 CPU，请加 --allow-cpu。")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "python": sys.version.split()[0], "executable": sys.executable,
        "platform": platform.platform(), "device": str(device),
        "versions": {name: version(name) for name in [
            "torch", "torchvision", "numpy", "Pillow", "opencv-python",
            "matplotlib", "tqdm", "PyYAML", "scikit-image",
        ]},
        "cuda_build": torch.version.cuda,
    }
    torch.manual_seed(20260910)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        report.update(gpu=props.name, gpu_memory_gib=round(props.total_memory / 1024**3, 2))
        torch.cuda.reset_peak_memory_stats(device)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    # 固定小批次仅检查能否学习，不作为水下数据集实验。
    y, x = torch.meshgrid(torch.linspace(0, 1, 256, device=device), torch.linspace(0, 1, 256, device=device), indexing="ij")
    batch = torch.stack([x, y, (x + y) * 0.5])[None].repeat(2, 1, 1, 1)
    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    before = next(model.parameters()).detach().clone()
    losses = []
    start = time.perf_counter()
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            prediction = model(batch)
            loss = nn.functional.mse_loss(prediction, batch)
        if not torch.isfinite(loss):
            raise RuntimeError("前向传播出现 NaN/Inf")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if not all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()):
            raise RuntimeError("反向传播产生了无效梯度")
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.detach().item())
    if torch.equal(before, next(model.parameters()).detach()):
        raise RuntimeError("优化器没有更新模型参数")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report["training_check"] = {
        "steps": 8, "batch_size": 2, "size": 256, "losses": losses,
        "parameters_updated": True, "finite_gradients": True,
        "elapsed_s": round(time.perf_counter() - start, 3),
        "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1) if device.type == "cuda" else None,
    }

    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=torch.float32, device=device)
    kept = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8], device=device), 0.5)
    if kept.tolist() != [0]:
        raise RuntimeError("Torchvision NMS 测试失败")
    report["torchvision_native_ops"] = "passed"
    with tempfile.TemporaryDirectory(prefix="semantic-env-") as directory:
        temporary = Path(directory)
        checkpoint = temporary / "self_check.pt"
        torch.save(model.state_dict(), checkpoint)
        restored = ConvAutoencoder().to(device)
        restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        restored.eval()
        with torch.inference_mode():
            torch.testing.assert_close(model(batch), restored(batch))
        report["checkpoint_reload"] = "passed"
        image = np.full((64, 96, 3), (30, 120, 210), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("OpenCV PNG 编码失败")
        image_path = temporary / "中文路径图像.png"
        encoded.tofile(image_path)
        decoded = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if not np.array_equal(image, decoded):
            raise RuntimeError("OpenCV 图像读写不一致")
        report["unicode_image_io"] = "passed"
    if video:
        capture = cv2.VideoCapture(str(video.resolve()))
        try:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"无法读取视频首帧：{video}")
            report["video_read"] = {"source": str(video.resolve()), "width": frame.shape[1], "height": frame.shape[0], "status": "passed"}
        finally:
            capture.release()
    report["status"] = "passed"
    report["note"] = "环境检查和少量参数更新，不代表正式训练质量或实时性能。"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/environment_check.json"))
    parser.add_argument("--video", type=Path, help="可选：验证一个本地视频的首帧")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    # 清除旧的成功状态，异常退出时不会留下本次已通过的假象。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "running"}) + "\n", encoding="utf-8")
    try:
        report = check(args.output, args.video, args.allow_cpu)
    except Exception as error:
        args.output.write_text(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    stats = report["training_check"]
    print(f"PASS: {report['device']}, 8 steps, loss {stats['losses'][0]:.6f} -> {stats['losses'][-1]:.6f}")
    print(f"报告：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
