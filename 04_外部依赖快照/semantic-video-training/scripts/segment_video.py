"""使用训练好的 SUIM 分割模型逐帧处理视频。

脚本只负责本地视频推理和结果记录，不模拟信道，也不改变原视频的帧率。
默认会保存彩色类别视频和原图叠加视频；加上 --save-masks 可保存每帧的
0~7 类别编号 PNG，便于后续做重要区域压缩实验。
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from segmentation import (
    color_mask,
    fit_image,
    load_segmentation_model,
    palette_from_classes,
    write_json,
)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def codec_extension(codec: str) -> str:
    return ".mp4" if codec.lower() in {"mp4v", "avc1", "h264"} else ".avi"


def open_writer(path: Path, codec: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height), True)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            f"无法打开视频输出编码器 {codec!r}：{path}\n"
            "可尝试用 --codec MJPG 输出 AVI，或检查 OpenCV/FFmpeg 安装。"
        )
    return writer


def predict_frame(model, frame_rgb: np.ndarray, size: int, device: torch.device):
    """返回原尺寸类别图，以及预处理、前向和后处理耗时（毫秒）。"""
    height, width = frame_rgb.shape[:2]
    image = Image.fromarray(frame_rgb, mode="RGB")

    pre_start = time.perf_counter()
    tensor, _, geometry = fit_image(image, size)
    batch = tensor.unsqueeze(0).float().div(255).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    pre_ms = (time.perf_counter() - pre_start) * 1000

    model_start = time.perf_counter()
    with torch.inference_mode():
        logits = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_ms = (time.perf_counter() - model_start) * 1000

    post_start = time.perf_counter()
    g = geometry
    logits = logits[:, :, g["top"]:g["top"] + g["scaled_height"], g["left"]:g["left"] + g["scaled_width"]]
    logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
    prediction = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    post_ms = (time.perf_counter() - post_start) * 1000
    return prediction, {"preprocess_ms": pre_ms, "inference_ms": model_ms, "postprocess_ms": post_ms}


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def stabilize_prediction(history: deque[np.ndarray], current: np.ndarray, class_count: int) -> np.ndarray:
    """对最近几帧的类别图做因果投票。

    三帧是实时场景最常用的设置，使用专门的逐像素多数投票，避免创建
    class_count × H × W 的临时数组；五帧再使用带近期权重的通用实现。
    """
    if len(history) <= 1:
        return current
    if len(history) == 3:
        older, middle, recent = history
        same_older_middle = older == middle
        same_older_recent = older == recent
        same_middle_recent = middle == recent
        # 两票相同时取多数；三票都不同时取最新结果，减少突发目标的滞后。
        return np.where(
            same_older_middle,
            older,
            np.where(same_older_recent, older, np.where(same_middle_recent, middle, recent)),
        ).astype(np.uint8)
    height, width = current.shape
    scores = np.zeros((class_count, height, width), dtype=np.float32)
    # 越新的帧权重越大，避免投票平局时长期偏向低编号类别。
    for weight, mask in enumerate(history, start=1):
        for class_index in range(class_count):
            scores[class_index] += weight * (mask == class_index)
    return scores.argmax(axis=0).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="分割模型 segmentation_best.pt")
    parser.add_argument("--input", type=Path, required=True, help="AVI、MP4 等 OpenCV 可读取的视频")
    parser.add_argument("--output-dir", type=Path, help="结果目录；必须为空或不存在")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--size", type=int, help="模型输入边长，默认使用 checkpoint 中的训练尺寸")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理帧数，0 表示处理到视频结束")
    parser.add_argument("--save-masks", action="store_true", help="额外保存每帧 0~7 类别编号 PNG")
    parser.add_argument("--temporal-window", type=int, default=1, help="因果投票窗口，必须是 1、3 或 5；1 表示关闭稳定")
    parser.add_argument("--overlay-alpha", type=float, default=0.45, help="类别颜色叠加透明度，范围 0~1")
    parser.add_argument("--codec", default="mp4v", help="四字符 OpenCV 编码器，默认 mp4v；可用 MJPG 输出 AVI")
    args = parser.parse_args()

    if len(args.codec) != 4:
        parser.error("--codec 必须是 4 个字符，例如 mp4v 或 MJPG")
    if args.max_frames < 0:
        parser.error("--max-frames 不能小于 0")
    if args.temporal_window not in (1, 3, 5):
        parser.error("--temporal-window 只支持 1、3 或 5")
    if args.size is not None and (args.size < 32 or args.size % 8):
        parser.error("--size 必须是不小于 32 的 8 的倍数")
    if not 0 <= args.overlay_alpha <= 1:
        parser.error("--overlay-alpha 必须在 0~1 之间")
    if not args.checkpoint.is_file():
        parser.error(f"找不到模型文件：{args.checkpoint}")
    if not args.input.is_file():
        parser.error(f"找不到输入视频：{args.input}")

    torch.set_num_threads(4)
    device = choose_device(args.device)
    model, state = load_segmentation_model(args.checkpoint, device)
    size = args.size or int(state["size"])

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开输入视频：{args.input}")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    input_fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("输入视频没有有效的宽高信息")
    if not np.isfinite(input_fps) or input_fps <= 0:
        input_fps = 30.0
        fps_note = "输入视频未提供有效 FPS，输出按 30 FPS 写入"
    else:
        fps_note = None

    output = args.output_dir or Path("runs") / ("video_segmentation_" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        capture.release()
        parser.error(f"输出目录非空：{output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        (output / "masks").mkdir()

    extension = codec_extension(args.codec)
    color_path = output / f"segmentation_color{extension}"
    overlay_path = output / f"segmentation_overlay{extension}"
    color_writer = None
    overlay_writer = None
    try:
        color_writer = open_writer(color_path, args.codec, input_fps, width, height)
        overlay_writer = open_writer(overlay_path, args.codec, input_fps, width, height)
    except Exception:
        if color_writer is not None:
            color_writer.release()
        if overlay_writer is not None:
            overlay_writer.release()
        capture.release()
        raise
    palette = palette_from_classes(state["classes"])

    frame_rows: list[dict] = []
    inference_values: list[float] = []
    total_values: list[float] = []
    total_class_pixels = np.zeros(len(state["classes"]), dtype=np.int64)
    prediction_history: deque[np.ndarray] = deque(maxlen=args.temporal_window)
    previous_prediction: np.ndarray | None = None
    previous_raw_prediction: np.ndarray | None = None
    changed_values: list[float] = []
    raw_changed_values: list[float] = []
    processed = 0
    read_ms_values: list[float] = []
    write_ms_values: list[float] = []
    started = time.perf_counter()
    try:
        while args.max_frames == 0 or processed < args.max_frames:
            read_start = time.perf_counter()
            ok, frame_bgr = capture.read()
            read_ms = (time.perf_counter() - read_start) * 1000
            if not ok:
                break
            read_ms_values.append(read_ms)
            frame_start = time.perf_counter()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            raw_prediction, parts = predict_frame(model, frame_rgb, size, device)
            prediction_history.append(raw_prediction)
            prediction = stabilize_prediction(prediction_history, raw_prediction, len(state["classes"]))
            colors_rgb = color_mask(prediction, palette)
            overlay_rgb = cv2.addWeighted(frame_rgb, 1.0 - args.overlay_alpha, colors_rgb, args.overlay_alpha, 0)
            colors_bgr = cv2.cvtColor(colors_rgb, cv2.COLOR_RGB2BGR)
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

            write_start = time.perf_counter()
            color_writer.write(colors_bgr)
            overlay_writer.write(overlay_bgr)
            if args.save_masks:
                mask_path = output / "masks" / f"frame_{processed:06d}.png"
                if not cv2.imwrite(str(mask_path), prediction):
                    raise RuntimeError(f"无法保存类别图：{mask_path}")
            write_ms = (time.perf_counter() - write_start) * 1000
            write_ms_values.append(write_ms)

            total_ms = (time.perf_counter() - frame_start) * 1000
            total_values.append(total_ms)
            inference_values.append(parts["inference_ms"])
            total_class_pixels += np.bincount(prediction.ravel(), minlength=len(state["classes"]))
            changed_fraction = None
            if previous_prediction is not None:
                changed_fraction = float(np.mean(prediction != previous_prediction))
                changed_values.append(changed_fraction)
            raw_changed_fraction = None
            if previous_raw_prediction is not None:
                raw_changed_fraction = float(np.mean(raw_prediction != previous_raw_prediction))
                raw_changed_values.append(raw_changed_fraction)
            previous_prediction = prediction
            previous_raw_prediction = raw_prediction
            frame_rows.append({
                "frame_index": processed,
                "source_timestamp_ms": round(processed * 1000 / input_fps, 3),
                "preprocess_ms": round(parts["preprocess_ms"], 3),
                "inference_ms": round(parts["inference_ms"], 3),
                "postprocess_ms": round(parts["postprocess_ms"], 3),
                "write_ms": round(write_ms, 3),
                "total_ms": round(total_ms, 3),
                "raw_changed_fraction": None if raw_changed_fraction is None else round(raw_changed_fraction, 6),
                "changed_fraction": None if changed_fraction is None else round(changed_fraction, 6),
            })
            processed += 1
    finally:
        capture.release()
        if color_writer is not None:
            color_writer.release()
        if overlay_writer is not None:
            overlay_writer.release()
    elapsed = time.perf_counter() - started

    with (output / "frame_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]) if frame_rows else ["frame_index"])
        writer.writeheader()
        writer.writerows(frame_rows)

    fractions = (total_class_pixels / total_class_pixels.sum()).tolist() if total_class_pixels.sum() else [0.0] * len(total_class_pixels)
    processing_fps = processed / elapsed if elapsed > 0 else None
    summary = {
        "input": str(args.input.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "model_name": state["model_name"],
        "model_input_size": size,
        "input_width": width,
        "input_height": height,
        "input_fps": input_fps,
        "reported_frame_count": reported_frames,
        "processed_frames": processed,
        "elapsed_seconds": elapsed,
        "processing_fps": processing_fps,
        "realtime_ratio": (processing_fps / input_fps) if processing_fps and input_fps else None,
        "realtime_capable_on_this_run": bool(processing_fps and processing_fps >= input_fps),
        "temporal_window": args.temporal_window,
        "timing_ms": {
            "preprocess_mean": float(np.mean([r["preprocess_ms"] for r in frame_rows])) if frame_rows else None,
            "inference_mean": float(np.mean(inference_values)) if inference_values else None,
            "inference_p50": percentile(inference_values, 50),
            "inference_p95": percentile(inference_values, 95),
            "total_mean": float(np.mean(total_values)) if total_values else None,
            "total_p50": percentile(total_values, 50),
            "total_p95": percentile(total_values, 95),
            "total_max": float(np.max(total_values)) if total_values else None,
            "read_mean": float(np.mean(read_ms_values)) if read_ms_values else None,
            "write_mean": float(np.mean(write_ms_values)) if write_ms_values else None,
        },
        "temporal_stability": {
            "mean_changed_fraction": float(np.mean(changed_values)) if changed_values else None,
            "p95_changed_fraction": percentile(changed_values, 95),
            "raw_mean_changed_fraction": float(np.mean(raw_changed_values)) if raw_changed_values else None,
            "raw_p95_changed_fraction": percentile(raw_changed_values, 95),
            "definition": "相邻输出类别图中类别发生变化的像素比例；raw 是模型原始预测，输出值是保存结果。没有真值时只能作为抖动参考。",
        },
        "classes": [
            {**row, "predicted_pixels": int(total_class_pixels[i]), "predicted_fraction": float(fractions[i])}
            for i, row in enumerate(state["classes"])
        ],
        "outputs": {
            "color_video": str(color_path.resolve()),
            "overlay_video": str(overlay_path.resolve()),
            "frame_metrics": str((output / "frame_metrics.csv").resolve()),
            "masks": str((output / "masks").resolve()) if args.save_masks else None,
        },
        "codec": args.codec,
        "fps_note": fps_note,
        "note": "模型仍逐帧独立推理；temporal_window 只对类别图做因果投票，不是时序神经网络。结果不含网络传输、编码码率控制或丢包恢复。processing_fps 是本机从读帧到写结果的端到端处理速度。",
    }
    write_json(output / "summary.json", summary)
    print(f"输出目录：{output.resolve()}")
    print(f"处理帧数：{processed}，输入 FPS：{input_fps:.2f}，端到端处理 FPS：{processing_fps:.2f}" if processing_fps else f"处理帧数：{processed}")
    print(f"彩色视频：{color_path.resolve()}")
    print(f"叠加视频：{overlay_path.resolve()}")
    print(f"汇总：{(output / 'summary.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
