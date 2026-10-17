"""用分割结果做重要区域保护的语义感知压缩原型。

脚本同时写出普通 OpenCV 视频和“前景保留、背景降采样”的视频，便于比较
文件大小与画面质量。它是语义感知压缩实验，不是神经压缩或语义通信码流。
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity

from segment_video import (
    choose_device,
    codec_extension,
    load_segmentation_model,
    open_writer,
    predict_frame,
    stabilize_prediction,
)
from segmentation import write_json


def parse_class_indices(value: str, class_count: int) -> list[int]:
    try:
        result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as error:
        raise ValueError("--important-classes 必须是逗号分隔的整数，例如 1,3,4,6") from error
    if not result or any(item < 0 or item >= class_count for item in result):
        raise ValueError(f"--important-classes 中的类别必须在 0~{class_count - 1} 之间，且至少有一个")
    return result


def protect_foreground(frame: np.ndarray, prediction: np.ndarray, important: list[int], background_scale: float, dilate: int) -> np.ndarray:
    foreground = np.isin(prediction, important)
    if dilate:
        kernel_size = dilate * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        foreground = cv2.dilate(foreground.astype(np.uint8), kernel, iterations=1).astype(bool)
    if background_scale >= 0.999:
        degraded = frame.copy()
    else:
        height, width = frame.shape[:2]
        small_width = max(1, round(width * background_scale))
        small_height = max(1, round(height * background_scale))
        small = cv2.resize(frame, (small_width, small_height), interpolation=cv2.INTER_AREA)
        degraded = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.where(foreground[..., None], frame, degraded).astype(np.uint8)


def psnr_from_mse(mse: float | None) -> float | None:
    if mse is None:
        return None
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10((255.0 * 255.0) / mse))


def evaluate_encoded(source: Path, encoded: Path, masks_dir: Path, important: list[int], fps: float, ssim_sample_step: int = 1) -> dict:
    source_cap = cv2.VideoCapture(str(source))
    encoded_cap = cv2.VideoCapture(str(encoded))
    if not source_cap.isOpened() or not encoded_cap.isOpened():
        source_cap.release()
        encoded_cap.release()
        raise RuntimeError(f"无法重新打开待评估视频：{encoded}")
    frame_count = 0
    overall_mse: list[float] = []
    foreground_mse: list[float] = []
    background_mse: list[float] = []
    ssim_values: list[float] = []
    foreground_pixels = 0
    background_pixels = 0
    try:
        while True:
            source_ok, source_frame = source_cap.read()
            encoded_ok, encoded_frame = encoded_cap.read()
            if not source_ok or not encoded_ok:
                if source_ok != encoded_ok:
                    raise RuntimeError(f"编码结果帧数与源视频不一致：{encoded}")
                break
            if source_frame.shape != encoded_frame.shape:
                raise RuntimeError(f"编码结果分辨率与源视频不一致：{encoded}")
            mask_path = masks_dir / f"frame_{frame_count:06d}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != source_frame.shape[:2]:
                raise RuntimeError(f"类别掩码缺失或尺寸不匹配：{mask_path}")
            foreground = np.isin(mask, important)
            background = ~foreground
            difference = source_frame.astype(np.float32) - encoded_frame.astype(np.float32)
            squared = np.mean(difference * difference, axis=2)
            overall_mse.append(float(np.mean(squared)))
            if foreground.any():
                foreground_mse.append(float(np.mean(squared[foreground])))
                foreground_pixels += int(foreground.sum())
            if background.any():
                background_mse.append(float(np.mean(squared[background])))
                background_pixels += int(background.sum())
            if frame_count % max(1, ssim_sample_step) == 0:
                source_rgb = cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB)
                encoded_rgb = cv2.cvtColor(encoded_frame, cv2.COLOR_BGR2RGB)
                ssim_values.append(float(structural_similarity(source_rgb, encoded_rgb, channel_axis=2, data_range=255)))
            frame_count += 1
    finally:
        source_cap.release()
        encoded_cap.release()
    if not frame_count:
        raise RuntimeError(f"待评估视频没有有效帧：{encoded}")
    duration = frame_count / fps if fps > 0 else None
    size_bytes = encoded.stat().st_size
    return {
        "path": str(encoded.resolve()),
        "size_bytes": size_bytes,
        "duration_seconds": duration,
        "bitrate_kbps_including_container": (size_bytes * 8 / duration / 1000) if duration else None,
        "frames": frame_count,
        "mean_mse": float(np.mean(overall_mse)),
        "mean_psnr_db": psnr_from_mse(float(np.mean(overall_mse))),
        "foreground_mse": float(np.mean(foreground_mse)) if foreground_mse else None,
        "foreground_psnr_db": psnr_from_mse(float(np.mean(foreground_mse))) if foreground_mse else None,
        "background_mse": float(np.mean(background_mse)) if background_mse else None,
        "background_psnr_db": psnr_from_mse(float(np.mean(background_mse))) if background_mse else None,
        "mean_ssim": float(np.mean(ssim_values)) if ssim_values else None,
        "ssim_sample_step": ssim_sample_step,
        "foreground_pixel_fraction": foreground_pixels / max(1, foreground_pixels + background_pixels),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--temporal-window", type=int, default=3, help="分割掩码稳定窗口：1、3 或 5；默认 3")
    parser.add_argument("--important-classes", default="1,2,3,4,5,6,7", help="重要类别编号，默认保护所有非背景类别")
    parser.add_argument("--background-scale", type=float, default=0.5, help="背景降采样比例，0.25、0.5、1.0 等")
    parser.add_argument("--mask-dilate", type=int, default=2, help="重要区域向外膨胀的像素数，保护目标边缘")
    parser.add_argument("--codec", default="mp4v", help="四字符 OpenCV 编码器，默认 mp4v；可用 MJPG 输出 AVI")
    parser.add_argument("--no-save-masks", action="store_true", help="不保存逐帧类别图；评估指标仍需要掩码，因此会跳过区域指标")
    parser.add_argument("--ssim-sample-step", type=int, default=2, help="每隔多少帧计算一次 SSIM，默认每 2 帧")
    args = parser.parse_args()

    if len(args.codec) != 4:
        parser.error("--codec 必须是 4 个字符，例如 mp4v 或 MJPG")
    if args.temporal_window not in (1, 3, 5):
        parser.error("--temporal-window 只支持 1、3 或 5")
    if not 0 < args.background_scale <= 1:
        parser.error("--background-scale 必须大于 0 且不超过 1")
    if args.mask_dilate < 0:
        parser.error("--mask-dilate 不能为负数")
    if args.ssim_sample_step < 1:
        parser.error("--ssim-sample-step 必须至少为 1")
    if not args.checkpoint.is_file():
        parser.error(f"找不到模型文件：{args.checkpoint}")
    if not args.input.is_file():
        parser.error(f"找不到输入视频：{args.input}")

    torch.set_num_threads(4)
    device = choose_device(args.device)
    model, state = load_segmentation_model(args.checkpoint, device)
    important = parse_class_indices(args.important_classes, len(state["classes"]))
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开输入视频：{args.input}")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    input_fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if width <= 0 or height <= 0 or not np.isfinite(input_fps) or input_fps <= 0:
        capture.release()
        raise RuntimeError("输入视频缺少有效的宽高或 FPS 信息")

    output = args.output_dir or Path("runs") / ("semantic_compress_" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        capture.release()
        parser.error(f"输出目录非空：{output}")
    output.mkdir(parents=True, exist_ok=True)
    masks_dir = output / "masks"
    if not args.no_save_masks:
        masks_dir.mkdir()
    extension = codec_extension(args.codec)
    baseline_path = output / f"baseline{extension}"
    semantic_path = output / f"semantic{extension}"
    baseline_writer = None
    semantic_writer = None
    try:
        baseline_writer = open_writer(baseline_path, args.codec, input_fps, width, height)
        semantic_writer = open_writer(semantic_path, args.codec, input_fps, width, height)
    except Exception:
        if baseline_writer is not None:
            baseline_writer.release()
        if semantic_writer is not None:
            semantic_writer.release()
        capture.release()
        raise
    history: deque[np.ndarray] = deque(maxlen=args.temporal_window)
    previous_prediction: np.ndarray | None = None
    changed_values: list[float] = []
    class_pixels = np.zeros(len(state["classes"]), dtype=np.int64)
    frame_rows: list[dict] = []
    processed = 0
    started = time.perf_counter()
    try:
        while True:
            read_started = time.perf_counter()
            ok, frame = capture.read()
            read_ms = (time.perf_counter() - read_started) * 1000
            if not ok:
                break
            frame_started = time.perf_counter()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_prediction, parts = predict_frame(model, frame_rgb, int(state["size"]), device)
            history.append(raw_prediction)
            prediction = stabilize_prediction(history, raw_prediction, len(state["classes"]))
            semantic_frame = protect_foreground(frame, prediction, important, args.background_scale, args.mask_dilate)
            baseline_writer.write(frame)
            semantic_writer.write(semantic_frame)
            if not args.no_save_masks:
                mask_path = masks_dir / f"frame_{processed:06d}.png"
                if not cv2.imwrite(str(mask_path), prediction):
                    raise RuntimeError(f"无法保存类别图：{mask_path}")
            if previous_prediction is not None:
                changed_values.append(float(np.mean(prediction != previous_prediction)))
            previous_prediction = prediction
            class_pixels += np.bincount(prediction.ravel(), minlength=len(state["classes"]))
            frame_rows.append({
                "frame_index": processed,
                "read_ms": round(read_ms, 3),
                "inference_ms": round(parts["inference_ms"], 3),
                "total_ms": round((time.perf_counter() - frame_started) * 1000, 3),
                "changed_fraction": None if not changed_values else round(changed_values[-1], 6),
            })
            processed += 1
    finally:
        capture.release()
        if baseline_writer is not None:
            baseline_writer.release()
        if semantic_writer is not None:
            semantic_writer.release()
    if not processed:
        raise RuntimeError("输入视频没有可处理的帧")

    with (output / "frame_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)

    elapsed = time.perf_counter() - started
    baseline_metrics = evaluate_encoded(args.input, baseline_path, masks_dir, important, input_fps, args.ssim_sample_step) if not args.no_save_masks else None
    semantic_metrics = evaluate_encoded(args.input, semantic_path, masks_dir, important, input_fps, args.ssim_sample_step) if not args.no_save_masks else None
    # 未保存掩码时仍提供文件大小和帧数，避免 --no-save-masks 变成不可用模式。
    if args.no_save_masks:
        baseline_metrics = {"path": str(baseline_path.resolve()), "size_bytes": baseline_path.stat().st_size}
        semantic_metrics = {"path": str(semantic_path.resolve()), "size_bytes": semantic_path.stat().st_size}
    total_pixels = max(1, int(class_pixels.sum()))
    summary = {
        "input": str(args.input.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "model_name": state["model_name"],
        "model_input_size": int(state["size"]),
        "input_width": width,
        "input_height": height,
        "input_fps": input_fps,
        "reported_frame_count": reported_frames,
        "processed_frames": processed,
        "processing_fps": processed / max(elapsed, 1e-9),
        "elapsed_seconds": elapsed,
        "temporal_window": args.temporal_window,
        "important_classes": important,
        "important_class_names": [state["classes"][i]["name_zh"] for i in important],
        "background_scale": args.background_scale,
        "mask_dilate": args.mask_dilate,
        "predicted_class_fractions": [int(value) / total_pixels for value in class_pixels],
        "temporal_stability": {
            "mean_changed_fraction": float(np.mean(changed_values)) if changed_values else None,
            "p95_changed_fraction": float(np.percentile(changed_values, 95)) if changed_values else None,
        },
        "baseline": baseline_metrics,
        "semantic": semantic_metrics,
        "size_reduction_fraction": (
            1 - semantic_metrics["size_bytes"] / baseline_metrics["size_bytes"]
            if baseline_metrics and baseline_metrics.get("size_bytes") else None
        ),
        "outputs": {
            "baseline": str(baseline_path.resolve()),
            "semantic": str(semantic_path.resolve()),
            "masks": str(masks_dir.resolve()) if not args.no_save_masks else None,
            "frame_metrics": str((output / "frame_metrics.csv").resolve()),
        },
        "note": "普通和语义视频使用同一 OpenCV 编码器；语义版本只在编码前降低分割判定的背景空间细节。它不是神经压缩、熵编码或可传输的语义码流，文件大小受编码器和输入内容影响。PSNR/SSIM 相对于源视频计算，前景和背景按预测类别掩码划分。",
    }
    write_json(output / "summary.json", summary)
    print(f"输出目录：{output.resolve()}")
    print(f"处理帧数：{processed}，处理 FPS：{summary['processing_fps']:.2f}")
    print(f"普通视频：{baseline_path.resolve()}")
    print(f"语义视频：{semantic_path.resolve()}")
    print(f"汇总：{(output / 'summary.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
