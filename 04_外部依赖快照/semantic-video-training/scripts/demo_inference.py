"""演示页面的单任务推理。与命令行共用模型、缩放和时序投票实现。"""
from __future__ import annotations

import csv
import hashlib
import math
import subprocess
import time
import zipfile
from collections import deque
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from segment_video import choose_device, predict_frame, stabilize_prediction
from segmentation import color_mask, load_segmentation_model, palette_from_classes, write_json


class Cancelled(Exception):
    pass


def check_cancel(event):
    if event.is_set():
        raise Cancelled("已停止处理；原文件保留，可以重新拖入。")


class VideoWriter:
    """使用本机 FFmpeg 写 H.264，避免 mp4v 在浏览器中没有画面。"""

    def __init__(self, executable, path, width, height, fps):
        self.log_path = path.with_suffix(".log")
        self.log = self.log_path.open("wb")
        command = [str(executable), "-hide_banner", "-loglevel", "error", "-n",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                   "-r", f"{fps:.8f}", "-i", "pipe:0", "-an",
                   "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                   "-threads", "2", "-preset", "veryfast", "-crf", "18",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        try:
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            self.log.close()
            raise

    def write(self, rgb):
        try:
            self.process.stdin.write(np.ascontiguousarray(rgb).tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"视频写入失败，请查看 {self.log_path.name}") from exc

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        error = None
        try:
            if kind is not None:
                self.process.kill()
            try:
                self.process.stdin.close()
            except OSError as exc:
                error = exc
            try:
                result = self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise RuntimeError("视频封装超时")
        finally:
            self.log.close()
        if kind is None and (result or error):
            message = self.log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
            raise RuntimeError(f"FFmpeg 输出失败：{message}")


class DemoRunner:
    def __init__(self, checkpoint: Path, device: str, ffmpeg: Path | None, semantic_checkpoint: Path | None = None):
        torch.set_num_threads(4)
        self.device = choose_device(device)
        self.model, self.state = load_segmentation_model(checkpoint, self.device)
        self.checkpoint = checkpoint
        self.ffmpeg = ffmpeg
        self.palette = palette_from_classes(self.state["classes"])
        self.classes = self.state["classes"]
        self.image_codec = None
        self.semantic_checkpoint = semantic_checkpoint
        self.semantic_codec = None
        # 一次预热留在启动阶段，不计入上传素材的速度。
        predict_frame(self.model, np.zeros((256, 256, 3), dtype=np.uint8),
                      int(self.state["size"]), self.device)

    def info(self):
        return {"device": str(self.device),
                "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU",
                "model_name": self.state["model_name"], "input_size": int(self.state["size"]),
                "classes": self.classes}

    def run(self, source: Path, output: Path, job: dict, progress, cancel):
        start = time.perf_counter()
        check_cancel(cancel)
        options = job["options"]
        semantic_mode = options.get("mode") == "semantic_codec"
        codec_mode = options.get("mode") in ("image_codec", "semantic_codec")
        if codec_mode and job["kind"] != "image":
            raise ValueError("图片压缩与传输模式暂时只支持图片")
        if semantic_mode:
            summary = self.semantic_transfer(source, output, options, progress, cancel)
        elif codec_mode:
            summary = self.image_transfer(source, output, options, progress, cancel)
        elif job["kind"] == "image":
            summary = self.image(source, output, options, progress, cancel)
        else:
            summary = self.video(source, output, options, progress, cancel)
        check_cancel(cancel)
        summary.update({"filename": job["filename"], "kind": job["kind"], "options": options,
                        "model_name": self.state["model_name"], "checkpoint": str(self.checkpoint.resolve()),
                        "device": str(self.device), "model_input_size": options["size"],
                        "elapsed_seconds": time.perf_counter() - start,
                        "note": ("发送端预测语义参与编码；接收端从量化潜变量直接解出 RGB 和类别图。"
                                 "训练使用码率、像素与类别损失；这是理想信道下的联合语义源编码原型。"
                                 if semantic_mode else "图片使用原尺寸神经编码和实际 rANS 码流，经本机 TCP 限速收发后解码。"
                                 "语义检查使用缩放后的 SegFormer 预测，一致性不是真值准确率。"
                                 if codec_mode else "图片分割 / 视频逐帧推理。模型按长边缩放并补边，预测再还原至输入尺寸；"
                                 "不是原生高清推理。上传内容没有真值标签，占比不是准确率。")})
        write_json(output / "summary.json", summary)
        progress(phase="正在打包结果", progress=99)
        # 不把上传原文件再复制进压缩包；预览、预测、参数和类别编号足够复查。
        with zipfile.ZipFile(output / "results.zip", "w", compression=zipfile.ZIP_STORED) as archive:
            for path in sorted(output.rglob("*")):
                check_cancel(cancel)
                if path.is_file() and path.name not in {"results.zip", "job.json"} and path.suffix != ".log":
                    archive.write(path, path.relative_to(output).as_posix())
        return summary

    def semantic_transfer(self, source, output, options, progress, cancel):
        from file_transfer import loopback_transfer
        from image_codec import quality_metrics, read_rgb
        from semantic_image_codec import SemanticImageCodec
        from segmentation import confusion_matrix, metrics_from_confusion

        if not self.semantic_checkpoint or not self.semantic_checkpoint.is_file():
            raise ValueError("未指定训练后的联合语义编码权重，请使用 run_semantic_codec_demo.cmd")
        if options["size"] != int(self.state["size"]) or options["codec_quality"] != 3:
            raise ValueError("当前联合模型固定使用训练时的 256 分割输入和 q3 初始化；调档位需要另训模型")
        progress(phase="正在读取图片和加载联合模型", progress=5)
        frame = read_rgb(source)
        Image.fromarray(frame).save(output / "input.png")
        if self.semantic_codec is None:
            self.semantic_codec = SemanticImageCodec(self.semantic_checkpoint)
        digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        if self.semantic_codec.state["sender_teacher_sha256"] != digest:
            raise ValueError("发送端分割模型与联合模型的训练条件不一致")
        check_cancel(cancel)
        progress(phase="发送端正在提取编码所需的语义", progress=10)
        before, before_timing = predict_frame(self.model, frame, options["size"], self.device)
        Image.fromarray(color_mask(before, self.palette)).save(output / "semantic_before.png")
        Image.fromarray(before).save(output / "sender_semantic_index.png")
        check_cancel(cancel)
        progress(phase="正在联合编码图像和语义", progress=15)
        encoded = self.semantic_codec.encode(frame, before, output / "encoded.svic")
        check_cancel(cancel)

        def transfer_progress(sent, total, elapsed):
            progress(phase=f"TCP 传输：{sent / 1024:.1f} / {total / 1024:.1f} KiB · {elapsed:.1f} 秒",
                     progress=20 + 55 * sent / total, transfer_bytes=sent, transfer_total_bytes=total,
                     transfer_elapsed_seconds=elapsed)

        transferred = loopback_transfer(output / "encoded.svic", output / "received.svic",
                                        options["transfer_kbps"], event=cancel, progress=transfer_progress,
                                        trace=output / "transfer.csv")
        check_cancel(cancel)
        progress(phase="接收端从码流直接解码图片和类别", progress=80)
        reconstructed, received_classes, decoded = self.semantic_codec.decode(
            output / "received.svic", output / "reconstructed.png", output / "prediction_index.png")
        colors = color_mask(received_classes, self.palette)
        Image.fromarray(colors).save(output / "prediction_color.png")
        Image.fromarray(cv2.addWeighted(reconstructed, 1-options["alpha"], colors, options["alpha"], 0)).save(output / "overlay.png")
        check_cancel(cancel)
        consistency = metrics_from_confusion(confusion_matrix(torch.from_numpy(received_classes), torch.from_numpy(before)), self.classes)
        progress(phase="正在保存联合语义编码报告", progress=95)
        return {"mode": "semantic_codec", "width": encoded["width"], "height": encoded["height"],
                "processed_frames": 1, "source_file_bytes": source.stat().st_size,
                "codec": {**encoded, **decoded, "architecture": self.semantic_codec.state["model_name"],
                          "checkpoint": str(self.semantic_checkpoint.resolve()), "training_epoch": self.semantic_codec.state["epoch"],
                          "device": "cpu"}, "transfer": transferred, "quality": quality_metrics(frame, reconstructed),
                "semantic_consistency": {"pixel_agreement": consistency["pixel_accuracy"], "prediction_miou": consistency["miou"],
                                         "definition": "发送端预测与接收端直接解码类别的一致性；没有真值，不是准确率"},
                "original_semantic_timing_ms": before_timing, "semantic_decoding": "direct_latent_decoder",
                "receiver_uses_segmentation_model": False,
                "classes": self.class_counts(np.bincount(received_classes.ravel(), minlength=8)),
                "outputs": {"input": "input.png", "reconstructed": "reconstructed.png", "overlay": "overlay.png",
                            "color": "prediction_color.png", "index": "prediction_index.png",
                            "semantic_before": "semantic_before.png", "encoded": "encoded.svic", "received": "received.svic"},
                "warnings": ["接收端类别图由联合解码器直接输出；没有对还原图片再次运行分割模型。",
                             "模型训练尺寸为 256。原尺寸编码是跨分辨率试用，效果与验证集指标可能不同。",
                             "这是首轮联合语义源编码实验；尚未证明比普通压缩节省码量，也未加入水声信道。",
                             "传输限速含应用层头，不含 TCP/IP 包头；两端模型预先部署，权重不随图片传输。"]}

    def image_transfer(self, source, output, options, progress, cancel):
        from file_transfer import loopback_transfer
        from image_codec import ImageCodec, quality_metrics, read_rgb

        progress(phase="正在读取图片", progress=5)
        frame = read_rgb(source)
        Image.fromarray(frame).save(output / "input.png")
        check_cancel(cancel)
        progress(phase="正在加载图片编码器", progress=10)
        if self.image_codec is None:
            self.image_codec = ImageCodec("cpu")
        load_started = time.perf_counter()
        self.image_codec.model(options["codec_quality"])
        model_load_seconds = time.perf_counter() - load_started
        check_cancel(cancel)
        progress(phase="正在编码为压缩文件", progress=15)
        encoded = self.image_codec.encode(frame, output / "encoded.svic", options["codec_quality"])
        check_cancel(cancel)

        def transfer_progress(sent, total, elapsed):
            progress(phase=f"TCP 传输：{sent / 1024:.1f} / {total / 1024:.1f} KiB · {elapsed:.1f} 秒",
                     progress=20 + 55 * sent / total, transfer_bytes=sent, transfer_total_bytes=total,
                     transfer_elapsed_seconds=elapsed)

        progress(phase=f"正在以 {options['transfer_kbps']:g} kbps 发送文件", progress=20)
        transferred = loopback_transfer(output / "encoded.svic", output / "received.svic",
                                        options["transfer_kbps"], event=cancel, progress=transfer_progress,
                                        trace=output / "transfer.csv")
        check_cancel(cancel)
        progress(phase="接收端正在解码图片", progress=78)
        reconstructed, decoded = self.image_codec.decode(output / "received.svic", output / "reconstructed.png")
        check_cancel(cancel)
        progress(phase="正在检查画质和语义变化", progress=85)
        metrics = quality_metrics(frame, reconstructed)
        before, before_timing = predict_frame(self.model, frame, options["size"], self.device)
        check_cancel(cancel)
        after, after_timing = predict_frame(self.model, reconstructed, options["size"], self.device)
        before_colors, after_colors = color_mask(before, self.palette), color_mask(after, self.palette)
        overlay = cv2.addWeighted(reconstructed, 1 - options["alpha"], after_colors, options["alpha"], 0)
        for name, pixels in (("semantic_before.png", before_colors), ("prediction_color.png", after_colors),
                             ("prediction_index.png", after), ("overlay.png", overlay)):
            Image.fromarray(pixels).save(output / name)
        from segmentation import confusion_matrix, metrics_from_confusion
        consistency = metrics_from_confusion(confusion_matrix(torch.from_numpy(after), torch.from_numpy(before)), self.classes)
        progress(phase="正在保存报告", progress=95)
        return {"mode": "image_codec", "width": encoded["width"], "height": encoded["height"],
                "processed_frames": 1, "source_file_bytes": source.stat().st_size,
                "codec": {**encoded, **decoded, "architecture": "bmshj2018-hyperprior", "device": str(self.image_codec.device),
                          "model_load_seconds": model_load_seconds},
                "transfer": transferred, "quality": metrics,
                "semantic_consistency": {"pixel_agreement": consistency["pixel_accuracy"],
                                         "prediction_miou": consistency["miou"],
                                         "definition": "同一分割模型在原图与还原图上的预测一致性；没有人工真值，不是准确率"},
                "timing_ms": after_timing, "original_semantic_timing_ms": before_timing,
                "classes": self.class_counts(np.bincount(after.ravel(), minlength=8)),
                "outputs": {"input": "input.png", "reconstructed": "reconstructed.png", "overlay": "overlay.png",
                            "color": "prediction_color.png", "index": "prediction_index.png",
                            "semantic_before": "semantic_before.png", "encoded": "encoded.svic", "received": "received.svic"},
                "warnings": ["传输限速是应用层发送速度，不含 TCP/IP 包头；ZIP 和还原 PNG 的大小不作为通信码量。",
                             "压缩模型为通用预训练模型，尚未进行 SUIM 微调或语义损失训练。"]}

    def class_counts(self, counts):
        total = int(counts.sum())
        return [{**row, "predicted_pixels": int(counts[index]),
                 "predicted_fraction": float(counts[index] / total) if total else 0}
                for index, row in enumerate(self.classes)]

    @staticmethod
    def check_dimensions(width, height):
        if width <= 0 or height <= 0 or width * height > 12_000_000:
            raise ValueError("素材尺寸无效或超过 1200 万像素。演示建议使用 720p 或 1080p。")

    def image(self, source, output, options, progress, cancel):
        progress(phase="正在读取图片", progress=10)
        with Image.open(source) as original:
            self.check_dimensions(*original.size)
            frame = np.array(ImageOps.exif_transpose(original).convert("RGB"))
        check_cancel(cancel)
        progress(phase="正在分割图片", progress=35)
        prediction, timing = predict_frame(self.model, frame, options["size"], self.device)
        check_cancel(cancel)
        colors = color_mask(prediction, self.palette)
        overlay = cv2.addWeighted(frame, 1 - options["alpha"], colors, options["alpha"], 0)
        for name, array in (("input.png", frame), ("overlay.png", overlay),
                            ("prediction_color.png", colors), ("prediction_index.png", prediction)):
            Image.fromarray(array).save(output / name)
        progress(phase="正在保存图片", progress=90)
        return {"width": int(frame.shape[1]), "height": int(frame.shape[0]), "processed_frames": 1,
                "timing_ms": timing, "classes": self.class_counts(np.bincount(prediction.ravel(), minlength=8)),
                "outputs": {"input": "input.png", "overlay": "overlay.png", "color": "prediction_color.png",
                            "index": "prediction_index.png"}}

    def video(self, source, output, options, progress, cancel):
        if not self.ffmpeg:
            raise ValueError("未找到本机 FFmpeg，视频预览不可用。请按演示说明配置 FFMPEG_PATH。")
        capture = cv2.VideoCapture(str(source))
        try:
            if not capture.isOpened():
                raise ValueError("无法解码这个视频。请确认文件完整，或先换一个 MP4 / AVI 文件。")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            fps_note = None
            if not math.isfinite(fps) or fps <= 0:
                fps = 30.0
                fps_note = "源文件未提供有效帧率，预览按 30 FPS 保存。"
            raw_frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            reported = int(raw_frames) if math.isfinite(raw_frames) and raw_frames > 0 else 0
            target = min(reported, options["max_frames"]) if reported and options["max_frames"] else reported
            masks_dir = output / "masks"
            if options["save_masks"]:
                masks_dir.mkdir()
            counts = np.zeros(8, dtype=np.int64)
            timings = np.zeros(3, dtype=np.float64)
            history = deque(maxlen=options["temporal_window"])
            previous = None
            changed = 0.0
            processed = 0
            last_update = 0.0
            progress(phase="正在逐帧分割", progress=2, total_frames=target)
            loop_start = time.perf_counter()
            with ExitStack() as stack:
                csv_file = stack.enter_context((output / "frames.csv").open("w", newline="", encoding="utf-8"))
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["frame", "time_seconds", "preprocess_ms", "inference_ms", "postprocess_ms", "pipeline_ms"])
                writers = None
                while not options["max_frames"] or processed < options["max_frames"]:
                    check_cancel(cancel)
                    frame_start = time.perf_counter()
                    ok, bgr = capture.read()
                    if not ok:
                        break
                    height, width = bgr.shape[:2]
                    self.check_dimensions(width, height)
                    if writers is None:
                        dimensions = (width, height)
                        writers = [stack.enter_context(VideoWriter(self.ffmpeg, output / name, width, height, fps))
                                   for name in ("input.mp4", "overlay.mp4", "prediction_color.mp4")]
                    elif (width, height) != dimensions:
                        raise ValueError("视频中途改变了画面尺寸，请先转为固定分辨率视频。")
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    prediction, timing = predict_frame(self.model, rgb, options["size"], self.device)
                    history.append(prediction)
                    prediction = stabilize_prediction(history, prediction, 8)
                    colors = color_mask(prediction, self.palette)
                    overlay = cv2.addWeighted(rgb, 1 - options["alpha"], colors, options["alpha"], 0)
                    for writer, frame in zip(writers, (rgb, overlay, colors)):
                        check_cancel(cancel)
                        writer.write(frame)
                    if processed == 0:
                        Image.fromarray(overlay).save(output / "poster.jpg", quality=90)
                    if options["save_masks"]:
                        Image.fromarray(prediction).save(masks_dir / f"frame_{processed:06d}.png")
                    counts += np.bincount(prediction.ravel(), minlength=8)
                    if previous is not None:
                        changed += float(np.mean(prediction != previous))
                    previous = prediction
                    row = [timing[name] for name in ("preprocess_ms", "inference_ms", "postprocess_ms")]
                    timings += row
                    csv_writer.writerow([processed, processed / fps, *row, (time.perf_counter() - frame_start) * 1000])
                    processed += 1
                    now = time.perf_counter()
                    if now - last_update >= 0.3:
                        progress(phase="正在逐帧分割", progress=min(94, 2 + 92 * processed / target) if target else 25,
                                 processed_frames=processed, total_frames=target, live_fps=processed / (now - loop_start))
                        last_update = now
                loop_seconds = time.perf_counter() - loop_start
                progress(phase="正在封装浏览器预览", progress=96, processed_frames=processed)
            if not processed:
                raise ValueError("视频没有可以读取的帧。")
            truncated = bool(options["max_frames"] and processed >= options["max_frames"])
            warnings = ["预览保留读取到的帧，按源帧率保存为无声 H.264 视频。奇数宽高补齐到偶数；可变帧率视频按名义帧率播放。"]
            if fps_note:
                warnings.append(fps_note)
            if reported and processed < reported and not truncated:
                warnings.append(f"文件报告 {reported} 帧，但仅解码到 {processed} 帧；可能是元数据不准或文件损坏。")
            if truncated:
                warnings.append(f"本次设置最多处理 {options['max_frames']} 帧；请将上限设为 0 以处理全部。")
            return {"width": width, "height": height, "input_fps": fps, "reported_frames": reported,
                    "processed_frames": processed, "duration_seconds": processed / fps,
                    "processing_fps": processed / loop_seconds, "pipeline_seconds": loop_seconds,
                    "timing_ms": dict(zip(("preprocess_ms", "inference_ms", "postprocess_ms"), (timings / processed).tolist())),
                    "timing_note": "处理 FPS 包含读帧、分割、投票、统计和写入三路视频，不含上传、模型启动预热及最终封装打包；不是网络传输帧率。",
                    "mean_changed_fraction": changed / (processed - 1) if processed > 1 else None,
                    "stability_note": "相邻预测变化包含实际运动，不能直接视为抖动或分割误差；三帧投票可能造成运动边界滞后。",
                    "classes": self.class_counts(counts), "warnings": warnings,
                    "outputs": {"input": "input.mp4", "overlay": "overlay.mp4", "color": "prediction_color.mp4"}}
        finally:
            capture.release()
