from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
PALETTE = [[25, 45, 65], [255, 110, 50], [75, 205, 100], [255, 205, 65],
           [55, 210, 225], [180, 95, 230], [40, 240, 155], [195, 160, 120]]


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run(config):
    import numpy as np
    from PIL import Image, ImageOps

    sys.path.insert(0, str(HERE.parents[1] / "09_新阶段D_语义目标增强"))
    from infer_d1 import FrozenSegFormer, letterbox_diver, torch

    folder, source = Path(config["folder"]), Path(config["source"])
    options, ffmpeg = config["parameters"], config["ffmpeg"]
    started = time.perf_counter()

    def progress(stage, percent, **extra):
        save_json(folder / "progress.json", {"stage": stage, "progress": percent, **extra})
        print(stage, flush=True)

    progress("加载语义分割模型", 3)
    if not torch.cuda.is_available():
        raise RuntimeError("语义模型环境未检测到 CUDA，请检查显卡运行状态")
    torch.set_num_threads(4)
    torch.manual_seed(20261013)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    checkpoint = Path(config["checkpoint"])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != "ff00cb439a4af290dc3efde8ae76d4acfad797784483b669ab6db79aaf5b1476":
        raise ValueError("SUIM 模型权重与已登记版本不一致")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if tuple(c["symbol"] for c in state["classes"]) != ("BW", "HD", "PF", "WR", "RO", "RI", "FV", "SR"):
        raise ValueError("模型类别映射不一致")
    model = FrozenSegFormer()
    model.load_state_dict(state["model"], strict=True)
    model = model.to("cuda").eval().requires_grad_(False)
    classes = [{"id": c["index"], "symbol": c["symbol"], "name": c["name_zh"],
                "color": PALETTE[c["index"]], "suim_rgb": c["rgb"]} for c in state["classes"]]
    del state
    palette = np.asarray(PALETTE, dtype=np.uint8)
    histogram = np.zeros(8, dtype=np.int64)
    inference_seconds = 0.0
    inference_mode = options["mode"]

    def infer(rgb):
        nonlocal inference_seconds
        height, width = rgb.shape[:2]
        torch.cuda.synchronize()
        tick = time.perf_counter()
        with torch.inference_mode():
            if inference_mode == "fast256":
                labels, _ = letterbox_diver(model, Image.fromarray(rgb))
            else:
                total = torch.zeros((1, 8, height, width), device="cuda")
                counts = torch.zeros((1, 1, height, width), device="cuda")
                xs = sorted(set(list(range(0, max(1, width - 512), 384)) + [max(0, width - 512)]))
                ys = sorted(set(list(range(0, max(1, height - 512), 384)) + [max(0, height - 512)]))
                for y in ys:
                    for x in xs:
                        crop = torch.from_numpy(rgb[y:y+512, x:x+512].copy()).permute(2, 0, 1)[None]
                        total[:, :, y:y+512, x:x+512] += model(crop.float().to("cuda") / 255)
                        counts[:, :, y:y+512, x:x+512] += 1
                labels = (total / counts).argmax(1)[0].cpu().numpy().astype(np.uint8)
        torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - tick
        histogram[:] += np.bincount(labels.ravel(), minlength=8)
        colors = palette[labels]
        alpha = np.where(labels == 0, 0.0, options["alpha"])[..., None]
        overlay = np.rint(rgb * (1 - alpha) + colors * alpha).astype(np.uint8)
        return labels, colors, overlay

    files = {}

    def add(key, path, name=None):
        files[key] = {"path": str(path), "name": name or path.name}

    def stills(rgb, labels, colors, overlay):
        for key, pixels in (("source_image", rgb), ("mask_image", labels), ("color_image", colors), ("overlay_image", overlay)):
            path = folder / (key + ".png")
            Image.fromarray(pixels).save(path)
            add(key, path)

    video = config["kind"] == "video"
    frame_count, fps, duration = 0, None, None
    timeline = []
    source_duration, source_fps = None, None
    preview_width, preview_height = None, None
    if not video:
        with Image.open(source) as image:
            if image.width * image.height > 16_777_216:
                raise ValueError("图片过大，请使用不超过 1600 万像素的图片")
            if getattr(image, "is_animated", False):
                raise ValueError("请上传静态图片，或转换为视频后上传")
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
        height, width = rgb.shape[:2]
        progress("正在分割图片", 10)
        labels, colors, overlay = infer(rgb)
        stills(rgb, labels, colors, overlay)
        frame_count = 1
        mask_file = folder / "mask_image.png"
    else:
        from fractions import Fraction
        command = [config["ffprobe"], "-v", "error", "-select_streams", "v:0", "-show_entries",
                   "stream=width,height,avg_frame_rate,duration:stream_side_data=rotation:format=duration", "-of", "json", str(source)]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                creationflags=FLAGS, timeout=45)
        if result.returncode:
            raise ValueError(result.stderr[-1000:] or "无法读取视频")
        info = json.loads(result.stdout)
        if not info.get("streams"):
            raise ValueError("文件中没有可读取的视频")
        stream = info["streams"][0]
        width, height = stream["width"], stream["height"]
        rotation = next((s["rotation"] for s in stream.get("side_data_list", []) if "rotation" in s), 0)
        if abs(round(rotation)) % 180 == 90:
            width, height = height, width
        if width * height > 16_777_216:
            raise ValueError("视频画面过大，请使用不超过 1600 万像素的视频")
        source_fps = float(Fraction(stream["avg_frame_rate"]))
        source_duration = float(stream.get("duration", info.get("format", {}).get("duration", 0)))
        if not 0 < source_fps <= 120 or not 0 < source_duration:
            raise ValueError("视频帧率或时长无效（最高 120 fps）")
        fps = source_fps if options["fps"] == "source" else min(float(options["fps"]), source_fps)
        fps_text = stream["avg_frame_rate"] if options["fps"] == "source" or fps == source_fps else str(fps)
        wanted = min(source_duration, 10) if options["duration"] == "10" else source_duration
        expected = max(1, round(wanted * fps))
        preview_height = max(2, min(540, height) // 2 * 2)
        preview_width = max(2, round(width * preview_height / height / 2) * 2)
        mask_file = folder / "labels.zip"
        decoder, encoders = None, []
        log = (folder / "ffmpeg.log").open("a", encoding="utf-8")
        try:
            decoder = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(source),
                "-map", "0:v:0", "-an", "-t", str(wanted), "-vf", f"fps={fps_text},format=rgb24", "-f", "rawvideo", "pipe:1"],
                stdout=subprocess.PIPE, stderr=log, creationflags=FLAGS)
            for key in ("source_video", "color_video", "overlay_video"):
                path = folder / (key + ".mp4")
                proc = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo",
                    "-pixel_format", "rgb24", "-video_size", f"{width}x{height}", "-framerate", fps_text, "-i", "pipe:0", "-an",
                    "-vf", f"scale={preview_width}:{preview_height}:flags={'neighbor' if key == 'color_video' else 'lanczos'}",
                    "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "15", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log, creationflags=FLAGS)
                encoders.append(proc)
                add(key, path)
            frame_bytes = width * height * 3
            with zipfile.ZipFile(mask_file, "w", compression=zipfile.ZIP_STORED) as bundle:
                while True:
                    raw = bytearray()
                    while len(raw) < frame_bytes:
                        block = decoder.stdout.read(frame_bytes - len(raw))
                        if not block:
                            break
                        raw.extend(block)
                    if not raw:
                        break
                    if len(raw) != frame_bytes:
                        raise RuntimeError("视频帧读取不完整")
                    rgb = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
                    labels, colors, overlay = infer(rgb)
                    if frame_count == 0:
                        stills(rgb, labels, colors, overlay)
                    png = io.BytesIO()
                    Image.fromarray(labels).save(png, format="PNG")
                    bundle.writestr(f"labels/{frame_count:06d}.png", png.getvalue())
                    timeline.append({"frame": frame_count, "timestamp_seconds": frame_count / fps})
                    for proc, pixels in zip(encoders, (rgb, colors, overlay)):
                        proc.stdin.write(pixels.tobytes())
                    frame_count += 1
                    progress(f"语义分割中 · {frame_count} / 约 {expected} 帧", min(94, 10 + 84 * frame_count / expected),
                             frames_completed=frame_count)
                if frame_count == 0:
                    raise ValueError("视频没有可处理的画面")
                bundle.writestr("classes.json", json.dumps(classes, ensure_ascii=False, indent=2))
                bundle.writestr("timeline.json", json.dumps(timeline, indent=2))
            if decoder.wait(timeout=60):
                raise RuntimeError("视频解码失败，请查看日志")
            progress("整理分割视频和标签文件", 96)
            for proc in encoders:
                proc.stdin.close()
            for proc in encoders:
                if proc.wait(timeout=120):
                    raise RuntimeError("分割预览视频生成失败，请查看日志")
        finally:
            for proc in [decoder, *encoders]:
                if proc is not None and proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=10)
            log.close()
        for key in ("source_video", "color_video", "overlay_video"):
            verified = subprocess.run([config["ffprobe"], "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=nb_frames", "-of", "json", files[key]["path"]], capture_output=True, text=True, creationflags=FLAGS, timeout=30)
            if verified.returncode or int(json.loads(verified.stdout)["streams"][0]["nb_frames"]) != frame_count:
                raise RuntimeError("预览视频与标签帧数不一致")
        duration = frame_count / fps
        add("labels", mask_file, "逐帧类别标签.zip")
    total_pixels = int(histogram.sum())
    for item in classes:
        item["pixels"] = int(histogram[item["id"]])
        item["fraction"] = item["pixels"] / total_pixels
    metrics = {"width": width, "height": height, "frames": frame_count, "fps": fps,
               "duration_seconds": duration, "source_duration_seconds": source_duration, "source_fps": source_fps,
               "preview_width": preview_width, "preview_height": preview_height,
               "source_bytes": source.stat().st_size, "mask_bytes": mask_file.stat().st_size,
               "inference_seconds": inference_seconds, "pipeline_seconds": time.perf_counter() - started,
               "inference_fps": frame_count / inference_seconds, "device": torch.cuda.get_device_name(0)}
    report = {"source_name": config["source_name"], "kind": config["kind"], "parameters": options,
              "model": "SegFormer-B0 / SUIM", "checkpoint_sha256": digest, "classes": classes, "metrics": metrics,
              "label_format": "uint8 PNG, class IDs 0-7, original display dimensions",
              "inference": "512 tiles / stride 384 / mean overlapping logits" if inference_mode == "tiles512" else "long edge 256 / center letterbox / bilinear logits",
              "background_overlay_alpha": 0, "training": False, "ground_truth_used": False}
    save_json(folder / "report.json", report)
    add("report", folder / "report.json", "分割统计.json")
    add("source", source, config["source_name"])
    save_json(folder / "result.json", {"metrics": metrics, "classes": classes, "files": files,
                                      "kind": config["kind"], "model": report["model"]})
    progress("分割完成", 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(json.loads(args.config.read_text("utf-8")))
    except Exception as exc:
        save_json(args.config.with_name("failure.json"), {"error": str(exc)})
        raise
