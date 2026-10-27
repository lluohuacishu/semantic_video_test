from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import time
import zipfile

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
LEGACY = WORKSPACE / "01_项目/运行环境/旧MLVC工程"
PYTHON = LEGACY / ".venv/Scripts/python.exe"
SYSTEM = LEGACY / "semantic_mvp/system.py"
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def execute(app, record, folder, args, stage, progress):
    command = [str(PYTHON), "-u", str(SYSTEM), *map(str, args)]
    app.update(record, stage=stage, progress=progress)
    with app.lock:
        record["commands"].append(command)
        record["logs"] = (record["logs"] + [stage])[-25:]
    env = {**os.environ, "PYTHONUTF8": "1", "FFMPEG_PATH": str(WORKSPACE / "01_项目/运行环境/FFmpeg/ffmpeg.exe")}
    with (folder / "pipeline.log").open("a", encoding="utf-8") as log:
        log.write(stage + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=LEGACY, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", creationflags=FLAGS)
        for line in process.stdout:
            log.write(line)
            log.flush()
            line = line.strip()
            if not line:
                continue
            with app.lock:
                record["logs"] = (record["logs"] + [line])[-25:]
            match = re.search(r"语义提取 (\d+)/(\d+)", line)
            if match:
                done, total = map(int, match.groups())
                app.update(record, stage=line if done < total else "编码背景与目标区域", progress=round(12 + 45 * done / total))
        code = process.wait()
    if code:
        raise RuntimeError((folder / "pipeline.log").read_text("utf-8", errors="replace")[-2400:])


def run(app, record, source, folder, probe, write_json):
    started = time.perf_counter()
    if not PYTHON.is_file() or not SYSTEM.is_file():
        raise RuntimeError("旧项目的语义编码环境不存在")
    app.update(record, status="running", started_at=time.time(), stage="准备语义编码输入", progress=2)
    original = probe(source)
    p = record["parameters"]
    duration = min(original["duration"], float(p["duration"])) if p["duration"] != "full" else original["duration"]
    if duration * original["fps"] > 10000:
        raise ValueError("旧语义协议最多处理 10000 帧，请选择前 10 秒或缩短视频")
    height = original["height"] if p["resolution"] == "source" else min(original["height"], int(p["resolution"]))
    height = height // 2 * 2
    width = round(original["width"] * height / original["height"] / 2) * 2
    if min(width, height) < 32 or max(width, height) > 4096:
        raise ValueError("编码尺寸须在 32–4096 像素之间，请选择较小尺寸")
    processing = folder / "processing_input.mkv"
    app.ffmpeg(record, folder, ["-i", source, "-map", "0:v:0", "-an", "-t", duration,
        "-vf", f"fps={original['fps_string']},scale={width}:{height}:flags=lanczos,format=bgr0",
        "-c:v", "ffv1", "-level", "3", "-map_metadata", "-1", processing], "准备无损编码输入", 3, 11, duration)
    sender, receiver = folder / "sender", folder / "receiver"
    execute(app, record, folder, ["encode", "--input", processing, "--output", sender,
        "--mode", "video", "--target", p["target"], "--frames", "10000", "--device", "cuda",
        "--background-scale", p["background_scale"], "--background-crf", p["background_crf"],
        "--roi-crf", p["roi_crf"], "--roi-layout", p["roi_layout"], "--segment-frames", "32",
        "--padding", "16", "--min-pixels", "64"], "加载旧项目模型并提取语义", 12)
    encoded = json.loads((sender / "encode.json").read_text("utf-8"))
    manifest = json.loads((sender / "manifest.json").read_text("utf-8"))
    packet = sender / "message.scv"
    execute(app, record, folder, ["decode", "--input", packet, "--output", receiver], "独立接收端解码与重建", 67)
    decoded = json.loads((receiver / "decode.json").read_text("utf-8"))
    if encoded["file_sha256"] != decoded["file_sha256"] or encoded["frames"] != decoded["frames"]:
        raise RuntimeError("发送包与接收重建校验不一致")
    frames, duration = encoded["frames"], encoded["duration_seconds"]
    source_preview = folder / "source_preview.mp4"
    app.ffmpeg(record, folder, ["-framerate", manifest["fps"], "-i", sender / "private_frames/%05d.png", "-an",
        "-frames:v", frames, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", source_preview], "生成编码输入预览", 83, 90, duration)
    reconstruction = receiver / "reconstructed.mp4"
    overlay = receiver / "semantic_overlay.mp4"
    for video in (source_preview, reconstruction, overlay):
        checked = probe(video)
        if int(checked["nb_frames"]) != frames or (checked["width"], checked["height"]) != (width, height):
            raise RuntimeError("预览帧数或尺寸不一致")
    raw_folder = folder / "raw_h265"
    raw_folder.mkdir()
    stream_entries = []
    names = ["context.mp4"] + [item["file"] for item in manifest["patches"]]
    for index, name in enumerate(names):
        path = sender / "sender_streams" / name
        checked = probe(path)
        if checked["codec_name"] != "hevc":
            raise RuntimeError("分层视频不是 H.265 码流")
        raw = raw_folder / (Path(name).stem + ".h265")
        app.ffmpeg(record, folder, ["-i", path, "-map", "0:v:0", "-c:v", "copy", "-bsf:v", "hevc_mp4toannexb",
            "-f", "hevc", raw], f"导出各层裸码流 {index + 1}/{len(names)}", 90, 96, duration)
        stream_entries.append({"name": name, "codec": checked["codec_name"], "bytes": path.stat().st_size,
            "raw_name": raw.name, "raw_bytes": raw.stat().st_size})
    raw_zip = folder / "h265_streams.zip"
    with zipfile.ZipFile(raw_zip, "x", compression=zipfile.ZIP_STORED) as archive:
        for item in stream_entries:
            archive.write(raw_folder / item["raw_name"], item["raw_name"])
    raw_bytes = sum(item["raw_bytes"] for item in stream_entries)
    packet_bytes = packet.stat().st_size
    if packet_bytes != sum(encoded[key] for key in ("context_bytes", "roi_stream_bytes", "semantic_bytes", "manifest_and_zip_bytes")):
        raise RuntimeError("传输包字节统计不一致")
    metrics = {"duration_seconds": duration, "width": width, "height": height, "fps": frames / duration, "frames": frames,
        "source_width": original["width"], "source_height": original["height"], "source_bytes": source.stat().st_size,
        "reconstruction_bytes": reconstruction.stat().st_size, "stream_bytes": raw_bytes, "package_bytes": packet_bytes,
        "stream_kbps": raw_bytes * 8 / duration / 1000, "package_kbps": packet_bytes * 8 / duration / 1000,
        "preview_kbps": reconstruction.stat().st_size * 8 / duration / 1000,
        **{key: encoded[key] for key in ("context_bytes", "roi_stream_bytes", "semantic_bytes", "manifest_and_zip_bytes",
            "target_present_frames", "roi_segments", "average_roi_rectangle_fraction")}}
    timings = {"compression_seconds": encoded["encode_and_inference_seconds"], "preview_seconds": decoded["decode_and_preview_seconds"],
        "pipeline_seconds": time.perf_counter() - started}
    stem = Path(record["source_name"]).stem
    entries = {"source": (source, record["source_name"]), "packet": (packet, stem + "_语义H265.scv"),
        "raw_streams": (raw_zip, stem + "_各层H265裸码流.zip"),
        "context": (sender / "sender_streams/context.mp4", stem + "_背景层_H265.mp4"),
        "reconstruction": (reconstruction, stem + "_语义重建预览_H264.mp4"),
        "overlay": (overlay, stem + "_预测目标框.mp4"),
        "semantics": (receiver / "target_information.json", "target_information.json"),
        "manifest": (sender / "manifest.json", "manifest.json"),
        "decode_audit": (receiver / "receiver_read_audit.json", "receiver_read_audit.json"),
        "metrics": (folder / "metrics.json", "metrics.json"), "log": (folder / "pipeline.log", "pipeline.log"),
        "source_preview": (source_preview, "source_preview.mp4"),
        "reconstruction_preview": (reconstruction, "reconstruction_preview.mp4"),
        "overlay_preview": (overlay, "semantic_overlay.mp4")}
    metadata = {"source_name": record["source_name"], "codec": "semantic-h265", "parameters": p,
        "profile": manifest["profile"], "teacher": manifest["teacher"], "streams": stream_entries,
        "packet_sha256": encoded["file_sha256"], "receiver_packet_sha256": decoded["file_sha256"],
        "receiver_process_id": decoded["independent_process_pid"], "receiver_inputs": decoded["inputs"],
        "preprocessing": "CFR at source average fps; selected even dimensions; lossless FFV1 input; video only",
        **metrics, **timings}
    write_json(folder / "metrics.json", metadata)
    app.update(record, status="done", stage="语义重建完成", progress=100, metrics=metrics, timings=timings,
        teacher=manifest["teacher"], profile=manifest["profile"], streams=stream_entries, finished_at=time.time(),
        files={key: {"path": str(path), "name": name} for key, (path, name) in entries.items()})
    processing.unlink()
