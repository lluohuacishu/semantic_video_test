"""PTS-based sample-and-hold RGB reader. Legacy OpenCV sampling is unchanged."""
from __future__ import annotations

from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

FPS = {"detail_motion": Fraction(25), "detail": Fraction(25, 2),
       "motion": Fraction(25), "compact": Fraction(25, 2)}
WIDTHS = {"detail_motion": 960, "detail": 960, "motion": 640, "compact": 640}
MAX_SOURCE_FRAMES = 250000


def hidden():
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def fraction(value, name):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"Invalid {name}")
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"Invalid {name}") from None
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def clip_options(value):
    value = value or {"mode": "first10"}
    if not isinstance(value, dict) or set(value) - {"mode", "start", "end"}:
        raise ValueError("Invalid time selection")
    mode = value.get("mode", "first10")
    if mode not in ("first10", "segment", "full"):
        raise ValueError("Unknown time selection")
    if mode != "segment":
        if set(value) - {"mode"}:
            raise ValueError("Only segment selection accepts start/end")
        return {"mode": mode}
    start, end = fraction(value.get("start"), "start"), fraction(value.get("end"), "end")
    if end <= start:
        raise ValueError("End must be after start")
    return {"mode": mode, "start": str(start), "end": str(end)}


def video_options(value):
    """Explicit adaptive settings; missing settings retain historical sampling."""
    if not isinstance(value, dict) or set(value) - {"fps_mode", "fps_limit", "size_mode", "long_edge"}:
        raise ValueError("无效视频处理参数")
    fps_mode, size_mode = value.get("fps_mode", "source"), value.get("size_mode", "source")
    if fps_mode not in ("source", "half", "limit") or size_mode not in ("source", "half", "limit"):
        raise ValueError("请选择有效的帧率和分辨率处理方式")
    result = {"fps_mode": fps_mode, "size_mode": size_mode}
    if fps_mode == "limit":
        fps = fraction(value.get("fps_limit"), "输出帧率上限")
        if not 0 < fps <= 240:
            raise ValueError("帧率上限须大于 0 且不超过 240 fps")
        result["fps_limit"] = str(fps)
    elif "fps_limit" in value:
        raise ValueError("只有帧率上限模式接受 fps_limit")
    if size_mode == "limit":
        edge = value.get("long_edge")
        if type(edge) is not int or not 1 <= edge <= 4096:
            raise ValueError("最长边上限须为 1–4096 的整数")
        result["long_edge"] = edge
    elif "long_edge" in value:
        raise ValueError("只有尺寸上限模式接受 long_edge")
    return result


def source_timing(probe):
    pts = [Fraction(p) for p in probe["pts"]]
    duration = Fraction(probe["duration"])
    tick = Fraction(probe.get("source_time_base", "0"))
    try:
        nominal = Fraction(probe.get("nominal_fps") or "0")
    except (ValueError, ZeroDivisionError):
        nominal = Fraction(0)
    measured = Fraction(len(pts), 1) / duration
    rate = nominal if nominal > 0 else measured
    tolerance = max(min(tick, 1 / (4 * rate)), Fraction(1, 1000000))
    # Timestamp ticks can round CFR frame intervals (e.g. 29.97 fps in Matroska).
    variable = any(abs((b - a) - 1 / rate) > tolerance for a, b in zip(pts, pts[1:]))
    return (measured if variable else rate), variable, tolerance


def probe_video(source, ffmpeg, cancel=None):
    source, ffmpeg = Path(source), Path(ffmpeg)
    if not source.is_file() or not 0 < source.stat().st_size <= 1024**3:
        raise ValueError("Input must be a nonempty video of at most 1 GiB")
    ffprobe = ffmpeg.with_name("ffprobe.exe" if ffmpeg.suffix.lower() == ".exe" else "ffprobe")
    base = [str(ffprobe), "-v", "error", "-select_streams", "v:0"]
    raw = subprocess.check_output(base + ["-show_streams", "-of", "json", str(source)],
                                  timeout=30, creationflags=hidden())
    streams = json.loads(raw).get("streams", [])
    if len(streams) != 1:
        raise ValueError("No video stream")
    stream = streams[0]
    width, height = stream["width"], stream["height"]
    if width * height > 4096 * 2160:
        raise ValueError("Source exceeds the 4096x2160 pixel budget")
    for side in stream.get("side_data_list", []):
        if side.get("rotation", 0):
            raise ValueError("Rotated source: normalize display rotation before time sampling")
    timebase = Fraction(stream["time_base"])
    if timebase <= 0:
        raise ValueError("Invalid source time base")
    command = base + ["-show_frames", "-show_entries",
        "frame=best_effort_timestamp,duration,pkt_duration,width,height", "-of", "compact=p=0:nk=0", str(source)]
    pts, durations = [], []
    # Only numeric frame metadata, capped at MAX_SOURCE_FRAMES; no RGB is buffered.
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          text=True, encoding="utf-8", creationflags=hidden()) as process:
        watchdog = threading.Timer(120, process.kill)
        watchdog.daemon = True
        watchdog.start()
        try:
            for line in process.stdout:
                if cancel:
                    cancel()
                fields = dict(item.split("=", 1) for item in line.strip().split("|") if "=" in item)
                if "width" not in fields:
                    continue
                if (int(fields["width"]), int(fields["height"])) != (width, height):
                    raise ValueError("Changing source dimensions are unsupported")
                try:
                    timestamp = int(fields["best_effort_timestamp"]) * timebase
                    duration = int(fields.get("duration", fields.get("pkt_duration", "0"))) * timebase
                except (KeyError, ValueError):
                    raise ValueError("Missing or invalid frame presentation timestamp") from None
                if pts and timestamp <= pts[-1]:
                    raise ValueError("Source presentation timestamps must be strictly increasing")
                pts.append(timestamp)
                durations.append(duration)
                if len(pts) > MAX_SOURCE_FRAMES:
                    raise ValueError("Source frame metadata exceeds the bounded scan limit")
            if process.wait(timeout=5):
                raise ValueError("FFprobe frame scan failed or exceeded 120 seconds")
        finally:
            watchdog.cancel()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    if not pts:
        raise ValueError("No decoded video frames")
    inferred = durations[-1] <= 0
    last_duration = durations[-1] if not inferred else (pts[-1] - pts[-2] if len(pts) > 1 else 0)
    if last_duration <= 0:
        raise ValueError("Cannot determine final video frame duration")
    origin = pts[0]
    return {"width": width, "height": height, "source_time_base": str(timebase),
            "source_pts_origin": str(origin), "pts": [str(p - origin) for p in pts],
            "duration": str(pts[-1] - origin + last_duration),
            "last_duration_inferred_from_previous_interval": inferred,
            "nominal_fps": stream.get("avg_frame_rate"), "source_frames": len(pts)}


def plan_samples(probe, preset, clip=None, video=None):
    if preset not in FPS:
        raise ValueError("Choose one of the four quality presets")
    clip = clip_options(clip)
    duration, fps = Fraction(probe["duration"]), FPS[preset]
    source_fps, variable_fps, tick = source_timing(probe)
    width = min(probe["width"], WIDTHS[preset])
    height = max(1, round(probe["height"] * width / probe["width"]))
    if video is not None:
        video = video_options(video)
        fps = source_fps
        if video["fps_mode"] == "half":
            fps /= 2
        elif video["fps_mode"] == "limit":
            fps = min(fps, Fraction(video["fps_limit"]))
        scale = Fraction(1)
        if video["size_mode"] == "half":
            scale = Fraction(1, 2)
        elif video["size_mode"] == "limit":
            scale = min(scale, Fraction(video["long_edge"], max(probe["width"], probe["height"])))
        width, height = (max(1, round(probe[key] * scale)) for key in ("width", "height"))
        if width > 4096 or height > 2160:
            raise ValueError("当前码流支持宽度 ≤4096、高度 ≤2160，请降低最长边上限")
        if not 0 < fps <= 240:
            raise ValueError("当前码流支持 0–240 fps，请降低帧率上限")
    start = Fraction(clip.get("start", 0))
    requested_end = Fraction(clip["end"]) if clip["mode"] == "segment" else (
        Fraction(10) if clip["mode"] == "first10" else duration)
    end = min(requested_end, duration)
    if start >= end:
        raise ValueError("Selected range is empty or beyond actual video EOF")
    count = math.ceil((end - start) * fps)
    if not 1 <= count <= 10000:
        raise ValueError("Selected range exceeds the 10000 output-frame limit")
    pts = [Fraction(p) for p in probe["pts"]]
    targets = [start + Fraction(i, 1) / fps for i in range(count)]
    # For adaptive CFR, account for rounded PTS without duplicating a preceding
    # frame just because its successor is one clock tick beyond the ideal grid.
    tolerance = tick if video is not None and not variable_fps else 0
    indices = [max(0, bisect_right(pts, target + (tolerance if (target * source_fps).denominator == 1 else 0)) - 1)
               for target in targets]
    if video is not None and not variable_fps:
        samples = [(index, target) for i, (index, target) in enumerate(zip(indices, targets))
                   if not i or index != indices[i - 1]]
        indices, targets = map(list, zip(*samples))
        count = len(indices)
    return {"schema_version": 1, "sampling_mode": "time", "clip": clip, "preset": preset,
            "video": video, "source_fps": str(source_fps), "source_variable_fps": variable_fps,
            "source_width": probe["width"], "source_height": probe["height"],
            "source_duration_seconds": float(duration), "source_frames": len(pts),
            "policy": ("adaptive CFR with clock-tick tolerance; no upsampling" if video is not None and not variable_fps
                       else "source frame visible at target PTS; hold until next PTS; left-closed right-open"),
            "probe": probe, "width": width, "height": height, "target_fps": str(fps), "frames": count,
            "start_seconds": float(start), "end_seconds": float(end),
            "selected_duration_seconds": float(end - start), "duration_seconds": float(count / fps),
            "requested_end_seconds": float(requested_end), "clipped_to_eof": requested_end > duration,
            "sampled_source_frame_indices": indices,
            "sampled_source_pts": [str(pts[i]) for i in indices],
            "target_source_pts": list(map(str, targets)),
            "duplicated_output_frames": len(indices) - len(set(indices))}


def prepare(source, ffmpeg, preset, clip=None, cancel=None, video=None):
    return plan_samples(probe_video(source, ffmpeg, cancel), preset, clip, video)


class TimedFrames:
    """FFmpeg decodes in PTS order; a two-frame queue bounds raw RGB memory."""
    def __init__(self, args, cancel=None):
        import numpy as np
        self.np, self.args, self.cancel = np, args, cancel
        self.plan = args.time_plan
        self.count, self.seconds = 0, 0.0
        self.hashes, self.indices = [], []
        self.decoded_index, self.last = -1, None
        self.queue, self.stopped = queue.Queue(2), threading.Event()
        self.error = None
        p = self.plan["probe"]
        self.shape = (p["height"], p["width"], 3)
        self.frame_bytes = p["width"] * p["height"] * 3
        self.log = (args.workdir / "source_decode.log").open("xb")
        argv = [str(args.ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-noautorotate",
                "-i", str(args.input), "-map", "0:v:0", "-an", "-sn", "-dn",
                "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
        try:
            self.process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=self.log,
                                            bufsize=0, creationflags=hidden())
        except BaseException:
            self.log.close()
            raise
        self.thread = threading.Thread(target=self._read, daemon=True, name="c5-source-frames")
        self.thread.start()

    def _put(self, value):
        while not self.stopped.is_set():
            try:
                self.queue.put(value, timeout=0.1)
                return
            except queue.Full:
                pass

    def _read(self):
        try:
            while not self.stopped.is_set():
                raw = bytearray()
                while len(raw) < self.frame_bytes:
                    chunk = self.process.stdout.read(self.frame_bytes - len(raw))
                    if not chunk:
                        if raw:
                            raise ValueError("Truncated decoded RGB frame")
                        if self.process.wait(timeout=10):
                            raise ValueError("FFmpeg source decoding failed")
                        self._put(None)
                        return
                    raw.extend(chunk)
                self._put(bytes(raw))
        except BaseException as exc:
            self.error = exc
            self._put(None)

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.plan["frames"]:
            raise StopIteration
        started = time.perf_counter()
        try:
            desired = self.plan["sampled_source_frame_indices"][self.count]
            while self.decoded_index < desired:
                deadline = time.monotonic() + 30
                while True:
                    if self.cancel:
                        self.cancel()
                    try:
                        raw = self.queue.get(timeout=0.1)
                        break
                    except queue.Empty:
                        if time.monotonic() > deadline:
                            raise TimeoutError("Source decode stalled for 30 seconds")
                if raw is None:
                    raise ValueError("Decoded source ended before its probed timestamp mapping") from self.error
                self.last = self.np.frombuffer(raw, dtype=self.np.uint8).reshape(self.shape)
                self.decoded_index += 1
            rgb = self.last
            if self.shape[:2] != (self.plan["height"], self.plan["width"]):
                import cv2
                rgb = cv2.resize(rgb, (self.plan["width"], self.plan["height"]), interpolation=cv2.INTER_AREA)
            self.indices.append(desired)
            self.hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
            self.count += 1
            return rgb
        finally:
            self.seconds += time.perf_counter() - started

    def metadata(self):
        if not self.count:
            raise ValueError("No time-sampled frames")
        p, plan = self.plan["probe"], self.plan
        return {"input_path": str(self.args.input.resolve()), "input_bytes": self.args.input.stat().st_size,
                "source_width": p["width"], "source_height": p["height"], "width": plan["width"],
                "height": plan["height"], "fps": float(Fraction(plan["target_fps"])),
                "reported_source_frames": p["source_frames"], "processed_frames": self.count,
                "sampled_source_frame_indices": self.indices, "rgb_sha256": self.hashes,
                "duration_seconds": self.count / float(Fraction(plan["target_fps"])),
                "audio": "not processed", "timebase": "source PTS resampled to constant output fps",
                "sampling_mode": "time", "sampling_plan": plan}

    def close(self):
        self.stopped.set()
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=10)
        self.thread.join(timeout=10)
        self.process.stdout.close()
        self.log.close()
