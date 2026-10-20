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


def plan_samples(probe, preset, clip=None):
    if preset not in FPS:
        raise ValueError("Choose one of the four quality presets")
    clip = clip_options(clip)
    duration, fps = Fraction(probe["duration"]), FPS[preset]
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
    indices = [max(0, bisect_right(pts, target) - 1) for target in targets]
    width = min(probe["width"], WIDTHS[preset])
    height = max(1, round(probe["height"] * width / probe["width"]))
    return {"schema_version": 1, "sampling_mode": "time", "clip": clip, "preset": preset,
            "policy": "source frame visible at target PTS; hold until next PTS; left-closed right-open",
            "probe": probe, "width": width, "height": height, "target_fps": str(fps), "frames": count,
            "start_seconds": float(start), "end_seconds": float(end),
            "selected_duration_seconds": float(end - start), "duration_seconds": float(count / fps),
            "requested_end_seconds": float(requested_end), "clipped_to_eof": requested_end > duration,
            "sampled_source_frame_indices": indices,
            "sampled_source_pts": [str(pts[i]) for i in indices],
            "target_source_pts": list(map(str, targets)),
            "duplicated_output_frames": len(indices) - len(set(indices))}


def prepare(source, ffmpeg, preset, clip=None, cancel=None):
    return plan_samples(probe_video(source, ffmpeg, cancel), preset, clip)


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
