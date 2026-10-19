"""Bounded RGB iteration and rawvideo previews; no neural/entropy changes.

OpenCV sampling deliberately follows glc_video_probe.extract_frames. Preview
pixels are viewing copies only; correctness is checked on RGB before FFmpeg.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

import cv2
import numpy as np


class MemoryFrames:
    """One decoded frame at a time; actual EOF, not metadata, sets the count."""

    def __init__(self, args):
        self.args = args
        self.reader = cv2.VideoCapture(str(args.input))
        self.indices, self.hashes = [], []
        self.original_size = None
        self.count = 0
        self.seconds = 0.0
        try:
            if not self.reader.isOpened():
                raise ValueError(f"Cannot open video: {args.input}")
            self.fps = self.reader.get(cv2.CAP_PROP_FPS)
            self.reported = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(self.fps) or self.fps <= 0:
                raise ValueError("Input has no usable nominal frame rate.")
            for _ in range(args.start_frame):
                if not self.reader.grab():
                    raise ValueError("start-frame lies beyond the video.")
        except BaseException:
            self.close()
            raise

    def __iter__(self):
        return self

    def __next__(self):
        started = time.perf_counter()
        try:
            if self.count >= self.args.frames:
                raise StopIteration
            # Skip *between* sampled frames, using the same grab/read sequence
            # as the reference. A final partial stride is handled by read/EOF.
            if self.count:
                for _ in range(self.args.frame_stride - 1):
                    if not self.reader.grab():
                        break
            ok, frame = self.reader.read()
            if not ok:
                raise StopIteration
            h, w = frame.shape[:2]
            if self.original_size is None:
                self.original_size = (w, h)
                self.width = min(w, self.args.width) if self.args.width else w
                self.height = max(1, round(h * self.width / w))
            if (w, h) != self.original_size:
                raise ValueError("Changing frame dimensions are not supported.")
            if (w, h) != (self.width, self.height):
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.indices.append(self.args.start_frame + self.count * self.args.frame_stride)
            self.hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
            self.count += 1
            return rgb
        finally:
            self.seconds += time.perf_counter() - started

    def metadata(self):
        if not self.count:
            raise ValueError("No frames were read.")
        args = self.args
        return {
            "input_path": str(args.input.resolve()), "input_bytes": args.input.stat().st_size,
            "source_width": self.original_size[0], "source_height": self.original_size[1],
            "width": self.width, "height": self.height, "fps": self.fps / args.frame_stride,
            "reported_source_frames": self.reported, "processed_frames": self.count,
            "start_frame": args.start_frame, "frame_stride": args.frame_stride,
            "sampled_source_frame_indices": self.indices,
            "duration_seconds": self.count / (self.fps / args.frame_stride),
            "audio": "not processed", "timebase": "nominal constant frame rate",
            "rgb_sha256": self.hashes,
        }

    def close(self):
        self.reader.release()


class RawPreview:
    """FFmpeg stdin with at most two queued RGB copies and one writer frame.

    stderr is a file (never an undrained PIPE). Final MP4 publication happens
    only after finish(), allowing sequence verification to finish first.
    """

    def __init__(self, ffmpeg, output, width, height, fps):
        self.output = Path(output)
        self.part = self.output.with_name(self.output.stem + ".part.mp4")
        self.log_path = self.output.with_suffix(".ffmpeg.log")
        if any(p.exists() for p in (self.output, self.part, self.log_path)):
            raise FileExistsError(self.output)
        self.shape = (height, width, 3)
        self.pending = queue.Queue(maxsize=2)
        self.stopped = threading.Event()
        self.error = None
        self.closed = False
        self.frames = 0
        self.feed_seconds = 0.0
        self.log = self.log_path.open("xb")
        argv = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin",
                "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
                "-framerate", str(fps), "-i", "pipe:0", "-an",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-n", str(self.part)]
        try:
            self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                            stderr=self.log, bufsize=0)
        except BaseException:
            self.log.close()
            raise
        self.thread = threading.Thread(target=self._drain, name="glc-preview", daemon=True)
        self.thread.start()

    def _drain(self):
        try:
            while not self.stopped.is_set():
                try:
                    raw = self.pending.get(timeout=0.1)
                except queue.Empty:
                    continue
                if raw is None:
                    break
                view = memoryview(raw)
                while view:
                    written = self.process.stdin.write(view)
                    if not written:
                        raise BrokenPipeError("FFmpeg closed its rawvideo input")
                    view = view[written:]
        except BaseException as exc:
            self.error = exc
        finally:
            self.process.stdin.close()

    def _put(self, value, cancel=None):
        deadline = time.monotonic() + 60
        while True:
            if cancel is not None:
                cancel()
            if self.error is not None or not self.thread.is_alive():
                raise RuntimeError(f"Preview writer failed; see {self.log_path}") from self.error
            if time.monotonic() > deadline:
                raise TimeoutError("FFmpeg preview input stalled for 60 seconds")
            try:
                self.pending.put(value, timeout=0.1)
                return
            except queue.Full:
                pass

    def add(self, rgb, cancel=None):
        if self.closed or rgb.dtype != np.uint8 or rgb.shape != self.shape:
            raise ValueError("Invalid RGB frame or closed preview")
        started = time.perf_counter()
        self._put(rgb.tobytes(), cancel)
        self.frames += 1
        self.feed_seconds += time.perf_counter() - started

    def finish(self, cancel=None):
        if self.closed:
            raise RuntimeError("Preview already closed")
        self._put(None, cancel)
        deadline = time.monotonic() + 60
        while self.thread.is_alive() or self.process.poll() is None:
            if cancel is not None:
                cancel()
            if time.monotonic() > deadline:
                raise TimeoutError("FFmpeg preview completion stalled for 60 seconds")
            self.thread.join(timeout=0.05)
            if not self.thread.is_alive() and self.process.poll() is None:
                time.sleep(0.05)
        self.log.close()
        self.closed = True
        if self.error is not None or self.process.returncode or not self.part.stat().st_size:
            raise RuntimeError(f"FFmpeg preview failed; see {self.log_path}") from self.error
        # Same-volume, no-clobber publication, matching StreamWriter.
        os.link(self.part, self.output)
        self.part.unlink()

    def abort(self):
        self.stopped.set()
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=10)
        self.thread.join(timeout=10)
        self.log.close()
        self.closed = True
        # Keep the partial video and log as failed-task evidence.
