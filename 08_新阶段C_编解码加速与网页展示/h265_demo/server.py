from __future__ import annotations

import argparse
import copy
from datetime import datetime
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from urllib.parse import parse_qs, quote, urlsplit
import uuid
from semantic_pipeline import run as run_semantic

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
STORE = WORKSPACE / "02_实验数据" / "C_编解码加速与网页展示" / "H265_web_jobs"
FFMPEG = WORKSPACE / "01_项目" / "运行环境" / "FFmpeg" / "ffmpeg.exe"
FFPROBE = FFMPEG.with_name("ffprobe.exe")
SAMPLE = STORE / "assets" / "video131_420-469_鱼类演示.mp4"
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
MAX_UPLOAD = 1024 ** 3


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def probe(path):
    result = subprocess.run([str(FFPROBE), "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames,duration:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=FLAGS, timeout=45)
    if result.returncode:
        raise ValueError(result.stderr[-800:] or "无法读取视频")
    value = json.loads(result.stdout)
    if not value.get("streams"):
        raise ValueError("文件中没有可读取的视频")
    video = value["streams"][0]
    video["duration"] = float(video.get("duration", value.get("format", {}).get("duration", 0)))
    fps = Fraction(video.get("avg_frame_rate", "0/1"))
    if not 0 < fps <= 120 or video["duration"] <= 0:
        raise ValueError("视频时长或帧率无效（支持最高 120 fps）")
    video["fps_string"], video["fps"] = str(fps), float(fps)
    return video


def parameters(query):
    background_crf = int(query.get("background_crf", ["42"])[0])
    roi_crf = int(query.get("roi_crf", ["32"])[0])
    background_scale = float(query.get("background_scale", ["0.5"])[0])
    target = query.get("target", ["fish"])[0]
    roi_layout = query.get("roi_layout", ["single"])[0]
    resolution = query.get("resolution", ["source"])[0]
    duration = query.get("duration", ["full"])[0]
    if not (0 <= background_crf <= 51 and 0 <= roi_crf <= 51 and background_scale in {.25, .5, .75, 1}
            and target in {"fish", "diver"} and roi_layout in {"single", "halves"}
            and resolution in {"source", "360", "540", "720"} and duration in {"full", "10"}):
        raise ValueError("编码参数无效")
    return {"background_crf": background_crf, "roi_crf": roi_crf, "background_scale": background_scale,
        "target": target, "roi_layout": roi_layout, "resolution": resolution, "duration": duration,
        "preset": "medium", "gop": 32, "bframes": 0, "segment_frames": 32, "padding": 16, "min_pixels": 64}


class App:
    def __init__(self):
        STORE.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.processing = threading.Lock()
        self.records = {}
        for path in (STORE / "jobs").glob("*/state.json"):
            try:
                record = json.loads(path.read_text("utf-8"))
                record.setdefault("codec", "hevc")
                if record["status"] not in {"done", "failed"}:
                    record.update(status="failed", stage="处理已中断", error="服务重启，请重新运行")
                    write_json(path, record)
                self.records[record["id"]] = record
            except (OSError, ValueError, KeyError):
                continue

    def update(self, record, **values):
        with self.lock:
            record.update(values)
            write_json(STORE / "jobs" / record["id"] / "state.json", record)

    def public(self, record):
        with self.lock:
            result = copy.deepcopy({key: value for key, value in record.items() if key != "files"})
            result["downloads"] = [{"key": key, "name": file["name"], "bytes": Path(file["path"]).stat().st_size,
                "url": f"/files/{record['id']}/{key}?download=1"} for key, file in record["files"].items()
                if not key.endswith("_preview")]
            result["media"] = {key: f"/files/{record['id']}/{key}" for key in
                ("source_preview", "reconstruction_preview", "overlay_preview") if key in record["files"]}
            return result

    def listing(self):
        with self.lock:
            return {"app": "h265-web", "pipeline": "semantic-h265", "busy": self.processing.locked(), "sample_available": SAMPLE.is_file(),
                "jobs": [{key: record[key] for key in ("id", "source_name", "status", "stage", "created_at", "parameters", "codec")}
                    for record in sorted(self.records.values(), key=lambda r: r["created_at"], reverse=True)]}

    def ffmpeg(self, record, folder, args, stage, start, end, duration):
        began = time.perf_counter()
        command = [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-progress", "pipe:1", "-nostats", *map(str, args)]
        with self.lock:
            record["commands"].append(command)
            record["logs"] = (record["logs"] + [stage])[-25:]
        self.update(record, stage=stage, progress=start)
        with (folder / "pipeline.log").open("a", encoding="utf-8") as log:
            log.write(stage + "\n")
            log.flush()
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, creationflags=FLAGS,
                text=True, encoding="utf-8", errors="replace", cwd=folder)
            for line in process.stdout:
                if line.startswith("out_time_us="):
                    try:
                        fraction = min(.99, max(0, int(line.split("=", 1)[1]) / 1e6 / duration))
                        with self.lock:
                            record["progress"] = round(start + (end - start) * fraction)
                    except ValueError:
                        pass
            code = process.wait()
        if code:
            tail = (folder / "pipeline.log").read_text("utf-8", errors="replace")[-1600:]
            raise RuntimeError(tail or f"FFmpeg 处理失败（{code}）")
        elapsed = time.perf_counter() - began
        with self.lock:
            record["logs"] = (record["logs"] + [f"{stage} · {elapsed:.2f} s"])[-25:]
        self.update(record, progress=end)
        return elapsed

    def worker(self, record, source, folder):
        try:
            run_semantic(self, record, source, folder, probe, write_json)
        except Exception as exc:
            log = folder / "pipeline.log"
            files = {"log": {"path": str(log), "name": "pipeline.log"}} if log.exists() else {}
            self.update(record, status="failed", stage="处理失败", error=str(exc), files=files, finished_at=time.time())
        finally:
            self.processing.release()



class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def log_message(self, *args):
        pass

    def json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_file(self, path, name=None):
        size = path.stat().st_size
        start, end = 0, size - 1
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
            valid = match and any(match.groups())
            if valid:
                a, b = match.groups()
                if a:
                    start, end = int(a), min(int(b), end) if b else end
                else:
                    start = max(0, size - int(b))
                valid = 0 <= start <= end < size
            if not valid:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        self.send_response(206 if requested else 200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(max(0, end - start + 1)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("X-Content-Type-Options", "nosniff")
        if requested:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if name:
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(name)}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as file:
            file.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                block = file.read(min(remaining, 256 * 1024))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            parsed = urlsplit(self.path)
            path = parsed.path
            app = self.server.app
            if path == "/api/jobs":
                return self.json(app.listing())
            if path.startswith("/api/jobs/"):
                with app.lock:
                    record = app.records.get(path.split("/")[-1])
                return self.json(app.public(record)) if record else self.json({"error": "记录不存在"}, 404)
            parts = path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "files":
                with app.lock:
                    file = app.records.get(parts[1], {}).get("files", {}).get(parts[2])
                if not file:
                    return self.json({"error": "文件不存在"}, 404)
                return self.send_file(Path(file["path"]), file["name"] if "download" in parse_qs(parsed.query) else None)
            static = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
            if path in static:
                return self.send_file(HERE / static[path])
            return self.json({"error": "页面不存在"}, 404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:
            self.json({"error": str(exc)}, 500)

    def do_POST(self):
        app = self.server.app
        acquired, folder = False, None
        self.close_connection = True
        try:
            host = self.headers.get("Host", "")
            if host not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"} or self.headers.get("Origin") not in {None, "http://" + host}:
                return self.json({"error": "仅支持本机页面提交"}, 403)
            parsed = urlsplit(self.path)
            if parsed.path != "/api/jobs":
                return self.json({"error": "接口不存在"}, 404)
            query = parse_qs(parsed.query)
            options = parameters(query)
            sample = query.get("sample", [""])[0] == "1"
            length = int(self.headers.get("Content-Length", "0"))
            name = SAMPLE.name if sample else Path(query.get("name", ["video.mp4"])[0].replace("\\", "/")).name
            if Path(name).suffix.lower() not in SUFFIXES or (not sample and not 0 < length <= MAX_UPLOAD):
                raise ValueError("请选择有效视频（最大 1 GB）")
            acquired = app.processing.acquire(blocking=False)
            if not acquired:
                return self.json({"error": "已有任务正在处理"}, 409)
            job_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
            folder = STORE / "jobs" / job_id
            folder.mkdir(parents=True)
            source = SAMPLE if sample else folder / ("source" + Path(name).suffix.lower())
            if not sample:
                self.connection.settimeout(120)
                with source.open("wb") as file:
                    remaining = length
                    while remaining:
                        block = self.rfile.read(min(remaining, 1024 * 1024))
                        if not block:
                            raise ValueError("上传中断，请重试")
                        file.write(block)
                        remaining -= len(block)
            probe(source)
            record = {"id": job_id, "source_name": name, "codec": "semantic-h265", "parameters": options, "status": "queued", "stage": "准备处理",
                "progress": 0, "created_at": datetime.now().isoformat(timespec="seconds"), "logs": [], "commands": [], "files": {}}
            with app.lock:
                app.records[job_id] = record
                app.update(record)
            threading.Thread(target=app.worker, args=(record, source, folder), daemon=True).start()
            acquired = False
            self.json(app.public(record), 202)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            self.json({"error": str(exc)}, 400)
        finally:
            if acquired:
                app.processing.release()
                if folder and not (folder / "state.json").exists():
                    shutil.rmtree(folder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8777)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    server.app = App()
    print(f"H.265: http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
