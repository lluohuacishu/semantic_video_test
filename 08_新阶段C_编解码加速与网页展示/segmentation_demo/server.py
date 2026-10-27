from __future__ import annotations

import argparse
import copy
from datetime import datetime
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

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
STORE = WORKSPACE / "02_实验数据/C_编解码加速与网页展示/Segmentation_web_jobs"
RUNTIME = WORKSPACE / "01_项目/运行环境"
PYTHON = Path(os.environ.get("GLC_SEMANTIC_PYTHON", str(RUNTIME / "语义模型_Python环境/Scripts/python.exe")))
CHECKPOINT = Path(os.environ.get("GLC_SEMANTIC_CHECKPOINT", str(RUNTIME / "旧训练工程/runs/suim_segformer_b0_20260915/segmentation_best.pt")))
FFMPEG = RUNTIME / "FFmpeg/ffmpeg.exe"
SAMPLE = WORKSPACE / "汇报展示材料/01_可上传视频/021_原视频.mp4"
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SUFFIXES = IMAGES | {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
MAX_UPLOAD = 1024 ** 3
MEDIA = {"source_image", "mask_image", "color_image", "overlay_image", "source_video", "color_video", "overlay_video"}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def parameters(query):
    mode = query.get("mode", ["tiles512"])[0]
    fps = query.get("fps", ["5"])[0]
    duration = query.get("duration", ["full"])[0]
    alpha = float(query.get("alpha", ["0.45"])[0])
    if mode not in {"tiles512", "fast256"} or fps not in {"5", "10", "source"} or duration not in {"full", "10"} or alpha not in {.45, .65, .85}:
        raise ValueError("分割参数无效")
    return {"mode": mode, "fps": fps, "duration": duration, "alpha": alpha}


class App:
    def __init__(self):
        STORE.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.processing = threading.Lock()
        self.records = {}
        for path in (STORE / "jobs").glob("*/state.json"):
            try:
                record = json.loads(path.read_text("utf-8"))
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
                "url": f"/files/{record['id']}/{key}?download=1"} for key, file in record["files"].items() if Path(file["path"]).is_file()]
            result["media"] = {key: f"/files/{record['id']}/{key}" for key in MEDIA if key in record["files"]}
            return result

    def listing(self):
        with self.lock:
            return {"app": "segmentation-web", "busy": self.processing.locked(), "sample_available": SAMPLE.is_file(),
                "jobs": [{key: record[key] for key in ("id", "source_name", "status", "stage", "created_at", "parameters", "kind")}
                    for record in sorted(self.records.values(), key=lambda r: r["created_at"], reverse=True)]}

    def worker(self, record, source, folder):
        process = None
        started = time.perf_counter()
        try:
            for path in (PYTHON, CHECKPOINT, FFMPEG):
                if not path.is_file():
                    raise FileNotFoundError(f"运行文件不存在：{path}")
            config = {"source": str(source), "source_name": record["source_name"], "folder": str(folder),
                      "kind": record["kind"], "parameters": record["parameters"], "checkpoint": str(CHECKPOINT),
                      "ffmpeg": str(FFMPEG), "ffprobe": str(FFMPEG.with_name("ffprobe.exe"))}
            write_json(folder / "config.json", config)
            self.update(record, status="running", started_at=time.time(), stage="启动模型环境", progress=1)
            with (folder / "pipeline.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen([str(PYTHON), str(HERE / "worker.py"), "--config", str(folder / "config.json")],
                    cwd=HERE, env={**os.environ, "PYTHONUTF8": "1"}, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=log, creationflags=FLAGS)
                previous = None
                while process.poll() is None:
                    try:
                        progress = json.loads((folder / "progress.json").read_text("utf-8"))
                        if progress != previous:
                            self.update(record, **progress)
                            previous = progress
                    except (OSError, ValueError):
                        pass
                    time.sleep(.5)
            if process.returncode:
                failure = folder / "failure.json"
                message = json.loads(failure.read_text("utf-8"))["error"] if failure.is_file() else (folder / "pipeline.log").read_text("utf-8", errors="replace")[-1200:]
                raise RuntimeError(message or "分割进程退出，请查看运行日志")
            result = json.loads((folder / "result.json").read_text("utf-8"))
            result["files"]["log"] = {"path": str(folder / "pipeline.log"), "name": "pipeline.log"}
            result["metrics"]["pipeline_seconds"] = time.perf_counter() - started
            report_path = folder / "report.json"
            report = json.loads(report_path.read_text("utf-8"))
            report["metrics"] = result["metrics"]
            write_json(report_path, report)
            self.update(record, **result, status="done", stage="分割完成", progress=100, finished_at=time.time())
        except Exception as exc:
            files = {}
            if (folder / "pipeline.log").is_file():
                files["log"] = {"path": str(folder / "pipeline.log"), "name": "pipeline.log"}
            self.update(record, status="failed", stage="分割失败", error=str(exc), files=files, finished_at=time.time())
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
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
                raise ValueError("请选择支持的图片或视频（最大 1 GB）")
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
            kind = "image" if source.suffix.lower() in IMAGES else "video"
            record = {"id": job_id, "source_name": name, "parameters": options, "kind": kind, "status": "queued", "stage": "准备处理",
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
    parser.add_argument("--port", type=int, default=8778)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    server.app = App()
    print(f"语义分割: http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
