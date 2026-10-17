"""本机图片／视频分割与图片神经编解码演示，无额外 Web 依赖。"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import mimetypes
import os
import re
import shutil
import threading
import traceback
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, unquote, urlsplit
from urllib.request import ProxyHandler, build_opener


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "console"
DEFAULT_CHECKPOINT = ROOT / "runs" / "suim_transfer_20260911" / "segmentation_best.pt"
MAX_UPLOAD = 1024 ** 3
APP_ID = "semantic-video-demo-v1"
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_TYPES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
TERMINAL = {"done", "error", "cancelled", "interrupted"}
ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
ASSETS = {"input.png", "overlay.png", "prediction_color.png", "prediction_index.png",
          "input.mp4", "overlay.mp4", "prediction_color.mp4", "poster.jpg", "summary.json", "frames.csv", "results.zip",
          "reconstructed.png", "encoded.svic", "received.svic", "transfer.csv", "semantic_before.png"}


def now():
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def atomic_json(path: Path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def find_ffmpeg(explicit=None):
    if explicit or os.environ.get("FFMPEG_PATH"):
        path = Path(explicit or os.environ["FFMPEG_PATH"]).expanduser()
        return path.resolve() if path.is_file() else None
    for pattern in ("ffmpeg.exe", "*/bin/ffmpeg.exe", "bin/ffmpeg.exe"):
        matches = sorted(path for path in (ROOT / "tools").glob(pattern) if path.is_file())
        if matches:
            return matches[-1]
    on_path = shutil.which("ffmpeg")
    if on_path:
        return Path(on_path)
    previous = ROOT.parent.parent / "Python" / "video-baseline" / "tools"
    matches = sorted(path for path in previous.glob("*/bin/ffmpeg.exe") if path.is_file())
    if matches:
        return matches[-1]
    return None


def parse_options(query):
    def one(key, default):
        values = query.get(key, [str(default)])
        if len(values) != 1:
            raise ValueError(f"参数 {key} 重复")
        return values[0]

    size = int(one("size", 256))
    window = int(one("temporal_window", 3))
    alpha = float(one("alpha", 0.45))
    frames = int(one("max_frames", 0))
    masks = one("save_masks", "false")
    mode = one("mode", "segmentation")
    quality = int(one("codec_quality", 3))
    rate = float(one("transfer_kbps", 20))
    if size not in (256, 384, 512):
        raise ValueError("模型输入边长只支持 256、384、512")
    if window not in (1, 3):
        raise ValueError("时序窗口只支持 1 或 3")
    if not 0 <= alpha <= 1:
        raise ValueError("叠加比例必须在 0～1 之间")
    if not 0 <= frames <= 1_000_000 or masks not in ("true", "false"):
        raise ValueError("帧数或标签保存参数无效")
    if mode not in ("segmentation", "image_codec", "semantic_codec") or quality not in (1, 3, 5) or not 1 <= rate <= 1000:
        raise ValueError("图片压缩模式、质量档位或传输限速无效")
    if mode == "semantic_codec" and (quality != 3 or size != 256):
        raise ValueError("联合语义编码当前固定为 256 分割输入和 q3 初始化")
    return {"size": size, "temporal_window": window, "alpha": alpha,
            "max_frames": frames, "save_masks": masks == "true", "mode": mode,
            "codec_quality": quality, "transfer_kbps": rate}


def filename_info(name):
    name = name.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not name or len(name) > 200 or any(ord(ch) < 32 for ch in name):
        raise ValueError("文件名为空、过长或含无效字符")
    extension = Path(name).suffix.lower()
    if extension not in IMAGE_TYPES | VIDEO_TYPES:
        raise ValueError("请选择 JPG、PNG、WebP、BMP 图片，或 MP4、AVI、MOV、MKV、WebM 视频")
    return name, extension, "image" if extension in IMAGE_TYPES else "video"


def byte_range(header, size):
    """单区间 HTTP Range，供浏览器拖动视频时间轴。"""
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header or "")
    if not match or not any(match.groups()) or size <= 0:
        raise ValueError("无效 Range")
    first, last = match.groups()
    if first:
        start = int(first)
        end = min(int(last), size - 1) if last else size - 1
    else:
        suffix = int(last)
        if suffix <= 0:
            raise ValueError("无效 Range")
        start, end = max(0, size - suffix), size - 1
    if start > end or start >= size:
        raise ValueError("Range 超出文件范围")
    return start, end


class JobStore:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
        self.events = {}
        for record in self.directory.glob("*/job.json"):
            try:
                job = json.loads(record.read_text(encoding="utf-8"))
                identifier = job["id"]
                if not ID_PATTERN.fullmatch(identifier) or record.parent.name != identifier:
                    continue
                if job["status"] not in TERMINAL:
                    job.update(status="interrupted", phase="上次服务退出时任务未完成，请重新拖入素材")
                    atomic_json(record, job)
                self.jobs[identifier] = job
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def create(self, filename, kind, options):
        with self.lock:
            if sum(job["status"] not in TERMINAL for job in self.jobs.values()) >= 3:
                raise ValueError("已有 3 个任务在上传或等待，请先完成或停止任务")
            identifier = uuid.uuid4().hex
            folder = self.directory / identifier
            folder.mkdir()
            (folder / "result").mkdir()
            job = {"id": identifier, "filename": filename, "kind": kind, "options": options,
                   "status": "uploading", "phase": "正在上传", "progress": 0, "created_at": now(),
                   "processed_frames": 0, "total_frames": 0}
            self.jobs[identifier] = job
            self.events[identifier] = threading.Event()
            atomic_json(folder / "job.json", job)
            return copy.deepcopy(job)

    def update(self, identifier, **changes):
        with self.lock:
            job = self.jobs[identifier]
            job.update(changes, updated_at=now())
            atomic_json(self.directory / identifier / "job.json", job)
            return copy.deepcopy(job)

    def get(self, identifier):
        with self.lock:
            if not ID_PATTERN.fullmatch(identifier) or identifier not in self.jobs:
                raise KeyError(identifier)
            return copy.deepcopy(self.jobs[identifier])

    def list(self):
        with self.lock:
            return [copy.deepcopy(row) for row in sorted(self.jobs.values(), key=lambda row: row["created_at"], reverse=True)[:30]]

    def cancel(self, identifier):
        with self.lock:
            job = self.get(identifier)
            if job["status"] not in TERMINAL:
                self.events[identifier].set()
                return self.update(identifier, status="cancelled", phase="已请求停止", error=None)
            return job


class DemoApp:
    def __init__(self, checkpoint, data_dir, device, ffmpeg, default_mode="segmentation", semantic_checkpoint=None):
        self.checkpoint = checkpoint
        self.ffmpeg = ffmpeg
        self.default_mode = default_mode
        self.semantic_checkpoint = semantic_checkpoint
        self.store = JobStore(data_dir)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="segmentation")
        self.runner = None
        self.model_info = {"state": "loading", "message": "正在加载模型并预热…"}
        self.executor.submit(self.initialize, device)

    def initialize(self, device):
        try:
            import sys
            sys.path.insert(0, str(ROOT / "scripts"))
            from demo_inference import DemoRunner
            if not self.checkpoint.is_file():
                raise FileNotFoundError("找不到训练权重。请保留 runs 下的训练结果，或用 --checkpoint 指定本机权重。")
            self.runner = DemoRunner(self.checkpoint, device, self.ffmpeg, self.semantic_checkpoint)
            self.model_info = {"state": "ready", "message": "模型已就绪", **self.runner.info()}
        except Exception as exc:
            traceback.print_exc()
            self.model_info = {"state": "error", "message": str(exc)}

    def config(self):
        return {"app": APP_ID, "model": self.model_info, "video_available": self.ffmpeg is not None,
                "default_mode": self.default_mode,
                "semantic_checkpoint": str(self.semantic_checkpoint) if self.semantic_checkpoint else None,
                "semantic_codec_available": (self.semantic_checkpoint is not None and self.semantic_checkpoint.is_file()
                                             and importlib.util.find_spec("compressai") is not None),
                "image_codec_available": (importlib.util.find_spec("compressai") is not None
                    and all((ROOT / f"checkpoints/pretrained/image-codec/hyperprior-q{quality}.pth.tar").is_file()
                            for quality in (1, 3, 5))),
                "max_upload_bytes": MAX_UPLOAD, "output_directory": str(self.store.directory),
                "extensions": sorted(IMAGE_TYPES | VIDEO_TYPES)}

    def run(self, identifier, source):
        event = self.store.events[identifier]
        if event.is_set():
            return
        try:
            if self.runner is None:
                raise RuntimeError(self.model_info["message"])
            self.store.update(identifier, status="running", phase="开始处理", progress=1)
            summary = self.runner.run(source, self.store.directory / identifier / "result", self.store.get(identifier),
                                      lambda **fields: self.store.update(identifier, **fields), event)
            with self.store.lock:
                if event.is_set():
                    self.store.update(identifier, status="cancelled", phase="已停止处理")
                    return
                self.store.update(identifier, status="done", phase="处理完成", progress=100, summary=summary)
        except Exception as exc:
            if event.is_set():
                self.store.update(identifier, status="cancelled", phase="已停止；可以重新拖入素材")
                return
            traceback.print_exc()
            self.store.update(identifier, status="error", phase="处理失败", error=str(exc))

    def close(self):
        for event in list(self.store.events.values()):
            event.set()
        self.executor.shutdown(wait=False, cancel_futures=True)


class Handler(BaseHTTPRequestHandler):
    server_version = "SemanticDemo/1"

    @property
    def app(self):
        return self.server.app

    def log_message(self, fmt, *args):
        if args and str(args[1] if len(args) > 1 else "") not in {"200", "206"}:
            super().log_message(fmt, *args)

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        super().end_headers()

    def check_origin(self):
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in hosts:
            raise PermissionError("此页面仅供本机访问")
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{host}" for host in hosts}:
            raise PermissionError("不接受其他网页发起的请求")

    def json(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def file(self, path, *, download=False):
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        if range_header:
            try:
                start, end = byte_range(range_header, size)
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        self.send_response(206 if range_header else 200)
        mime = {".js": "text/javascript", ".css": "text/css", ".mp4": "video/mp4"}.get(path.suffix) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            self.check_origin()
            url = urlsplit(self.path)
            route = unquote(url.path)
            if route == "/api/config":
                return self.json(self.app.config())
            if route == "/api/jobs":
                return self.json(self.app.store.list())
            if route.startswith("/api/jobs/"):
                return self.json(self.app.store.get(route.rsplit("/", 1)[-1]))
            if route.startswith("/media/"):
                parts = route.split("/")
                if len(parts) != 4 or parts[3] not in ASSETS:
                    raise KeyError(route)
                job = self.app.store.get(parts[2])
                if job["status"] != "done":
                    raise KeyError(route)
                path = (self.app.store.directory / parts[2] / "result" / parts[3]).resolve()
                if not path.is_relative_to(self.app.store.directory) or not path.is_file():
                    raise KeyError(route)
                return self.file(path, download="download" in parse_qs(url.query))
            files = {"/": "index.html", "/app.js": "app.js", "/styles.css": "styles.css", "/favicon.svg": "favicon.svg"}
            if route not in files:
                raise KeyError(route)
            return self.file(STATIC / files[route])
        except PermissionError as exc:
            self.json({"error": str(exc)}, 403)
        except (KeyError, FileNotFoundError):
            self.json({"error": "未找到文件或任务"}, 404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_POST(self):
        identifier = None
        try:
            self.check_origin()
            url = urlsplit(self.path)
            if re.fullmatch(r"/api/jobs/[0-9a-f]{32}/cancel", url.path):
                return self.json(self.app.store.cancel(url.path.split("/")[3]))
            if re.fullmatch(r"/api/jobs/[0-9a-f]{32}/rerun", url.path):
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("不支持分块请求")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("参数长度无效")
                self.connection.settimeout(15)
                settings = json.loads(self.rfile.read(length))
                if not isinstance(settings, dict):
                    raise ValueError("参数应为 JSON 对象")
                options = parse_options({key: [str(value).lower()] for key, value in settings.items()})
                previous = self.app.store.get(url.path.split("/")[3])
                if previous["status"] != "done":
                    raise ValueError("请先等当前任务完成，再重新处理")
                if self.app.model_info["state"] == "error":
                    raise ValueError(self.app.model_info["message"])
                _, extension, kind = filename_info(previous["filename"])
                if options["mode"] in ("image_codec", "semantic_codec") and kind != "image":
                    raise ValueError("图片压缩与传输模式请使用图片素材")
                if kind == "video" and not self.app.ffmpeg:
                    raise ValueError("未找到 FFmpeg，暂时无法处理视频")
                old_source = self.app.store.directory / previous["id"] / f"source{extension}"
                if not old_source.is_file():
                    raise ValueError("本机原文件已被移走，请重新拖入素材")
                job = self.app.store.create(previous["filename"], kind, options)
                identifier = job["id"]
                source = self.app.store.directory / identifier / f"source{extension}"
                shutil.copyfile(old_source, source)
                job = self.app.store.update(identifier, status="queued", phase="等待模型处理",
                                            bytes=source.stat().st_size, parent_id=previous["id"])
                self.app.executor.submit(self.app.run, identifier, source)
                return self.json(job, 202)
            if url.path != "/api/jobs":
                raise KeyError(url.path)
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("请使用带 Content-Length 的文件上传")
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_UPLOAD:
                raise ValueError("文件为空或超过 1 GiB 上传上限")
            query = parse_qs(url.query)
            options = parse_options(query)
            filename, extension, kind = filename_info(query.get("filename", [""])[0])
            if options["mode"] in ("image_codec", "semantic_codec") and kind != "image":
                raise ValueError("图片压缩与传输模式暂时只支持图片，视频可以使用分割演示")
            if self.app.model_info["state"] == "error":
                raise ValueError(self.app.model_info["message"])
            if kind == "video" and not self.app.ffmpeg:
                raise ValueError("未找到 FFmpeg，暂时只能处理图片；配置方法见演示说明")
            job = self.app.store.create(filename, kind, options)
            identifier = job["id"]
            source = self.app.store.directory / identifier / f"source{extension}"
            temporary = source.with_suffix(extension + ".part")
            self.connection.settimeout(60)
            remaining = length
            with temporary.open("xb") as output:
                while remaining:
                    if self.app.store.events[identifier].is_set():
                        raise ValueError("上传已停止")
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("上传中断，请重新拖入文件")
                    output.write(chunk)
                    remaining -= len(chunk)
            if self.app.store.events[identifier].is_set():
                raise ValueError("上传已停止")
            temporary.replace(source)
            job = self.app.store.update(identifier, status="queued", phase="等待模型处理", bytes=length)
            self.app.executor.submit(self.app.run, identifier, source)
            self.json(job, 202)
        except PermissionError as exc:
            self.json({"error": str(exc)}, 403)
        except KeyError:
            self.json({"error": "未找到任务"}, 404)
        except (ValueError, OSError) as exc:
            if identifier:
                cancelled = self.app.store.events[identifier].is_set()
                self.app.store.update(identifier, status="cancelled" if cancelled else "error",
                                      phase="上传已停止" if cancelled else "上传失败", error=str(exc))
            try:
                self.json({"error": str(exc)}, 400)
            except (BrokenPipeError, ConnectionError):
                pass


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, app):
        self.app = app
        super().__init__(address, Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "runs" / "demo_console")
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--open", action="store_true", help="启动后打开默认浏览器；已启动时复用本机页面")
    parser.add_argument("--default-mode", choices=("segmentation", "image_codec", "semantic_codec"), default="segmentation")
    parser.add_argument("--semantic-checkpoint", type=Path, help="训练后的联合语义图片编码权重")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("端口应为 1024～65535")
    url = f"http://127.0.0.1:{args.port}/"
    try:
        with build_opener(ProxyHandler({})).open(url + "api/config", timeout=1) as response:
            running = json.load(response)
        if isinstance(running, dict) and running.get("app") == APP_ID:
            if args.semantic_checkpoint and running.get("semantic_checkpoint") != str(args.semantic_checkpoint.resolve()):
                print("这个端口已有使用其他权重的演示服务。请关闭对应启动窗口后重开，或用 --port 8770。", flush=True)
                return 1
            print(f"演示界面已启动：{url}", flush=True)
            if args.open:
                webbrowser.open(url)
            return 0
    except (OSError, ValueError, URLError):
        pass
    # 先绑定端口，再加载模型，端口冲突时不占用 GPU。
    try:
        server = DemoServer(("127.0.0.1", args.port), None)
    except OSError as exc:
        print(f"无法启动：端口 {args.port} 可能被占用。可用 --port 8768 改端口。\n{exc}", flush=True)
        return 1
    app = DemoApp(args.checkpoint.resolve(), args.data_dir, args.device, find_ffmpeg(args.ffmpeg), args.default_mode,
                  args.semantic_checkpoint.resolve() if args.semantic_checkpoint else None)
    server.app = app
    print(f"演示界面：{url}\n结果目录：{app.store.directory}\n关闭本窗口或按 Ctrl+C 停止服务。", flush=True)
    if args.open:
        threading.Timer(0.6, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("正在停止…", flush=True)
    finally:
        app.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
