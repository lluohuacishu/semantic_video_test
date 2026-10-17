"""GLC 视频本机控制台；复用已验证的文件编解码流程，无额外 Web 依赖。"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit
from urllib.request import ProxyHandler, build_opener
import webbrowser

from demo_server import Handler as BaseHandler, JobStore, MAX_UPLOAD, TERMINAL, filename_info, now

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "glc_console"
APP_ID = "glc-video-console-v1"
EXAMPLE_ROOT = ROOT / "runs/glc_stream_720p_final_20261002"
RUN_ID = r"(?:[0-9a-f]{32}|example-q(?:00|63))"
ASSETS = {
    "input.mp4": "sender/source_preview.mp4",
    "input.png": "sender/source/im00001.png",
    "reconstructed.mp4": "receiver/reconstruction_preview.mp4",
    "reconstructed.png": "receiver/frames/im00001.png",
    "comparison.mp4": "comparison_preview.mp4",
    "comparison.png": "comparison.png",
    "encoded.glcv": "sender.glcv", "received.glcv": "received.glcv",
    "encode.json": "sender/encode.json", "decode.json": "receiver/decode.json",
    "transfer.csv": "transfer.csv", "transfer.json": "transfer.json",
    "verification.json": "verification.json", "frames.csv": "frames.csv",
}


def parse_options(query):
    defaults = {"q": 63, "width": 1280, "frames": 50, "start_frame": 0,
                "kbps": 20, "intra_period": 32, "reset_interval": 32}
    result = {}
    for key, default in defaults.items():
        values = query.get(key, [str(default)])
        if len(values) != 1:
            raise ValueError(f"参数 {key} 重复")
        try:
            result[key] = float(values[0]) if key == "kbps" else int(values[0])
        except (ValueError, TypeError):
            raise ValueError(f"参数 {key} 不是有效数字") from None
    if (not 0 <= result["q"] <= 63 or result["width"] not in (320, 640, 960, 1280, 1920)
            or not 1 <= result["frames"] <= 10000 or not 0 <= result["start_frame"] <= 1_000_000
            or not math.isfinite(result["kbps"]) or not 1 <= result["kbps"] <= 1000
            or not 1 <= result["intra_period"] <= 10000 or not 0 <= result["reset_interval"] <= 10000):
        raise ValueError("参数超出范围：q 为 0～63，传输速度为 1～1000 kbps，帧数为 1～10000")
    return result


def compact_result(result):
    enc, dec, transfer = result["encode"], result["decode"], result["transfer"]
    info = enc["input"]
    return {
        "width": info["width"], "height": info["height"], "fps": info["fps"],
        "frames": info["processed_frames"], "duration_seconds": info["duration_seconds"],
        "source_bytes": info["input_bytes"], "file_bytes": enc["file_bytes"],
        "q": enc["header"]["q"], "actual_bpp": enc["actual_bpp"],
        "actual_video_kbps": enc["actual_video_kbps"], "psnr_db": enc["mean_psnr_db"],
        "encode_seconds": enc["frame_processing_seconds"], "decode_seconds": dec["frame_processing_seconds"],
        "encode_pipeline_seconds": enc["pipeline_seconds_before_preview"],
        "decode_pipeline_seconds": dec["pipeline_seconds_before_preview"],
        "transfer_seconds": transfer["elapsed_seconds"], "limit_kbps": transfer["limit_kbps"],
        "average_application_kbps": transfer["average_application_kbps"],
        "peak_allocated_mib": enc["peak_allocated_mib"],
        "verified": bool(result["verification"]["all_rgb_frames_identical"]
                         and result["verification"]["all_reference_latents_identical"]
                         and transfer["verified"] and dec["all_frame_checksums_match"]
                         and not dec["source_media_reads"]),
    }


def frame_rows(result):
    decoded = {row["frame"]: row for row in result["decode"]["frames"]}
    return [{"frame": row["frame"], "type": row["type"], "bytes": row["record_bytes"],
             "psnr_db": row["psnr_db"], "encode_seconds": row["seconds"],
             "decode_seconds": decoded[row["frame"]]["seconds"]}
            for row in result["encode"]["frames"]]


def stop_process_tree(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW, timeout=15)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


class GLCStore(JobStore):
    def cancel(self, identifier):
        with self.lock:
            job = self.get(identifier)
            if job["status"] not in TERMINAL:
                self.events[identifier].set()
                if job["status"] == "uploading" and not job.get("receiving"):
                    return self.update(identifier, status="cancelled", phase="已停止上传")
                return self.update(identifier, status="cancelling", phase="正在停止任务并释放资源")
            return job


class GLCApp:
    def __init__(self, directory, python=None):
        self.store = GLCStore(directory)
        self.python = Path(python or ROOT / ".venv-glc/Scripts/python.exe").resolve()
        self.model_info = {"state": "checking", "message": "正在检查 GLC 环境与权重"}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="glc-job")
        self.process_lock = threading.Lock()
        self.process = None
        self.closed = False
        self.examples = {}
        try:
            for result in json.loads((EXAMPLE_ROOT / "summary.json").read_text(encoding="utf-8"))["results"]:
                identifier = f"example-q{result['encode']['header']['q']:02d}"
                compact_result(result)
                self.examples[identifier] = result
        except (OSError, ValueError, KeyError, TypeError):
            self.examples = {}
        self.executor.submit(self.initialize)

    def initialize(self):
        try:
            if not self.python.is_file() or not (ROOT / "tools/ffmpeg/bin/ffmpeg.exe").is_file():
                raise FileNotFoundError("缺少 .venv-glc 环境或 FFmpeg，请参照 docs/GLC-Video.md 配置")
            sys.path.insert(0, str(ROOT / "scripts"))
            from glc_video_probe import check_source, check_weights
            check_source(ROOT / "tools/GLC")
            check_weights(ROOT / "checkpoints/glc")
            code = ("import json,torch; print(json.dumps({'cuda':torch.cuda.is_available(),"
                    "'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,"
                    "'torch':str(torch.__version__)}))")
            checked = subprocess.run([str(self.python), "-c", code], capture_output=True,
                                     text=True, encoding="utf-8", errors="replace", timeout=45,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if checked.returncode:
                raise RuntimeError("GLC Python 环境无法启动：" + checked.stderr[-600:])
            info = json.loads(checked.stdout)
            if not info["cuda"]:
                raise RuntimeError("没有检测到 CUDA。请检查 NVIDIA 驱动与 .venv-glc 环境")
            self.model_info = {"state": "ready", "message": "环境就绪，开始任务后加载模型", **info}
        except Exception as exc:
            self.model_info = {"state": "error", "message": str(exc)}

    def config(self):
        return {"app": APP_ID, "model": self.model_info, "max_upload_bytes": MAX_UPLOAD,
                "output_directory": str(self.store.directory), "defaults": parse_options({}),
                "examples": [{"id": key, "q": value["encode"]["header"]["q"],
                              "label": f"720p · q={value['encode']['header']['q']} · 已有实验"}
                             for key, value in self.examples.items()]}

    def require_ready(self):
        if self.closed or self.model_info["state"] != "ready":
            raise ValueError(self.model_info["message"] if not self.closed else "服务正在停止")

    def get_run(self, identifier):
        if identifier in self.examples:
            result = self.examples[identifier]
            return {"id": identifier, "is_example": True, "status": "done", "phase": "已完成的 720p 实验",
                    "filename": Path(result["encode"]["input"]["input_path"]).name,
                    "summary": compact_result(result), "result_directory": str(EXAMPLE_ROOT / result["folder"])}
        return self.store.get(identifier)

    def result_folder(self, identifier):
        if identifier in self.examples:
            return EXAMPLE_ROOT / self.examples[identifier]["folder"]
        job = self.store.get(identifier)
        return self.store.directory / identifier / "result" / f"q{job['options']['q']:02d}"

    def full_result(self, identifier):
        if identifier in self.examples:
            return self.examples[identifier]
        job = self.store.get(identifier)
        if job["status"] != "done":
            raise KeyError(identifier)
        return json.loads((self.store.directory / identifier / "result/summary.json").read_text(encoding="utf-8"))["results"][0]

    def submit(self, identifier, source):
        with self.store.lock:
            if self.store.events[identifier].is_set():
                return self.store.update(identifier, status="cancelled", phase="已停止上传")
            self.store.update(identifier, status="queued", phase="等待处理", bytes=source.stat().st_size)
            self.executor.submit(self.run, identifier, source)
            return self.store.get(identifier)

    def run(self, identifier, source):
        event = self.store.events[identifier]
        folder = self.store.directory / identifier
        process = None
        try:
            if event.is_set() or self.closed:
                raise InterruptedError()
            job = self.store.get(identifier)
            options = job["options"]
            self.store.update(identifier, status="running", stage="encoding", phase="准备视频片段与模型",
                              started_at=now(), progress=0)
            command = [str(self.python), "-u", str(ROOT / "scripts/glc_video_roundtrip.py"),
                       "--input", str(source), "--output", str(folder / "result"),
                       "--progress-file", str(folder / "progress.json")]
            for key, value in options.items():
                command.extend(["--" + key.replace("_", "-"), str(value)])
            environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
            with (folder / "job.log").open("x", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                                           start_new_session=os.name != "nt")
                with self.process_lock:
                    self.process = process
                previous = None
                while process.poll() is None:
                    if event.wait(0.2) or self.closed:
                        raise InterruptedError()
                    try:
                        text = (folder / "progress.json").read_text(encoding="utf-8")
                        if text != previous:
                            fields = json.loads(text)
                            if fields["stage"] == "done":
                                fields.update(stage="verifying", phase="正在整理结果", progress=99)
                            with self.store.lock:
                                if not event.is_set():
                                    self.store.update(identifier, **fields)
                            previous = text
                    except (OSError, ValueError):
                        pass
                if event.is_set():
                    raise InterruptedError()
                if process.returncode:
                    raise RuntimeError("编解码进程未完成，请查看运行日志")
            result = json.loads((folder / "result/summary.json").read_text(encoding="utf-8"))["results"][0]
            summary = compact_result(result)
            if not summary["verified"]:
                raise ValueError("接收结果未通过一致性检查")
            rows = frame_rows(result)
            with (self.result_folder(identifier) / "frames.csv").open("x", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.store.lock:
                if event.is_set():
                    raise InterruptedError()
                self.store.update(identifier, status="done", stage="done", phase="编码、传输与独立解码完成",
                                  progress=100, summary=summary, completed_at=now(),
                                  result_directory=str(self.result_folder(identifier)))
        except InterruptedError:
            if process is not None:
                stop_process_tree(process)
            self.store.update(identifier, status="cancelled", phase="已停止，可以调整参数后重新运行", error=None)
        except Exception as exc:
            if process is not None:
                stop_process_tree(process)
            detail = str(exc)
            if (folder / "job.log").exists():
                tail = (folder / "job.log").read_text(encoding="utf-8", errors="replace")[-2400:]
                if "out of memory" in tail.lower():
                    detail = "显存不足，请减小最大宽度，并关闭占用显存的其他任务"
                else:
                    detail += "\n" + tail
            self.store.update(identifier, status="error", phase="处理失败", error=detail)
        finally:
            with self.process_lock:
                if self.process is process:
                    self.process = None

    def close(self):
        self.closed = True
        for event in list(self.store.events.values()):
            event.set()
        self.executor.shutdown(wait=True, cancel_futures=True)


class Handler(BaseHandler):
    server_version = "GLCDemo/1"

    def do_GET(self):
        try:
            self.check_origin()
            url = urlsplit(self.path)
            route = unquote(url.path)
            if route == "/api/config":
                return self.json(self.app.config())
            if route == "/api/jobs":
                return self.json(self.app.store.list())
            match = re.fullmatch(rf"/api/runs/({RUN_ID})(/frames)?", route)
            if match:
                identifier, frames = match.groups()
                if not frames:
                    return self.json(self.app.get_run(identifier))
                query = parse_qs(url.query)
                offset = int(query.get("offset", [0])[0])
                if not 0 <= offset <= 10000:
                    raise ValueError("页码无效")
                rows = frame_rows(self.app.full_result(identifier))
                return self.json({"total": len(rows), "offset": offset, "rows": rows[offset:offset + 50]})
            match = re.fullmatch(rf"/media/({RUN_ID})/([^/]+)", route)
            if match:
                identifier, name = match.groups()
                job = self.app.get_run(identifier)
                if name == "log.txt" and not job.get("is_example"):
                    path = self.app.store.directory / identifier / "job.log"
                else:
                    if job["status"] != "done" or name not in ASSETS:
                        raise KeyError(route)
                    folder = self.app.result_folder(identifier).resolve()
                    path = (folder / ASSETS[name]).resolve()
                    if not path.is_relative_to(folder):
                        raise KeyError(route)
                if not path.is_file():
                    raise KeyError(route)
                return self.file(path, download="download" in parse_qs(url.query))
            files = {"/": "index.html", "/app.js": "app.js", "/styles.css": "styles.css", "/favicon.svg": "favicon.svg"}
            if route not in files:
                raise KeyError(route)
            return self.file(STATIC / files[route])
        except PermissionError as exc:
            self.json({"error": str(exc)}, 403)
        except (KeyError, FileNotFoundError):
            self.json({"error": "未找到任务或文件"}, 404)
        except (ValueError, OSError) as exc:
            try:
                self.json({"error": str(exc)}, 400)
            except (BrokenPipeError, ConnectionError):
                pass

    def do_POST(self):
        identifier = None
        temporary = None
        submitted = False
        try:
            self.check_origin()
            url = urlsplit(self.path)
            match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/cancel", url.path)
            if match:
                return self.json(self.app.store.cancel(match.group(1)))
            self.app.require_ready()
            if url.path == "/api/jobs/reserve":
                length = int(self.headers.get("Content-Length", "0"))
                if self.headers.get("Transfer-Encoding") or not 0 < length <= 4096:
                    raise ValueError("上传参数长度无效")
                self.connection.settimeout(15)
                settings = json.loads(self.rfile.read(length))
                if (not isinstance(settings, dict) or not isinstance(settings.get("options"), dict)
                        or not isinstance(settings.get("filename"), str)):
                    raise ValueError("上传参数无效")
                filename, extension, kind = filename_info(settings.get("filename", ""))
                size = settings.get("bytes")
                if kind != "video" or type(size) is not int or not 0 < size <= MAX_UPLOAD:
                    raise ValueError("请选择大小在 1 GiB 以内的视频")
                options = parse_options({key: [str(value)] for key, value in settings["options"].items()})
                job = self.app.store.create(filename, kind, options)
                return self.json(self.app.store.update(job["id"], expected_bytes=size), 202)
            match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/rerun", url.path)
            if match:
                length = int(self.headers.get("Content-Length", "0"))
                if self.headers.get("Transfer-Encoding") or not 0 < length <= 4096:
                    raise ValueError("参数长度无效")
                self.connection.settimeout(15)
                settings = json.loads(self.rfile.read(length))
                if not isinstance(settings, dict):
                    raise ValueError("参数应为 JSON 对象")
                options = parse_options({key: [str(value)] for key, value in settings.items()})
                previous = self.app.store.get(match.group(1))
                if previous["status"] not in TERMINAL:
                    raise ValueError("请先完成或停止原任务")
                _, extension, _ = filename_info(previous["filename"])
                old = self.app.store.directory / previous["id"] / f"source{extension}"
                if not old.is_file():
                    raise ValueError("原文件不在结果目录中，请重新上传")
                job = self.app.store.create(previous["filename"], "video", options)
                identifier = job["id"]
                source = self.app.store.directory / identifier / f"source{extension}"
                try:
                    os.link(old, source)
                except OSError:
                    shutil.copyfile(old, source)
                queued = self.app.submit(identifier, source)
                submitted = True
                return self.json(queued, 202)
            upload_match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/source", url.path)
            if not upload_match and url.path != "/api/jobs":
                raise KeyError(url.path)
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("请使用带 Content-Length 的上传")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_UPLOAD:
                raise ValueError("文件为空或超过 1 GiB")
            if upload_match:
                with self.app.store.lock:
                    job = self.app.store.get(upload_match.group(1))
                    if job["status"] != "uploading" or job.get("receiving") or job["expected_bytes"] != length:
                        raise ValueError("任务不再接受上传，或文件大小与登记时不同")
                    _, extension, _ = filename_info(job["filename"])
                    identifier = job["id"]
                    self.app.store.update(identifier, receiving=True)
            else:
                query = parse_qs(url.query)
                names = query.get("filename", [""])
                if len(names) != 1:
                    raise ValueError("文件名重复")
                filename, extension, kind = filename_info(names[0])
                if kind != "video":
                    raise ValueError("GLC 视频页面请选择 AVI、MP4 等视频文件")
                options = parse_options(query)
                job = self.app.store.create(filename, kind, options)
                identifier = job["id"]
                self.app.store.update(identifier, receiving=True)
            source = self.app.store.directory / identifier / f"source{extension}"
            temporary = source.with_suffix(extension + ".part")
            self.connection.settimeout(60)
            remaining = length
            with temporary.open("xb") as output:
                while remaining:
                    if self.app.store.events[identifier].is_set():
                        raise InterruptedError("上传已停止")
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("上传中断，请重新选择视频")
                    output.write(chunk)
                    remaining -= len(chunk)
            if self.app.store.events[identifier].is_set():
                raise InterruptedError("上传已停止")
            temporary.replace(source)
            queued = self.app.submit(identifier, source)
            submitted = True
            self.json(queued, 202)
        except PermissionError as exc:
            self.json({"error": str(exc)}, 403)
        except KeyError:
            self.json({"error": "未找到任务"}, 404)
        except (ValueError, OSError) as exc:
            if temporary is not None and temporary.exists():
                temporary.unlink()
            if identifier and not submitted:
                cancelled = self.app.store.events[identifier].is_set()
                self.app.store.update(identifier, status="cancelled" if cancelled else "error",
                                      phase="已停止上传" if cancelled else "上传或准备失败", error=None if cancelled else str(exc))
            try:
                self.json({"error": str(exc)}, 400)
            except (BrokenPipeError, ConnectionError):
                pass


class GLCServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, app):
        self.app = app
        super().__init__(address, Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "runs/glc_console")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("端口应为 1024～65535")
    url = f"http://127.0.0.1:{args.port}/"
    try:
        with build_opener(ProxyHandler({})).open(url + "api/config", timeout=1) as response:
            existing = json.load(response)
        if isinstance(existing, dict) and existing.get("app") == APP_ID:
            if existing.get("output_directory") != str(args.data_dir.resolve()):
                print("此端口的 GLC 界面使用了其他结果目录，请关闭后再启动或换端口。", flush=True)
                return 1
            print(f"GLC 页面已启动：{url}", flush=True)
            if args.open:
                webbrowser.open(url)
            return 0
    except (OSError, ValueError):
        pass
    try:
        server = GLCServer(("127.0.0.1", args.port), None)
    except OSError:
        print(f"端口 {args.port} 被占用。请关闭旧服务，或使用 --port 指定其他端口。", flush=True)
        return 1
    app = GLCApp(args.data_dir)
    server.app = app
    print(f"GLC 网页界面：{url}\n结果目录：{app.store.directory}\n按 Ctrl+C 停止服务及运行中的任务。", flush=True)
    if args.open:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
