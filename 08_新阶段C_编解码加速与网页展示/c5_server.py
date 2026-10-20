"""C5 independent endpoints: authenticated control HTTP and framed TCP data."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import http.client
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socketserver
import subprocess
import sys
import threading
import time
import traceback
from urllib.parse import parse_qs, urlsplit
from http.server import ThreadingHTTPServer

from c3_server import C3App, Handler as OldHandler, HERE, REPO, TERMINAL, MAX_UPLOAD, now, read_json, sha256, write
from c4_config import UploadCatalog
from c5_config import load_settings, peer_url
from c5_compat import POLICIES, check_arithmetic
from c5_transport import acknowledge, file_info, read_header, receive_file, send_file, rate_limit
from c5_timebase import clip_options, prepare

sys.path.insert(0, str(REPO / "04_外部依赖快照/semantic-video-training/scripts"))
from glc_container import read_stream, validate_header

STATIC = HERE / "c5_console"
ASSETS = {"source.mp4": "encode/encoder/source_preview.mp4",
          "decoded.mp4": "decode/receiver/reconstruction_preview.mp4",
          "stream.glcv": "stream.glcv", "received.glcv": "received.glcv",
          "encode.json": "encode/encoder/encode.json", "decode.json": "decode/receiver/decode.json",
          "encode.log": "encode_process/worker.log", "decode.log": "decode_process/worker.log",
          "encode-timing.json": "encode/timing.json", "decode-timing.json": "decode/timing.json",
          "result.json": "result.json", "failure.json": "failure.json", "manifest.json": "manifest.json",
          "transfer.json": "transfer.json", "transfer.csv": "transfer.csv", "events.jsonl": "events.jsonl"}


def inspect_stream(path):
    header, packets, integrity = read_stream(path, with_integrity=True)
    del packets
    if integrity["container_version"] != 2:
        raise ValueError("C5 双端入口需要容器 v2")
    return {**file_info(path), "header": header, "integrity": integrity}


def options(value, presets):
    if not isinstance(value, dict) or set(value) - {"preset", "backend", "sampling_mode", "clip", "frames"}:
        raise ValueError("无效编码参数")
    preset, backend = value.get("preset"), value.get("backend", "native")
    mode = value.get("sampling_mode", "time")
    if preset not in presets or backend not in ("native", "python"):
        raise ValueError("请选择画质档位和有效后端")
    result = {"preset": preset, "backend": backend, "sampling_mode": mode}
    if mode == "time":
        if value.get("frames") is not None:
            raise ValueError("按时间处理不能同时设置帧数")
        clip_options(value.get("clip"))
        result["clip"] = value.get("clip") or {"mode": "first10"}
    elif mode == "legacy_frame_stride":
        frames = value.get("frames")
        if type(frames) is not int or not 1 <= frames <= 10000:
            raise ValueError("旧模式需要指定 1–10000 帧")
        result["frames"] = frames
    else:
        raise ValueError("未知时间采样模式")
    return result


class Peer:
    def __init__(self, url, token):
        self.url, self.token = peer_url(url), token
        self.control_bytes = 0
        self.lock = threading.Lock()

    def call(self, method, path, value=None, timeout=180):
        parsed = urlsplit(self.url)
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode() if value is not None else b""
        headers = {"Authorization": "Bearer " + self.token, "Content-Type": "application/json",
                   "Content-Length": str(len(data))}
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)
        try:
            connection.request(method, path, body=data, headers=headers)
            response = connection.getresponse()
            body = response.read(2 * 1024 * 1024 + 1)
            if len(body) > 2 * 1024 * 1024:
                raise ValueError("接收端控制响应过大")
            # Body bytes only; never count these as constrained forward data bytes.
            with self.lock:
                self.control_bytes += len(data) + len(body)
            value = json.loads(body)
            if response.status >= 400:
                raise ValueError(value.get("error", "接收端请求失败"))
            return value
        finally:
            connection.close()


class Endpoint(C3App):
    def __init__(self, settings):
        self.settings = settings
        self.role = settings["role"]
        super().__init__(settings["data_dir"], UploadCatalog(), settings["runtime"])
        self.environment = None
        self.environment_lock = threading.Lock()
        self.tickets = {}
        self.receiving = set()
        self.controls = {}
        self.rate_mode = settings["rate_mode"]
        self.policy = settings["compatibility_policy"]
        self.backend = settings["backend"]
        self.peer = None
        self.preference_file = self.directory / "endpoint.local.json"
        saved = read_json(self.preference_file) if self.preference_file.exists() else {}
        token_path = self.directory / "connection-token.txt"
        if self.role == "receiver":
            self.token = settings["token"] or (token_path.read_text("ascii").strip() if token_path.exists() else secrets.token_urlsafe(32))
            if not token_path.exists():
                token_path.write_text(self.token, encoding="ascii")
            self.rate_mode = saved.get("rate_mode", self.rate_mode)
            self.policy = saved.get("compatibility_policy", self.policy)
            self.backend = saved.get("backend", self.backend)
        else:
            self.token = ""
            self.set_peer(saved.get("peer_url", settings["peer_url"]), saved.get("token", settings["token"]), save=False)

    def set_peer(self, url, token, save=True):
        if self.role != "sender" or not isinstance(token, str):
            raise ValueError("仅发送端设置连接")
        if token and len(token) < 24:
            raise ValueError("请粘贴接收端完整连接口令")
        self.peer = Peer(url, token)
        if save:
            write(self.preference_file, {"peer_url": self.peer.url, "token": token})
        return {"peer_url": self.peer.url, "token_configured": bool(token)}

    def preferences(self, value):
        if self.role != "receiver" or set(value) - {"rate_mode", "compatibility_policy", "backend"}:
            raise ValueError("无效接收设置")
        rate = value.get("rate_mode", self.rate_mode)
        policy = value.get("compatibility_policy", self.policy)
        backend = value.get("backend", self.backend)
        rate_limit(rate)
        if policy not in POLICIES or backend not in ("native", "python"):
            raise ValueError("无效接收设置")
        with self.store.lock:
            self.rate_mode, self.policy, self.backend = rate, policy, backend
            write(self.preference_file, {"rate_mode": rate, "compatibility_policy": policy, "backend": backend})
        return self.config()

    def config(self):
        value = super().config()
        value.update(app="glc-c5-" + self.role, version="C5", role=self.role, data_port=self.settings["data_port"],
                     rate_mode=self.rate_mode, compatibility_policy=self.policy, backend=self.backend,
                     preflight_ready=self.environment is not None, worker_mode=self.settings["worker_mode"],
                     peer_url=self.peer.url if self.peer else None, token_configured=bool(self.peer and self.peer.token))
        return value

    def preflight(self):
        with self.environment_lock:
            if self.environment is None:
                folder = self.directory / ("preflight_" + secrets.token_hex(8))
                folder.mkdir()
                write(folder / "runtime.json", {"runtime": {k: str(v) for k, v in self.runtime.items()}})
                with (folder / "preflight.log").open("w", encoding="utf-8") as log:
                    result = subprocess.run([str(self.runtime["python"]), "-B", str(HERE / "c5_preflight.py"),
                        "--runtime-json", str(folder / "runtime.json"), "--output", str(folder / "environment.json")],
                        cwd=REPO, stdout=log, stderr=subprocess.STDOUT, timeout=150,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                if result.returncode:
                    raise RuntimeError("环境预检失败，请查看 " + str(folder / "preflight.log"))
                self.environment = read_json(folder / "environment.json")
            return self.environment

    def admit(self, description, policy=None, backend=None):
        header = description["header"]
        validate_header(header)
        if description["integrity"]["container_version"] != 2:
            raise ValueError("双端只接收 v2")
        environment = self.preflight()
        for key, target in (("revision", "model_revision"), ("weights", "weights"),
                            ("entropy", "entropy"), ("table_sha256", "table_sha256")):
            if header[key] != environment[target]:
                raise ValueError("接收端模型/熵表不匹配：" + key)
        if not environment["backends"][backend or self.backend]:
            raise ValueError("接收端所选熵后端不可用")
        return check_arithmetic(header["arithmetic"], environment["arithmetic"], policy or self.policy, 2)

    def reserve(self, value):
        if self.role != "sender":
            raise ValueError("请在发送端上传")
        opts = options(value.get("options"), self.catalog.presets["presets"])
        filename = value.get("filename", "").replace("\\", "/").rsplit("/", 1)[-1].strip()
        extension = Path(filename).suffix.lower()
        if not filename or len(filename) > 200 or any(ord(c) < 32 for c in filename):
            raise ValueError("无效文件名")
        if extension not in (".mp4", ".mkv", ".mov", ".avi", ".webm", ".glcv"):
            raise ValueError("请选择视频或 .glcv 码流")
        size = value.get("bytes")
        if type(size) is not int or not 0 < size <= (256 * 2**20 if extension == ".glcv" else MAX_UPLOAD):
            raise ValueError("视频最大 1 GiB，码流最大 256 MiB")
        job = self.store.create(filename, "stream" if extension == ".glcv" else "video", opts)
        return self.store.update(job["id"], expected_bytes=size, extension=extension, receiving=False)

    def reserve_receive(self, value):
        if self.role != "receiver" or self.closed or self.blockers:
            raise ValueError("接收服务不可用")
        description = value["stream"]
        sender_id = value["sender_id"]
        if not re.fullmatch("[0-9a-f]{32}", sender_id):
            raise ValueError("无效发送任务编号")
        size, digest = description["bytes"], description["sha256"]
        if type(size) is not int or not 0 < size <= 256 * 2**20 or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("无效码流声明")
        # Snapshot policy/rate/backend: UI changes affect subsequent reservations only.
        with self.store.lock:
            rate, policy, backend = self.rate_mode, self.policy, self.backend
        admission = self.admit(description, policy, backend)
        with self.store.lock:
            for job in self.store.jobs.values():
                if job.get("sender_id") == sender_id:
                    if job["stream"] != description:
                        raise ValueError("重复任务携带不同码流")
                    return self.reservation(job)
            job = self.store.create("received.glcv", "stream", {"backend": backend, "compatibility_policy": policy})
            job = self.store.update(job["id"], status="staged", stage="receive", phase="等待码流",
                                    sender_id=sender_id, stream=description, rate_mode=rate, admission=admission)
            self.tickets[job["id"]] = secrets.token_urlsafe(32)
            return self.reservation(job)

    def reservation(self, job):
        return {"id": job["id"], "status": job["status"], "rate_mode": job["rate_mode"],
                "ticket": self.tickets.get(job["id"]), "data_port": self.settings["data_port"]}

    def progress(self, identifier):
        last = [0.]
        def update(done, total, seconds):
            if time.monotonic() - last[0] > .4 or done == total:
                self.store.update(identifier, transferred_bytes=done, total_bytes=total,
                                  transfer_seconds=seconds, progress=round(done * 100 / total, 1))
                last[0] = time.monotonic()
        return update

    def receive(self, connection):
        identifier = None
        try:
            header, header_bytes = read_header(connection)
            candidate = header["job_id"]
            with self.store.lock:
                job = self.store.get(candidate)
                if (not hmac.compare_digest(header["ticket"], self.tickets.get(candidate, ""))
                        or job["status"] != "staged" or candidate in self.receiving):
                    raise ValueError("未知、过期或已使用的传输凭证")
                if any(header[k] != job["stream"][k] for k in ("bytes", "sha256")) or header["rate_mode"] != job["rate_mode"]:
                    raise ValueError("传输与预约不一致")
                identifier = candidate  # Only an authenticated claim can mutate this job.
                self.receiving.add(identifier)
                self.store.update(identifier, status="running", phase="正在接收码流", stage="receive")
            stop = self.store.events[identifier]
            folder = self.directory / identifier
            transfer = receive_file(connection, header, folder / "received.glcv", event=stop,
                                    progress=self.progress(identifier))
            actual = inspect_stream(folder / "received.glcv")
            if actual != job["stream"]:
                raise ValueError("完整码流与预约元数据不一致")
            transfer.update(header_bytes=header_bytes, application_bytes=header_bytes + header["bytes"],
                            rate_mode=header["rate_mode"], network_transfer_performed=True)
            write(folder / "transfer.json", transfer)
            # Commit receipt before ACK. Decode proceeds even when ACK is lost.
            with self.store.lock:
                if stop.is_set():
                    raise InterruptedError("接收已取消")
                self.store.update(identifier, status="queued", stage="decode", phase="码流校验通过，等待解码",
                                  receipt_verified=True, queued_at=now())
                self.executor.submit(self.run_decode, identifier)
            try:
                acknowledge(connection, header, stop)
            except (OSError, InterruptedError):
                self.store.update(identifier, acknowledgement_lost=True)
        except BaseException as exc:
            if identifier:
                self.fail(identifier, exc)
        finally:
            if identifier:
                with self.store.lock:
                    self.receiving.discard(identifier)
                    self.tickets.pop(identifier, None)

    def start(self, identifier):
        if self.role != "sender":
            raise ValueError("接收端由完整码流触发解码")
        return super().start(identifier)

    def run_decode(self, identifier):
        stop = self.store.events[identifier]
        folder = self.directory / identifier
        try:
            if stop.is_set():
                return
            started = time.perf_counter()
            job = self.store.get(identifier)
            self.store.update(identifier, status="running", stage="decode", phase="独立解码中", started_at=now())
            write(folder / "manifest.json", {"role": "receiver", "stream": job["stream"],
                  "options": job["options"], "rate_mode": job["rate_mode"],
                  "code_sha256": code_hashes(), "source_media_provided": False})
            task = {"id": identifier, "input": str(folder / "received.glcv"), "output": str(folder / "decode"),
                    "output_mode": "display", **job["options"]}
            self.execute("decode", task, folder)
            dec = read_json(folder / "decode/receiver/decode.json")
            stream = job["stream"]
            if (dec["input_sha256"] != stream["sha256"] or not dec["independent_decoder"]
                    or dec["source_media_reads"] or not dec["all_frame_checksums_match"]
                    or not dec["whole_file_checksum_verified"] or
                    dec["sequence_reconstruction_checksums"] != stream["integrity"]["sequence_reconstruction_checksums"]):
                raise ValueError("接收端完整流/RGB/latent/独立解码核验失败")
            result = {"role": "receiver", "verified": True, "stream_sha256": stream["sha256"],
                      "file_bytes": stream["bytes"], "frames": len(dec["frames"]), "header": dec["header"],
                      "source_media_reads": [], "sequence_reconstruction_checksums": dec["sequence_reconstruction_checksums"],
                      "arithmetic_compatibility": dec["arithmetic_compatibility"],
                      "service_seconds": time.perf_counter() - started,
                      "timing": read_json(folder / "decode/timing.json"),
                      "transfer": read_json(folder / "transfer.json")}
            self.complete(identifier, result)
        except BaseException as exc:
            self.fail(identifier, exc)

    def run(self, identifier):
        folder = self.directory / identifier
        stop = self.store.events[identifier]
        peer = Peer(self.peer.url, self.peer.token)  # Snapshot connection; per-task accounting.
        self.controls[identifier] = peer
        reservation = None
        try:
            if stop.is_set():
                return
            started = time.perf_counter()
            job = self.store.get(identifier)
            self.store.update(identifier, status="running", stage="preflight", phase="检查接收端环境", started_at=now(),
                              peer_url=peer.url)
            remote = peer.call("POST", "/control/preflight", {})
            source = self.source(job)
            if job["kind"] == "video":
                local = self.preflight()
                check_arithmetic(local["arithmetic"], remote["environment"]["arithmetic"], remote["compatibility_policy"], 2)
                if any(local[k] != remote["environment"][k] for k in ("weights", "model_revision", "table_sha256", "entropy")):
                    raise ValueError("两端模型或熵表不一致")
                self.store.update(identifier, stage="encode", phase="编码中")
                self.execute("encode", {"id": identifier, "input": str(source), "output": str(folder / "encode"),
                                       "output_mode": "display", **job["options"]}, folder)
                shutil.copyfile(folder / "encode/stream.glcv", folder / "stream.glcv")
            else:
                shutil.copyfile(source, folder / "stream.glcv")
            if stop.is_set():
                raise InterruptedError("已取消")
            description = inspect_stream(folder / "stream.glcv")
            write(folder / "manifest.json", {"role": "sender", "source_sha256": sha256(source),
                  "stream": description, "options": job["options"], "code_sha256": code_hashes(),
                  "encoded_this_task": job["kind"] == "video"})
            reservation = peer.call("POST", "/control/reserve", {"sender_id": identifier, "stream": description})
            self.store.update(identifier, receiver_job_id=reservation["id"], rate_mode=reservation["rate_mode"],
                              stage="send", phase="正在发送码流")
            transfer = None
            if reservation["status"] == "staged":
                try:
                    transfer = send_file(folder / "stream.glcv", urlsplit(peer.url).hostname, reservation["data_port"],
                        reservation, event=stop, progress=self.progress(identifier), trace=folder / "transfer.csv")
                    write(folder / "transfer.json", transfer)
                except (OSError, ValueError):
                    if stop.is_set():
                        raise
                    remote_job = peer.call("GET", "/control/jobs/" + reservation["id"], timeout=10)
                    if not remote_job.get("receipt_verified"):
                        raise
                    transfer = {"verified": True, "acknowledgement_lost": True, "network_transfer_performed": True,
                                "sender_measurement_complete": False}
                    write(folder / "transfer.json", transfer)
            self.store.update(identifier, stage="remote_decode", phase="码流已接收，等待接收端解码")
            deadline, last_contact = time.monotonic() + 6 * 3600, time.monotonic()
            while time.monotonic() < deadline:
                if stop.is_set():
                    raise InterruptedError("已请求停止")
                try:
                    remote_job = peer.call("GET", "/control/jobs/" + reservation["id"], timeout=5)
                    last_contact = time.monotonic()
                except OSError:
                    if time.monotonic() - last_contact > 30:
                        raise ConnectionError("接收端暂时不可达；远端状态未知，请恢复连接后查询")
                    stop.wait(1)
                    continue
                self.store.update(identifier, remote_status=remote_job["status"], remote_heartbeat=remote_job.get("heartbeat"))
                if remote_job["status"] in TERMINAL:
                    break
                stop.wait(.8)
            else:
                raise TimeoutError("等待接收端超时；远端状态未知")
            if remote_job["status"] != "done":
                raise RuntimeError("接收端：" + remote_job.get("error", remote_job["phase"]))
            remote_result = remote_job["result"]
            if (not remote_result["verified"] or remote_result["stream_sha256"] != description["sha256"]
                    or remote_result["sequence_reconstruction_checksums"] != description["integrity"]["sequence_reconstruction_checksums"]):
                raise ValueError("两端结果摘要不一致")
            result = {"role": "sender", "verified": True, "file_bytes": description["bytes"],
                      "stream_sha256": description["sha256"], "frames": description["header"]["frames"],
                      "header": description["header"], "receiver_job_id": reservation["id"],
                      "encoded_this_task": job["kind"] == "video", "transfer": transfer,
                      "remote": remote_result, "control_body_bytes": peer.control_bytes,
                      "control_byte_definition": "HTTP request/response bodies only; separate from constrained TCP bytes",
                      "service_seconds": time.perf_counter() - started}
            self.complete(identifier, result)
        except BaseException as exc:
            if reservation and stop.is_set():
                try:
                    remote_job = peer.call("POST", "/control/jobs/" + reservation["id"] + "/cancel", {}, timeout=5)
                    self.store.update(identifier, remote_status=remote_job["status"])
                except (OSError, ValueError):
                    self.store.update(identifier, remote_status="unknown")
            self.fail(identifier, exc)
        finally:
            self.controls.pop(identifier, None)

    def complete(self, identifier, result):
        with self.store.lock:
            if self.store.events[identifier].is_set():
                raise InterruptedError("任务已取消")
            write(self.directory / identifier / "result.json", result)
            self.store.update(identifier, status="done", stage="done", phase="已完成：码流、RGB 和 latent 校验通过",
                              finished_at=now(), progress=100)

    def fail(self, identifier, exc):
        stop = self.store.events[identifier]
        cancelled = stop.is_set() or isinstance(exc, InterruptedError)
        write(self.directory / identifier / "failure.json",
              {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        self.store.update(identifier, status="cancelled" if cancelled else "error", error=str(exc),
                          phase="已停止，未完成结果不计入成功" if cancelled else str(exc), finished_at=now())

    def get_job(self, identifier, remote=False):
        job = self.store.get(identifier)
        folder = self.directory / identifier
        stage = job.get("stage")
        if stage in ("encode", "decode"):
            try:
                heartbeat = read_json(folder / (stage + "_process") / "heartbeat.json")
                job["heartbeat"] = {k: heartbeat.get(k) for k in ("state", "last_progress", "task_state")}
            except (OSError, ValueError):
                pass
        if job["status"] == "done":
            job["result"] = read_json(folder / "result.json")
            if remote:  # No media, paths or large frame arrays on the control channel.
                job["result"].pop("timing", None)
        if not remote:
            job["downloads"] = [{"name": name, "url": f"/jobs/{identifier}/{name}?download=1"}
                for name, path in ASSETS.items() if (folder / path).is_file()
                and (job["status"] == "done" or name not in ("source.mp4", "decoded.mp4", "result.json"))]
        return job

    def resend(self, identifier):
        if self.role != "sender":
            raise ValueError("只在发送端重发")
        previous = self.store.get(identifier)
        stream = self.directory / identifier / "stream.glcv"
        inspect_stream(stream)
        job = self.store.create("cached.glcv", "stream", previous["options"])
        shutil.copyfile(stream, self.directory / job["id"] / "source.glcv")
        return self.store.update(job["id"], status="staged", phase="码流已复用，点击开始发送",
                                 extension=".glcv", bytes=stream.stat().st_size, parent_id=identifier)

    def remote_status(self, identifier):
        job = self.store.get(identifier)
        if not job.get("receiver_job_id"):
            raise ValueError("该任务尚未预约接收端")
        if job.get("peer_url") != self.peer.url:
            raise ValueError("请先恢复该任务原接收端地址")
        remote = self.peer.call("GET", "/control/jobs/" + job["receiver_job_id"], timeout=5)
        self.store.update(identifier, remote_status=remote["status"], remote_checked_at=now())
        return remote


def code_hashes():
    files = list(HERE.glob("c5_*.py")) + [HERE / name for name in
        ("c1_runtime.py", "c1_io.py", "c2_worker.py", "c2_native.py", "c2_watch.py")]
    return {p.name: sha256(p) for p in files}


class Handler(OldHandler):
    server_version = "GLC-C5/1"

    def authorize(self, control=False):
        if control:
            if self.app.role != "receiver" or not hmac.compare_digest(
                    self.headers.get("Authorization", ""), "Bearer " + self.app.token):
                raise PermissionError("接收端连接口令不匹配")
            if self.headers.get("Origin"):
                raise PermissionError("控制接口仅供发送服务调用")
        else:
            if self.client_address[0] not in ("127.0.0.1", "::1"):
                raise PermissionError("网页在各自电脑本机打开")
            self.check_origin()

    def body(self):
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("不支持分块请求")
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= 65536:
            raise ValueError("无效请求长度")
        self.connection.settimeout(15)
        raw = self.rfile.read(size)
        if len(raw) != size:
            raise ValueError("请求不完整")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("请求需要 JSON 对象")
        return value

    def do_GET(self):
        try:
            parsed = urlsplit(self.path)
            route = parsed.path
            control = route.startswith("/control/")
            self.authorize(control)
            if route == "/api/config":
                return self.json(self.app.config())
            if route == "/api/pairing" and self.app.role == "receiver":
                return self.json({"token": self.app.token, "http_port": self.server.server_port,
                                  "data_port": self.app.settings["data_port"]})
            if route == "/api/jobs":
                return self.json(self.app.store.list())
            match = re.fullmatch(r"/(?:api|control)/jobs/([0-9a-f]{32})", route)
            if match:
                return self.json(self.app.get_job(match[1], remote=control))
            match = re.fullmatch(r"/jobs/([0-9a-f]{32})/([^/]+)", route)
            if match:
                job = self.app.store.get(match[1])
                name = match[2]
                if name not in ASSETS or (name in ("source.mp4", "decoded.mp4", "result.json") and job["status"] != "done"):
                    raise KeyError(name)
                return self.file(self.app.directory / match[1] / ASSETS[name], download="download" in parse_qs(parsed.query))
            if route in ("/", "/app.js", "/styles.css"):
                return self.file(STATIC / ("index.html" if route == "/" else route[1:]))
            raise KeyError(route)
        except BaseException as exc:
            self.error(exc)

    def do_POST(self):
        try:
            route = urlsplit(self.path).path
            control = route.startswith("/control/")
            self.authorize(control)
            value = self.body()
            if route == "/api/shutdown":
                self.json({"stopping": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if route in ("/api/preflight", "/control/preflight"):
                return self.json({"environment": self.app.preflight(),
                                  "compatibility_policy": self.app.policy, "rate_mode": self.app.rate_mode})
            if route == "/api/peer":
                return self.json(self.app.set_peer(value["url"], value["token"]))
            if route == "/api/peer/check":
                return self.json(self.app.peer.call("POST", "/control/preflight", {}))
            if route == "/api/preferences":
                return self.json(self.app.preferences(value))
            if route == "/api/jobs":
                return self.json(self.app.reserve(value), 201)
            if route == "/control/reserve":
                return self.json(self.app.reserve_receive(value), 201)
            match = re.fullmatch(r"/(?:api|control)/jobs/([0-9a-f]{32})/(start|cancel|resend|remote|probe)", route)
            if match:
                identifier, action = match.groups()
                if control and action != "cancel":
                    raise PermissionError("远程不支持该操作")
                if action == "probe":
                    job = self.app.store.get(identifier)
                    if job["status"] != "staged" or job["kind"] != "video":
                        raise ValueError("请先准备视频")
                    plan = prepare(self.app.source(job), self.app.runtime["ffmpeg"],
                                   job["options"]["preset"], job["options"].get("clip"))
                    return self.json({k: v for k, v in plan.items() if k not in
                                      ("sampled_source_frame_indices", "sampled_source_pts", "target_source_pts", "probe")})
                function = {"start": self.app.start, "cancel": self.app.cancel,
                            "resend": self.app.resend, "remote": self.app.remote_status}[action]
                return self.json(function(identifier))
            raise KeyError(route)
        except BaseException as exc:
            self.error(exc)

    def do_PUT(self):
        identifier = None
        try:
            self.authorize()
            match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/upload", urlsplit(self.path).path)
            if not match or self.app.role != "sender":
                raise ValueError("无效上传地址")
            identifier = match[1]
            with self.app.store.lock:
                job = self.app.store.get(identifier)
                if job["status"] != "uploading" or job.get("receiving"):
                    identifier = None
                    raise ValueError("任务不可再次上传")
                length = int(self.headers.get("Content-Length", "0"))
                if self.headers.get("Transfer-Encoding") or length != job["expected_bytes"]:
                    raise ValueError("上传长度与声明不一致")
                self.app.store.update(identifier, receiving=True)
            source = self.app.source(job)
            part = source.with_name(source.name + ".part")
            self.connection.settimeout(30)
            remaining = length
            with part.open("xb") as handle:
                while remaining:
                    if self.app.store.events[identifier].is_set():
                        raise InterruptedError("上传已取消")
                    data = self.rfile.read(min(remaining, 1024 * 1024))
                    if not data:
                        raise ValueError("上传提前断开")
                    handle.write(data)
                    remaining -= len(data)
                handle.flush()
                os.fsync(handle.fileno())
            with self.app.store.lock:
                if self.app.store.events[identifier].is_set():
                    raise InterruptedError("上传已取消")
                os.link(part, source)
                part.unlink()
                job = self.app.store.update(identifier, status="staged", receiving=False, bytes=length,
                                            phase="已准备，点击开始")
            return self.json(job)
        except BaseException as exc:
            if identifier:
                self.app.fail(identifier, exc)
            self.error(exc)

    def error(self, exc):
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return
        status = 403 if isinstance(exc, PermissionError) else 404 if isinstance(exc, (KeyError, FileNotFoundError)) else 400
        try:
            self.json({"error": str(exc)}, status)
        except OSError:
            pass


class Server(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, app=None):
        self.app = app
        super().__init__(address, Handler)


class DataHandler(socketserver.BaseRequestHandler):
    def handle(self):
        if not self.server.slots.acquire(blocking=False):
            return
        try:
            self.server.app.receive(self.request)
        finally:
            self.server.slots.release()


class DataServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = False
    def __init__(self, address, app=None):
        self.app = app
        self.slots = threading.BoundedSemaphore(4)
        super().__init__(address, DataHandler)


def main(role=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if role is None:
        parser.add_argument("role", choices=("sender", "receiver"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    settings = load_settings(role or args.role, args.config)
    if args.check:
        missing = [k for k, p in settings["runtime"].items() if not p.exists()]
        print(json.dumps({"role": settings["role"], "missing": missing, "data_dir": str(settings["data_dir"]),
                          "http_port": settings["http_port"], "data_port": settings["data_port"],
                          "gpu_started": False}, ensure_ascii=False))
        return int(bool(set(missing) - {"native_library"}))
    # Bind ports before opening/reclassifying the persistent job store.
    with Server((settings["host"], settings["http_port"])) as server:
        data = DataServer((settings["host"], settings["data_port"])) if settings["role"] == "receiver" else None
        try:
            app = Endpoint(settings)
            server.app = app
            if data:
                data.app = app
                threading.Thread(target=data.serve_forever, kwargs={"poll_interval": .25}, daemon=True).start()
            url = "http://127.0.0.1:" + str(settings["http_port"]) + "/"
            print("C5_READY " + settings["role"] + " " + url, flush=True)
            if args.open:
                import webbrowser
                webbrowser.open(url)
            try:
                server.serve_forever(poll_interval=.25)
            except KeyboardInterrupt:
                pass
            finally:
                if data:
                    data.shutdown()
                app.close()
        finally:
            if data:
                data.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
