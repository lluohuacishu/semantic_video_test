"""TCP 文件收发；限速统计应用层发送字节，包含文件与传输头。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import socket
import struct
import threading
import time
from pathlib import Path

MAX_FILE_BYTES = 256 * 1024 * 1024


def check_cancel(event):
    if event is not None and event.is_set():
        raise InterruptedError("文件传输已停止")


def recv_exact(connection, count, event=None):
    chunks = bytearray()
    last_data = time.monotonic()
    while len(chunks) < count:
        check_cancel(event)
        try:
            chunk = connection.recv(min(count - len(chunks), 65536))
        except socket.timeout:
            if time.monotonic() - last_data > 30:
                raise TimeoutError("连接超过 30 秒没有收到数据")
            continue
        if not chunk:
            raise ConnectionError("文件没有收完整，连接已断开")
        chunks.extend(chunk)
        last_data = time.monotonic()
    return bytes(chunks)


def receive_connection(connection, output: Path, event=None):
    connection.settimeout(0.5)
    length = struct.unpack("!I", recv_exact(connection, 4, event))[0]
    if not 0 < length <= 1024:
        raise ValueError("文件传输头长度无效")
    info = json.loads(recv_exact(connection, length, event))
    if not isinstance(info, dict):
        raise ValueError("文件传输头应为对象")
    size, expected_hash = info.get("bytes"), info.get("sha256")
    if (type(size) is not int or not 0 < size <= MAX_FILE_BYTES
            or not isinstance(expected_hash, str) or len(expected_hash) != 64
            or any(c not in "0123456789abcdef" for c in expected_hash)):
        raise ValueError("文件传输头无效")
    temporary = output.with_name(output.name + ".part")
    if output.exists():
        raise FileExistsError(f"接收目标已经存在：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        remaining = size
        while remaining:
            chunk = recv_exact(connection, min(remaining, 4096), event)
            handle.write(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
    check_cancel(event)
    if digest.hexdigest() != expected_hash:
        raise ValueError("收到的文件校验失败，保留未完成文件供检查")
    temporary.replace(output)
    acknowledgement = json.dumps({"ok": True, "bytes": size, "sha256": expected_hash}).encode()
    connection.sendall(struct.pack("!I", len(acknowledgement)) + acknowledgement)
    return {"bytes": size, "sha256": expected_hash, "verified": True}


def send_file(source: Path, host: str, port: int, kbps: float, *, event=None, progress=None, trace=None):
    if not math.isfinite(kbps) or not 1 <= kbps <= 100000:
        raise ValueError("传输速度应为 1～100000 kbps")
    size = source.stat().st_size
    if not 0 < size <= MAX_FILE_BYTES:
        raise ValueError("传输文件为空或超过 256 MiB")
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    header = json.dumps({"bytes": size, "sha256": digest}, separators=(",", ":")).encode()
    prefix = struct.pack("!I", len(header)) + header
    rows = []
    with socket.create_connection((host, port), timeout=5) as connection, source.open("rb") as handle:
        connection.settimeout(0.5)
        started = time.monotonic()
        sent, payload_sent = 0, 0

        def paced_send(data):
            nonlocal sent
            deadline = started + (sent + len(data)) * 8 / (kbps * 1000)
            while True:
                check_cancel(event)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if event is None:
                    time.sleep(min(remaining, 0.1))
                else:
                    event.wait(min(remaining, 0.1))
            check_cancel(event)
            connection.sendall(data)
            sent += len(data)

        paced_send(prefix)
        while chunk := handle.read(1024):
            paced_send(chunk)
            payload_sent += len(chunk)
            elapsed = time.monotonic() - started
            rows.append((elapsed, payload_sent, sent, sent * 8 / elapsed / 1000))
            if progress:
                progress(payload_sent, size, elapsed)
        length = struct.unpack("!I", recv_exact(connection, 4, event))[0]
        if not 0 < length <= 1024:
            raise ValueError("接收确认无效")
        acknowledgement = json.loads(recv_exact(connection, length, event))
        elapsed = time.monotonic() - started
        if (not isinstance(acknowledgement, dict) or acknowledgement.get("ok") is not True or acknowledgement.get("sha256") != digest
                or acknowledgement.get("bytes") != size):
            raise ValueError("接收端未确认完整文件")
    if trace is not None:
        with Path(trace).open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("elapsed_seconds", "file_bytes_sent", "application_bytes_sent", "average_kbps"))
            writer.writerows(rows)
    return {"protocol": "TCP", "host": host, "port": port, "limit_kbps": kbps,
            "file_bytes": size, "application_bytes": sent, "sha256": digest, "verified": True,
            "elapsed_seconds": elapsed, "average_file_kbps": size * 8 / elapsed / 1000,
            "average_application_kbps": sent * 8 / elapsed / 1000,
            "minimum_application_seconds": sent * 8 / (kbps * 1000),
            "rate_definition": "十进制 kbps；发送方向的应用层文件和传输头，不含 TCP/IP 包头、链路层和反向确认"}


def loopback_transfer(source, output, kbps, *, event=None, progress=None, trace=None):
    stopped = threading.Event()

    class ReceiverCancellation:
        def is_set(self):
            return stopped.is_set() or (event is not None and event.is_set())

    receiver_cancel = ReceiverCancellation()
    errors = []
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(0.5)
    port = listener.getsockname()[1]

    def receiver():
        try:
            while True:
                check_cancel(receiver_cancel)
                try:
                    connection, _ = listener.accept()
                    break
                except socket.timeout:
                    continue
            with connection:
                receive_connection(connection, Path(output), receiver_cancel)
        except Exception as exc:
            errors.append(exc)
        finally:
            listener.close()

    worker = threading.Thread(target=receiver, name="file-receiver", daemon=True)
    worker.start()
    try:
        result = send_file(Path(source), "127.0.0.1", port, kbps, event=event, progress=progress, trace=trace)
        worker.join(timeout=2)
        if worker.is_alive():
            raise TimeoutError("接收线程没有正常结束")
        if errors:
            raise errors[0]
        return result
    except Exception:
        stopped.set()
        worker.join(timeout=2)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    receiver = subparsers.add_parser("receive")
    receiver.add_argument("--host", default="127.0.0.1")
    receiver.add_argument("--port", type=int, default=8771)
    receiver.add_argument("--output", type=Path, required=True)
    sender = subparsers.add_parser("send")
    sender.add_argument("--host", default="127.0.0.1")
    sender.add_argument("--port", type=int, default=8771)
    sender.add_argument("--input", type=Path, required=True)
    sender.add_argument("--kbps", type=float, default=20)
    args = parser.parse_args()
    if args.command == "send":
        result = send_file(args.input, args.host, args.port, args.kbps)
    else:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind((args.host, args.port))
            listener.listen(1)
            print(f"等待文件：{args.host}:{args.port}", flush=True)
            connection, _ = listener.accept()
            with connection:
                result = receive_connection(connection, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
