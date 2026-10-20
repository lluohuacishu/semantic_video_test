"""C5 task envelope over TCP; legacy B transport entry points remain unchanged."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_主工程"))
from semantic_mvp.transport import MAX_FILE_BYTES, check_cancel, recv_exact

RATES = {"20": 20, "40": 40, "max": None}
ID = re.compile(r"[0-9a-f]{32}")


def rate_limit(mode):
    if not isinstance(mode, str) or mode not in RATES:
        raise ValueError("Choose 20, 40 or max")
    return RATES[mode]


def file_info(source):
    source = Path(source)
    size = source.stat().st_size
    if not 0 < size <= MAX_FILE_BYTES:
        raise ValueError("Bitstream must be nonempty and at most 256 MiB")
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"bytes": size, "sha256": digest}


def validate_header(info):
    if not isinstance(info, dict) or set(info) != {"version", "job_id", "ticket", "bytes", "sha256", "rate_mode"}:
        raise ValueError("Invalid C5 transfer header fields")
    if type(info["version"]) is not int or info["version"] != 2:
        raise ValueError("Unsupported C5 transfer version")
    if not isinstance(info["job_id"], str) or not ID.fullmatch(info["job_id"]):
        raise ValueError("Invalid receiver job ID")
    if not isinstance(info["ticket"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", info["ticket"]):
        raise ValueError("Invalid transfer ticket")
    if type(info["bytes"]) is not int or not 0 < info["bytes"] <= MAX_FILE_BYTES:
        raise ValueError("Invalid transfer size")
    if not isinstance(info["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", info["sha256"]):
        raise ValueError("Invalid transfer SHA-256")
    rate_limit(info["rate_mode"])
    return info


def wire_header(info):
    validate_header(info)
    raw = json.dumps(info, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return struct.pack("!I", len(raw)) + raw


def read_header(connection, event=None):
    connection.settimeout(0.25)
    size = struct.unpack("!I", recv_exact(connection, 4, event))[0]
    if not 0 < size <= 4096:
        raise ValueError("Invalid C5 transfer header length")
    raw = recv_exact(connection, size, event)
    return validate_header(json.loads(raw)), 4 + size


def send_all(connection, data, event=None):
    remaining = memoryview(data)
    last_progress = time.monotonic()
    while remaining:
        check_cancel(event)
        try:
            sent = connection.send(remaining)
        except socket.timeout:
            if time.monotonic() - last_progress > 30:
                raise TimeoutError("Send stalled for 30 seconds")
            continue
        if not sent:
            raise ConnectionError("Peer closed transfer connection")
        remaining = remaining[sent:]
        last_progress = time.monotonic()


class Pacer:
    def __init__(self, mode, event=None, clock=time.perf_counter, sleep=time.sleep):
        self.limit, self.event = rate_limit(mode), event
        self.clock, self.sleep = clock, sleep
        self.started, self.sent = clock(), 0

    def wait(self, size):
        if self.limit is None:
            check_cancel(self.event)
            return None
        scheduled = (self.sent + size) * 8 / (self.limit * 1000)
        deadline = self.started + scheduled
        while True:
            check_cancel(self.event)
            remaining = deadline - self.clock()
            if remaining <= 0:
                return scheduled
            if self.event is not None:
                self.event.wait(min(remaining, 0.1))
            else:
                self.sleep(min(remaining, 0.1))


def send_file(source, host, port, reservation, *, event=None, progress=None, trace=None):
    info = {**file_info(source), "version": 2, "job_id": reservation["id"],
            "ticket": reservation["ticket"], "rate_mode": reservation["rate_mode"]}
    prefix = wire_header(info)
    trace_file = Path(trace).open("x", encoding="utf-8", newline="") if trace else None
    writer = csv.writer(trace_file) if trace_file else None
    if writer:
        writer.writerow(["chunk_bytes", "application_bytes", "scheduled_seconds", "send_started", "send_finished"])
    connection_started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=5) as connection, Path(source).open("rb") as handle:
            connection.settimeout(0.25)
            pacer = Pacer(info["rate_mode"], event)
            connect_seconds = pacer.started - connection_started
            payload = 0

            def send(data):
                scheduled = pacer.wait(len(data))
                started = time.perf_counter() - pacer.started
                send_all(connection, data, event)
                pacer.sent += len(data)
                if writer:
                    writer.writerow([len(data), pacer.sent, scheduled, started, time.perf_counter() - pacer.started])

            send(prefix)
            while data := handle.read(1024):
                send(data)
                payload += len(data)
                if progress:
                    progress(payload, info["bytes"], time.perf_counter() - pacer.started)
            sent_seconds = time.perf_counter() - pacer.started
            size = struct.unpack("!I", recv_exact(connection, 4, event))[0]
            if not 0 < size <= 4096:
                raise ValueError("Invalid transfer acknowledgement")
            acknowledgement = json.loads(recv_exact(connection, size, event))
            if (not isinstance(acknowledgement, dict) or acknowledgement.get("ok") is not True
                    or acknowledgement.get("job_id") != info["job_id"]
                    or acknowledgement.get("bytes") != info["bytes"]
                    or acknowledgement.get("sha256") != info["sha256"]):
                raise ValueError("Receiver did not acknowledge this complete file")
            elapsed = time.perf_counter() - pacer.started
            return {"protocol": "C5-TCP-v2", "file_bytes": info["bytes"], "sha256": info["sha256"],
                    "job_id": info["job_id"], "rate_mode": info["rate_mode"], "limit_kbps": pacer.limit,
                    "application_bytes": pacer.sent, "header_bytes": len(prefix), "ack_bytes": size + 4,
                    "connect_seconds": connect_seconds, "send_seconds": sent_seconds,
                    "ack_wait_seconds": elapsed - sent_seconds, "elapsed_seconds": elapsed,
                    "average_application_kbps": pacer.sent * 8 / elapsed / 1000,
                    "minimum_application_seconds": pacer.sent * 8 / (pacer.limit * 1000) if pacer.limit else None,
                    "verified": True, "network_transfer_performed": True,
                    "rate_definition": "forward TCP application header + glcv; excludes control, ACK and network headers"}
    finally:
        if trace_file:
            trace_file.close()


def receive_file(connection, info, output, *, event=None, progress=None):
    validate_header(info)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_name(output.name + ".part")
    digest = hashlib.sha256()
    started = time.perf_counter()
    with part.open("xb") as handle:
        remaining = info["bytes"]
        while remaining:
            data = recv_exact(connection, min(remaining, 4096), event)
            handle.write(data)
            digest.update(data)
            remaining -= len(data)
            if progress:
                progress(info["bytes"] - remaining, info["bytes"], time.perf_counter() - started)
        handle.flush()
        os.fsync(handle.fileno())
    check_cancel(event)
    if digest.hexdigest() != info["sha256"]:
        raise ValueError("Received SHA-256 differs; partial retained")
    # Atomic no-clobber publication on the same filesystem.
    os.link(part, output)
    part.unlink()
    return {"bytes": info["bytes"], "sha256": info["sha256"], "verified": True,
            "payload_receive_seconds": time.perf_counter() - started}


def acknowledge(connection, info, event=None):
    raw = json.dumps({"ok": True, "job_id": info["job_id"],
                      "bytes": info["bytes"], "sha256": info["sha256"]}, separators=(",", ":")).encode()
    send_all(connection, struct.pack("!I", len(raw)) + raw, event)
    return 4 + len(raw)
