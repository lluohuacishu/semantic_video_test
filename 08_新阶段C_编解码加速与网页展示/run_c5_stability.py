"""Finite real endpoint recovery and ten-success sequence; max speed only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import time

from c2_watch import inspect, write
from c4_config import HERE, REPO, load_settings
from c5_server import Peer, inspect_stream
from c5_transport import wire_header
from run_c5_endpoints_validation import call, port

TERMINAL = {"done", "cancelled", "error", "interrupted"}


def wait_job(number, identifier, predicate, timeout=1200):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = call(number, "GET", "/api/jobs/" + identifier)
        if predicate(job):
            return job
        if job["status"] in TERMINAL:
            raise RuntimeError("Unexpected terminal state: " + json.dumps(job, ensure_ascii=False))
        time.sleep(.25)
    raise TimeoutError(identifier)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--stream", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    runtime = load_settings()["runtime"]
    long_source = root / "source_60s_320.mp4"
    subprocess.run([str(runtime["ffmpeg"]), "-v", "error", "-nostdin", "-n", "-stream_loop", "-1",
                    "-i", str(args.second), "-t", "60", "-vf", "scale=320:-2", "-r", "25",
                    "-an", "-c:v", "libx264", "-crf", "18", str(long_source)], check=True, timeout=120,
                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    token = secrets.token_urlsafe(32)
    txport, rxport, dataport = port(), port(), port()
    processes, logs, rows, cancellations = [], [], [], []

    def submit(source, preset="compact", clip=None):
        job = call(txport, "POST", "/api/jobs", {"filename": source.name, "bytes": source.stat().st_size,
                   "options": {"preset": preset, "backend": "native", "sampling_mode": "time",
                               "clip": clip or {"mode": "first10"}}})
        call(txport, "PUT", f"/api/jobs/{job['id']}/upload", upload=source)
        call(txport, "POST", f"/api/jobs/{job['id']}/start", {})
        return job["id"]

    def finish(identifier):
        job = wait_job(txport, identifier, lambda j: j["status"] in TERMINAL)
        if job["status"] != "done":
            raise RuntimeError(json.dumps(job, ensure_ascii=False))
        result = job["result"]
        receiver = call(rxport, "GET", "/api/jobs/" + result["receiver_job_id"])
        if receiver["status"] != "done" or result["remote"]["source_media_reads"] or not result["verified"]:
            raise ValueError("Independent receiver checks failed")
        return result

    try:
        for role, http_port in (("receiver", rxport), ("sender", txport)):
            config = root / (role + ".local.json")
            write(config, {"runtime": {k: str(v) for k, v in runtime.items()}, "data_dir": str(root / role),
                  "host": "127.0.0.1", "http_port": http_port, "data_port": dataport, "token": token,
                  "peer_url": f"http://127.0.0.1:{rxport}", "rate_mode": "max", "worker_mode": "cold"})
            log = (root / (role + ".log")).open("w", encoding="utf-8")
            logs.append(log)
            process = subprocess.Popen([str(runtime["python"]), "-X", "utf8", "-B",
                str(HERE / ("c5_" + role + ".py")), "--config", str(config)], cwd=REPO,
                stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            processes.append((process, http_port))
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError("Endpoint exited")
                try:
                    call(http_port, "GET", "/api/config")
                    break
                except OSError:
                    time.sleep(.1)
            else:
                raise TimeoutError("Endpoint startup")
        # Real encoding cancellation after observed neural progress.
        identifier = submit(args.source, clip={"mode": "full"})
        job = wait_job(txport, identifier, lambda j: j.get("stage") == "encode" and
                       ((j.get("heartbeat") or {}).get("last_progress") or {}).get("frame_completed", 0) >= 2)
        started = time.perf_counter()
        call(txport, "POST", f"/api/jobs/{identifier}/cancel", {})
        job = wait_job(txport, identifier, lambda j: j["status"] in TERMINAL)
        if job["status"] != "cancelled":
            raise ValueError("Encode cancellation failed")
        cancellations.append({"stage": "encode", "id": identifier, "seconds": time.perf_counter()-started,
                              "status": job["status"]})
        print("C5_CANCEL " + json.dumps(cancellations[-1]), flush=True)
        # Real receiving cancellation: send a partial file at max, then stop the producer.
        peer = Peer(f"http://127.0.0.1:{rxport}", token)
        description = inspect_stream(args.stream)
        reservation = peer.call("POST", "/control/reserve", {"sender_id": secrets.token_hex(16), "stream": description})
        header = {"version": 2, "job_id": reservation["id"], "ticket": reservation["ticket"], "rate_mode": "max",
                  "bytes": description["bytes"], "sha256": description["sha256"]}
        with socket.create_connection(("127.0.0.1", dataport)) as connection:
            connection.sendall(wire_header(header) + args.stream.read_bytes()[:4096])
            wait_job(rxport, reservation["id"], lambda j: j.get("transferred_bytes", 0) >= 4096)
            started = time.perf_counter()
            call(rxport, "POST", f"/api/jobs/{reservation['id']}/cancel", {})
            job = wait_job(rxport, reservation["id"], lambda j: j["status"] in TERMINAL)
        if job["status"] != "cancelled" or (root / "receiver" / job["id"] / "received.glcv").exists():
            raise ValueError("Transfer cancellation published an incomplete stream")
        cancellations.append({"stage": "receive", "id": job["id"], "seconds": time.perf_counter()-started,
                              "status": job["status"], "rate_mode": "max", "partial_sender_stopped": True})
        print("C5_CANCEL " + json.dumps(cancellations[-1]), flush=True)
        # Real decoder cancellation requested from the sender after remote frame progress.
        identifier = submit(args.stream)
        job = wait_job(txport, identifier, lambda j:
                       ((j.get("remote_heartbeat") or {}).get("last_progress") or {}).get("frame_completed", 0) >= 2)
        receiver_id = job["receiver_job_id"]
        started = time.perf_counter()
        call(txport, "POST", f"/api/jobs/{identifier}/cancel", {})
        job = wait_job(txport, identifier, lambda j: j["status"] in TERMINAL)
        remote = wait_job(rxport, receiver_id, lambda j: j["status"] in TERMINAL)
        if job["status"] != "cancelled" or remote["status"] != "cancelled":
            raise ValueError("Remote decode cancellation did not reach both endpoints")
        cancellations.append({"stage": "decode", "sender_id": identifier, "receiver_id": receiver_id,
                              "seconds": time.perf_counter()-started, "status": remote["status"]})
        print("C5_CANCEL " + json.dumps(cancellations[-1]), flush=True)
        write(root / "cancellations.json", cancellations)
        cases = [(args.source, "compact", {"mode": "first10"}), (args.second, "compact", {"mode": "first10"}),
                 (args.source, "compact", {"mode": "first10"})]
        cases += [(args.second, preset, {"mode": "segment", "start": .2, "end": .36})
                  for preset in ("compact", "motion", "detail", "detail_motion", "compact", "motion")]
        cases += [(long_source, "compact", {"mode": "full"})]
        for number, (source, preset, clip) in enumerate(cases, 1):
            started = time.perf_counter()
            identifier = submit(source, preset, clip)
            result = finish(identifier)
            row = {"number": number, "id": identifier, "receiver_id": result["receiver_job_id"],
                   "frames": result["frames"], "bytes": result["file_bytes"], "width": result["header"]["width"],
                   "height": result["header"]["height"], "seconds": result["frames"]/result["header"]["fps"],
                   "preset": preset, "clip": clip, "wall_seconds": time.perf_counter()-started,
                   "rate_mode": "max", "verified": True}
            rows.append(row)
            write(root / "summary.json", {"status": "running", "successes": rows, "cancellations": cancellations})
            print("C5_STABILITY_DONE " + json.dumps(row), flush=True)
        states = {str(p.relative_to(root)): inspect(p.parent)["observed_state"]
                  for p in root.glob("*/*/*_process/process.json")}
        if any(s in ("running", "orphan_running", "liveness_unknown") for s in states.values()):
            raise ValueError("A worker remains active/uncertain")
        write(root / "summary.json", {"status": "passed", "successes": rows, "cancellations": cancellations,
              "process_states": states, "two_physical_computers": False, "rate_limited_experiments_performed": False,
              "long_video_definition": "60 seconds, 320x180 source, compact 750 output frames"})
    finally:
        for process, http_port in processes:
            if process.poll() is None:
                call(http_port, "POST", "/api/shutdown", {})
                process.wait(timeout=90)
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
