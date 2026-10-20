"""Bounded independent-service localhost validation. Always unlimited transfer."""
import argparse
import http.client
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import time

from c4_config import HERE, REPO, load_settings
from c2_watch import write
from c5_preflight import digest


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def call(port_number, method, route, value=None, upload=None):
    conn = http.client.HTTPConnection("127.0.0.1", port_number, timeout=180)
    try:
        if upload:
            with Path(upload).open("rb") as body:
                conn.request(method, route, body, {"Content-Length": str(Path(upload).stat().st_size)})
        else:
            conn.request(method, route, json.dumps(value).encode() if value is not None else None)
        res = conn.getresponse()
        data = res.read()
        if res.status >= 400:
            raise ValueError(data.decode())
        return json.loads(data)
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--stream", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    settings = load_settings()
    token = secrets.token_urlsafe(32)
    txport, rxport, dataport = port(), port(), port()
    processes, logs = [], []
    results = []
    try:
        for role, http_port in (("receiver", rxport), ("sender", txport)):
            conf = {"runtime": {k: str(v) for k, v in settings["runtime"].items()},
                    "data_dir": str(root / role), "host": "127.0.0.1", "http_port": http_port,
                    "data_port": dataport, "peer_url": f"http://127.0.0.1:{rxport}",
                    "token": token, "rate_mode": "max", "worker_mode": "cold"}
            config = root / (role + ".local.json")
            write(config, conf)
            log = (root / (role + ".log")).open("w", encoding="utf-8")
            logs.append(log)
            process = subprocess.Popen([str(settings["runtime"]["python"]), "-X", "utf8", "-B",
                str(HERE / ("c5_" + role + ".py")), "--config", str(config)], cwd=REPO,
                stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            processes.append((process, http_port))
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError(role + " service exited")
                try:
                    call(http_port, "GET", "/api/config")
                    break
                except OSError:
                    time.sleep(.1)
            else:
                raise TimeoutError(role + " startup")
        for name, source in (("reuse8", args.stream), ("encode10", args.source)):
            started = time.perf_counter()
            job = call(txport, "POST", "/api/jobs", {"filename": source.name, "bytes": source.stat().st_size,
                       "options": {"preset": "compact", "backend": "native", "sampling_mode": "time", "clip": {"mode": "first10"}}})
            identifier = job["id"]
            call(txport, "PUT", f"/api/jobs/{identifier}/upload", upload=source)
            call(txport, "POST", f"/api/jobs/{identifier}/start", {})
            previous = None
            for _ in range(1800):
                job = call(txport, "GET", "/api/jobs/" + identifier)
                state = (job["status"], job.get("stage"))
                if state != previous:
                    print(name, state, flush=True)
                    previous = state
                if job["status"] in ("done", "error", "cancelled", "interrupted"):
                    break
                time.sleep(.5)
            if job["status"] != "done":
                raise RuntimeError(json.dumps(job, ensure_ascii=False))
            result = job["result"]
            receiver = call(rxport, "GET", "/api/jobs/" + result["receiver_job_id"])
            if not result["verified"] or receiver["status"] != "done" or result["remote"]["source_media_reads"]:
                raise ValueError("Independent receiver checks failed")
            # Verify actual downloaded bytes, not merely presence of a link.
            conn = http.client.HTTPConnection("127.0.0.1", rxport, timeout=30)
            conn.request("GET", f"/jobs/{receiver['id']}/received.glcv?download=1")
            response = conn.getresponse()
            body = response.read()
            conn.close()
            import hashlib
            if response.status != 200 or hashlib.sha256(body).hexdigest() != result["stream_sha256"]:
                raise ValueError("Downloaded stream differs")
            results.append({"name": name, "sender_id": identifier, "receiver_id": receiver["id"],
                            "frames": result["frames"], "bytes": result["file_bytes"], "verified": True,
                            "rate_mode": "max", "wall_seconds": time.perf_counter() - started,
                            "transfer": result["transfer"], "source_media_reads": [], "download_sha_verified": True,
                            "two_physical_computers": False})
            write(root / "summary.json", {"status": "running", "results": results})
            print("C5_ENDPOINT_DONE " + json.dumps(results[-1]), flush=True)
        write(root / "summary.json", {"status": "passed", "results": results,
              "code_sha256": {p.name: digest(p) for p in HERE.glob("c5_*.py")},
              "rate_limited_experiments_performed": False, "two_physical_computers": False})
    finally:
        for process, http_port in processes:
            if process.poll() is None:
                call(http_port, "POST", "/api/shutdown", {})
                process.wait(timeout=90)
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
