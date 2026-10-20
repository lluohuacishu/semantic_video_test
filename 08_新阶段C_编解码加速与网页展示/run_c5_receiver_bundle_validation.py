"""Run copied receiver assets through real TCP, independent decode and downloads.

Uses this machine's Python environment, not a newly installed target environment.
No source video, historical experiment directory, or sender task path is passed
to the receiver. Evidence and temporary endpoint configuration stay outside it.
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import subprocess
import time

from c2_watch import write
from c4_config import HERE, load_settings
from c5_server import Peer, inspect_stream
from c5_transport import send_file
from package_c5_receiver import verify_bundle
from run_c5_endpoints_validation import call, port


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stream", type=Path, required=True, action="append")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    integrity = verify_bundle(bundle)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    stage = bundle / "semantic_video_test" / HERE.name
    python = load_settings()["runtime"]["python"]
    http_port, data_port = port(), port()
    token = secrets.token_urlsafe(32)
    config = root / "receiver.local.json"
    write(config, {"base_config": str(stage / "runtime.local.json"), "runtime": {"python": str(python)},
                   "data_dir": str(root / "receiver"), "host": "127.0.0.1", "http_port": http_port,
                   "data_port": data_port, "token": token, "rate_mode": "max", "worker_mode": "cold"})
    results = []
    with (root / "receiver.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([str(python), "-X", "utf8", "-B", str(stage / "c5_receiver.py"),
                                    "--config", str(config)], cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        ready = False
        try:
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError("Copied receiver exited; inspect receiver.log")
                try:
                    public = call(http_port, "GET", "/api/config")
                    ready = True
                    break
                except OSError:
                    time.sleep(.1)
            if not ready:
                raise TimeoutError("Copied receiver startup")
            if public["role"] != "receiver":
                raise ValueError("Incorrect endpoint role")
            peer = Peer(f"http://127.0.0.1:{http_port}", token)
            preflight = peer.call("POST", "/control/preflight", {})
            write(root / "environment.json", preflight)
            for source in args.stream:
                description = inspect_stream(source)
                job = peer.call("POST", "/control/reserve", {"sender_id": secrets.token_hex(16), "stream": description})
                started = time.perf_counter()
                transfer = send_file(source, "127.0.0.1", data_port, job)
                deadline = time.monotonic() + 300
                while time.monotonic() < deadline:
                    job = call(http_port, "GET", "/api/jobs/" + job["id"])
                    if job["status"] in ("done", "error", "cancelled", "interrupted"):
                        break
                    time.sleep(.25)
                if job["status"] != "done":
                    raise ValueError("Copied receiver failed: " + json.dumps(job, ensure_ascii=False))
                folder = root / "receiver" / job["id"]
                decode = json.loads((folder / "decode/receiver/decode.json").read_text("utf-8"))
                if (not decode["all_frame_checksums_match"] or not decode["whole_file_checksum_verified"]
                        or not decode["independent_decoder"] or decode["source_media_reads"]
                        or decode["input_sha256"] != description["sha256"]
                        or decode["sequence_reconstruction_checksums"] != description["integrity"]["sequence_reconstruction_checksums"]):
                    raise ValueError("Copied receiver reconstruction or isolation check failed")
                downloaded = {}
                for name, local in (("received.glcv", folder / "received.glcv"),
                                    ("decoded.mp4", folder / "decode/receiver/reconstruction_preview.mp4")):
                    connection = http.client.HTTPConnection("127.0.0.1", http_port, timeout=30)
                    try:
                        connection.request("GET", f"/jobs/{job['id']}/{name}?download=1")
                        response = connection.getresponse()
                        data = response.read()
                        digest = hashlib.sha256(data).hexdigest()
                        if response.status != 200 or digest != hashlib.sha256(local.read_bytes()).hexdigest():
                            raise ValueError("Downloaded bundle result differs: " + name)
                        downloaded[name] = {"bytes": len(data), "sha256": digest}
                    finally:
                        connection.close()
                rows = {"frames": description["header"]["frames"], "bytes": description["bytes"],
                        "receiver_id": job["id"], "verified": True, "downloads": downloaded,
                        "transfer": transfer, "seconds": time.perf_counter() - started, "source_media_reads": []}
                results.append(rows)
                print("C5_BUNDLE_DONE " + json.dumps({k: rows[k] for k in ("frames", "bytes", "verified", "seconds")}), flush=True)
        finally:
            if process.poll() is None:
                if ready:
                    call(http_port, "POST", "/api/shutdown", {})
                    process.wait(timeout=90)
                else:
                    process.terminate()
                    process.wait(timeout=15)
    write(root / "summary.json", {"status": "passed", "bundle": str(bundle), "integrity": integrity,
          "results": results, "existing_python_environment_reused": True, "fresh_target_install_tested": False,
          "receiver_assets_from_bundle_only": True, "two_physical_computers": False,
          "rate_limited_experiments_performed": False})


if __name__ == "__main__":
    main()
