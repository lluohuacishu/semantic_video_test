"""Preserve a live TCP receipt and defer its decoder before GPU contention.

This is a recorded orchestration interruption, never an accepted decode.
The original receiver and runner directories remain intact for recovery.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

from b2_common import git, now, read, require, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receiver", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=7200)
    args = parser.parse_args()
    receiver = args.receiver.resolve()
    require(not args.output.exists(), "Deferral record already exists")
    started = now()
    deadline = time.monotonic() + args.timeout
    while not (receiver / "receive.json").exists():
        require(time.monotonic() < deadline, "No complete receipt before timeout")
        require(not (receiver / "failure.json").exists(), "Receiver failed before complete receipt")
        time.sleep(0.02)
    receipt = read(receiver / "receive.json")
    ready = read(receiver / "ready.json")
    require(receipt["verified"] and receipt["source"] == "actual TCP reception", "Not a verified TCP receipt")
    require(receipt["pid"] == ready["pid"], "Receiver PID mismatch")
    # Terminate this known experiment child only. Do not stop the sending process;
    # its transfer acknowledgement and trace must be allowed to finish normally.
    observed_start = (receiver / "decode_started.json").exists()
    observed_completion = (receiver / "completion.json").exists()
    require(not observed_completion, "Decoder already completed; do not interrupt it")
    command = ["taskkill.exe", "/PID", str(receipt["pid"]), "/F"]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    packet = receiver / "message.scv"
    digest = sha(packet)
    require(packet.stat().st_size == receipt["bytes"] and digest == receipt["sha256"], "Preserved receipt changed")
    write(args.output, {
        "status": "controlled_decoder_deferral" if result.returncode == 0 else "deferral_command_failed",
        "reason": "Independent public runners had no cross-run GPU lock; defer complete receipts and recover serially",
        "started_at": started, "deferred_at": now(), "receiver": str(receiver),
        "receiver_pid": receipt["pid"], "decode_started_observed": observed_start,
        "completion_observed": observed_completion, "command": command,
        "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
        "receipt_sha256": sha(receiver / "receive.json"), "packet_sha256": digest,
        "packet_bytes": packet.stat().st_size, "retransmission": False,
        "accepted_decode": False, "code_commit": git("rev-parse", "HEAD"),
        "script_sha256": sha(__file__), "argv": [sys.executable, *sys.argv],
    })
    require(result.returncode == 0, "Could not defer receiver; inspect record")
    print(f"Preserved {packet.stat().st_size} received bytes; deferred PID {receipt['pid']}", flush=True)


if __name__ == "__main__":
    main()
