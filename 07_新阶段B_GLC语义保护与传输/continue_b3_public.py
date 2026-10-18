"""Continue public TCP trials with preserved receipts and one GPU decoder at a time."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from b2_common import HERE, ROOT, Journal, git, now, read, require, sha, write
from run_b0_b1 import child_env
from run_b3_public import deployment_config, verify_one


def validate_receipt(directory, packet):
    """Require the original receiver's verified bytes and sender acknowledgement."""
    receipt = read(directory / "receiver/receive.json")
    transfer = read(directory / "transfer.json")
    received = directory / "receiver/message.scv"
    digest = sha(packet)
    require(receipt["verified"] is True and transfer["verified"] is True, "Unverified TCP receipt")
    require(receipt["source"] == "actual TCP reception", "Missing original TCP provenance")
    require(sha(received) == receipt["sha256"] == transfer["sha256"] == digest, "TCP receipt hash mismatch")
    require(received.stat().st_size == receipt["bytes"] == transfer["file_bytes"] == packet.stat().st_size,
            "TCP receipt size mismatch")
    require(transfer["application_bytes"] > transfer["file_bytes"], "Transmission header not counted")
    require((directory / "send_schedule.csv").is_file() and (directory / "transfer.csv").is_file(),
            "Original transfer traces missing")
    return receipt, transfer


@contextmanager
def decode_lease(lock_path, directory):
    """Windows releases this byte-range lock automatically if the holder exits."""
    import msvcrt
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lease = {"requested_at": now(), "lock_path": str(lock_path), "holder_pid": os.getpid(),
             "mode": "Windows cross-process byte-range lock", "status": "waiting"}
    write(directory / "gpu_lease.json", lease)
    with lock_path.open("a+b") as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + 3500
        while True:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                require(time.monotonic() < deadline, "GPU decode lease timed out")
                time.sleep(0.2)
        lease.update(acquired_at=now(), status="decoding")
        write(directory / "gpu_lease.json", lease)
        try:
            yield lease
            lease["status"] = "completed"
        except BaseException:
            lease["status"] = "failed"
            raise
        finally:
            lease["released_at"] = now()
            write(directory / "gpu_lease.json", lease)
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def copy_receipt(previous, directory, packet):
    receipt, transfer = validate_receipt(previous, packet)
    copies = ["transfer.json", "transfer.csv", "send_schedule.csv", "receiver/message.scv", "receiver/receive.json"]
    records = []
    for relative in copies:
        original, target = previous / relative, directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists(), "Recovery target already exists")
        shutil.copyfile(original, target)
        require(sha(original) == sha(target), "Receipt copy changed bytes")
        records.append({"source": str(original), "copy": str(target), "sha256": sha(target)})
    write(directory / "recovery_provenance.json", {"original_directory": str(previous), "files": records,
          "original_receiver_pid": receipt["pid"], "retransmission": False, "cached_RGB_used": False})
    return transfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sender", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--decode-lock", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, default=HERE / "configs/b0_baseline.json")
    parser.add_argument("--wait-for-prior", action="store_true")
    args = parser.parse_args()
    baseline = read(args.baseline_config)
    require(Path(sys.executable).resolve() == Path(baseline["runtime_dependencies"]["glc_python"]).resolve(),
            "Use frozen GLC Python")
    require(not git("status", "--porcelain"), "Commit recovery code before execution")
    prior = args.prior_run.resolve() / "network"
    if args.wait_for_prior:
        deadline = time.monotonic() + 7200
        while not any((prior / name).exists() for name in ("failure.json", "summary.json")):
            require(time.monotonic() < deadline, "Previous runner did not finish")
            time.sleep(0.5)
    require(any((prior / name).exists() for name in ("failure.json", "summary.json")), "Previous runner is still active")
    sender, run = args.sender.resolve(), args.run.resolve()
    require(read(sender / "encode.json")["status"] == "B3_public_candidate_sender_complete", "Sender incomplete")
    run.mkdir(parents=True, exist_ok=False)
    network = run / "network"
    network.mkdir()
    os.environ["FFMPEG_PATH"] = baseline["runtime_dependencies"]["ffmpeg"]
    sys.path.insert(0, str(ROOT / "01_主工程/semantic_mvp"))
    from transport import send_file
    from glc_layered import inspect_packet
    packet, manifest = sender / "message.scv", read(sender / "manifest.json")
    inspect_packet(packet)
    deployment = deployment_config(baseline)
    journal = Journal(network)
    invocation = journal.begin([sys.executable, *sys.argv], code_commit=git("rev-parse", "HEAD"), pid=os.getpid())
    write(network / "environment.json", {"code_commit": git("rev-parse", "HEAD"), "runner_sha256": sha(__file__),
          "deployment": deployment, "source_id": args.source_id, "prior_network": str(prior),
          "decode_lock": str(args.decode_lock.resolve()), "python": sys.version, "executable": sys.executable,
          "network_scope": "same-host TCP loopback; per-flow sender application-byte limit"})
    results, directories, actions = [], {}, {}
    process = None
    started = time.monotonic()
    try:
        for rate in (40, 20, 30):
            previous = prior / f"rate_{rate}kbps"
            result_path = previous / "result.json"
            if result_path.exists():
                validate_receipt(previous, packet)
                result = read(result_path)
                require(result["status"] == "passed" and result["frames"] == manifest["frames"]
                        and result["packet_sha256"] == sha(packet), "Previous accepted result differs")
                require(read(previous / "receiver/completion.json")["success"], "Previous receiver incomplete")
                directories[str(rate)] = str(previous)
                actions[str(rate)] = {"action": "retain_previous_pass", "result_sha256": sha(result_path),
                                      "retransmission": False, "new_decode": False}
                results.append(result)
                print(f"{args.source_id} {rate} kbps: retained previous accepted trial", flush=True)
                continue
            directory = network / f"rate_{rate}kbps"
            receiver = directory / "receiver"
            receiver.mkdir(parents=True)
            recovered = (previous / "receiver/receive.json").exists()
            if recovered:
                transfer = copy_receipt(previous, directory, packet)
            else:
                require(not (previous / "transfer.json").exists(), "Incomplete previous transfer evidence requires inspection")
            write(receiver / "deployment.json", deployment)
            argv = [sys.executable, "-B", str(HERE / "receiver_b2.py"), "--config", str(receiver / "deployment.json"),
                    "--output", str(receiver)] + (["--decode-existing"] if recovered else [])
            command = journal.begin(argv, cwd=str(receiver))
            tick = time.monotonic()
            with (directory / "receiver_process.log").open("x", encoding="utf-8") as log:
                process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=receiver, env=child_env())
                if not recovered:
                    deadline = time.monotonic() + 120
                    while not (receiver / "ready.json").exists():
                        require(process.poll() is None and time.monotonic() < deadline, "Receiver did not become ready")
                        time.sleep(0.2)
                    ready = read(receiver / "ready.json")
                    print(f"{args.source_id} {rate} kbps: sending {packet.stat().st_size} B", flush=True)
                    transfer = send_file(packet, "127.0.0.1", ready["port"], rate,
                                         trace=directory / "transfer.csv", schedule_trace=directory / "send_schedule.csv")
                    write(directory / "transfer.json", transfer)
                with decode_lease(args.decode_lock.resolve(), directory) as lease:
                    require(process.poll() is None, "Receiver exited before decode gate")
                    lease["receiver_launcher_pid"] = process.pid
                    write(receiver / "decode_allowed.json", {"allowed": True, "at": now(),
                          "scheduler_pid": os.getpid(), "decode_lock": str(args.decode_lock.resolve())})
                    print(f"{args.source_id} {rate} kbps: GPU lease acquired; independent decode", flush=True)
                    code = process.wait(timeout=3600)
                    journal.finish(command, code, time.monotonic() - tick)
                    require(code == 0, f"Receiver failed at {rate} kbps")
                process = None
            result = verify_one(rate, directory, packet, manifest, journal, sender / "private_frames",
                                sender / "private_masks", args.source_id)
            results.append(result)
            directories[str(rate)] = str(directory)
            actions[str(rate)] = {"action": "recover_received_packet" if recovered else "new_TCP_transfer",
                                  "retransmission": False, "new_decode": True}
            write(network / "progress.json", {"completed_rates": [r["rate_kbps"] for r in results], "actions": actions})
            print(f"{args.source_id} {rate} kbps: passed ({transfer['elapsed_seconds']:.3f} s original transfer)", flush=True)
        ordered = sorted(results, key=lambda r: r["rate_kbps"])
        require(len({r["enhanced_rgb_sequence_sha256"] for r in ordered}) == 1, "Cross-rate RGB mismatch")
        summary = {"status": "B3_public_candidate_three_rate_engineering_passed", "source_id": args.source_id,
                   "sender": str(sender), "packet_sha256": sha(packet), "packet_bytes": packet.stat().st_size,
                   "frames": manifest["frames"], "results": ordered, "result_directories": directories,
                   "actions": actions, "prior_network": str(prior), "all_rates_identical_enhanced_RGB": True,
                   "semantic_benefit": "not_evaluated; predicted masks are not ground truth; human review pending",
                   "limitations": ["public pretraining exposure unverified", "same-host TCP loopback",
                                   "Python audit is not OS isolation", "per-flow application bytes exclude TCP/IP and link bytes",
                                   "controlled decoder deferrals retained; original TCP receipts reused without retransmission"]}
        write(network / "summary.json", summary)
        journal.finish(invocation, 0, time.monotonic() - started)
        print(f"{args.source_id}: three rates accepted with GPU decode serialization", flush=True)
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        write(network / "failure.json", {"type": type(exc).__name__, "message": str(exc),
              "passed_rates": [r["rate_kbps"] for r in results], "result_directories": directories})
        journal.finish(invocation, 1, time.monotonic() - started)
        raise


if __name__ == "__main__":
    main()
