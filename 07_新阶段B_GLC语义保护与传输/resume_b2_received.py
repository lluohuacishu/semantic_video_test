"""Re-decode already received B2 files in fresh audited receiver directories.

No transfer timings or receiving PIDs are invented; original failed run is kept.
No RGB reconstruction cache is reused.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from b2_common import HERE, ROOT, Journal, git, now, read, require, sha, write
from run_b0_b1 import child_env
from run_b2 import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-network", default="network_retry")
    parser.add_argument("--previous-recovery", help="Keep passed rates at their original paths and decode only missing rates")
    args = parser.parse_args()
    root = args.run.resolve()
    sender, original = root / "sender", root / "network"
    require((original / "failure.json").exists(), "Original attempt has not ended with a recorded failure")
    require(not git("status", "--porcelain"), "Commit recovery code first")
    saved = read(sender / "config.json")
    deps = saved["baseline"]["runtime_dependencies"]
    require(Path(sys.executable).resolve() == Path(deps["glc_python"]).resolve(), "Use frozen GLC Python")
    require(Path(args.output_network).name == args.output_network, "Output must be a direct child of the run")
    output = root / args.output_network
    output.mkdir()
    os.environ["FFMPEG_PATH"] = deps["ffmpeg"]
    sys.path.insert(0, str(ROOT / "01_主工程/semantic_mvp"))
    original_environment = read(original / "environment.json")
    deployment = original_environment["deployment"]
    deployment["code_hashes"] = {path: sha(path) for path in deployment["code_hashes"]}
    deployment["wait_for_decode_gate"] = False
    journal = Journal(output)
    invocation = journal.begin([sys.executable, *sys.argv], code_commit=git("rev-parse", "HEAD"), cwd=str(Path.cwd()), pid=os.getpid())
    results, directories, decode_commits, reused = [], {}, {}, []
    previous = None
    if args.previous_recovery:
        require(Path(args.previous_recovery).name == args.previous_recovery, "Previous recovery must be a direct child")
        previous = root / args.previous_recovery
        require(previous != output and (previous / "failure.json").exists(), "Previous recovery must be a preserved failed attempt")
        previous_env = read(previous / "environment.json")
        require(previous_env["deployment"]["code_hashes"] == deployment["code_hashes"] and
                previous_env["deployment"]["shared_models"] == deployment["shared_models"], "Passed rates used different decoder dependencies")
        for rate in (40, 20, 30):
            directory = previous / f"rate_{rate}kbps"
            result_path = directory / "result.json"
            if not result_path.exists():
                continue
            result = read(result_path)
            decoded = read(directory / "receiver/decode.json")
            require(result["status"] == "passed" and result["rate_kbps"] == rate and result["decoded_frames"] == 250
                    and result["outside_roi_unchanged"] and result["composed_roi_matches_decoded_patch"] and result["playback_verified"]
                    and result["source_or_sender_cache_reads"] == 0 and read(directory / "receiver/completion.json")["success"], "Previous result incomplete")
            require(sha(directory / "receiver/message.scv") == sha(sender / "message.scv") == result["packet_sha256"]
                    and decoded["enhanced_rgb_sequence_sha256"] == result["enhanced_rgb_sequence_sha256"], "Previous accepted receipt changed")
            results.append(result)
            directories[str(rate)] = str(directory)
            decode_commits[str(rate)] = previous_env["code_commit"]
            reused.append({"rate_kbps": rate, "result_path": str(result_path), "result_sha256": sha(result_path),
                           "meaning": "Already independently decoded and passed; preserved in place, not input to another decoder"})
            print(f"B2 recovery keeps previously passed {rate} kbps result in place", flush=True)
    write(output / "environment.json", {"code_commit": invocation["code_commit"], "recovery_script_sha256": sha(__file__),
          "python": sys.version, "executable": sys.executable, "deployment": deployment,
          "original_network": str(original), "original_failure": read(original / "failure.json"),
          "transfer_reused": True, "RGB_cache_reused": False, "recovery_started_at": now(),
          "original_environment_sha256": sha(original / "environment.json"), "monotonic_clock": original_environment["monotonic_clock"],
          "previous_recovery": str(previous) if previous else None, "reused_completed_results": reused})
    current = None
    tick = time.perf_counter()
    fresh_decodes = 0
    try:
        for rate in (40, 20, 30):
            if str(rate) in directories:
                continue
            prior = original / f"rate_{rate}kbps"
            transfer = read(prior / "transfer.json")
            receipt = read(prior / "receiver/receive.json")
            received = prior / "receiver/message.scv"
            digest = sha(received)
            require(transfer["verified"] and receipt["verified"] and digest == transfer["sha256"] == receipt["sha256"] == sha(sender / "message.scv"), "Original receipt evidence invalid")
            directory = output / f"rate_{rate}kbps"
            receiver = directory / "receiver"
            receiver.mkdir(parents=True)
            shutil.copyfile(received, receiver / "message.scv")
            require(sha(receiver / "message.scv") == digest, "Copied received file changed")
            for name in ("transfer.json", "transfer.csv", "send_schedule.csv"):
                shutil.copyfile(prior / name, directory / name)
            shutil.copyfile(prior / "receiver/receive.json", receiver / "receive.json")
            write(receiver / "deployment.json", deployment)
            provenance = {"actual_receipt_directory": str(prior / "receiver"), "original_receiving_pid": receipt["pid"],
                          "received_file_sha256": digest, "actual_transfer_result_sha256": sha(prior / "transfer.json"),
                          "transfer_reused": True, "cached_RGB_reused": False, "copied_at": now()}
            write(directory / "receipt_provenance.json", provenance)
            argv = [sys.executable, "-B", "-u", "-X", "faulthandler", str(HERE / "receiver_b2.py"), "--config", str(receiver / "deployment.json"), "--output", str(receiver), "--decode-existing"]
            row = journal.begin(argv, cwd=str(receiver))
            started = time.perf_counter()
            print(f"B2 recovery {rate} kbps: re-decode actual received file; no retransmission or RGB cache reuse", flush=True)
            with (directory / "receiver_process.log").open("x", encoding="utf-8") as log:
                current = subprocess.Popen(argv, cwd=receiver, env=child_env(), stdout=log, stderr=subprocess.STDOUT)
                code = current.wait(timeout=3600)
            journal.finish(row, code, time.perf_counter() - started)
            require(code == 0, f"Recovery receiver failed: {rate}")
            result = verify(rate, directory, sender, journal)
            result["receipt_provenance"] = provenance
            write(directory / "result.json", result)
            results.append(result)
            directories[str(rate)] = str(directory)
            decode_commits[str(rate)] = invocation["code_commit"]
            fresh_decodes += 1
            write(output / "progress.json", {"completed_rates": [r["rate_kbps"] for r in results]})
            print(f"B2 recovery {rate} kbps passed: 250 frames", flush=True)
        results.sort(key=lambda r: r["rate_kbps"])
        require([r["rate_kbps"] for r in results] == [20, 30, 40] and len({r["enhanced_rgb_sequence_sha256"] for r in results}) == 1, "Rates incomplete or RGB differs")
        write(output / "summary.json", {"status": "B2_three_rate_engineering_passed", "code_commit": invocation["code_commit"], "results": results,
              "result_directories": directories, "independent_decode_commits": decode_commits,
              "all_rates_identical_enhanced_RGB": True, "recovery": {"original_network": str(original), "original_failure": read(original / "failure.json"),
              "transfer_results_reused": True, "cached_RGB_reused": False, "fresh_independent_decodes": fresh_decodes,
              "previous_recovery": str(previous) if previous else None, "reused_completed_results": reused,
              "accepted_independent_decodes_total": len(results)},
              "semantic_benefit": "B3_controls_and_human_review_pending",
              "limits": ["Same-host TCP loopback; sender application bytes only", "Python audit is not OS isolation", "Fresh receiver decoding of preserved actual TCP receipts; no repeated transmission"]})
        write(root / "accepted_network.json", {"directory": args.output_network, "summary_sha256": sha(output / "summary.json")})
        journal.finish(invocation, 0, time.perf_counter() - tick)
    except BaseException as exc:
        write(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        journal.finish(invocation, 1, time.perf_counter() - tick)
        raise
    finally:
        if current is not None and current.poll() is None:
            current.terminate()
            current.wait(timeout=30)


if __name__ == "__main__":
    main()
