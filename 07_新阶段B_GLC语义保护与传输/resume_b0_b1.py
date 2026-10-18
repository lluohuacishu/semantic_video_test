"""Resume missing B1 trials without replacing accepted results or inventing logs."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import os
from pathlib import Path
import sys

import run_b0_b1 as baseline


def now():
    return datetime.now().astimezone().isoformat()


def verify_existing(config, frozen, packets, mapping, rate_dir):
    """Read-only recheck of all retained frames and original acceptance evidence."""
    import numpy as np
    from PIL import Image
    result = baseline.read_json(rate_dir / "result.json")
    receiver = rate_dir / "receiver"
    decoded = baseline.read_json(receiver / "decoded/decode.json")
    audit = baseline.read_json(receiver / "receiver_read_audit.json")
    completion = baseline.read_json(receiver / "completion.json")
    require = baseline.require
    require(completion["success"] and completion["pid"] == result["receiver_pid"], "Receiver completion differs")
    require(result["receiver_pid"] != result["evaluator_pid"], "Receiver was not separate")
    require(not audit["denied_reads"] and not audit["external_source_or_sender_cache_reads"], "Read audit failed")
    require(all(row["allowed"] for row in audit["subprocesses"]), "Subprocess audit failed")
    require({row["path"] for row in audit["python_reads"] if row["category"] == "declared_shared_model"}
            == set(audit["shared_model_allowlist"]), "Model audit incomplete")
    require(decoded["independent_decoder"] and decoded["encoder_modules_disabled"] and not decoded["source_media_reads"], "Decoder contract failed")
    require(baseline.sha(receiver / "received.glcv") == decoded["input_sha256"] == result["received_sha256"]
            == config["baseline_stream"]["sha256"], "Stream checksum differs")
    require(len(decoded["frames"]) == len(packets) == result["decoded_frames"], "Frame count differs")
    frames = sorted((receiver / "decoded/frames").glob("*.png"))
    require([p.name for p in frames] == [row["decoded_filename"] for row in mapping], "Frame order differs")
    hashes = []
    for index, path in enumerate(frames):
        with Image.open(path) as image:
            require(image.size == (config["source"]["width"], config["source"]["height"]), "Frame geometry differs")
            digest = hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()
        require(digest == packets[index]["rgb_sha256"] == decoded["frames"][index]["rgb_sha256"], "RGB mismatch")
        require(decoded["frames"][index]["latent_sha256"] == packets[index]["latent_sha256"], "Latent mismatch")
        hashes.append(digest)
    require(hashlib.sha256("".join(hashes).encode()).hexdigest() == result["rgb_sequence_sha256"], "Sequence mismatch")
    transfer = baseline.read_json(rate_dir / "transfer.json")
    require(transfer == result["transfer"] and transfer["verified"], "Transfer evidence differs")
    pacing = baseline.validate_schedule(rate_dir / "send_schedule.csv", result["rate_kbps"], transfer["application_bytes"],
                                        result["pacing"]["clock_resolution_seconds"])
    require(pacing == result["pacing"], "Pacing evidence differs")
    require(result["playback_verified"] and (rate_dir / "playback_decode.log").exists()
            and (rate_dir / "playback_decode.log").stat().st_size == 0, "Playback verification absent")
    require(Path(result["preview"]).is_file(), "Preview absent")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    output = args.run.resolve()
    baseline.require(not (output / "summary.json").exists(), "Run already completed")
    baseline.require(not (output / "recovery.json").exists(), "Recovery already attempted; inspect its evidence first")
    config = baseline.read_json(output / "config.json")
    frozen = baseline.read_json(output / "b0_frozen.json")
    baseline.require(config == frozen["config"], "Frozen config differs")
    baseline.require(Path(sys.executable).resolve() == Path(config["runtime_dependencies"]["glc_python"]).resolve(), "Use pinned Python")
    baseline.require(baseline.git("rev-parse", "HEAD") == frozen["code_commit"], "Execution commit changed")
    for path, digest in frozen["code_hashes"].items():
        baseline.require(baseline.sha(path) == digest, f"Pinned code changed: {path}")
    baseline.require(baseline.sha(config["baseline_stream"]["path"]) == config["baseline_stream"]["sha256"], "Baseline changed")
    sys.path.insert(0, str((baseline.ROOT / config["adapter"]["repository_path"]).parent))
    from glc_container import read_stream
    from glc_video_codec import configure
    from glc_video_probe import check_source, check_weights
    deps = config["runtime_dependencies"]
    baseline.require(configure("cuda") == config["codec"]["arithmetic_profile"], "Arithmetic environment changed")
    check_source(Path(deps["glc_repo"]))
    baseline.require(check_weights(Path(deps["weights"])) == frozen["shared_weights"], "Shared weights changed")
    _, packets = read_stream(config["baseline_stream"]["path"])
    mapping = baseline.read_json(output / "input_frame_mapping.json")["frames"]
    sys.path.insert(0, str((baseline.ROOT / config["transport_plan"]["implementation"]).parent))
    commands = [{"argv": [sys.executable, *sys.argv], "cwd": str(Path.cwd()), "started_at": now(),
                 "baseline_code_commit": frozen["code_commit"], "recovery_script_sha256": baseline.sha(__file__)}]
    recovery = {"started_at": now(), "pid": os.getpid(), "baseline_code_commit": frozen["code_commit"],
                "helper_sha256": baseline.sha(__file__), "retained_rates": [], "rerun_rates": [],
                "preserved_incomplete_attempts": [],
                "original_commands_log_missing": not (output / "commands.json").exists(),
                "observation": "After interruption no matching runner/receiver OS process remained; 40 kbps reception existed but completion/result did not. Original process exit codes and in-memory command log were not captured."}
    baseline.write_json(output / "recovery.json", recovery)
    try:
        results = []
        for rate in config["transport_plan"]["rates_kbps"]:
            rate_dir = output / f"rate_{rate}kbps"
            if (rate_dir / "result.json").exists():
                results.append(verify_existing(config, frozen, packets, mapping, rate_dir))
                recovery["retained_rates"].append(rate)
                print(f"Retained {rate} kbps: all 250 existing RGB frames rechecked without rewriting evidence", flush=True)
            else:
                if rate_dir.exists():
                    destination = output / f"interrupted_rate_{rate}kbps"
                    baseline.require(not destination.exists(), "Interrupted archive already exists")
                    inventory = [{"relative_path": str(p.relative_to(rate_dir)), "bytes": p.stat().st_size, "sha256": baseline.sha(p)}
                                 for p in sorted(rate_dir.rglob("*")) if p.is_file()]
                    rate_dir.rename(destination)
                    recovery["preserved_incomplete_attempts"].append({"original_path": str(rate_dir), "preserved_path": str(destination), "files": inventory})
                recovery["rerun_rates"].append(rate)
                baseline.write_json(output / "recovery.json", recovery)
                results.append(baseline.trial(config, frozen, packets, mapping, rate, output, commands))
            baseline.write_json(output / "progress.json", {"completed_rates": [r["rate_kbps"] for r in results]})
            baseline.write_json(output / "recovery.json", recovery)
        baseline.require(len({r["rgb_sequence_sha256"] for r in results}) == 1, "RGB differs across rates")
        recovery.update(status="completed", finished_at=now())
        baseline.write_json(output / "recovery.json", recovery)
        baseline.write_json(output / "summary.json", {
            "status": "B0_B1_passed", "code_commit": frozen["code_commit"], "results": results,
            "all_rates_identical_received_RGB": True, "semantic_benefit": "not_evaluated_B2_B3_pending",
            "recovery": {"record": "recovery.json", "retained_rates": recovery["retained_rates"], "rerun_rates": recovery["rerun_rates"],
                         "original_command_log_complete": False},
            "limits": ["Same-host TCP loopback, software application-rate limit only; no physical acoustic link or cross-device claim.",
                       "TCP/IP, link-layer overhead and reverse ACK bytes excluded.", "Python audit and subprocess argument checks are not an OS sandbox.",
                       "GLC weights and matching GPU/software environment must be pre-shared.", "Preview quality is not a human viewing score.",
                       "Original command-log finalization was interrupted; retained per-rate evidence is intact and rechecked; 40 kbps was rerun. Missing original command exit codes/timings are not reconstructed."]})
        print("B0+B1 complete: three accepted transfers and 750 independently decoded frames", flush=True)
    except BaseException as exc:
        recovery.update(status="failed", failure_type=type(exc).__name__, failure_message=str(exc), finished_at=now())
        baseline.write_json(output / "recovery.json", recovery)
        raise
    finally:
        baseline.write_json(output / "recovery_commands.json", commands)
        if recovery["original_commands_log_missing"]:
            baseline.write_json(output / "commands.json", {
                "original_runner_log": {"available": False, "reason": "Abrupt interruption before original finally block; no original return codes or command timings invented."},
                "recovery_commands": commands,
                "original_per_rate_evidence": ["rate_20kbps/result.json", "rate_30kbps/result.json"],
                "reproduction_instructions": "B0_B1复现说明.md"})


if __name__ == "__main__":
    main()
