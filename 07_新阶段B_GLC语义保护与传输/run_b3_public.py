"""Run actual B2-format TCP delivery and independent decode for one public clip."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from b2_common import HERE, ROOT, Journal, git, now, read, require, sha, write
from run_b0_b1 import child_env, validate_schedule


def deployment_config(baseline):
    deps = baseline["runtime_dependencies"]
    adapter = (ROOT / baseline["adapter"]["repository_path"]).resolve()
    paths = [adapter.parent / name for name in ("glc_video_codec.py", "glc_container.py", "glc_entropy.py", "glc_video_probe.py")]
    paths += [ROOT / "01_主工程/semantic_mvp" / name for name in
              ("transport.py", "runtime.py", "roi_layout.py", "system.py", "glc_layered.py", "glc_layered_decoder.py")]
    paths += [HERE / name for name in ("receiver_b2.py", "receiver_baseline.py", "b2_common.py")]
    return {
        "adapter": str(adapter), "glc_repo": str(Path(deps["glc_repo"]).resolve()),
        "weights": str(Path(deps["weights"]).resolve()), "ffmpeg": str(Path(deps["ffmpeg"]).resolve()),
        "transport": str(ROOT / "01_主工程/semantic_mvp/transport.py"),
        "shared_models": {str((Path(deps["weights"]) / name).resolve()): digest
                          for name, digest in baseline["codec"]["expected_weight_hashes_from_a"].items()},
        "code_hashes": {str(p.resolve()): sha(p) for p in paths},
        "wait_for_decode_gate": True,
    }


def verify_one(rate, directory, packet, expected_manifest, journal, source_dir, sender_masks, source_id):
    import numpy as np
    from PIL import Image
    receiver = directory / "receiver"
    decoded = read(receiver / "decode.json")
    manifest = read(receiver / "received_manifest.json")
    require(manifest == expected_manifest, "Received manifest differs from frozen sender manifest")
    require(read(receiver / "completion.json")["success"], "Receiver incomplete")
    require(sha(receiver / "message.scv") == sha(packet) == decoded["packet_sha256"], "Packet checksum differs")
    audit = read(receiver / "receiver_read_audit.json")
    require(not audit["denied_reads"] and not audit["external_source_or_sender_cache_reads"]
            and all(row["allowed"] for row in audit["subprocesses"]), "Receiver read audit failed")
    shared = {row["path"] for row in audit["python_reads"] if row["category"] == "declared_shared_model"}
    require(shared == set(audit["shared_model_allowlist"]), "Shared-model access audit incomplete")
    require(decoded["receiver_pid"] != os.getpid() and decoded["glc_encoder_modules_disabled"]
            and not decoded["glc_source_media_reads"], "Independent decoder checks failed")
    count, width, height = manifest["frames"], manifest["width"], manifest["height"]
    require(decoded["frames"] == count and len(decoded["enhanced_rgb_sha256"]) == count,
            "Decoded frame count mismatch")
    require([path.name for path in sorted((receiver / "rgb").glob("*.png"))]
            == [f"{i:05d}.png" for i in range(count)], "Enhanced RGB sequence incomplete")
    rows, enhanced_hashes = [], []
    for index in range(count):
        with Image.open(source_dir / f"{index:05d}.png") as image:
            source = np.asarray(image.convert("RGB"))
        with Image.open(receiver / "base_decoded/frames" / f"im{index+1:05d}.png") as image:
            base = np.asarray(image.convert("RGB"))
        with Image.open(receiver / "rgb" / f"{index:05d}.png") as image:
            require(image.size == (width, height), "Enhanced frame geometry mismatch")
            enhanced = np.asarray(image.convert("RGB"))
        base_hash = hashlib.sha256(base.tobytes()).hexdigest()
        enhanced_hash = hashlib.sha256(enhanced.tobytes()).hexdigest()
        require(base_hash == decoded["base_rgb_sha256"][index], "Saved GLC RGB hash mismatch")
        require(enhanced_hash == decoded["enhanced_rgb_sha256"][index], "Saved enhanced RGB hash mismatch")
        protected = np.zeros((height, width), dtype=bool)
        for number, patch in enumerate(manifest["patches"]):
            if patch["start"] <= index < patch["start"] + patch["frames"]:
                x, y, patch_width, patch_height = patch["box"]
                protected[y:y+patch_height, x:x+patch_width] = True
                with Image.open(receiver / f"roi_rgb_{number:04d}" / f"{index-patch['start']:05d}.png") as image:
                    roi = np.asarray(image.convert("RGB"))
                require(np.array_equal(enhanced[y:y+patch_height, x:x+patch_width], roi), "Composed ROI differs from received ROI decode")
        require(np.array_equal(base[~protected], enhanced[~protected]), "Pixels outside transmitted ROI changed")
        mae = float(np.abs(source.astype(np.int16) - enhanced.astype(np.int16)).mean())
        mask_diagnostics = {}
        for target in ("fish", "diver"):
            with Image.open(sender_masks / target / "raw" / f"{index:05d}.png") as image:
                mask = np.asarray(image.convert("L")) > 0
            if mask.any():
                error = source[mask].astype(np.float64) - enhanced[mask].astype(np.float64)
                mse = float(np.square(error).mean())
                mask_diagnostics[target] = {"prediction_pixels": int(mask.sum()),
                                            "psnr_db": float(10 * np.log10(255**2 / mse)) if mse else None}
            else:
                mask_diagnostics[target] = {"prediction_pixels": 0, "psnr_db": None}
        rows.append({"frame": index, "source_vs_enhanced_mae": mae, "predicted_region_diagnostics": mask_diagnostics,
                     "outside_roi_unchanged": True, "received_patch_matches_composite": True})
        enhanced_hashes.append(enhanced_hash)
    require(hashlib.sha256("".join(enhanced_hashes).encode()).hexdigest()
            == decoded["enhanced_rgb_sequence_sha256"], "Enhanced RGB sequence digest mismatch")
    transfer = read(directory / "transfer.json")
    pacing = validate_schedule(directory / "send_schedule.csv", rate, transfer["application_bytes"],
                               time.get_clock_info("monotonic").resolution)
    import runtime
    runtime.run = journal.run
    probe = runtime.probe(receiver / "enhanced_preview.mp4")
    require([probe["width"], probe["height"], int(probe["nb_read_frames"])]
            == [width, height, count] and runtime.fps_string(probe["avg_frame_rate"]) == manifest["fps"],
            "Enhanced playback properties differ")
    journal.run([runtime.FFMPEG, "-v", "error", "-i", receiver / "enhanced_preview.mp4", "-f", "null", "-"])
    write(directory / "frame_verification.json", {"source_id": source_id, "source_read_by_evaluator": True,
          "source_read_by_receiver": False, "frames": rows})
    write(directory / "pacing_verification.json", pacing)
    result = {"status": "passed", "rate_kbps": rate, "application_bytes": transfer["application_bytes"],
              "elapsed_seconds": transfer["elapsed_seconds"], "frames": count, "packet_sha256": decoded["packet_sha256"],
              "enhanced_rgb_sequence_sha256": decoded["enhanced_rgb_sequence_sha256"],
              "receiver_pid": decoded["receiver_pid"], "evaluator_pid": os.getpid(),
              "shared_models_read": len(shared), "external_source_or_cache_reads": 0,
              "outside_roi_unchanged": True, "received_patch_matches_composite": True,
              "pacing": pacing, "playback_verified": True}
    write(directory / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sender", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, default=HERE / "configs/b0_baseline.json")
    parser.add_argument("--rates", type=int, nargs="+", default=[40, 20, 30])
    parser.add_argument("--source-id", required=True)
    args = parser.parse_args()
    baseline = read(args.baseline_config)
    require(Path(sys.executable).resolve() == Path(baseline["runtime_dependencies"]["glc_python"]).resolve(),
            "Run network/receiver orchestration from pinned GLC Python")
    require(not git("status", "--porcelain"), "Commit run_b3_public.py before public-candidate transport")
    sender, run = args.sender.resolve(), args.run.resolve()
    require(read(sender / "encode.json")["status"] == "B3_public_candidate_sender_complete", "Sender is incomplete")
    run.mkdir(parents=True, exist_ok=False)
    network = run / "network"
    network.mkdir()
    os.environ["FFMPEG_PATH"] = baseline["runtime_dependencies"]["ffmpeg"]
    sys.path.insert(0, str(ROOT / "01_主工程/semantic_mvp"))
    from transport import send_file
    import runtime
    from glc_layered import inspect_packet
    packet = sender / "message.scv"
    manifest = read(sender / "manifest.json")
    inspect_packet(packet)
    deployment = deployment_config(baseline)
    journal = Journal(network)
    invocation = journal.begin([sys.executable, *sys.argv], code_commit=git("rev-parse", "HEAD"), pid=os.getpid())
    order = list(args.rates)
    require(set(order) == {20, 30, 40} and len(order) == 3, "Use all frozen B application-layer rates once")
    write(network / "environment.json", {"code_commit": git("rev-parse", "HEAD"), "runner_sha256": sha(__file__),
          "execution_order_kbps": order, "python": sys.version, "executable": sys.executable,
          "monotonic_clock": vars(time.get_clock_info("monotonic")), "deployment": deployment,
          "network_scope": "same-host TCP loopback; sender application bytes", "source_id": args.source_id})
    results = []
    try:
        for rate in order:
            directory = network / f"rate_{rate}kbps"
            receiver = directory / "receiver"
            receiver.mkdir(parents=True)
            write(receiver / "deployment.json", deployment)
            argv = [sys.executable, "-B", str(HERE / "receiver_b2.py"), "--config",
                    str(receiver / "deployment.json"), "--output", str(receiver)]
            command = journal.begin(argv, cwd=str(receiver))
            log = (directory / "receiver_process.log").open("x", encoding="utf-8")
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=receiver, env=child_env())
            deadline = time.monotonic() + 120
            while not (receiver / "ready.json").exists():
                require(process.poll() is None and time.monotonic() < deadline, "Receiver failed before ready")
                time.sleep(0.2)
            ready = read(receiver / "ready.json")
            print(f"B3 {args.source_id} {rate} kbps sending {packet.stat().st_size} B", flush=True)
            transfer = send_file(packet, "127.0.0.1", ready["port"], rate,
                                 trace=directory / "transfer.csv", schedule_trace=directory / "send_schedule.csv")
            write(directory / "transfer.json", transfer)
            write(receiver / "decode_allowed.json", {"allowed": True, "at": now(), "prior_receiver_pid": None})
            code = process.wait(timeout=3600)
            log.close()
            journal.finish(command, code, transfer["elapsed_seconds"])
            require(code == 0, f"Receiver failed at {rate} kbps")
            source_dir = sender / "private_frames"
            masks = sender / "private_masks"
            result = verify_one(rate, directory, packet, manifest, journal, source_dir, masks, args.source_id)
            results.append(result)
            write(network / "progress.json", {"completed_rates": [r["rate_kbps"] for r in results]})
            print(f"B3 {args.source_id} {rate} kbps passed independent decode ({result['elapsed_seconds']:.1f}s)", flush=True)
        ordered = sorted(results, key=lambda row: row["rate_kbps"])
        require(len({row["enhanced_rgb_sequence_sha256"] for row in ordered}) == 1,
                "Received RGB differs across application rates")
        summary = {"status": "B3_public_candidate_three_rate_engineering_passed", "source_id": args.source_id,
                   "sender": str(sender), "packet_sha256": sha(packet), "packet_bytes": packet.stat().st_size,
                   "frames": manifest["frames"], "results": ordered,
                   "all_rates_identical_enhanced_RGB": True,
                   "semantic_benefit": "not_evaluated; prediction masks are not ground truth and human review is pending",
                   "limitations": ["public-pretraining exposure is unverified", "same-host loopback", "Python audit is not OS isolation",
                                   "application-byte rate excludes TCP/IP and link-layer bytes"]}
        write(network / "summary.json", summary)
        journal.finish(invocation, 0, sum(r["elapsed_seconds"] for r in results))
        print(f"B3 {args.source_id}: all three rates independently decoded", flush=True)
    except BaseException as exc:
        write(network / "failure.json", {"type": type(exc).__name__, "message": str(exc),
              "passed_rates": [row["rate_kbps"] for row in results]})
        journal.finish(invocation, 1, 0)
        raise


if __name__ == "__main__":
    main()
