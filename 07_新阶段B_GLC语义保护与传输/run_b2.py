"""Three actual B2 transfers; serial GPU decoding may overlap later transfers."""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from b2_common import HERE, ROOT, Journal, git, now, read, require, sha, write
from run_b0_b1 import child_env, validate_schedule


def verify(rate, directory, sender, journal):
    import numpy as np
    from PIL import Image
    receiver = directory / "receiver"
    decoded = read(receiver / "decode.json")
    manifest = read(receiver / "received_manifest.json")
    require(manifest == read(sender / "manifest.json"), "Received manifest differs")
    require(read(receiver / "completion.json")["success"], "Receiver incomplete")
    require(sha(receiver / "message.scv") == sha(sender / "message.scv") == decoded["packet_sha256"], "Packet checksum differs")
    audit = read(receiver / "receiver_read_audit.json")
    require(not audit["denied_reads"] and not audit["external_source_or_sender_cache_reads"] and all(r["allowed"] for r in audit["subprocesses"]), "Receiver audit failed")
    shared = {r["path"] for r in audit["python_reads"] if r["category"] == "declared_shared_model"}
    require(shared == set(audit["shared_model_allowlist"]), "Shared model audit incomplete")
    require(decoded["receiver_pid"] != os.getpid() and decoded["glc_encoder_modules_disabled"] and not decoded["glc_source_media_reads"], "Decoder independence failed")
    baseline_result = read(HERE / "evidence_b0_b1_20261018/rate_20kbps/result.json")
    require(decoded["base_rgb_sequence_sha256"] == baseline_result["rgb_sequence_sha256"], "B2 base differs from B1")
    n, w, h = (manifest[k] for k in ("frames", "width", "height"))
    require(len(decoded["enhanced_rgb_sha256"]) == n and [p.name for p in sorted((receiver / "rgb").glob("*.png"))] == [f"{i:05d}.png" for i in range(n)], "Enhanced sequence incomplete")
    rows, hashes = [], []
    for index in range(n):
        with Image.open(receiver / "base_decoded/frames" / f"im{index + 1:05d}.png") as image:
            base = np.asarray(image.convert("RGB"))
        with Image.open(receiver / "rgb" / f"{index:05d}.png") as image:
            require(image.size == (w, h), "Enhanced dimensions differ")
            enhanced = np.asarray(image.convert("RGB"))
        digest = hashlib.sha256(enhanced.tobytes()).hexdigest()
        require(digest == decoded["enhanced_rgb_sha256"][index], "Saved enhanced RGB differs")
        require(hashlib.sha256(base.tobytes()).hexdigest() == decoded["base_rgb_sha256"][index], "Saved base RGB differs")
        protected = np.zeros((h, w), dtype=bool)
        for number, patch in enumerate(manifest["patches"]):
            if patch["start"] <= index < patch["start"] + patch["frames"]:
                x, y, pw, ph = patch["box"]
                protected[y:y+ph, x:x+pw] = True
                with Image.open(receiver / f"roi_rgb_{number:04d}" / f"{index - patch['start']:05d}.png") as image:
                    roi = np.asarray(image.convert("RGB"))
                require(np.array_equal(enhanced[y:y+ph, x:x+pw], roi), "ROI composition mismatch")
        require(np.array_equal(base[~protected], enhanced[~protected]), "Pixels outside transmitted ROI changed")
        rows.append({"frame": index, "enhanced_rgb_sha256": digest, "protected_pixels": int(protected.sum()), "outside_roi_unchanged": True, "patch_pixels_match": True})
        hashes.append(digest)
    require(hashlib.sha256("".join(hashes).encode()).hexdigest() == decoded["enhanced_rgb_sequence_sha256"], "Enhanced sequence checksum differs")
    write(directory / "frame_verification.json", {"evaluator_pid": os.getpid(), "source_media_read": False, "frames": rows})
    transfer = read(directory / "transfer.json")
    pacing = validate_schedule(directory / "send_schedule.csv", rate, transfer["application_bytes"], time.get_clock_info("monotonic").resolution)
    write(directory / "pacing_verification.json", pacing)
    import runtime
    runtime.run = journal.run
    preview = receiver / "enhanced_preview.mp4"
    probe = runtime.probe(preview)
    require([probe["width"], probe["height"], int(probe["nb_read_frames"])] == [w, h, n] and runtime.fps_string(probe["avg_frame_rate"]) == manifest["fps"], "Enhanced playback metadata differs")
    journal.run([runtime.FFMPEG, "-v", "error", "-i", preview, "-f", "null", "-"])
    write(directory / "playback_probe.json", probe)
    result = {"status": "passed", "rate_kbps": rate, "decoded_frames": n, "packet_sha256": decoded["packet_sha256"],
              "base_identical_to_B1": True, "enhanced_rgb_sequence_sha256": decoded["enhanced_rgb_sequence_sha256"],
              "outside_roi_unchanged": True, "composed_roi_matches_decoded_patch": True, "source_or_sender_cache_reads": 0,
              "shared_models_read": len(shared), "receiver_pid": decoded["receiver_pid"], "evaluator_pid": os.getpid(),
              "transfer": transfer, "pacing": pacing, "playback_verified": True, "decode_and_preview_seconds": decoded["decode_and_preview_seconds"],
              "preview": str(preview)}
    write(directory / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    sender = run / "sender"
    saved = read(sender / "config.json")
    baseline, config = saved["baseline"], saved["b2"]
    deps = baseline["runtime_dependencies"]
    require(Path(sys.executable).resolve() == Path(deps["glc_python"]).resolve(), "Use pinned GLC Python")
    require(read(sender / "encode.json")["status"] == "B2_sender_complete", "Sender not complete")
    require(not git("status", "--porcelain"), "Commit executed code before running B2")
    network = run / "network"
    network.mkdir()
    os.environ["FFMPEG_PATH"] = deps["ffmpeg"]
    sys.path.insert(0, str(ROOT / "01_主工程/semantic_mvp"))
    from transport import send_file
    import runtime
    from glc_layered import inspect_packet
    packet = sender / "message.scv"
    inspect_packet(packet)
    for path, digest in read(sender / "code_hashes.json").items():
        require(sha(path) == digest, f"Sender dependency changed before receiver run: {path}")
    adapter = (ROOT / baseline["adapter"]["repository_path"]).resolve()
    paths = [adapter.parent / name for name in ("glc_video_codec.py", "glc_container.py", "glc_entropy.py", "glc_video_probe.py")]
    paths += [ROOT / "01_主工程/semantic_mvp" / name for name in ("transport.py", "runtime.py", "roi_layout.py", "system.py", "glc_layered.py", "glc_layered_decoder.py")]
    paths += [HERE / name for name in ("receiver_b2.py", "receiver_baseline.py", "b2_common.py")]
    deployment = {"adapter": str(adapter), "glc_repo": str(Path(deps["glc_repo"]).resolve()), "weights": str(Path(deps["weights"]).resolve()),
                  "ffmpeg": str(Path(deps["ffmpeg"]).resolve()), "transport": str(ROOT / "01_主工程/semantic_mvp/transport.py"),
                  "shared_models": {str((Path(deps["weights"]) / name).resolve()): digest for name, digest in baseline["codec"]["expected_weight_hashes_from_a"].items()},
                  "code_hashes": {str(p): sha(p) for p in paths}, "wait_for_decode_gate": True}
    journal = Journal(network)
    invocation = journal.begin([sys.executable, *sys.argv], cwd=str(Path.cwd()), code_commit=git("rev-parse", "HEAD"), pid=os.getpid())
    order = [40, 20, 30]
    require(set(order) == set(config["rates_kbps"]), "Rate configuration differs")
    write(network / "environment.json", {"code_commit": git("rev-parse", "HEAD"), "runner_sha256": sha(__file__), "execution_order_kbps": order,
          "schedule": "Independent transfers may overlap a previous decode; only one GPU receiver decodes at a time, gated on previous process success.",
          "monotonic_clock": vars(time.get_clock_info("monotonic")), "python": sys.version, "executable": sys.executable,
          "ffmpeg_sha256": sha(deps["ffmpeg"]), "deployment": deployment})
    processes, logs, threads = [], [], []
    tick = time.perf_counter()
    try:
        previous = None
        for rate in order:
            directory = network / f"rate_{rate}kbps"
            receiver = directory / "receiver"
            receiver.mkdir(parents=True)
            write(receiver / "deployment.json", deployment)
            argv = [sys.executable, "-B", str(HERE / "receiver_b2.py"), "--config", str(receiver / "deployment.json"), "--output", str(receiver)]
            command = journal.begin(argv, cwd=str(receiver))
            log = (directory / "receiver_process.log").open("x", encoding="utf-8")
            logs.append(log)
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=receiver, env=child_env())
            processes.append((rate, process, command, time.perf_counter()))
            def permit_after_prior(prior=previous, destination=receiver):
                if prior is None or prior.wait() == 0:
                    write(destination / "decode_allowed.json", {"allowed": True, "at": now(), "prior_receiver_pid": prior.pid if prior else None})
            thread = threading.Thread(target=permit_after_prior, daemon=True)
            thread.start()
            threads.append(thread)
            previous = process
            deadline = time.monotonic() + 120
            while not (receiver / "ready.json").exists():
                require(process.poll() is None and time.monotonic() < deadline, f"Receiver not ready: {directory}")
                time.sleep(0.2)
            ready = read(receiver / "ready.json")
            write(network / "progress.json", {"phase": "sending", "rate_kbps": rate, "receiver_pid": ready["pid"]})
            print(f"B2 {rate} kbps sending {packet.stat().st_size} bytes", flush=True)
            transfer = send_file(packet, "127.0.0.1", ready["port"], rate, trace=directory / "transfer.csv", schedule_trace=directory / "send_schedule.csv")
            write(directory / "transfer.json", transfer)
            print(f"B2 {rate} kbps transfer complete: {transfer['elapsed_seconds']:.3f}s", flush=True)
        results = []
        for rate, process, command, started in processes:
            code = process.wait(timeout=3600)
            journal.finish(command, code, time.perf_counter() - started)
            require(code == 0, f"Receiver failed: rate_{rate}kbps")
            result = verify(rate, network / f"rate_{rate}kbps", sender, journal)
            results.append(result)
            write(network / "progress.json", {"phase": "verification", "completed_rates": [r["rate_kbps"] for r in results]})
            print(f"B2 {rate} kbps passed: all 250 composed frames and read audit verified", flush=True)
        results.sort(key=lambda r: r["rate_kbps"])
        require(len({r["enhanced_rgb_sequence_sha256"] for r in results}) == 1, "Enhanced RGB differs across rates")
        write(network / "summary.json", {"status": "B2_three_rate_engineering_passed", "code_commit": invocation["code_commit"], "results": results,
              "all_rates_identical_enhanced_RGB": True, "semantic_benefit": "B3_controls_and_human_review_pending",
              "limits": ["same-host TCP loopback and sender application bytes only", "Python read audit is not OS isolation", "shared GLC models/environment required"]})
        journal.finish(invocation, 0, time.perf_counter() - tick)
        print("B2 three-rate engineering checks complete", flush=True)
    except BaseException as exc:
        write(network / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        journal.finish(invocation, 1, time.perf_counter() - tick)
        raise
    finally:
        for _rate, process, _command, _started in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
