"""Freeze the existing A stream, then run three paced TCP/independent-decode trials.

Run with the existing GLC Python environment. Output must be a new directory.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text("utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, encoding="utf-8").strip()


def command(argv, log, commands):
    argv = list(map(str, argv))
    tick = time.perf_counter()
    with Path(log).open("x", encoding="utf-8") as handle:
        result = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, env=child_env())
    commands.append({"argv": argv, "log": str(log), "returncode": result.returncode,
                     "seconds": time.perf_counter() - tick})
    require(result.returncode == 0, f"Command failed; see {log}")


def child_env():
    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8",
            "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}


def validate_schedule(path, rate, expected_bytes, resolution):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), "Missing send schedule")
    cumulative, lags = 0, []
    for row in rows:
        chunk = int(row["chunk_bytes"])
        require(chunk > 0, "Empty send")
        cumulative += chunk
        require(cumulative == int(row["cumulative_application_bytes"]), "Incorrect cumulative send bytes")
        deadline = cumulative * 8 / (rate * 1000)
        require(abs(float(row["scheduled_seconds"]) - deadline) <= max(resolution, 1e-8), "Incorrect schedule deadline")
        start, finish = float(row["send_started_seconds"]), float(row["send_finished_seconds"])
        require(finish >= start, "Invalid send timestamps")
        lags.append(start - deadline)
    require(cumulative == expected_bytes, "Schedule omits header or payload")
    early = max(0.0, -min(lags))
    require(early <= resolution, "Sender released bytes before their cumulative deadline")
    return {"calls_including_header": len(rows), "application_bytes": cumulative,
            "clock_resolution_seconds": resolution, "maximum_early_seconds": early,
            "minimum_start_lag_seconds": min(lags), "maximum_start_lag_seconds": max(lags),
            "all_calls_on_or_after_deadline": early == 0, "within_clock_resolution": early <= resolution,
            "measurement": "timestamps immediately before and after sendall; includes initial transmission header"}


def freeze(config, output, commands):
    import numpy as np
    from PIL import Image
    deps = config["runtime_dependencies"]
    adapter = ROOT / config["adapter"]["repository_path"]
    for path in deps.values():
        require(Path(path).exists(), f"Missing dependency: {path}")
    require(Path(sys.executable).resolve() == Path(deps["glc_python"]).resolve(), "Use the configured GLC Python")
    code_paths = [adapter.parent / name for name in ("glc_video_codec.py", "glc_container.py", "glc_entropy.py", "glc_video_probe.py")]
    runtime_dir = Path(config["adapter"]["runtime_path"]).parent
    matches = []
    for path in code_paths:
        runtime = runtime_dir / path.name
        match = sha(path) == sha(runtime)
        require(match, f"Repository/runtime adapter mismatch: {path.name}")
        matches.append({"repository": str(path), "runtime": str(runtime), "sha256": sha(path), "match": match})
    require(sha(adapter) == config["adapter"]["sha256"], "A adapter hash changed")
    sys.path.insert(0, str(adapter.parent))
    from glc_container import read_stream
    from glc_video_codec import configure
    from glc_video_probe import check_source, check_weights
    arithmetic = configure("cuda")
    require(arithmetic == config["codec"]["arithmetic_profile"], "GLC arithmetic environment changed")
    check_source(Path(deps["glc_repo"]))
    weights = check_weights(Path(deps["weights"]))
    require({n: r["sha256"] for n, r in weights.items()} == config["codec"]["expected_weight_hashes_from_a"], "Weight inventory mismatch")
    source_stream = Path(config["baseline_stream"]["path"])
    require(sha(source_stream) == config["baseline_stream"]["sha256"], "Baseline stream hash mismatch")
    require(source_stream.stat().st_size == config["baseline_stream"]["bytes"], "Baseline size mismatch")
    header, packets = read_stream(source_stream)
    source = config["source"]
    expected = {k: source[k] for k in ("frames", "width", "height", "fps")}
    expected.update(q=config["codec"]["quality"], intra_period=config["codec"]["intra_period"], reset_interval=config["codec"]["reset_interval"])
    require(all(header[k] == v for k, v in expected.items()), "Baseline header differs from configuration")
    manifest_path = Path(source["a_inputs_manifest"])
    require(sha(manifest_path) == source["a_inputs_manifest_sha256"], "Input manifest hash mismatch")
    inputs = read_json(manifest_path)
    require(len(inputs["records"]) == source["frames"] == inputs["frames"], "Input frame count mismatch")
    require(inputs["source_start"] == source["source_start_index"] and inputs["video"] == int(source["video_id"].removeprefix("video")), "Wrong source clip")
    require(all(inputs[k] == source[k] for k in ("width", "height", "fps")), "Source geometry/timing mismatch")
    mapping = []
    for index, record in enumerate(inputs["records"]):
        require(record["source_frame"] == source["source_start_index"] + index, "Discontinuous source order")
        original = Path(inputs["source_dir"]) / f"{record['source_frame']:05d}.jpg"
        normalized = manifest_path.parent / "source" / f"im{index+1:05d}.png"
        encoder_input = source_stream.parent / "sender/source" / normalized.name
        require(sha(original) == record["source_sha256"], f"Original file changed at {index}")
        require(sha(normalized) == record["png_sha256"], f"Normalized file changed at {index}")
        for path in (normalized, encoder_input):
            with Image.open(path) as image:
                require(image.size == (source["width"], source["height"]), "Source dimensions differ")
                require(hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest() == record["rgb_sha256"], f"Input RGB differs at {index}")
        mapping.append({"index": index, "source_frame": record["source_frame"], "source_file": str(original),
                        "source_file_sha256": record["source_sha256"], "input_rgb_sha256": record["rgb_sha256"],
                        "decoded_filename": normalized.name, "expected_reconstruction_sha256": packets[index]["rgb_sha256"]})
    write_json(output / "input_frame_mapping.json", {"source_role": source["role"], "frames": mapping, "ground_truth_used": False})
    command([deps["segmentation_python"], "-B", HERE / "freeze_semantics.py", "--output", output / "semantic_models.json"], output / "freeze_semantics.log", commands)
    semantic = read_json(output / "semantic_models.json")
    for target, planned in config["semantic_plan"]["target_profiles"].items():
        actual = semantic["targets"][target]
        require(actual["model_sha256"] == planned["model_sha256"] and actual["threshold"] == planned["threshold"], "Semantic profile changed")
    code_paths += [ROOT / config["transport_plan"]["implementation"], HERE / "receiver_baseline.py", HERE / "run_b0_b1.py", HERE / "freeze_semantics.py",
                   ROOT / "01_主工程/semantic_mvp/teacher.py", ROOT / "01_主工程/semantic_mvp/roi_layout.py", ROOT / "01_主工程/adapters/semantic_teacher.py", ROOT / "01_主工程/adapters/common.py"]
    versions = {}
    for package in ("torch", "torchvision", "numpy", "Pillow", "timm", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "arithmetic": arithmetic, "packages": versions,
                   "ffmpeg_sha256": sha(deps["ffmpeg"]), "monotonic_clock": vars(time.get_clock_info("monotonic"))}
    command([deps["glc_python"], "-m", "pip", "freeze"], output / "glc_packages.txt", commands)
    command([deps["ffmpeg"], "-version"], output / "ffmpeg_version.txt", commands)
    command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], output / "gpu_driver.txt", commands)
    write_json(output / "environment.json", environment)
    z_bytes = sum(len(p["z"]) for p in packets)
    y_bytes = sum(sum(map(len, p["y"])) for p in packets)
    frozen = {"status": "B0_frozen", "config": config, "code_commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
              "git_status_at_start": git("status", "--porcelain"), "code_hashes": {str(p): sha(p) for p in code_paths},
              "adapter_matches": matches, "header": header, "checked_source_frames": len(mapping), "shared_weights": weights,
              "byte_ledger": {"glc_codebook_bytes": z_bytes, "glc_entropy_bytes": y_bytes,
                              "glc_container_and_checksums_bytes": source_stream.stat().st_size - z_bytes - y_bytes,
                              "glc_file_bytes": source_stream.stat().st_size, "roi_bytes": 0, "semantic_message_bytes": 0,
                              "roi_position_time_bytes": 0, "outer_package_bytes": 0,
                              "shared_GLC_weight_bytes_not_sent": sum(r["bytes"] for r in weights.values())}}
    write_json(output / "b0_frozen.json", frozen)
    return frozen, packets, mapping


def verify_received(config, rate_dir, packets, mapping):
    import numpy as np
    from PIL import Image
    receiver = rate_dir / "receiver"
    received = receiver / "received.glcv"
    require(sha(received) == config["baseline_stream"]["sha256"], "Received file hash mismatch")
    require(read_json(receiver / "completion.json")["success"], "Receiver did not complete")
    decoded = read_json(receiver / "decoded/decode.json")
    audit = read_json(receiver / "receiver_read_audit.json")
    require(not audit["denied_reads"] and not audit["external_source_or_sender_cache_reads"], "Receiver accessed a forbidden input")
    require(not decoded["source_media_reads"] and decoded["independent_decoder"] and decoded["encoder_modules_disabled"], "Independent decoding contract failed")
    shared_read = {r["path"] for r in audit["python_reads"] if r["category"] == "declared_shared_model"}
    require(shared_read == set(audit["shared_model_allowlist"]), "Shared model audit incomplete")
    require(all(c["allowed"] for c in audit["subprocesses"]), "Receiver subprocess contract failed")
    require(decoded["input_sha256"] == config["baseline_stream"]["sha256"], "Decoder used a different file")
    require(len(decoded["frames"]) == len(packets), "Decoded frame count mismatch")
    frames = sorted((receiver / "decoded/frames").glob("*.png"))
    require([p.name for p in frames] == [m["decoded_filename"] for m in mapping], "Output frame sequence mismatch")
    hashes, rows, squared, samples = [], [], 0.0, 0
    source_folder = Path(config["source"]["a_inputs_manifest"]).parent / "source"
    for index, path in enumerate(frames):
        with Image.open(path) as image:
            require(image.size == (config["source"]["width"], config["source"]["height"]), "Decoded dimensions differ")
            rgb = np.asarray(image.convert("RGB"))
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        require(digest == packets[index]["rgb_sha256"] == decoded["frames"][index]["rgb_sha256"], "Decoded RGB checksum mismatch")
        require(decoded["frames"][index]["latent_sha256"] == packets[index]["latent_sha256"], "Latent checksum mismatch")
        with Image.open(source_folder / path.name) as original:
            error = rgb.astype(np.float32) - np.asarray(original.convert("RGB"), dtype=np.float32)
        squared += float(np.square(error).sum(dtype=np.float64))
        samples += error.size
        hashes.append(digest)
        rows.append({"index": index, "source_frame": mapping[index]["source_frame"], "rgb_sha256": digest, "sender_checksum_matches": True})
    write_json(rate_dir / "frame_verification.json", {"frames": rows, "evaluation_process_pid": os.getpid(), "source_reads_are_only_in_this_separate_evaluator": True})
    return {"decoded_frames": len(frames), "source_frame_range": [mapping[0]["source_frame"], mapping[-1]["source_frame"]],
            "all_rgb_and_latent_checksums_match": True, "received_sha256": sha(received),
            "rgb_sequence_sha256": hashlib.sha256("".join(hashes).encode()).hexdigest(),
            "external_source_or_sender_cache_reads": 0, "shared_models_read": len(shared_read),
            "receiver_pid": read_json(receiver / "completion.json")["pid"], "evaluator_pid": os.getpid(),
            "decode_seconds_before_preview": decoded["pipeline_seconds_before_preview"],
            "full_psnr_db_from_received_RGB": 10 * math.log10(255**2 / (squared / samples)) if squared else None,
            "preview": str(receiver / "decoded/reconstruction_preview.mp4"),
            "preview_encoding": "viewing copy: H.264 CRF16 YUV420, original 15 fps; not transmitted; no human rating"}


def trial(config, frozen, packets, mapping, rate, output, commands):
    from transport import send_file
    rate_dir = output / f"rate_{rate}kbps"
    rate_dir.mkdir()
    receiver = rate_dir / "receiver"
    receiver.mkdir()
    deps = config["runtime_dependencies"]
    adapter = str((ROOT / config["adapter"]["repository_path"]).resolve())
    deployment = {"adapter": adapter, "glc_repo": str(Path(deps["glc_repo"]).resolve()),
                  "weights": str(Path(deps["weights"]).resolve()), "ffmpeg": str(Path(deps["ffmpeg"]).resolve()),
                  "transport": str((ROOT / config["transport_plan"]["implementation"]).resolve()),
                  "shared_models": {str((Path(deps["weights"]) / n).resolve()): r["sha256"] for n, r in frozen["shared_weights"].items()},
                  "code_hashes": {p: digest for p, digest in frozen["code_hashes"].items()
                                  if Path(p).parent == Path(adapter).parent or Path(p).name in {"transport.py", "receiver_baseline.py"}}}
    write_json(receiver / "deployment.json", deployment)
    argv = [deps["glc_python"], "-B", str(HERE / "receiver_baseline.py"), "--config", str(receiver / "deployment.json"), "--output", str(receiver)]
    tick = time.perf_counter()
    with (rate_dir / "receiver_process.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=receiver, env=child_env())
        try:
            deadline = time.monotonic() + 120
            while not (receiver / "ready.json").exists():
                require(process.poll() is None, f"Receiver failed before ready; see {rate_dir}")
                require(time.monotonic() < deadline, "Receiver ready timeout")
                time.sleep(0.2)
            ready = read_json(receiver / "ready.json")
            require(ready["pid"] != os.getpid(), "Receiver is not a separate process")
            print(f"B1 {rate} kbps: sending the existing .glcv file", flush=True)
            transfer = send_file(Path(config["baseline_stream"]["path"]), "127.0.0.1", ready["port"], rate,
                                 trace=rate_dir / "transfer.csv", schedule_trace=rate_dir / "send_schedule.csv")
            write_json(rate_dir / "transfer.json", transfer)
            print(f"B1 {rate} kbps: {transfer['application_bytes']} application bytes sent in {transfer['elapsed_seconds']:.3f}s; decoding", flush=True)
            code = process.wait(timeout=2400)
            require(code == 0, f"Receiver failed; see {rate_dir}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            commands.append({"argv": argv, "cwd": str(receiver), "returncode": process.returncode,
                             "log": str(rate_dir / "receiver_process.log"), "seconds": time.perf_counter() - tick})
    resolution = time.get_clock_info("monotonic").resolution
    pacing = validate_schedule(rate_dir / "send_schedule.csv", rate, transfer["application_bytes"], resolution)
    write_json(rate_dir / "pacing_verification.json", pacing)
    result = verify_received(config, rate_dir, packets, mapping)
    ffprobe = Path(deps["ffmpeg"]).with_name("ffprobe.exe")
    command([ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames,duration", "-of", "json", result["preview"]], rate_dir / "playback_probe.json", commands)
    stream = read_json(rate_dir / "playback_probe.json")["streams"][0]
    require([stream["width"], stream["height"], int(stream["nb_read_frames"])] == [config["source"]["width"], config["source"]["height"], config["source"]["frames"]], "Preview geometry/count mismatch")
    require(Fraction(stream["avg_frame_rate"]) == config["source"]["fps"], "Preview timing mismatch")
    command([deps["ffmpeg"], "-v", "error", "-i", result["preview"], "-f", "null", "-"], rate_dir / "playback_decode.log", commands)
    result.update(rate_kbps=rate, transfer=transfer, pacing=pacing,
                  byte_ledger={**frozen["byte_ledger"], "transmission_header_bytes": transfer["application_bytes"] - transfer["file_bytes"],
                               "total_sent_application_bytes": transfer["application_bytes"]}, playback_verified=True)
    write_json(rate_dir / "result.json", result)
    print(f"B1 {rate} kbps passed: {result['decoded_frames']} frames, hash and source-read audit verified", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "configs/b0_baseline.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = read_json(args.config)
    commands = [{"argv": [sys.executable, *sys.argv], "cwd": str(Path.cwd()), "code_commit": git("rev-parse", "HEAD"),
                 "started_at": datetime.now(timezone(timedelta(hours=8))).isoformat()}]
    write_json(output / "config.json", config)
    try:
        print("B0: checking pinned code, models, environment, and all 250 source frames", flush=True)
        frozen, packets, mapping = freeze(config, output, commands)
        print("B0: inputs and both existing semantic profiles frozen", flush=True)
        sys.path.insert(0, str((ROOT / config["transport_plan"]["implementation"]).parent))
        results = []
        for rate in config["transport_plan"]["rates_kbps"]:
            results.append(trial(config, frozen, packets, mapping, rate, output, commands))
            write_json(output / "progress.json", {"completed_rates": [r["rate_kbps"] for r in results]})
        require(len({r["rgb_sequence_sha256"] for r in results}) == 1, "Received RGB differs across rates")
        summary = {"status": "B0_B1_passed", "code_commit": frozen["code_commit"], "results": results,
                   "all_rates_identical_received_RGB": True, "semantic_benefit": "not_evaluated_B2_B3_pending",
                   "limits": ["Same-host TCP loopback, software application-rate limit only; no physical acoustic link or cross-device claim.",
                              "TCP/IP, link-layer overhead and reverse ACK bytes excluded.", "Python audit and subprocess argument checks are not an OS sandbox.",
                              "GLC weights and matching GPU/software environment must be pre-shared.", "Preview quality is not a human viewing score."]}
        write_json(output / "summary.json", summary)
        print("B0+B1 complete: three real transfers and 750 independently decoded frames", flush=True)
    except BaseException as exc:
        write_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write_json(output / "commands.json", commands)


if __name__ == "__main__":
    main()
