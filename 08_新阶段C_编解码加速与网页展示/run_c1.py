"""Finite, serialized C1 correctness and paired wall-clock experiments."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import platform
import subprocess
import sys
import time

from run_c0 import (HERE, REPO, WORKSPACE, SCRIPTS, RUNTIME, FFMPEG, SOURCE, B4, GOLD,
                    INPUT_SHA, read, write, sha, command_text, gpu_snapshot)
from c0_metrics import distribution
from glc_container import read_stream


def manifest(root):
    files = list(HERE.glob("c1_*.py")) + [Path(__file__), HERE / "run_c0.py"]
    files += [SCRIPTS / n for n in ("glc_video_codec.py", "glc_video_probe.py", "glc_entropy.py", "glc_container.py", "glc_presets.py", "glc_presets.json")]
    hashes = {str(p.relative_to(REPO)): sha(p) for p in files}
    target = root / "manifest.json"
    if target.exists():
        if read(target)["code_sha256"] != hashes:
            raise RuntimeError("Code changed; preserve prior evidence and use a fresh experiment directory")
        return
    if sha(SOURCE) != INPUT_SHA:
        raise ValueError("Frozen input changed")
    write(target, {"created_utc": datetime.now(timezone.utc).isoformat(),
          "base_commit": command_text(["git", "rev-parse", "HEAD"])["stdout"].strip(),
          "branch": command_text(["git", "branch", "--show-current"])["stdout"].strip(),
          "code_sha256": hashes, "input": {"path": str(SOURCE), "sha256": INPUT_SHA},
          "platform": platform.platform(), "python": sys.executable,
          "power_scheme": command_text(["powercfg", "/getactivescheme"]), "gpu": gpu_snapshot(),
          "battery": command_text(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Battery | Select-Object BatteryStatus,EstimatedChargeRemaining | ConvertTo-Json"]),
          "policy": "One role per worker; no concurrent GPU tasks; no profiler; AB/BA pairs; no result cache",
          "cold_definition": "New Python/CUDA process, OS disk caches not flushed",
          "reference": "Unchanged C0 adapter with input hash collection and resident model factory",
          "output_modes": {"research": "Stream, all PNGs, per-frame metrics and hashes, MP4",
                           "display": "Stream, per-frame metrics and hashes, MP4; legacy PNG intermediates deleted within timing"}})


def preview_info(path):
    result = subprocess.run([str(FFMPEG.with_name("ffprobe.exe")), "-v", "error", "-count_frames",
             "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate,duration", "-of", "json", str(path)],
             check=True, capture_output=True, text=True)
    return json.loads(result.stdout)["streams"][0]


def validate(task, role):
    folder, preset = Path(task["output"]), task["preset"]
    golden = read(B4 / GOLD[preset][1])
    path = folder / ("encoder/encode.json" if role == "encode" else "receiver/decode.json")
    result = read(path)
    n = len(result["frames"])
    for actual, expected in zip(result["frames"], golden["frames"]):
        for key in ("frame", "rgb_sha256", "latent_sha256"):
            if actual[key] != expected[key]:
                raise ValueError(f"{task['id']}: {key} mismatch at frame {actual['frame']}")
    if n != min(task.get("frames") or len(golden["frames"]), len(golden["frames"])):
        raise ValueError("Unexpected frame count")
    record = {"frames_checked": n, "all_rgb_and_latent_hashes_match_b4": True,
              "whole_stream_matches_b4": None, "input_rgb_matches_b4": None,
              "result_sha256": sha(path)}
    if role == "encode":
        stream = folder / "stream.glcv"
        header, packets = read_stream(stream)
        _, reference = read_stream(B4 / GOLD[preset][0])
        if not all((a["kind"], a["z"], a["y"]) == (b["kind"], b["z"], b["y"]) for a, b in zip(packets, reference)):
            raise ValueError("Neural payload differs from B4")
        if n == len(golden["frames"]):
            if sha(stream) != sha(B4 / GOLD[preset][0]):
                raise ValueError("Whole stream differs from B4")
            record["whole_stream_matches_b4"] = True
        record.update(stream_sha256=sha(stream), stream_bytes=stream.stat().st_size, prefix_payloads_match_b4=True)
        # C0 independently checked all these sampled input pixels, including
        # detail_motion's A2 source. Read frozen PNGs, outside measured time.
        source = B4 / GOLD[preset][2] if GOLD[preset][2] else WORKSPACE / "02_实验数据/A_压缩路线选型/A2_GLC验证/long108/glc_540_g64/sender/source"
        from PIL import Image
        expected_hashes = []
        for index in range(n):
            with Image.open(source / f"im{index + 1:05d}.png") as image:
                expected_hashes.append(hashlib.sha256(image.convert("RGB").tobytes()).hexdigest())
        if result["input"]["rgb_sha256"] != expected_hashes:
            raise ValueError("Input RGB differs from B4/A2")
        stride = 2 if preset in {"compact", "detail"} else 1
        if result["input"]["sampled_source_frame_indices"] != list(range(0, stride * n, stride)):
            raise ValueError("Sampled source frame mapping changed")
        record["input_rgb_matches_b4"] = True
    else:
        for key in ("independent_decoder", "encoder_modules_disabled", "whole_file_checksum_verified", "all_frame_checksums_match"):
            if result[key] is not True:
                raise ValueError(f"Decoder check failed: {key}")
        if result["source_media_reads"] != []:
            raise ValueError("Decoder read media")
        record.update(independent_decoder=True, source_media_reads=[], whole_file_checksum_verified=True)
    previews = list(path.parent.glob("*_preview.mp4"))
    if len(previews) != (2 if role == "encode" else 1):
        raise ValueError("Preview count mismatch")
    record["previews"] = {}
    for preview in previews:
        info = preview_info(preview)
        numerator, denominator = map(int, info["avg_frame_rate"].split("/"))
        if (int(info["nb_read_frames"]) != n or abs(numerator / denominator - result["header"]["fps"]) > 1e-8 or
                info["width"] != (result["header"]["width"] + 1) // 2 * 2 or
                info["height"] != (result["header"]["height"] + 1) // 2 * 2):
            raise ValueError("Preview shape/frame count/fps mismatch")
        record["previews"][preview.name] = info
    pngs = list(path.parent.rglob("*.png"))
    if len(pngs) != (n * (2 if role == "encode" else 1) if task["output_mode"] == "research" else 0):
        raise ValueError("Output scope mismatch")
    # Verify saved research evidence actually encodes the checked pixels.
    if pngs:
        from PIL import Image
        for png in pngs:
            index = int(png.stem[2:]) - 1
            expected = result["input"]["rgb_sha256"][index] if png.parent.name == "source" else result["frames"][index]["rgb_sha256"]
            with Image.open(png) as image:
                if hashlib.sha256(image.convert("RGB").tobytes()).hexdigest() != expected:
                    raise ValueError("Saved PNG does not match RGB evidence")
    record["png_count"] = len(pngs)
    record["png_bytes"] = sum(p.stat().st_size for p in pngs)
    write(folder / "validation.json", record)
    return record


def task(root, name, role, preset="compact", implementation="c1", mode="display", frames=None, **extra):
    return {"id": name, "output": str(root / name), "preset": preset, "implementation": implementation,
            "input": str(SOURCE if role == "encode" else B4 / GOLD[preset][0]),
            "output_mode": mode, "frames": frames, **extra}


def worker(root, name, role, tasks):
    folder = root / name
    folder.mkdir()
    config = {"role": role, "repo": str(RUNTIME / "tools/GLC"), "weights": str(RUNTIME / "checkpoints/glc"),
              "ffmpeg": str(FFMPEG), "tasks": tasks, "launch_perf_counter": time.perf_counter()}
    write(folder / "config.json", config)
    argv = [sys.executable, "-B", str(HERE / "c1_worker.py"), "--config", str(folder / "config.json")]
    command = {"argv": argv, "started_utc": datetime.now(timezone.utc).isoformat(), "returncode": None}
    write(folder / "command.json", command)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    started = time.perf_counter()
    with (folder / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(argv, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", env=env)
        command["pid"] = process.pid
        write(folder / "command.json", command)
        for line in process.stdout:
            log.write(line)
            log.flush()
            if line.startswith("C1_TASK_DONE") or "Error" in line or "Traceback" in line:
                print(name + ": " + line.strip(), flush=True)
            elif line.startswith(("encode ", "decode ")):
                index = int(line.split()[1].split("/")[0])
                if index == 1 or index % 50 == 0:
                    print(name + ": " + line.strip(), flush=True)
        command["returncode"] = process.wait()
    command["process_seconds"] = time.perf_counter() - started
    write(folder / "command.json", command)
    if command["returncode"]:
        raise RuntimeError(f"{name} failed: {folder / 'worker.log'}")
    for value in tasks:
        validate(value, role)
    print("C1_VALIDATED " + name, flush=True)


def gate(root, count):
    enc = task(root, f"encode{count}", "encode", frames=count, mode="research")
    worker(root, f"encode{count}_process", "encode", [enc])
    for implementation in ("c1", "legacy"):
        dec = task(root, f"decode{count}_{implementation}", "decode", frames=count,
                   implementation=implementation, input=str(Path(enc["output"]) / "stream.glcv"), mode="research")
        worker(root, f"decode{count}_{implementation}_process", "decode", [dec])


def paired(root):
    for role in ("encode", "decode"):
        tasks = [task(root, f"{role}_cold_legacy", role, implementation="legacy", mode="research", group="cold")]
        for mode in ("research", "display"):
            for repeat in (1, 2, 3):
                order = ["legacy", "memory_png6", "c1"] if mode == "research" else ["legacy", "c1"]
                if repeat % 2 == 0:
                    order.reverse()
                for variant in order:
                    tasks.append(task(root, f"{role}_{mode}_{repeat}_{variant}", role,
                         implementation="legacy" if variant == "legacy" else "c1", mode=mode,
                         group="ablation" if variant == "memory_png6" else "paired", pair=repeat,
                         png_level=6 if variant == "memory_png6" else 1, variant=variant))
        worker(root, role + "_paired_process", role, tasks)
        worker(root, role + "_cold_c1_process", role,
               [task(root, f"{role}_cold_c1", role, mode="research", group="cold")])


def residency(root):
    # Same three display jobs: one resident process versus three fresh processes.
    # Every first task/initialization and process shutdown is counted, no hidden warmup.
    for role in ("encode", "decode"):
        for policy in (("resident", "fresh") if role == "encode" else ("fresh", "resident")):
            tasks = [task(root, f"{role}_{policy}_{i}", role, mode="display", group="residency",
                          pair=i, variant=policy) for i in (1, 2, 3)]
            if policy == "resident":
                worker(root, f"{role}_resident_process", role, tasks)
            else:
                for value in tasks:
                    worker(root, value["id"] + "_process", role, [value])


def compatibility(root):
    for preset in ("detail", "motion", "detail_motion"):
        for role in ("encode", "decode"):
            worker(root, preset + "_" + role + "_process", role,
                   [task(root, preset + "_" + role, role, preset=preset, group="compatibility")])


def summarize(root):
    records = []
    for path in sorted(root.glob("*/timing.json")):
        record = read(path)
        validation = path.with_name("validation.json")
        if not validation.exists():
            continue
        record.update(validation=read(validation), evidence_path=str(path))
        result = read(record["result_path"])
        record.update(png_save_seconds=result.get("png_save_seconds"),
                      frame_processing_seconds=result["frame_processing_seconds"],
                      peak_allocated_mib=result["peak_allocated_mib"], peak_reserved_mib=result["peak_reserved_mib"])
        records.append(record)
    pairs = []
    for mode in ("research", "display"):
        for role in ("encode", "decode"):
            chosen = [r for r in records if r["group"] == "paired" and r["role"] == role and r["output_mode"] == mode]
            for repeat in (1, 2, 3):
                runs = {r["implementation"]: r for r in chosen if r["pair"] == repeat}
                if set(runs) != {"legacy", "c1"}:
                    continue
                before, after = runs["legacy"], runs["c1"]
                for boundary in ("evidence_ready_seconds", "offline_view_ready_seconds"):
                    a, b = before["task_seconds"][boundary], after["task_seconds"][boundary]
                    pairs.append({"output_mode": mode, "role": role, "repeat": repeat, "boundary": boundary,
                                  "before_id": before["run_id"], "after_id": after["run_id"],
                                  "before_seconds": a, "after_seconds": b, "saved_seconds": a - b,
                                  "reduction_percent": 100 * (a - b) / a, "speedup": a / b})
    groups = []
    for mode in ("research", "display"):
        for role in ("encode", "decode"):
            for boundary in ("evidence_ready_seconds", "offline_view_ready_seconds"):
                chosen = [p for p in pairs if (p["output_mode"], p["role"], p["boundary"]) == (mode, role, boundary)]
                if chosen:
                    groups.append({"output_mode": mode, "role": role, "boundary": boundary,
                                   **{key: distribution([p[key] for p in chosen]) for key in ("before_seconds", "after_seconds", "saved_seconds", "reduction_percent", "speedup")}})
    write(root / "summary.json", {"manifest": read(root / "manifest.json"), "records": records, "pairs": pairs, "groups": groups})
    print(json.dumps({"validated_tasks": len(records), "groups": groups}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["gate8", "gate66", "paired", "residency", "compatibility", "summary"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest(args.output)
    if args.phase.startswith("gate"):
        gate(args.output, int(args.phase[4:]))
    elif args.phase != "summary":
        # Gate evidence may live in a separate immutable experiment directory.
        gate_path = args.output.parent / "C1_gates_20261019" / "decode66_legacy/validation.json"
        if not read(gate_path)["all_rgb_and_latent_hashes_match_b4"]:
            raise RuntimeError("66-frame cross-decoding gate has not passed")
        {"paired": paired, "residency": residency, "compatibility": compatibility}[args.phase](args.output)
    summarize(args.output)


if __name__ == "__main__":
    main()
