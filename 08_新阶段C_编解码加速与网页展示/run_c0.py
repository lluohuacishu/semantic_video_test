"""Finite C0 matrix. GPU work is serialized; existing outputs are never overwritten."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from c0_metrics import distribution

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WORKSPACE = REPO.parents[2]
SCRIPTS = REPO / "04_外部依赖快照/semantic-video-training/scripts"
sys.path.insert(0, str(SCRIPTS))
from glc_container import read_stream

RUNTIME = Path("C:/Users/lenovo/Desktop/works/work/semantic-video-training")
FFMPEG = Path("C:/Users/lenovo/Desktop/works/Python/video-baseline/tools/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe")
SOURCE = WORKSPACE / "02_实验数据/A_压缩路线选型/A2_GLC验证/long108/input_540.mkv"
B4 = WORKSPACE / "02_实验数据/B_GLC通信与压缩/B4_四档最终结果"
INPUT_SHA = "6683af45254bdaed431658e5f7515cddaeb42df2382f18b983e6e1cba7ae436b"
GOLD = {
    "compact": ("joint_360p_12_5/received/stream.glcv", "joint_360p_12_5/receiver/decode.json", "joint_360p_12_5/encoder_work/source"),
    "detail": ("fps_only_540p_12_5/stream.glcv", "fps_only_540p_12_5/receiver/decode.json", "fps_only_540p_12_5/encoder_work/source"),
    "motion": ("resolution_only_360p_25/stream.glcv", "resolution_only_360p_25/receiver/decode.json", "resolution_only_360p_25/encoder_work/source"),
    "detail_motion": ("container_v2_baseline_repack/baseline_v2.glcv", "container_v2_baseline_decode/receiver/decode.json", None),
}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def command_text(argv):
    result = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def gpu_snapshot():
    return command_text(["nvidia-smi", "--query-gpu=name,driver_version,pstate,temperature.gpu,power.draw,power.limit,memory.used,memory.total,utilization.gpu,clocks.sm,clocks.mem", "--format=csv"])


def code_hashes():
    files = list(HERE.glob("c0_*.py")) + [Path(__file__)]
    files += [SCRIPTS / name for name in ["glc_video_codec.py", "glc_video_probe.py", "glc_entropy.py", "glc_container.py", "glc_presets.py", "glc_presets.json"]]
    return {str(path.relative_to(REPO)): sha(path) for path in files}


def create_manifest(root):
    current = code_hashes()
    if (root / "manifest.json").exists():
        old = read(root / "manifest.json")
        if current != old["code_sha256"]:
            raise RuntimeError("Measurement code changed: choose a fresh output directory; preserve earlier evidence.")
        return
    if sha(SOURCE) != INPUT_SHA:
        raise ValueError("Frozen B4 input digest mismatch")
    refs = {}
    for preset, (stream, evidence, source) in GOLD.items():
        result = read(B4 / evidence)
        digest = sha(B4 / stream)
        assert digest == result["input_sha256"]
        refs[preset] = {"stream": str(B4 / stream), "sha256": digest, "bytes": (B4 / stream).stat().st_size,
                       "decoder_evidence": str(B4 / evidence), "decoder_evidence_sha256": sha(B4 / evidence),
                       "header": result["header"]}
    machine = command_text(["powershell", "-NoProfile", "-Command",
        "[ordered]@{cpu=(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors);os=(Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,TotalVisibleMemorySize,FreePhysicalMemory);battery=(Get-CimInstance Win32_Battery | Select-Object BatteryStatus,EstimatedChargeRemaining)} | ConvertTo-Json -Depth 5"])
    write(root / "manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": command_text(["git", "rev-parse", "HEAD"])["stdout"].strip(),
        "branch": command_text(["git", "branch", "--show-current"])["stdout"].strip(),
        "code_sha256": current, "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "machine": machine,
        "power_scheme": command_text(["powercfg", "/getactivescheme"]), "gpu_before": gpu_snapshot(),
        "gpu_processes_before": command_text(["nvidia-smi"]),
        "ffmpeg": command_text([str(FFMPEG), "-version"]),
        "input": {"path": str(SOURCE), "sha256": INPUT_SHA, "bytes": SOURCE.stat().st_size},
        "reference_b4": refs, "output_scope": "full_evidence_png_checksums_mp4",
        "budget": "66-frame diagnosis encode/decode; compact 3 fresh + 3 resident each role; other presets 1 fresh each role",
        "cold_definition": "New Python process and CUDA context; OS disk caches are not flushed",
        "display_only_baseline": None, "reason": "C0 retains full original evidence; reduced-output mode is not implemented or compared"})


def validate_task(folder, role, preset, diagnostic):
    stream, evidence, source = GOLD[preset]
    golden = read(B4 / evidence)
    data = read(folder / ("encoder/encode.json" if role == "encode" else "receiver/decode.json"))
    expected_frames = golden["frames"][:66] if diagnostic else golden["frames"]
    assert len(data["frames"]) == len(expected_frames)
    for actual, expected in zip(data["frames"], expected_frames):
        for key in ("frame", "rgb_sha256", "latent_sha256"):
            assert actual[key] == expected[key], (folder, key, actual["frame"])
    validation = {"frames_checked": len(expected_frames), "all_rgb_and_latent_hashes_match_b4": True,
                  "whole_stream_matches_b4": None, "sampled_input_rgb_matches_b4": None}
    if role == "encode":
        actual = folder / "stream.glcv"
        validation["stream_sha256"] = sha(actual)
        validation["stream_bytes"] = actual.stat().st_size
        if diagnostic:
            _, packets = read_stream(actual)
            _, reference_packets = read_stream(B4 / stream)
            assert len(packets) == 66
            assert all((a["kind"], a["z"], a["y"]) == (b["kind"], b["z"], b["y"])
                       for a, b in zip(packets, reference_packets))
            validation["prefix_66_neural_payloads_equal_b4"] = True
        else:
            assert sha(actual) == sha(B4 / stream)
            validation["whole_stream_matches_b4"] = True
        stride = 2 if preset in {"compact", "detail"} else 1
        assert data["input"]["sampled_source_frame_indices"] == list(range(0, stride * len(expected_frames), stride))
        if source is not None:
            from PIL import Image
            digest = hashlib.sha256()
            for index in range(len(expected_frames)):
                name = f"im{index + 1:05d}.png"
                with Image.open(folder / "encoder/source" / name) as a, Image.open(B4 / source / name) as b:
                    raw = a.convert("RGB").tobytes()
                    assert raw == b.convert("RGB").tobytes()
                    digest.update(hashlib.sha256(raw).digest())
            validation["sampled_input_rgb_matches_b4"] = True
            validation["ordered_input_frame_digest"] = digest.hexdigest()
    else:
        assert data["independent_decoder"] and data["encoder_modules_disabled"]
        assert data["source_media_reads"] == [] and data["all_frame_checksums_match"]
        assert data["whole_file_checksum_verified"]
        validation.update(independent_decoder=True, source_media_reads=[], whole_file_checksum_verified=True)
    write(folder / "validation.json", validation)
    return validation


def run_worker(root, name, role, preset, tasks, diagnostic=False):
    folder = root / name
    folder.mkdir()  # no resume into partial or completed results
    config = {"role": role, "preset": preset, "diagnostic": diagnostic,
              "frames": 66 if diagnostic else None, "output": str(folder),
              "repo": str(RUNTIME / "tools/GLC"), "weights": str(RUNTIME / "checkpoints/glc"),
              "ffmpeg": str(FFMPEG), "tasks": tasks}
    argv = [sys.executable, "-B", str(HERE / "c0_worker.py"), "--config", str(folder / "config.json")]
    record = {"argv": argv, "start_utc": datetime.now(timezone.utc).isoformat(), "gpu_before": gpu_snapshot()}
    config["launch_perf_counter"] = time.perf_counter()
    write(folder / "config.json", config)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    with (folder / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(argv, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", env=env)
        record["pid"] = process.pid
        write(folder / "command.json", record)
        for line in process.stdout:
            log.write(line)
            log.flush()
            if line.startswith(("encode ", "decode ")):
                number = int(line.split()[1].split("/")[0])
                if number == 1 or number % 25 == 0:
                    print(name + ": " + line.strip(), flush=True)
            elif line.startswith("C0_TASK_DONE") or "Error" in line or "Traceback" in line:
                print(name + ": " + line.strip(), flush=True)
        record["returncode"] = process.wait()
    record["process_seconds"] = time.perf_counter() - config["launch_perf_counter"]
    record["gpu_after"] = gpu_snapshot()
    write(folder / "command.json", record)
    if record["returncode"]:
        raise RuntimeError(f"{name} failed: see {folder / 'worker.log'}")
    for task in tasks:
        validate_task(folder / task["id"], role, preset, diagnostic)
    print("C0_VALIDATED " + name, flush=True)


def diagnostics(root):
    run_worker(root, "diagnostic_encode", "encode", "compact", [{"id": "diag66", "input": str(SOURCE)}], True)
    run_worker(root, "diagnostic_decode", "decode", "compact",
               [{"id": "diag66", "input": str(root / "diagnostic_encode/diag66/stream.glcv")}], True)


def baselines(root):
    # First full run supplies the warm-up; it also counts as fresh repeat 1.
    # Sender and receiver processes never coexist on the GPU or exchange tensors.
    for role in ("encode", "decode"):
        source = SOURCE if role == "encode" else B4 / GOLD["compact"][0]
        tasks = [{"id": name, "input": str(source)} for name in ["cold1", "warm1", "warm2", "warm3"]]
        run_worker(root, f"compact_{role}_resident", role, "compact", tasks)
    for repeat in (2, 3):
        for role in ("encode", "decode"):
            source = SOURCE if role == "encode" else B4 / GOLD["compact"][0]
            run_worker(root, f"compact_{role}_cold{repeat}", role, "compact", [{"id": f"cold{repeat}", "input": str(source)}])
    for preset in ("detail", "motion", "detail_motion"):
        for role in ("encode", "decode"):
            source = SOURCE if role == "encode" else B4 / GOLD[preset][0]
            run_worker(root, f"{preset}_{role}", role, preset, [{"id": "cold1", "input": str(source)}])


def summarize(root):
    records = []
    diagnostics = []
    for path in sorted(root.glob("*/*/timing.json")):
        timing = read(path)
        validation = read(path.with_name("validation.json"))
        role = timing["role"]
        result = read(path.parent / ("encoder/encode.json" if role == "encode" else "receiver/decode.json"))
        record = {key: value for key, value in timing.items() if key not in {"frames", "profiler"}}
        record.update(evidence_path=str(path), validation=validation,
                      result_sha256=sha(path.parent / ("encoder/encode.json" if role == "encode" else "receiver/decode.json")),
                      peak_allocated_mib=result["peak_allocated_mib"], peak_reserved_mib=result["peak_reserved_mib"],
                      original_pipeline_seconds=result["pipeline_seconds_before_preview"],
                      original_frame_seconds=result["frame_processing_seconds"])
        (diagnostics if timing["diagnostic"] else records).append(record)
    groups = []
    for preset in GOLD:
        for mode in ("fresh_process", "resident_followup"):
            for role in ("encode", "decode"):
                chosen = [r for r in records if (r["preset"], r["mode"], r["role"]) == (preset, mode, role)]
                if not chosen:
                    continue
                groups.append({"preset": preset, "mode": mode, "role": role,
                    "evidence_ready_seconds": distribution([r["task_seconds"]["evidence_ready_seconds"] for r in chosen]),
                    "offline_view_ready_seconds": distribution([r["task_seconds"]["offline_view_ready_seconds"] for r in chosen]),
                    "frame_seconds": distribution([r["original_frame_seconds"] for r in chosen]),
                    "codec_acquire_seconds": distribution([r["codec_acquire_seconds"] for r in chosen]),
                    "process_launch_to_evidence_ready_seconds": distribution([r["process_launch_to_evidence_ready_seconds"] for r in chosen if r["process_launch_to_evidence_ready_seconds"] is not None])})
    value = {"manifest": read(root / "manifest.json"), "records": records, "groups": groups, "diagnostics": diagnostics,
             "formal_task_count": len(records), "diagnostic_task_count": len(diagnostics),
             "optimization_attribution": [{"change": name, "status": "not_implemented", "saved_seconds": None}
                for name in ["C1_resident_service", "C1_memory_frames", "C2_native_rans", "C2_gpu_submodule"]]}
    write(root / "summary.json", value)
    print(json.dumps({"formal_task_count": len(records), "diagnostic_task_count": len(diagnostics), "groups": groups}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["diagnostic", "baseline", "summary"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    create_manifest(args.output)
    if args.phase == "diagnostic":
        diagnostics(args.output)
    elif args.phase == "baseline":
        # No formal matrix unless the independent short decode has passed.
        assert read(args.output / "diagnostic_decode/diag66/validation.json")["all_rgb_and_latent_hashes_match_b4"]
        baselines(args.output)
    summarize(args.output)


if __name__ == "__main__":
    main()
