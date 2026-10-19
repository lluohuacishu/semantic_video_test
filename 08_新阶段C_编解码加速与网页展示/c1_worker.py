"""One role per resident process; JSON config or sequential JSON-lines stdin.

The legacy branch is a measurement control using the unchanged C0 loops. Only
model acquisition, readiness timestamps and input hashes are wrapped. Display
control deletes its private PNG intermediates *after* generating the MP4s.
"""
import time
ENTRY = time.perf_counter()
import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from unittest.mock import patch

from c1_runtime import ResidentRunner, ref, write_json
from run_c0 import gpu_snapshot


def reference_task(runner, task):
    with runner.task_scope(runner.role):
        folder = Path(task["output"]).resolve()
        args = runner.options(task, folder)
        folder.mkdir(parents=True, exist_ok=False)
        runner.synchronize()
        started = time.perf_counter()
        resident = runner.codec is not None
        marks, input_hashes = {}, []
        acquire_seconds = 0.0
        original_array, original_write, original_close = ref.np.array, ref.write_json, ref.StreamWriter.close

        def factory(repo, weights, device, decoder_only=False):
            nonlocal acquire_seconds
            if decoder_only != (runner.role == "decode"):
                raise ValueError("Reference worker role mismatch")
            tick = time.perf_counter()
            codec = runner.acquire()
            acquire_seconds += time.perf_counter() - tick
            return codec

        def array(value, *a, **kw):
            result = original_array(value, *a, **kw)
            if isinstance(value, ref.Image.Image):
                input_hashes.append(hashlib.sha256(result.tobytes()).hexdigest())
            return result

        def close(writer):
            result = original_close(writer)
            marks["bitstream_closed_seconds"] = time.perf_counter() - started
            return result

        def evidence(path, result):
            if Path(path).name in {"encode.json", "decode.json"}:
                if runner.role == "encode":
                    if len(input_hashes) != len(result["frames"]):
                        raise ValueError("Missing reference input RGB hashes")
                    result["input"]["rgb_sha256"] = input_hashes
                    for row, digest in zip(result["frames"], input_hashes):
                        row["source_rgb_sha256"] = digest
                runner.synchronize()
                marks["checks_complete_seconds"] = time.perf_counter() - started
                original_write(path, result)
                marks["evidence_ready_seconds"] = time.perf_counter() - started
            else:
                original_write(path, result)

        with patch.object(ref, "NeuralCodec", factory), patch.object(ref.np, "array", array), \
                patch.object(ref, "write_json", evidence), patch.object(ref.StreamWriter, "close", close):
            getattr(ref, runner.role)(args)
        if args.output_mode == "display":
            # Only the new task's own PNG trees; never touch historical outputs.
            for relative in ("encoder/source", "encoder/frames", "receiver/frames"):
                target = (folder / relative).resolve()
                if folder not in target.parents:
                    raise ValueError("Intermediate path escaped the task directory")
                if target.is_dir():
                    shutil.rmtree(target)
        runner.synchronize()
        marks["offline_view_ready_seconds"] = time.perf_counter() - started
        result_file = folder / ("encoder/encode.json" if runner.role == "encode" else "receiver/decode.json")
        value = {"role": runner.role, "preset": task.get("preset"), "pid": os.getpid(),
                 "implementation": "legacy", "output_mode": args.output_mode, "png_level": 6,
                 "model_was_resident": resident, "codec_acquire_seconds": acquire_seconds,
                 "task_seconds": marks, "task_timer_origin_perf_counter": started,
                 "detailed_instrumentation": False,
                 "browser_upload_queue_load_seconds": None, "transfer_measured_seconds": None,
                 "preview_overlaps_processing": False, "result_path": str(result_file),
                 "control_note": "Unchanged legacy loops; display removes PNG intermediates after previews"}
        write_json(folder / "timing.json", value)
        return value


def execute(runner, task, launch, first, environment):
    before = gpu_snapshot()
    task_start = time.perf_counter()
    folder = Path(task["output"])
    output_existed = folder.exists()
    try:
        result = reference_task(runner, task) if task.get("implementation", "c1") == "legacy" else runner.run(task)
    except BaseException as exc:
        # A rejected rerun must not append or overwrite prior task evidence.
        if not output_existed and not isinstance(exc, FileExistsError) and folder.is_dir():
            write_json(folder / "worker_failure.json", {"type": type(exc).__name__, "message": str(exc),
                       "traceback": traceback.format_exc(), "gpu_before": before, "gpu_after": gpu_snapshot()})
        raise
    total = time.perf_counter() - task_start
    origin = result["task_timer_origin_perf_counter"]
    result.update(run_id=task["id"], group=task.get("group"), pair=task.get("pair"),
                  variant=task.get("variant"), task_config=task,
                  outer_call_wall_seconds=total,
                  process_launch_to_evidence_ready_seconds=(origin - launch + result["task_seconds"]["evidence_ready_seconds"]) if first else None,
                  process_launch_to_view_ready_seconds=(origin - launch + result["task_seconds"]["offline_view_ready_seconds"]) if first else None,
                  process_launch_to_task_return_seconds=(task_start - launch + total) if first else None,
                  startup_before_first_task_seconds=origin - launch if first else None,
                  gpu_before=before, gpu_after=gpu_snapshot(), worker_environment=environment)
    result["torch_threads_during_inference"] = ref.torch.get_num_threads()
    write_json(folder / "timing.json", result)
    print("C1_TASK_DONE " + task["id"] + " " + json.dumps(result["task_seconds"]), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="role/repo/weights/ffmpeg/device and optional finite tasks list")
    parser.add_argument("--jsonl", action="store_true", help="Read sequential requests from stdin; stdout is JSON only")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    launch = config.get("launch_perf_counter", ENTRY)
    runner = ResidentRunner(config["role"], config["repo"], config["weights"], config["ffmpeg"], config.get("device", "cuda"))
    environment = {"python": sys.version, "torch": ref.torch.__version__, "cuda": ref.torch.version.cuda,
                   "pid": os.getpid(), "entry_to_imports_seconds": time.perf_counter() - ENTRY,
                   "torch_threads": ref.torch.get_num_threads(), "detailed_instrumentation": False}
    if args.jsonl:
        # C3 can feed this worker one job at a time. The Python API additionally
        # accepts a cancellation Event; the JSONL interface is deliberately serial.
        for position, line in enumerate(sys.stdin):
            try:
                task = json.loads(line)
                with redirect_stdout(sys.stderr):
                    result = execute(runner, task, launch, position == 0, environment)
                response = {"ok": True, "timing": result}
            except Exception as exc:
                response = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
            print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)
    else:
        for position, task in enumerate(config["tasks"]):
            execute(runner, task, launch, position == 0, environment)


if __name__ == "__main__":
    main()
