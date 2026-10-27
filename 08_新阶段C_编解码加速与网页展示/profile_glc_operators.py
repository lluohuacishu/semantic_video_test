"""Bounded FP32 profiling campaign; production codec and arithmetic stay unchanged."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
SAMPLED = {0, 1, 2, 33, 64}


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gpu_snapshot():
    fields = "timestamp,memory.used,utilization.gpu,utilization.memory,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu"
    p = subprocess.run(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, creationflags=FLAGS, timeout=10)
    return {"time": time.time(), "fields": fields, "value": p.stdout.strip(), "error": p.stderr.strip()}


class Observer:
    def __init__(self, torch, folder, role):
        self.torch, self.folder, self.role = torch, folder, role
        self.active = False
        self.mode = "warmup"
        self.frame_rows, self.profiles = [], []
        self.scopes, self.transfers = [], []
        self.frame_index = None

    def wrap(self, owner, attr, name, cuda=True):
        original = getattr(owner, attr)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            if not self.active:
                return original(*args, **kwargs)
            begin = self.torch.cuda.Event(enable_timing=True) if cuda else None
            end = self.torch.cuda.Event(enable_timing=True) if cuda else None
            if begin is not None:
                begin.record()
            tick = time.perf_counter()
            with self.torch.profiler.record_function("glc." + name):
                value = original(*args, **kwargs)
            wall_ms = (time.perf_counter() - tick) * 1000
            if end is not None:
                end.record()
            self.scopes.append({"name": name, "cpu_wall_ms": wall_ms, "begin": begin, "end": end})
            return value

        setattr(owner, attr, measured)

    def install_tensor_observers(self):
        for method in ("cpu", "to", "numpy", "item", "__bool__"):
            original = getattr(self.torch.Tensor, method)

            @functools.wraps(original)
            def measured(tensor, *args, _original=original, _method=method, **kwargs):
                if not self.active:
                    return _original(tensor, *args, **kwargs)
                source = str(tensor.device)
                caller = sys._getframe(1)
                location = f"{caller.f_code.co_filename}:{caller.f_lineno}"
                tick = time.perf_counter()
                with self.torch.profiler.record_function("glc.tensor." + _method):
                    value = _original(tensor, *args, **kwargs)
                wall_ms = (time.perf_counter() - tick) * 1000
                target = str(value.device) if isinstance(value, self.torch.Tensor) else "host_value"
                if source != target or _method in {"numpy", "item", "__bool__"}:
                    self.transfers.append({"method": _method, "source": source, "target": target,
                        "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                        "tensor_bytes": tensor.numel() * tensor.element_size(), "cpu_wall_ms": wall_ms,
                        "location": location})
                return value

            setattr(self.torch.Tensor, method, measured)

    def instrument(self, codec, ref):
        for prefix, model in (("I", codec.image), ("P", codec.video)):
            names = (["enc", "hyper_enc", "hyper_dec", "y_prior_fusion", "dec", "y_spatial_prior_reduction",
                      "y_spatial_prior", "y_spatial_prior_adaptor_1", "y_spatial_prior_adaptor_2", "y_spatial_prior_adaptor_3"]
                     if prefix == "I" else ["context_generation", "temporal_prior_encoder", "weight_generator",
                      "encoder", "temporal_cond", "y_temporal_combine", "hyper_tokenizer", "hyper_inv_tokenizer",
                      "res_prior_param_decoder", "decoder", "y_spatial_prior"])
            for name in names:
                member = getattr(model, name)
                self.wrap(member, "forward", f"{prefix}.{name}") if isinstance(member, self.torch.nn.Module) else self.wrap(model, name, f"{prefix}.{name}")
            for name in ("encoder", "generator"):
                self.wrap(getattr(model.vqgan, name), "forward", f"{prefix}.vqgan_{name}")
        for name in ("codebook", "prior"):
            self.wrap(codec, name, name)
        for name in ("indexes", "encode", "decode"):
            self.wrap(codec.entropy, name, "entropy." + name, cuda=False)
        for name in ("rgb_array", "reference_hash"):
            self.wrap(ref, name, name)
        original = codec.frame

        def frame(header, index, *args, **kwargs):
            self.frame_index = index
            sampled = self.mode == "profiled" and index in SAMPLED
            tick = time.perf_counter()
            if sampled:
                self.scopes, self.transfers = [], []
                with self.torch.profiler.profile(activities=[self.torch.profiler.ProfilerActivity.CPU,
                        self.torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
                    self.active = True
                    try:
                        with self.torch.profiler.record_function("glc.frame"):
                            result = original(header, index, *args, **kwargs)
                    finally:
                        self.active = False
                    self.torch.cuda.synchronize()
                measured_seconds = time.perf_counter() - tick
                self.save_profile(prof, index, measured_seconds)
            else:
                result = original(header, index, *args, **kwargs)
                measured_seconds = time.perf_counter() - tick
            self.frame_rows.append({"mode": self.mode, "index": index,
                "kind": "I" if index % header["intra_period"] == 0 else "P",
                "reset_boundary": index % header["reset_interval"] == 1,
                "profiled": sampled, "call_seconds_excluding_export": measured_seconds,
                "rgb_sha256": result[1]["rgb_sha256"], "latent_sha256": result[1]["latent_sha256"]})
            if index % 16 == 0 or index + 1 == header["frames"]:
                write(self.folder / "progress.json", {"role": self.role, "mode": self.mode, "frame": index + 1,
                    "total": header["frames"], "utc": datetime.now(timezone.utc).isoformat()})
                print(f"{self.role} {self.mode} {index + 1}/{header['frames']}", flush=True)
            return result

        codec.frame = frame

    def save_profile(self, prof, index, elapsed):
        folder = self.folder / "operators" / f"frame_{index:03d}"
        folder.mkdir(parents=True)
        operators = [{"name": e.key, "calls": e.count, "cpu_total_ms": e.cpu_time_total / 1000,
            "self_cpu_ms": e.self_cpu_time_total / 1000, "cuda_event_total_ms": e.device_time_total / 1000,
            "self_cuda_event_ms": e.self_device_time_total / 1000, "input_shapes": e.input_shapes}
            for e in prof.key_averages(group_by_input_shape=True)]
        operators.sort(key=lambda e: e["self_cuda_event_ms"], reverse=True)
        with (folder / "operators.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(operators[0]))
            writer.writeheader()
            writer.writerows(operators)
        events = prof.events()
        cuda_count = sum(str(e.device_type).endswith("CUDA") for e in events)
        scopes = [{"name": s["name"], "cpu_wall_ms": s["cpu_wall_ms"],
                   "cuda_stream_span_ms": s["begin"].elapsed_time(s["end"]) if s["begin"] is not None else None}
                  for s in self.scopes]
        by_scope = {}
        for e in events:
            if not e.name.startswith("aten::"):
                continue
            parent = e.cpu_parent
            while parent is not None and not parent.name.startswith("glc."):
                parent = parent.cpu_parent
            scope = parent.name if parent is not None else "unattributed"
            key = (scope, e.name)
            row = by_scope.setdefault(key, {"scope": scope, "name": e.name, "calls": 0,
                "self_cpu_ms": 0., "self_cuda_event_ms": 0.})
            row["calls"] += 1
            row["self_cpu_ms"] += e.self_cpu_time_total / 1000
            row["self_cuda_event_ms"] += e.self_device_time_total / 1000
        summary = {"frame": index, "kind": "I" if index % 64 == 0 else "P", "observed_seconds": elapsed,
            "measurement": "CUDA Event fallback; device spans may include stream idle time; not CUPTI kernel activity",
            "cuda_kernel_event_count": cuda_count, "cpu_event_count": len(events), "scopes": scopes,
            "python_tensor_handoffs": self.transfers,
            "operators": operators, "operators_by_scope": list(by_scope.values())}
        write(folder / "profile.json", summary)
        self.profiles.append({"frame": index, "folder": str(folder), "cuda_kernel_event_count": cuda_count,
            "observed_seconds": elapsed})


def worker(args, settings):
    from c1_runtime import ResidentRunner, ref
    from c2_native import NativeGaussianRans
    torch = ref.torch
    runtime = settings["runtime"]
    folder = args.output / args.role
    folder.mkdir()
    observer = Observer(torch, folder, args.role)

    def factory(*a, **kw):
        codec = ref.NeuralCodec(*a, **kw)
        codec.entropy = NativeGaussianRans(runtime["native_library"], codec.entropy)
        observer.instrument(codec, ref)
        return codec

    runner = ResidentRunner(args.role, runtime["repo"], runtime["weights"], runtime["ffmpeg"], codec_factory=factory)
    timings = {}
    for mode in ("warmup", "baseline", "profiled"):
        observer.mode = mode
        if mode == "baseline" and args.role == "encode" and args.resume_from:
            previous = args.resume_from / "encode/baseline"
            shutil.copytree(previous, folder / mode)
            timings[mode] = read(previous / "timing.json")
            encoded = read(previous / "encoder/encode.json")
            observer.frame_rows.extend({"mode": "baseline", "index": i, "kind": r["type"],
                "reset_boundary": i % encoded["header"]["reset_interval"] == 1,
                "profiled": False, "call_seconds_excluding_export": r["seconds"],
                "rgb_sha256": r["rgb_sha256"], "latent_sha256": r["latent_sha256"]}
                for i, r in enumerate(encoded["frames"]))
            print("encode baseline: reusing completed pre-interruption result", flush=True)
            continue
        if mode == "profiled":
            observer.install_tensor_observers()
        source = args.source if args.role == "encode" else args.output / "encode" / ("warmup" if mode == "warmup" else "baseline") / "stream.glcv"
        task = {"role": args.role, "id": mode, "input": str(source), "output": str(folder / mode),
            "preset": "compact", "output_mode": "display", "compatibility_policy": "strict"}
        if args.role == "encode":
            task["frames"] = 8 if mode == "warmup" else 66
        timings[mode] = runner.run(task)
        write(folder / (mode + "_checkpoint.json"), {"timing": timings[mode],
            "frames": [r for r in observer.frame_rows if r["mode"] == mode]})
    baseline = [r for r in observer.frame_rows if r["mode"] == "baseline"]
    profiled = [r for r in observer.frame_rows if r["mode"] == "profiled"]
    assert len(baseline) == len(profiled) == 66
    assert all((a["rgb_sha256"], a["latent_sha256"]) == (b["rgb_sha256"], b["latent_sha256"])
               for a, b in zip(baseline, profiled)), "Profiling changed reconstructed values"
    if args.role == "encode":
        assert digest(folder / "baseline/stream.glcv") == digest(folder / "profiled/stream.glcv"), "Profiling changed the bitstream"
    write(folder / "summary.json", {"role": args.role, "pid": os.getpid(), "timings": timings,
        "frames": observer.frame_rows, "profiles": observer.profiles, "all_66_frame_hashes_equal": True,
        "arithmetic": runner.codec.profile, "supported_activities": [str(a) for a in torch.profiler.supported_activities()],
        "parameter_dtypes": {"image": sorted({str(p.dtype) for p in runner.codec.image.parameters()}),
                             "video": sorted({str(p.dtype) for p in runner.codec.video.parameters()})}})


def campaign(args, settings):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Use a fresh campaign folder")
    runtime = settings["runtime"]
    source_paths = [Path(__file__), HERE / "c1_runtime.py", HERE / "c2_native.py",
                   HERE.parent / "04_外部依赖快照/semantic-video-training/scripts/glc_video_codec.py"]
    write(args.output / "manifest.json", {"utc": datetime.now(timezone.utc).isoformat(), "source": str(args.source),
        "source_sha256": digest(args.source), "runtime": {k: str(v) for k, v in runtime.items()},
        "source_hashes": {str(p): digest(p) for p in source_paths}, "frames": 66, "warmup_frames_per_role": 8,
        "profiled_frame_indices": sorted(SAMPLED), "preset": "compact", "entropy_backend": "native",
        "resumed_from": str(args.resume_from) if args.resume_from else None,
        "scope": "one matched baseline/profile campaign, separate encode/decode processes, no precision changes",
        "caveats": ["Single campaign, no performance confidence intervals", "Desktop applications remain running",
            "GPU telemetry is whole-device, not process-isolated", "CUDA Event spans are not hardware kernel busy time"]})
    stop = threading.Event()

    def monitor():
        with (args.output / "gpu_telemetry.jsonl").open("w", encoding="utf-8") as out:
            while not stop.is_set():
                try:
                    out.write(json.dumps(gpu_snapshot()) + "\n")
                    out.flush()
                except Exception as exc:
                    out.write(json.dumps({"error": str(exc)}) + "\n")
                stop.wait(1)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        for role in ("encode", "decode"):
            command = [str(runtime["python"]), "-u", str(Path(__file__)), "--output", str(args.output),
                       "--source", str(args.source), "--role", role]
            if args.resume_from:
                command += ["--resume-from", str(args.resume_from)]
            with (args.output / (role + ".log")).open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8",
                    errors="replace", creationflags=FLAGS, env={**os.environ, "PYTHONUTF8": "1"})
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line.rstrip(), flush=True)
                code = process.wait()
            if code:
                raise RuntimeError(f"{role} exited {code}; inspect {role}.log")
        encoded = read(args.output / "encode/baseline/encoder/encode.json")
        decoded = read(args.output / "decode/baseline/receiver/decode.json")
        assert decoded["source_media_reads"] == []
        assert len(encoded["frames"]) == len(decoded["frames"]) == 66
        assert all((a["rgb_sha256"], a["latent_sha256"]) == (b["rgb_sha256"], b["latent_sha256"])
                   for a, b in zip(encoded["frames"], decoded["frames"]))
        write(args.output / "validation.json", {"success": True, "frames": 66, "strict_independent_decode": True,
            "source_media_reads": [], "same_rgb_and_latent": True, "profiling_preserves_bitstream": True,
            "bitstream_sha256": digest(args.output / "encode/baseline/stream.glcv")})
    finally:
        stop.set()
        thread.join(timeout=12)


def main():
    from c4_config import load_settings
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path,
        default=settings["workspace"] / "02_实验数据/A_压缩路线选型/A2_GLC验证/long108/input_540.mkv")
    parser.add_argument("--role", choices=["encode", "decode"])
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    args.output, args.source = args.output.resolve(), args.source.resolve(strict=True)
    if args.resume_from:
        args.resume_from = args.resume_from.resolve(strict=True)
    try:
        worker(args, settings) if args.role else campaign(args, settings)
    except BaseException:
        write(args.output / ((args.role or "campaign") + "_failure.json"), {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
