"""Measure the unmodified GLC adapter. One process has one role and preset.

The optional resident factory only retains NeuralCodec, resets DPB, and invokes
the original encode/decode loops. It is a measurement control, not the C1 service.
All outputs remain full-evidence PNG + checksums + MP4. No training or cache of
results. Detailed profiler runs are explicitly excluded from formal baselines.
"""
import time
ENTRY = time.perf_counter()
import argparse
import functools
import json
import os
from pathlib import Path
import sys
import shutil
import tempfile
from contextlib import nullcontext

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "04_外部依赖快照/semantic-video-training/scripts"
sys.path.insert(0, str(SCRIPTS))
from c0_metrics import Timings
import glc_video_codec as ref
from glc_presets import apply_encode_preset
import cv2
import torch


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class Observer:
    def __init__(self, detailed, trace_dir):
        self.detailed, self.trace_dir = detailed, trace_dir
        self.cached = None
        self.profile_active = False
        self.reset()

    def reset(self):
        self.times = Timings()
        self.marks = {}
        self.frames = []
        self.cuda_scopes = []
        self.events = []
        self.profiler_reports = []
        self.started = None

    def wrap(self, owner, attr, name, cuda=False):
        original = getattr(owner, attr)
        @functools.wraps(original)
        def measured(*args, **kwargs):
            begin = end = None
            if cuda and self.detailed:
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
            annotation = torch.profiler.record_function(name) if self.profile_active else nullcontext()
            with self.times.scope(name), annotation:
                result = original(*args, **kwargs)
            if begin is not None:
                end.record()
                self.events.append((name, begin, end))
            return result
        setattr(owner, attr, measured)

    def install(self):
        for attr, name in [("extract_frames", "input_extract"), ("check_source", "source_revision_check"),
                           ("check_weights", "weight_hash_check"), ("load_weights", "load_state_dict"),
                           ("configure", "arithmetic_configuration"), ("make_preview", "preview_ffmpeg"),
                           ("read_stream", "stream_read_verify")]:
            self.wrap(ref, attr, name)
        # CPU-only, synchronous rANS operations: small timers also in formal runs.
        for attr in ("__init__", "indexes", "encode", "decode"):
            self.wrap(ref.GaussianRans, attr, "entropy." + attr)
        self.wrap(ref.StreamWriter, "add", "stream_packet_write")
        old_close = ref.StreamWriter.close
        def close(writer):
            result = old_close(writer)
            self.marks["bitstream_closed_seconds"] = time.perf_counter() - self.started
            return result
        ref.StreamWriter.close = close
        old_json = ref.write_json
        def write_json(path, value):
            if Path(path).name in {"encode.json", "decode.json"}:
                torch.cuda.synchronize()
                self.marks["checks_complete_seconds"] = time.perf_counter() - self.started
                result = old_json(path, value)
                self.marks["evidence_ready_seconds"] = time.perf_counter() - self.started
                return result
            return old_json(path, value)
        ref.write_json = write_json
        # These two are non-overlapping: conversion to source_rgb happens outside
        # Image.open; therefore read time includes only open/header, not all PNG I/O.
        old_save = ref.Image.Image.save
        def save(image, fp, *args, **kwargs):
            name = "source_png_save" if Path(fp).parent.name == "source" else "reconstruction_png_save"
            with self.times.scope(name):
                return old_save(image, fp, *args, **kwargs)
        ref.Image.Image.save = save
        if self.detailed:
            self.wrap(ref, "rgb_array", "rgb_convert_copy_wait")
            self.wrap(ref, "reference_hash", "latent_copy_hash_wait")
            self.wrap(ref.NeuralCodec, "codebook", "codebook", cuda=True)
            self.wrap(ref.NeuralCodec, "prior", "prior", cuda=True)
        self.original_type = ref.NeuralCodec
        old_frame = self.original_type.frame
        def frame(codec, header, index, *args, **kwargs):
            profiled = self.detailed and index in {1, 33, 64}
            before = self.times.snapshot() if self.detailed else None
            self.events = []
            prof = None
            if profiled:
                prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                          torch.profiler.ProfilerActivity.CUDA])
            tick = time.perf_counter()
            with prof if prof is not None else nullcontext():
                self.profile_active = profiled
                with self.times.scope("frame"):
                    result = old_frame(codec, header, index, *args, **kwargs)
                self.profile_active = False
            if self.detailed:
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - tick
                after = self.times.snapshot()
                scopes = {name: {key: row[key] - before.get(name, {}).get(key, 0)
                                 for key in row} for name, row in after.items()}
                spans = [{"name": name, "stream_elapsed_ms": begin.elapsed_time(end)}
                         for name, begin, end in self.events]
                self.frames.append({"index": index, "kind": "I" if index % header["intra_period"] == 0 else "P",
                                    "reset_boundary": index % header["reset_interval"] == 1,
                                    "profiled": profiled, "observed_seconds": elapsed,
                                    "scopes": scopes, "cuda_stream_spans": spans})
            if prof is not None:
                self.trace_dir.mkdir(parents=True, exist_ok=True)
                trace = self.trace_dir / f"frame_{index:03d}.json"
                # Kineto's Windows C++ writer cannot open Unicode paths. Write
                # through an ASCII temporary path; Python handles the final path.
                with tempfile.TemporaryDirectory(prefix="glc_c0_trace_") as temp:
                    temporary_trace = Path(temp) / "trace.json"
                    if not str(temporary_trace).isascii():
                        raise RuntimeError("Set TEMP to an ASCII path for Windows Kineto export")
                    prof.export_chrome_trace(str(temporary_trace))
                    if not temporary_trace.is_file() or not temporary_trace.stat().st_size:
                        raise RuntimeError("Profiler did not export its trace")
                    shutil.copy2(temporary_trace, trace)
                averages = prof.key_averages()
                events = prof.events()
                cuda_events = [event for event in events if str(event.device_type).endswith("CUDA")]
                top = sorted(averages, key=lambda row: row.self_cpu_time_total, reverse=True)[:30]
                self.profiler_reports.append({"index": index, "trace": str(trace),
                    "cuda_event_count": len(cuda_events),
                    "cuda_event_total_us": sum(event.self_device_time_total for event in cuda_events),
                    "top_cpu": [{"name": row.key, "calls": row.count, "self_cpu_us": row.self_cpu_time_total,
                                 "self_device_us": row.self_device_time_total} for row in top]})
            return result
        self.original_type.frame = frame

    def factory(self, repo, weights, device, decoder_only=False):
        with self.times.scope("codec_acquire"):
            if self.cached is None:
                self.cached = self.original_type(repo, weights, device, decoder_only=decoder_only)
                torch.cuda.synchronize()
                if self.detailed:
                    self.instrument_networks(self.cached)
            # State, masks and retained model-owned caches are treated exactly as
            # the reference models do; DPB is never shared across videos or roles.
            self.cached.dpb = {"ref_latent": None, "ref_y": None}
        return self.cached

    def instrument_networks(self, codec):
        for prefix, model in [("I", codec.image), ("P", codec.video)]:
            names = (["enc", "hyper_enc", "hyper_dec", "y_prior_fusion", "dec",
                      "y_spatial_prior_reduction", "y_spatial_prior", "y_spatial_prior_adaptor_1",
                      "y_spatial_prior_adaptor_2", "y_spatial_prior_adaptor_3"] if prefix == "I" else
                     ["context_generation", "temporal_prior_encoder", "weight_generator", "encoder",
                      "temporal_cond", "y_temporal_combine", "hyper_tokenizer", "hyper_inv_tokenizer",
                      "res_prior_param_decoder", "decoder", "y_spatial_prior"])
            for name in names:
                member = getattr(model, name)
                if isinstance(member, torch.nn.Module):
                    self.wrap(member, "forward", f"nn.{prefix}.{name}", cuda=True)
                else:
                    self.wrap(model, name, f"nn.{prefix}.{name}", cuda=True)
            for name in ("encoder", "generator"):
                self.wrap(getattr(model.vqgan, name), "forward", f"nn.{prefix}.vqgan_{name}", cuda=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    observer = Observer(config["diagnostic"], root / "traces")
    observer.install()
    ref.NeuralCodec = observer.factory
    # perf_counter uses the same monotonic system clock across Windows processes.
    environment = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                   "cudnn": torch.backends.cudnn.version(), "opencv": cv2.__version__,
                   "numpy": ref.np.__version__, "pillow": ref.Image.__version__,
                   "gpu": torch.cuda.get_device_name(), "device_capability": list(torch.cuda.get_device_capability()),
                   "pid": os.getpid(), "entry_to_imports_seconds": time.perf_counter() - ENTRY}
    dump(root / "worker_environment.json", environment)
    for position, task in enumerate(config["tasks"]):
        observer.reset()
        folder = root / task["id"]
        folder.mkdir()  # fail closed: never overwrite an earlier task
        common = dict(repo=Path(config["repo"]), weights=Path(config["weights"]),
                      ffmpeg=Path(config["ffmpeg"]), device="cuda", input=Path(task["input"]))
        if config["role"] == "encode":
            options = argparse.Namespace(**common, output=folder / "stream.glcv", workdir=folder / "encoder",
                preset=config["preset"], frames=config.get("frames"), start_frame=0, frame_stride=None,
                container_version=None, width=None, q=None, intra_period=None, reset_interval=None)
            apply_encode_preset(options)
        else:
            options = argparse.Namespace(**common, output=folder / "receiver")
        torch.cuda.synchronize()
        observer.started = time.perf_counter()
        getattr(ref, config["role"])(options)
        torch.cuda.synchronize()
        total = time.perf_counter() - observer.started
        scopes = observer.times.snapshot()
        init = scopes["codec_acquire"]["inclusive_seconds"]
        timing = {"run_id": task["id"], "role": config["role"], "preset": config["preset"],
                  "mode": "fresh_process" if position == 0 else "resident_followup", "pid": os.getpid(),
                  "diagnostic": config["diagnostic"], "output_scope": "full_evidence_png_checksums_mp4",
                  "warmup_policy": "No hidden warmup: first full task retained; later tasks reuse its models only",
                  "task_seconds": {**observer.marks, "offline_view_ready_seconds": total},
                  "codec_acquire_seconds": init,
                  "startup_before_first_task_seconds": (observer.started - config["launch_perf_counter"]) if position == 0 else None,
                  "process_launch_to_evidence_ready_seconds": (observer.started - config["launch_perf_counter"] + observer.marks["evidence_ready_seconds"]) if position == 0 else None,
                  "scopes": scopes, "browser_upload_queue_load_seconds": None, "transfer_measured_seconds": None,
                  "gpu_timing_note": "CUDA stream spans include waits/idle and may nest: never add them as GPU busy time.",
                  "frames": observer.frames, "profiler": observer.profiler_reports}
        dump(folder / "timing.json", timing)
        print("C0_TASK_DONE " + task["id"] + " " + json.dumps(timing["task_seconds"]), flush=True)


if __name__ == "__main__":
    main()
