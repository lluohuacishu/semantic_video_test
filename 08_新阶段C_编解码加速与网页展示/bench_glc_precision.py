"""Bounded GLC precision trial; production codec, weights, and settings stay untouched."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
MODES = ("fp32", "tf32", "amp_rgb", "amp_rgb_safe", "amp_full", "tf32_amp_rgb",
         "amp_core_fp32_rgb")
SAFE_PREFIX_BLOCKS = 22
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def configure(codec, mode, torch):
    tf32 = mode in {"tf32", "tf32_amp_rgb"}
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    hooks = []
    if mode in {"amp_rgb", "tf32_amp_rgb"}:
        for model in (codec.image, codec.video):
            generator = model.vqgan.generator
            original = generator.forward

            def forward(*args, _original=original, **kwargs):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return _original(*args, **kwargs)

            generator.forward = forward
    elif mode == "amp_rgb_safe":
        for model in (codec.image, codec.video):
            generator = model.vqgan.generator

            def safe_forward(x, _generator=generator):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    x = _generator.post_quant_conv(x)
                    for block in _generator.blocks[:SAFE_PREFIX_BLOCKS]:
                        x = block(x)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                    x = x.float()
                    for block in _generator.blocks[SAFE_PREFIX_BLOCKS:]:
                        x = block(x)
                    return x

            generator.forward = safe_forward
            for index, block in enumerate(generator.blocks):
                def check_finite(_module, _inputs, output, _index=index):
                    if isinstance(output, torch.Tensor) and not bool(torch.isfinite(output).all()):
                        raise ValueError(f"FP16 RGB generator overflow at block {_index}")

                hooks.append(block.register_forward_hook(check_finite))
    elif mode == "amp_full":
        original = codec.frame

        def frame(*args, **kwargs):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return original(*args, **kwargs)

        codec.frame = frame
    elif mode == "amp_core_fp32_rgb":
        for model in (codec.image, codec.video):
            generator = model.vqgan.generator
            original = generator.forward

            def fp32_forward(x, _original=original):
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                    return _original(x.float())

            def check_input(_module, inputs):
                if not bool(torch.isfinite(inputs[0]).all()):
                    raise ValueError("AMP produced non-finite RGB latent")

            def check_output(_module, _inputs, output):
                if not bool(torch.isfinite(output).all()):
                    raise ValueError("FP32 RGB generator produced non-finite pixels")

            generator.forward = fp32_forward
            hooks.append(generator.register_forward_pre_hook(check_input))
            hooks.append(generator.register_forward_hook(check_output))
        original_frame = codec.frame

        def frame(*args, **kwargs):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return original_frame(*args, **kwargs)

        codec.frame = frame
    if mode != "fp32":
        codec.profile.update({"precision_trial": mode, "tf32": tf32,
                              "amp": "float16_rgb_prefix_22_then_float32" if mode == "amp_rgb_safe" else
                                     "float16_neural_core_with_float32_rgb_generator" if mode == "amp_core_fp32_rgb" else
                                     "float16_rgb_generator" if "amp_rgb" in mode else
                                     "float16_full_frame" if mode == "amp_full" else None})
    return hooks


def worker(args, settings):
    from c1_runtime import ResidentRunner
    from c2_native import NativeGaussianRans
    import glc_video_codec as ref

    runtime = settings["runtime"]
    root = args.output / args.mode
    hooks = []

    def factory(*factory_args, **factory_kwargs):
        codec = ref.NeuralCodec(*factory_args, **factory_kwargs)
        if args.backend == "native":
            codec.entropy = NativeGaussianRans(runtime["native_library"], codec.entropy)
        hooks.extend(configure(codec, args.mode, ref.torch))
        write(root / f"{args.role}_runtime.json", {
            "entropy_backend": args.backend, "entropy_class": type(codec.entropy).__name__,
            "arithmetic": codec.profile, "torch_threads": ref.torch.get_num_threads(),
            "cudnn_benchmark": ref.torch.backends.cudnn.benchmark})
        return codec

    runner = ResidentRunner(args.role, runtime["repo"], runtime["weights"],
                            runtime["ffmpeg"], codec_factory=factory)
    if args.role == "encode":
        warmup = {"role": "encode", "id": "warmup", "input": str(args.source),
                  "output": str(root / "encode/warmup"), "preset": "compact",
                  "frames": 8, "output_mode": "display", "compatibility_policy": "strict"}
        runner.run(warmup)
        for hook in hooks:
            hook.remove()
        for iteration in range(args.repeats):
            task = {**warmup, "id": f"run_{iteration + 1}",
                    "output": str(root / f"encode/run_{iteration + 1}"), "frames": 66}
            runner.run(task)
    else:
        warmup = {"role": "decode", "id": "warmup", "input": str(root / "encode/warmup/stream.glcv"),
                  "output": str(root / "decode/warmup"), "output_mode": "display",
                  "compatibility_policy": "strict"}
        runner.run(warmup)
        for hook in hooks:
            hook.remove()
        for iteration in range(args.repeats):
            task = {**warmup, "id": f"run_{iteration + 1}",
                    "input": str(root / f"encode/run_{iteration + 1}/stream.glcv"),
                    "output": str(root / f"decode/run_{iteration + 1}")}
            runner.run(task)


def stream_payload_hash(path, scripts):
    sys.path.insert(0, str(scripts))
    from glc_container import read_stream

    header, frames = read_stream(path)
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(frame["z"])
        for part in frame["y"]:
            digest.update(part)
    return {"header": header, "payload_sha256": digest.hexdigest(),
            "payload_bytes": sum(len(frame["z"]) + sum(map(len, frame["y"])) for frame in frames)}


def run_child(args, role):
    command = [str(args.python), "-u", str(Path(__file__).resolve()), "--output", str(args.output),
               "--source", str(args.source), "--mode", args.mode, "--role", role,
               "--repeats", str(args.repeats), "--backend", args.backend]
    log_path = args.output / args.mode / f"{role}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace",
                                   creationflags=FLAGS, env={**os.environ, "PYTHONUTF8": "1"})
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(f"[{args.mode}/{role}] {line.rstrip()}", flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"{args.mode}/{role} exited {code}; see {log_path}")


def summarize(args, settings):
    from statistics import mean, stdev

    scripts = HERE.parent / "04_外部依赖快照/semantic-video-training/scripts"
    records = []
    for mode in args.modes:
        root = args.output / mode
        status = {"mode": mode, "success": False}
        try:
            for role in ("encode", "decode"):
                args.mode = mode
                run_child(args, role)
            runs = []
            for index in range(1, args.repeats + 1):
                folder = root / f"encode/run_{index}"
                encoded = read(folder / "encoder/encode.json")
                decoded = read(root / f"decode/run_{index}/receiver/decode.json")
                if not decoded["independent_decoder"] or decoded["source_media_reads"]:
                    raise RuntimeError("Decoder did not pass the independent-source audit")
                if len(encoded["frames"]) != 66 or len(decoded["frames"]) != 66:
                    raise RuntimeError("Expected 66 encoded and decoded frames")
                if not decoded["all_frame_checksums_match"] or any(
                    (a["rgb_sha256"], a["latent_sha256"]) != (b["rgb_sha256"], b["latent_sha256"])
                    for a, b in zip(encoded["frames"], decoded["frames"])):
                    raise RuntimeError("Encoder/decoder reconstruction mismatch")
                payload = stream_payload_hash(folder / "stream.glcv", scripts)
                enc_timing = read(folder / "timing.json")["task_seconds"]
                dec_timing = read(root / f"decode/run_{index}/timing.json")["task_seconds"]
                runs.append({"iteration": index, "stream_bytes": encoded["file_bytes"],
                    "payload_sha256": payload["payload_sha256"],
                    "payload_bytes": payload["payload_bytes"],
                    "encode_task_seconds": enc_timing["offline_view_ready_seconds"],
                    "decode_task_seconds": dec_timing["offline_view_ready_seconds"],
                    "mean_psnr_db": encoded["mean_psnr_db"],
                    "encode_core_seconds": encoded["frame_processing_seconds"],
                    "encode_core_fps": 66 / encoded["frame_processing_seconds"],
                    "decode_core_seconds": decoded["frame_processing_seconds"],
                    "decode_core_fps": 66 / decoded["frame_processing_seconds"],
                    "strict_decode_ok": decoded["all_frame_checksums_match"],
                    "rgb_hashes_match_encoder": all(
                        a["rgb_sha256"] == b["rgb_sha256"]
                        for a, b in zip(encoded["frames"], decoded["frames"])),
                    "latent_hashes_match_encoder": all(
                        a["latent_sha256"] == b["latent_sha256"]
                        for a, b in zip(encoded["frames"], decoded["frames"]))})
            status.update(success=True, runs=runs)
            for key in ("stream_bytes", "payload_bytes", "mean_psnr_db", "encode_core_fps", "decode_core_fps"):
                values = [run[key] for run in runs]
                status[key + "_mean"] = mean(values)
                status[key + "_stdev"] = stdev(values) if len(values) > 1 else None
            status["payload_sha256_identical_across_repeats"] = len({r["payload_sha256"] for r in runs}) == 1
        except Exception as exc:
            status.update(error=str(exc), traceback=traceback.format_exc())
        records.append(status)
        write(args.output / "summary.partial.json", {"modes": records})
        print(f"[{mode}] {'OK' if status['success'] else 'FAILED'}", flush=True)

    baseline = next((item for item in records if item["mode"] == "fp32" and item["success"]), None)
    if baseline:
        for item in records:
            if not item["success"]:
                continue
            item["encode_speedup_vs_fp32"] = item["encode_core_fps_mean"] / baseline["encode_core_fps_mean"]
            item["decode_speedup_vs_fp32"] = item["decode_core_fps_mean"] / baseline["decode_core_fps_mean"]
            item["psnr_delta_db_vs_fp32"] = item["mean_psnr_db_mean"] - baseline["mean_psnr_db_mean"]
            item["payload_matches_fp32"] = item["runs"][0]["payload_sha256"] == baseline["runs"][0]["payload_sha256"]
    write(args.output / "summary.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source), "source_sha256": hashlib.file_digest(args.source.open("rb"), "sha256").hexdigest(),
        "gpu": "RTX 4060 Laptop GPU", "frames": 66, "preset": "compact", "repeats": args.repeats,
        "entropy_backend": args.backend,
        "timing_scope": "per-frame neural codec processing only; excludes input, preview finalization, and model load",
        "modes": records})


def main():
    from c4_config import load_settings
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=settings["workspace"] /
                        "02_实验数据/A_压缩路线选型/A2_GLC验证/long108/input_540.mkv")
    parser.add_argument("--python", type=Path, default=settings["runtime"]["python"])
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--role", choices=("encode", "decode"))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--backend", choices=("native", "python"), default="native")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.source = args.source.resolve(strict=True)
    args.python = args.python.resolve(strict=True)
    if args.role:
        try:
            worker(args, settings)
        except BaseException:
            write(args.output / args.mode / f"{args.role}_failure.json", {"traceback": traceback.format_exc()})
            raise
    else:
        if (args.output / "manifest.json").exists():
            raise FileExistsError("Use a fresh experiment folder")
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "bench_glc_precision.snapshot.py").write_bytes(Path(__file__).read_bytes())
        write(args.output / "manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
            "modes": args.modes, "repeats": args.repeats, "source": str(args.source),
            "entropy_backend": args.backend,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "precision_scope": "FP32; TF32; RGB AMP; RGB prefix blocks[0:22] AMP and suffix FP32; full AMP; TF32/RGB AMP; core AMP with FP32 RGB",
            "production_files_modified": False})
        summarize(args, settings)


if __name__ == "__main__":
    main()
