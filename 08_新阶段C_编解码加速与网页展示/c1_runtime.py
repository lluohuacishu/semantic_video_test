"""C1 resident, single-role task executor. NeuralCodec.frame stays unchanged.

Research: source/reconstruction PNGs, per-frame metrics/hashes, stream and MP4.
Display: same stream, metrics/hashes and MP4; no PNG intermediate files.
Use a separate receiver process with role='decode'; it accepts only a bitstream.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "04_外部依赖快照/semantic-video-training/scripts"
sys.path.insert(0, str(SCRIPTS))
import glc_video_codec as ref
from glc_container import MAX_BYTES, container_overhead_breakdown
from glc_presets import apply_encode_preset
from c1_io import MemoryFrames, RawPreview


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


class TaskCancelled(Exception):
    pass


_audit_context = threading.local()


def _media_audit(event, values):
    reads = getattr(_audit_context, "reads", None)
    if reads is None or event != "open" or not isinstance(values[0], (str, bytes)):
        return
    path, mode, flags = values
    path = Path(os.fsdecode(path))
    reading = ((isinstance(mode, str) and "r" in mode) or
               (mode is None and not flags & (os.O_WRONLY | os.O_RDWR)))
    if reading and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".avi", ".mp4", ".mkv", ".npz"}:
        reads.append(str(path.resolve()))
        raise RuntimeError(f"Independent receiver attempted to read media: {path}")


# One hook, not one permanent closure per task. It is inactive outside decode.
sys.addaudithook(_media_audit)


@contextmanager
def receiver_audit():
    reads = []
    previous = getattr(_audit_context, "reads", None)
    _audit_context.reads = reads
    try:
        yield reads
    finally:
        _audit_context.reads = previous


class ResidentRunner:
    def __init__(self, role, repo, weights, ffmpeg, device="cuda", codec_factory=None):
        if role not in {"encode", "decode"}:
            raise ValueError("A worker has exactly one role: encode or decode")
        self.role, self.repo, self.weights = role, Path(repo), Path(weights)
        self.ffmpeg, self.device = Path(ffmpeg), device
        self.factory = codec_factory or ref.NeuralCodec
        self.codec = None
        self.lock = threading.Lock()
        self.tasks_started = 0

    def synchronize(self):
        if self.device == "cuda":
            ref.torch.cuda.synchronize()

    def acquire(self):
        if self.codec is None:
            self.codec = self.factory(self.repo, self.weights, self.device, decoder_only=self.role == "decode")
            self.synchronize()
        self.reset()
        return self.codec

    def reset(self):
        if self.codec is not None:
            self.codec.dpb = {"ref_latent": None, "ref_y": None}

    @contextmanager
    def task_scope(self, role):
        if role != self.role:
            raise ValueError("Sender and receiver must use separate workers")
        with self.lock:
            self.reset()
            self.tasks_started += 1
            try:
                yield
            finally:
                self.reset()  # also after cancellation, corrupt data, or I/O failure

    def options(self, task, folder, cancel_event=None):
        mode = task.get("output_mode", "research")
        level = task.get("png_level", 1)
        if mode not in {"research", "display"} or type(level) is not int or not 0 <= level <= 9:
            raise ValueError("Invalid output mode or PNG compression level")
        args = argparse.Namespace(input=Path(task["input"]), repo=self.repo, weights=self.weights,
                                  ffmpeg=self.ffmpeg, device=self.device, output_mode=mode, png_level=level)
        args.sampling_mode = task.get("sampling_mode", "legacy_frame_stride")
        args.compatibility_policy = task.get("compatibility_policy", "strict")
        if not args.input.is_file() or not self.ffmpeg.is_file():
            raise ValueError("Input and FFmpeg must exist")
        if self.role == "encode":
            args.preset = task.get("preset")
            if args.preset is None:
                raise ValueError("Select one of the four presets explicitly")
            args.frames = task.get("frames")
            args.start_frame = task.get("start_frame", 0)
            apply_encode_preset(args)
            args.prepared_transform = task.get("prepared_transform")
            if args.sampling_mode == "time":
                from c5_timebase import prepare
                if task.get("frames") is not None or task.get("start_frame", 0):
                    raise ValueError("Time sampling uses clip bounds, not frame-count overrides")
                def cancel_probe():
                    if cancel_event is not None and cancel_event.is_set():
                        raise TaskCancelled("Task cancelled during PTS scan")
                args.time_plan = prepare(args.input, self.ffmpeg, args.preset, task.get("clip"), cancel_probe,
                                         video=task.get("video"))
                args.frames = args.time_plan["frames"]
            elif args.sampling_mode != "legacy_frame_stride":
                raise ValueError("Unknown source sampling mode")
            if (type(args.frames) is not int or not 1 <= args.frames <= 10000 or
                    type(args.start_frame) is not int or args.start_frame < 0):
                raise ValueError("Invalid frame count or start frame")
            args.output, args.workdir = folder / "stream.glcv", folder / "encoder"
        else:
            args.output = folder / "receiver"
        return args

    def run(self, task, cancel_event=None, on_event=None):
        """Block on the serial queue; callback distinguishes evidence from video.

        Cancellation is cooperative at frame/preview boundaries. No result,
        packet, input tensor, frame number or sequence digest survives a task.
        """
        with self.task_scope(task.get("role", self.role)):
            folder = Path(task["output"]).resolve()
            args = self.options(task, folder, cancel_event)
            folder.mkdir(parents=True, exist_ok=False)
            self.synchronize()
            started = time.perf_counter()
            had_model = self.codec is not None
            marks, previews = {}, []

            def check_cancel():
                if cancel_event is not None and cancel_event.is_set():
                    raise TaskCancelled("Task cancelled at a frame/preview boundary")

            def event(state, **extra):
                value = {"state": state, "role": self.role, "elapsed_seconds": time.perf_counter() - started,
                         "task_seconds": dict(marks), **extra}
                write_json(folder / "status.json", value)
                if on_event:
                    on_event(value)

            event("running")
            try:
                check_cancel()
                tick = time.perf_counter()
                codec = self.acquire()
                init = time.perf_counter() - tick
                if self.device == "cuda":
                    ref.torch.cuda.reset_peak_memory_stats()
                if self.role == "encode":
                    result = self._encode(args, codec, previews, check_cancel, marks, started)
                    result_file = args.workdir / "encode.json"
                else:
                    with receiver_audit() as reads:
                        result = self._decode(args, codec, previews, check_cancel, marks, started)
                        result["source_media_reads"] = reads
                    result_file = args.output / "decode.json"
                self.synchronize()
                marks["checks_complete_seconds"] = time.perf_counter() - started
                result.update(output_mode=args.output_mode, png_compress_level=args.png_level if args.output_mode == "research" else None,
                              frame_path="ffmpeg_pts_rgb" if args.sampling_mode == "time" else "opencv_memory_rgb",
                              preview_path="ffmpeg_rawvideo_pipe",
                              pipeline_seconds_before_preview=marks["checks_complete_seconds"],
                              peak_allocated_mib=ref.torch.cuda.max_memory_allocated() / 2**20 if self.device == "cuda" else None,
                              peak_reserved_mib=ref.torch.cuda.max_memory_reserved() / 2**20 if self.device == "cuda" else None)
                write_json(result_file, result)
                marks["evidence_ready_seconds"] = time.perf_counter() - started
                event("preview_pending", result_path=str(result_file))
                for preview in previews:
                    preview.finish(check_cancel)
                self.synchronize()
                marks["offline_view_ready_seconds"] = time.perf_counter() - started
                timing = {"role": self.role, "preset": task.get("preset"), "pid": os.getpid(),
                          "implementation": "c1", "output_mode": args.output_mode,
                          "png_level": args.png_level if args.output_mode == "research" else None,
                          "model_was_resident": had_model, "codec_acquire_seconds": init,
                          "task_seconds": marks, "task_timer_origin_perf_counter": started,
                          "detailed_instrumentation": False,
                          "browser_upload_queue_load_seconds": None, "transfer_measured_seconds": None,
                          "preview_feed_seconds": sum(p.feed_seconds for p in previews),
                          "preview_overlaps_processing": True, "result_path": str(result_file)}
                write_json(folder / "timing.json", timing)
                event("ready", previews=[str(p.output) for p in previews])
                return timing
            except BaseException as exc:
                for preview in previews:
                    preview.abort()
                write_json(folder / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc(), "task_seconds": marks})
                event("cancelled" if isinstance(exc, TaskCancelled) else "failed", error=str(exc))
                raise

    def _preview(self, previews, path, header):
        value = RawPreview(self.ffmpeg, path, header["width"], header["height"], header["fps"])
        previews.append(value)
        return value

    def _encode(self, args, codec, previews, check_cancel, marks, started):
        args.workdir.mkdir()
        if args.output_mode == "research":
            (args.workdir / "source").mkdir()
            (args.workdir / "frames").mkdir()
        if args.sampling_mode == "time":
            from c5_timebase import TimedFrames
            reader = TimedFrames(args, cancel=check_cancel)
        else:
            reader = MemoryFrames(args)
        packets, records, buffered_bytes = [], [], 0
        png_seconds = 0.0
        try:
            first = next(reader, None)
            if first is None:
                raise ValueError("No frames were read")
            header = {key: reader.metadata()[key] for key in ("width", "height", "fps")}
            header.update(frames=args.frames, q=args.q, intra_period=args.intra_period,
                          reset_interval=args.reset_interval, revision=ref.REVISION, entropy=ref.ENTROPY_ID,
                          table_sha256=codec.entropy.table_sha256, arithmetic=codec.profile,
                          weights={key: value["sha256"] for key, value in ref.ASSETS.items()})
            if args.container_version == 2:
                header["container_version"] = 2
            ref.validate_header(header)
            source_preview = self._preview(previews, args.workdir / "source_preview.mp4", header)
            restored_preview = self._preview(previews, args.workdir / "reconstruction_preview.mp4", header)
            prepared, coding_preview = None, None
            if args.prepared_transform:
                from c5_enhancement import PreparedFrames
                prepared = PreparedFrames(args.prepared_transform, args.frames, header["width"], header["height"])
                coding_preview = self._preview(previews, args.workdir / "coding_input_preview.mp4", header)
            for index, source_rgb in enumerate(itertools.chain((first,), reader)):
                check_cancel()
                name = f"im{index + 1:05d}.png"
                coding_rgb = prepared.frame(index, source_rgb) if prepared else source_rgb
                # Exactly the reference's operation order and FP32 arithmetic.
                tensor = ref.torch.from_numpy(coding_rgb.copy()).to(args.device).permute(2, 0, 1).unsqueeze(0).float()
                tensor = tensor.div(255)
                tensor = ref.F.pad(tensor, (0, (-header["width"]) % 64, 0, (-header["height"]) % 64), mode="replicate")
                tensor = tensor.mul(2).sub(1)
                tick = time.perf_counter()
                rgb, packet = codec.frame(header, index, x=tensor)
                seconds = time.perf_counter() - tick
                buffered_bytes += len(packet["z"]) + sum(map(len, packet["y"])) + 128
                if buffered_bytes > MAX_BYTES:
                    raise ValueError("Encoded packet buffer exceeds the 256 MiB container limit")
                packets.append(packet)
                if args.output_mode == "research":
                    tick = time.perf_counter()
                    ref.Image.fromarray(source_rgb).save(args.workdir / "source" / name, compress_level=args.png_level)
                    ref.Image.fromarray(rgb).save(args.workdir / "frames" / name, compress_level=args.png_level)
                    png_seconds += time.perf_counter() - tick
                source_preview.add(source_rgb, check_cancel)
                if coding_preview:
                    coding_preview.add(coding_rgb, check_cancel)
                restored_preview.add(rgb, check_cancel)
                mse = float(ref.np.mean((source_rgb.astype(ref.np.float64) - rgb) ** 2))
                records.append({"frame": index + 1, "type": "I" if packet["kind"] == 0 else "P",
                                "z_bytes": len(packet["z"]), "y_bytes": sum(map(len, packet["y"])),
                                "seconds": seconds, "psnr_db": 10 * math.log10(255 ** 2 / mse) if mse else None,
                                "source_rgb_sha256": reader.hashes[index],
                                "coding_rgb_sha256": hashlib.sha256(coding_rgb.tobytes()).hexdigest(),
                                "rgb_sha256": packet["rgb_sha256"], "latent_sha256": packet["latent_sha256"]})
                print(f"encode {index + 1}/{args.frames} {seconds:.2f}s", flush=True)
            source = reader.metadata()
        finally:
            reader.close()
        # Buffer only bounded compressed packets, never an entire RGB video.
        # EOF determines the final header, preserving legacy short-input behavior.
        header["frames"] = len(records)
        check_cancel()
        writer = ref.StreamWriter(args.output, header, version=args.container_version)
        try:
            for packet, record in zip(packets, records):
                record["record_bytes"] = writer.add(packet)
            writer.finish()
        finally:
            writer.close()
        marks["bitstream_closed_seconds"] = time.perf_counter() - started
        size = args.output.stat().st_size
        psnr = [r["psnr_db"] for r in records if r["psnr_db"] is not None]
        return {"codec": "GLC official networks; C1 I/O; unchanged neural arithmetic and container",
                "container_version": args.container_version, "official_bitstream_format": False,
                "actual_bitstream": True, "training_performed": False, "input": source, "header": header,
                "weight_loading": codec.loading, "bitstream_path": str(args.output), "file_bytes": size,
                "file_sha256": ref.sha256_file(args.output), "actual_bpp": size * 8 / (header["width"] * header["height"] * header["frames"]),
                "actual_video_kbps": size * 8 / source["duration_seconds"] / 1000,
                "mean_psnr_db": float(ref.np.mean(psnr)) if psnr else None,
                "frame_processing_seconds": sum(r["seconds"] for r in records),
                "input_read_convert_hash_seconds": reader.seconds, "png_save_seconds": png_seconds,
                "compressed_packet_buffer_bytes_upper_bound": buffered_bytes,
                "container_overhead_bytes": size - sum(r["z_bytes"] + r["y_bytes"] for r in records),
                "container_overhead_breakdown": container_overhead_breakdown(header, args.container_version),
                "audio": "not encoded", "frames": records, "selected_preset": args.preset,
                "preprocessing": prepared.summary if prepared else None,
                "psnr_reference": "sampled/resized original RGB, before any lowpass or semantic protection",
                "input_transform": "RGB / 255; replicate right/bottom to multiples of 64; x * 2 - 1"}

    def _decode(self, args, codec, previews, check_cancel, marks, started):
        header, packets, integrity = ref.read_stream(args.input, with_integrity=True)
        if header["revision"] != ref.REVISION or header["entropy"] != ref.ENTROPY_ID:
            raise ValueError("Unsupported model or entropy coder version")
        if header["weights"] != {key: value["sha256"] for key, value in ref.ASSETS.items()}:
            raise ValueError("Stream requires different model weights")
        compatibility = None
        if args.compatibility_policy != "strict":
            from c5_compat import check_arithmetic
            compatibility = check_arithmetic(header["arithmetic"], codec.profile,
                                             args.compatibility_policy, integrity["container_version"])
        elif header["arithmetic"] != codec.profile:
            raise ValueError("Sender and receiver arithmetic profiles differ")
        if header["table_sha256"] != codec.entropy.table_sha256:
            raise ValueError("Entropy tables differ from sender")
        args.output.mkdir()
        if args.output_mode == "research":
            (args.output / "frames").mkdir()
        preview = self._preview(previews, args.output / "reconstruction_preview.mp4", header)
        records, png_seconds = [], 0.0
        checksums = ref.SequenceChecksums() if integrity["container_version"] == 2 else None
        verification = None
        for index, packet in enumerate(packets):
            check_cancel()
            tick = time.perf_counter()
            rgb, checked = codec.frame(header, index, packet=packet)
            seconds = time.perf_counter() - tick
            if args.output_mode == "research":
                tick = time.perf_counter()
                ref.Image.fromarray(rgb).save(args.output / "frames" / f"im{index + 1:05d}.png", compress_level=args.png_level)
                png_seconds += time.perf_counter() - tick
            preview.add(rgb, check_cancel)
            if checksums is not None:
                checksums.add(index, checked)
            records.append({"frame": index + 1, "seconds": seconds,
                            "rgb_sha256": checked["rgb_sha256"], "latent_sha256": checked["latent_sha256"],
                            "sender_checksums_match": True if checksums is None else None})
            print(f"decode {index + 1}/{header['frames']} {seconds:.2f}s", flush=True)
        if checksums is not None:
            from c7_diagnostics import verify_sequence
            verification = verify_sequence(checksums, integrity["sequence_reconstruction_checksums"],
                            args.output / "reconstruction_diagnostics.json", header=header,
                            receiver_profile=codec.profile, records=records,
                            stream_sha256=ref.sha256_file(args.input), policy=args.compatibility_policy)
            for record in records:
                record["sender_checksums_match"] = True if verification["rgb_verified"] else None
                record["sender_latent_checksums_match"] = True
        if compatibility is not None:
            compatibility["reconstruction_verified"] = verification["verified"] if verification else True
        marks["decode_verified_seconds"] = time.perf_counter() - started
        return {"input_bitstream": str(args.input.resolve()), "input_sha256": ref.sha256_file(args.input),
                "header": header, "independent_decoder": True, "encoder_modules_disabled": True,
                "whole_file_checksum_verified": True, "container_version": integrity["container_version"],
                "arithmetic_compatibility": compatibility,
                "reconstruction_verification": verification,
                "sequence_reconstruction_checksums": checksums.values() if checksums is not None else None,
                "all_frame_checksums_match": verification["verified"] if verification else True,
                "frame_checksum_verification": "ordered_aggregate" if checksums is not None else "per_frame",
                "frame_processing_seconds": sum(r["seconds"] for r in records),
                "png_save_seconds": png_seconds, "frames": records}
