"""Small B4 pilot for GLC container overhead and spatiotemporal reduction."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WORKSPACE = REPO.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "04_外部依赖快照" / "semantic-video-training" / "scripts"))
sys.path.insert(0, str(REPO / "01_主工程" / "semantic_mvp"))

from glc_container import read_stream
from repack_glc_v1_to_v2 import repack, transfer_estimate
from transport import loopback_transfer


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def console_line(value: str) -> None:
    """Write subprocess output without letting the Windows console codepage abort a run."""
    stream = sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    safe_value = value.encode(encoding, errors="backslashreplace").decode(encoding)
    stream.write(safe_value)
    stream.flush()


def is_progress_milestone(line: str) -> bool:
    fields = line.split(maxsplit=2)
    if len(fields) < 2 or fields[0] not in {"encode", "decode"} or "/" not in fields[1]:
        return True
    try:
        completed, total = (int(value) for value in fields[1].split("/", 1))
    except ValueError:
        return True
    return completed in {1, total} or completed % 25 == 0


def command_record(directory: Path, label: str, argv: list[str], cwd: Path) -> dict:
    log = directory / f"{label}.log"
    row = {"argv": argv, "cwd": str(cwd), "log": str(log),
           "started_at": datetime.now().astimezone().isoformat(), "status": "started"}
    write_json(directory / f"{label}.command.json", row)
    started = time.perf_counter()
    with log.open("w", encoding="utf-8", errors="replace", buffering=1) as handle:
        process = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        row["pid"] = process.pid
        write_json(directory / f"{label}.command.json", row)
        for line in process.stdout:
            handle.write(line)
            if is_progress_milestone(line):
                console_line(f"[{directory.name}/{label}] {line}")
        returncode = process.wait()
    row.update(returncode=returncode, seconds=time.perf_counter() - started,
               finished_at=datetime.now().astimezone().isoformat(),
               status="completed" if returncode == 0 else "failed")
    write_json(directory / f"{label}.command.json", row)
    if returncode:
        raise RuntimeError(f"B4 {label} failed (exit {returncode}); see {log}")
    return row


def mean_psnr_from_mse(mse: float) -> float | None:
    return None if mse <= 0 else 10 * math.log10((255.0 ** 2) / mse)


def resolution_and_time_quality(source_dir: Path, codec_input_dir: Path, decoded_dir: Path, *, stride: int,
                                reference_frames: int, width: int, height: int) -> dict:
    """Compare held reconstructed frames against the 960x540, 25-fps timeline."""
    import numpy as np
    from PIL import Image

    source_mse = 0.0
    codec_input_mse = 0.0
    source_pixels = 0
    codec_input_pixels = 0
    for reference_index in range(reference_frames):
        frame_index = min(reference_index // stride, (reference_frames - 1) // stride)
        source_path = source_dir / f"im{reference_index + 1:05d}.png"
        decoded_path = decoded_dir / f"im{frame_index + 1:05d}.png"
        sampled_source_path = codec_input_dir / f"im{frame_index + 1:05d}.png"
        with Image.open(source_path) as image:
            reference = np.asarray(image.convert("RGB"), dtype=np.float64)
        with Image.open(sampled_source_path) as image:
            codec_input = np.asarray(image.convert("RGB"), dtype=np.float64)
        with Image.open(decoded_path) as image:
            decoded = image.convert("RGB")
            if decoded.size != (width, height):
                raise ValueError(f"Unexpected decoded frame dimensions: {decoded.size}")
            decoded_reference = np.asarray(
                decoded.resize((960, 540), Image.Resampling.LANCZOS), dtype=np.float64)
            decoded_native = np.asarray(decoded, dtype=np.float64)
        if reference.shape[:2] != (540, 960):
            raise ValueError(f"Unexpected source reference dimensions: {reference.shape}")
        source_mse += float(np.square(reference - decoded_reference).mean())
        codec_input_mse += float(np.square(codec_input - decoded_native).mean())
        source_pixels += 1
        codec_input_pixels += 1
    return {
        "reference_timeline": "250 source frames at 960x540, 25 fps; lower-rate reconstructions held for two source intervals",
        "mean_psnr_db_vs_960x540_25fps_timeline_after_lanczos_upscale": mean_psnr_from_mse(source_mse / source_pixels),
        "mean_psnr_db_vs_sampled_encoder_input_at_native_resolution": mean_psnr_from_mse(codec_input_mse / codec_input_pixels),
        "frames_compared_at_reference_timeline": source_pixels,
        "metric_note": "RGB PNGs before viewing-video encoding; frame-hold playback captures temporal sampling changes",
    }


def create_comparison(ffmpeg: Path, source_dir: Path, baseline_dir: Path, candidate_dir: Path,
                      output: Path, label: str, stride: int) -> dict:
    from PIL import Image, ImageDraw, ImageFont

    frame_count = 250
    tile_width, tile_height = 640, 360
    canvas_width, canvas_height = tile_width * 3, tile_height + 40
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 18) if font_path.is_file() else ImageFont.load_default()
    command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{canvas_width}x{canvas_height}",
               "-framerate", "25", "-i", "pipe:0", "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
    labels = ["Source · 960×540 / 25 fps", "GLC q0 · 960×540 / 25 fps", label]
    try:
        for index in range(frame_count):
            folders = (source_dir, baseline_dir, candidate_dir)
            lookup = (index, index, min(index // stride, (frame_count - 1) // stride))
            canvas = Image.new("RGB", (canvas_width, canvas_height), "#18242e")
            draw = ImageDraw.Draw(canvas)
            for column, (folder, frame_index) in enumerate(zip(folders, lookup, strict=True)):
                with Image.open(folder / f"im{frame_index + 1:05d}.png") as image:
                    resized = image.convert("RGB").resize((tile_width, tile_height), Image.Resampling.LANCZOS)
                canvas.paste(resized, (column * tile_width, 40))
                draw.text((column * tile_width + 10, 10), labels[column], font=font, fill="white")
            process.stdin.write(canvas.tobytes())
    finally:
        process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", "replace")
    returncode = process.wait()
    if returncode:
        raise RuntimeError(f"Comparison preview failed ({returncode}): {stderr[-2000:]}")
    return {"path": str(output), "bytes": output.stat().st_size, "sha256": sha256(output),
            "frames": frame_count, "fps": 25, "duration_seconds": frame_count / 25,
            "is_transmitted_glc": False, "ffmpeg_args": command}


def run(args) -> dict:
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"Preserve existing experiment output: {output}")
    output.mkdir(parents=True, exist_ok=args.resume)
    status_path = output / "pilot_status.json"
    prior_status = json.loads(status_path.read_text(encoding="utf-8")) if args.resume and status_path.is_file() else {}
    if args.resume and not prior_status:
        raise FileNotFoundError("Cannot resume: existing B4 output has no pilot_status.json.")
    if args.resume and prior_status.get("status") == "complete":
        raise ValueError("Cannot resume a B4 pilot already marked complete.")
    config = json.loads((HERE / "configs" / "b0_baseline.json").read_text(encoding="utf-8"))
    runtime = config["runtime_dependencies"]
    python = Path(runtime["glc_python"]).resolve()
    adapter = (REPO / "04_外部依赖快照" / "semantic-video-training" / "scripts" / "glc_video_codec.py").resolve()
    input_video = args.input.resolve() if args.input else (
        WORKSPACE / "work" / "stage_a_glc_round2_20261017" / "long108" / "input_540.mkv")
    baseline_root = WORKSPACE / "work" / "stage_a_glc_round2_20261017" / "long108" / "glc_540_g64"
    baseline_encode_path = baseline_root / "sender" / "encode.json"
    baseline_decode_path = baseline_root / "receiver" / "decode.json"
    baseline_stream = baseline_root / "stream.glcv"
    glc_repo, weights, ffmpeg = (Path(runtime[key]).resolve() for key in ("glc_repo", "weights", "ffmpeg"))
    required_files = {"GLC Python": python, "GLC adapter": adapter, "source video": input_video,
                      "source stream": baseline_stream, "baseline encode summary": baseline_encode_path,
                      "baseline decode summary": baseline_decode_path, "GLC repository": glc_repo,
                      "weights directory": weights, "FFmpeg": ffmpeg}
    missing = [f"{label}: {path}" for label, path in required_files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing B4 dependency: " + "; ".join(missing))
    baseline_encode = json.loads(baseline_encode_path.read_text(encoding="utf-8"))
    baseline_decode = json.loads(baseline_decode_path.read_text(encoding="utf-8"))
    baseline_header, baseline_packets, baseline_integrity = read_stream(baseline_stream, with_integrity=True)
    if (baseline_integrity["container_version"] != 1 or baseline_encode["file_bytes"] != 98172 or
            baseline_header["width"] != 960 or baseline_header["height"] != 540 or
            baseline_header["fps"] != 25 or baseline_header["frames"] != 250 or
            baseline_header["q"] != 0 or not baseline_decode["all_frame_checksums_match"] or
            len(baseline_packets) != 250):
        raise ValueError("Existing 540p/25 baseline failed frozen metadata and decode checks.")
    recorded_source = Path(baseline_encode["input"]["input_path"])
    if recorded_source.resolve() != input_video.resolve():
        raise ValueError(f"B4 input is not the same file used by the existing baseline: {recorded_source}")
    baseline_source_dir = baseline_root / "sender" / "source"
    baseline_decoded_dir = baseline_root / "receiver" / "frames"
    if len(list(baseline_source_dir.glob("im*.png"))) != 250 or len(list(baseline_decoded_dir.glob("im*.png"))) != 250:
        raise ValueError("Baseline source/reconstruction PNG sequence is incomplete.")
    baseline_v1 = {
        "stream": str(baseline_stream), "file_bytes": baseline_stream.stat().st_size,
        "sha256": sha256(baseline_stream), "header": baseline_header,
        "decode_verified": baseline_decode["all_frame_checksums_match"],
        "encode_seconds": baseline_encode["pipeline_seconds_before_preview"],
        "decode_seconds": baseline_decode["pipeline_seconds_before_preview"],
        "input_mean_psnr_db": baseline_encode["mean_psnr_db"],
    }
    baseline_v1["transfer_estimate"] = transfer_estimate(baseline_v1["file_bytes"], baseline_v1["sha256"])
    input_digest = sha256(input_video)
    if args.resume and prior_status.get("input_sha256") != input_digest:
        raise ValueError("Cannot resume B4 against a different input video.")
    baseline_v2_dir = output / "container_v2_baseline_repack"
    baseline_v2_dir.mkdir(exist_ok=args.resume)
    baseline_v2_path = baseline_v2_dir / "baseline_v2.glcv"
    baseline_v2_report_path = baseline_v2_path.with_suffix(baseline_v2_path.suffix + ".repack.json")
    if args.resume and baseline_v2_path.is_file() and baseline_v2_report_path.is_file():
        baseline_v2_report = json.loads(baseline_v2_report_path.read_text(encoding="utf-8"))
        _, _, repack_integrity = read_stream(baseline_v2_path, with_integrity=True)
        if (baseline_v2_report.get("source_sha256") != baseline_v1["sha256"] or
                baseline_v2_report.get("output_sha256") != sha256(baseline_v2_path) or
                baseline_v2_report.get("output_bytes") != baseline_v2_path.stat().st_size or
                repack_integrity["container_version"] != 2):
            raise ValueError("Existing B4 v2 baseline repack failed resume validation.")
    else:
        baseline_v2_report = repack(baseline_stream, baseline_v2_path)
    status = prior_status or {"stage": "B4", "started_at": datetime.now().astimezone().isoformat(),
                              "input_video": str(input_video), "input_sha256": input_digest,
                              "baseline_v1": baseline_v1, "baseline_repack_v2": baseline_v2_report,
                              "cases": [], "gpu_experiment_started": True}
    status.update({"status": "running", "updated_at": datetime.now().astimezone().isoformat(),
                   "resume_count": status.get("resume_count", 0) + int(args.resume),
                   "recovery_note": ("Reused completed encoder/decoder artifacts after correcting B4 post-processing issues."
                                     if args.resume else None)})
    write_json(status_path, status)
    cases = [
        {"id": "fps_only_540p_12_5", "width": 0, "frames": 125, "frame_stride": 2,
         "label": "GLC v2 · 960×540 / 12.5 fps"},
        {"id": "resolution_only_360p_25", "width": 640, "frames": 250, "frame_stride": 1,
         "label": "GLC v2 · 640×360 / 25 fps"},
        {"id": "joint_360p_12_5", "width": 640, "frames": 125, "frame_stride": 2,
         "label": "GLC v2 · 640×360 / 12.5 fps"},
    ]
    completed = []
    prior_cases = {row["id"]: row for row in status.get("cases", [])}
    for specification in cases:
        case_id = specification["id"]
        case_dir = output / case_id
        if case_id in prior_cases:
            row = prior_cases[case_id]
            stream_path = Path(row["stream"])
            case_root = stream_path.parent.parent if stream_path.parent.name == "received" else stream_path.parent
            decoder_summary_path = case_root / "receiver" / "decode.json"
            decoder_summary = json.loads(decoder_summary_path.read_text(encoding="utf-8"))
            if (not stream_path.is_file() or row.get("sha256") != sha256(stream_path) or
                    not decoder_summary.get("all_frame_checksums_match") or
                    not decoder_summary.get("whole_file_checksum_verified")):
                raise ValueError(f"Previously accepted B4 case failed resume validation: {case_id}")
            completed.append(row)
            print(f"B4 case reused: {case_id}; verified existing stream and independent decode.", flush=True)
            continue
        existing_case = case_dir.exists()
        if existing_case and not args.resume:
            raise FileExistsError(f"Preserve existing case output: {case_dir}")
        case_dir.mkdir(exist_ok=args.resume)
        encoder_work = case_dir / "encoder_work"
        stream = case_dir / "stream.glcv"
        encoder_summary_path = encoder_work / "encode.json"
        if existing_case and encoder_summary_path.is_file() and stream.is_file():
            encoder_summary = json.loads(encoder_summary_path.read_text(encoding="utf-8"))
            encode_command_path = case_dir / "encode.command.json"
            encode_command = json.loads(encode_command_path.read_text(encoding="utf-8")) if encode_command_path.is_file() else None
            if encoder_summary.get("file_bytes") != stream.stat().st_size:
                raise ValueError(f"Existing B4 encoder summary and stream differ: {case_id}")
            print(f"B4 encoder artifacts reused: {case_id}.", flush=True)
        elif existing_case:
            raise FileExistsError(f"Existing case is incomplete; preserving it for review: {case_dir}")
        else:
            encoder = [str(python), "-B", str(adapter), "encode", "--input", str(input_video),
                       "--output", str(stream), "--workdir", str(encoder_work), "--repo", str(glc_repo),
                       "--weights", str(weights), "--ffmpeg", str(ffmpeg), "--device", "cuda",
                       "--frames", str(specification["frames"]), "--start-frame", "0",
                       "--frame-stride", str(specification["frame_stride"]), "--width", str(specification["width"]),
                       "--q", "0", "--intra-period", "64", "--reset-interval", "32",
                       "--container-version", "2"]
            encode_command = command_record(case_dir, "encode", encoder, REPO)
            encoder_summary = json.loads(encoder_summary_path.read_text(encoding="utf-8"))
        if (encoder_summary["container_version"] != 2 or encoder_summary["input"]["processed_frames"] != specification["frames"] or
                encoder_summary["input"]["width"] != (960 if specification["width"] == 0 else specification["width"]) or
                encoder_summary["input"]["height"] != (540 if specification["width"] == 0 else 360) or
                abs(encoder_summary["input"]["fps"] - 25 / specification["frame_stride"]) > 1e-9):
            raise ValueError(f"Encoded input metadata does not match the frozen B4 case: {case_id}")
        receive_record = None
        decode_input = stream
        if case_id == "joint_360p_12_5":
            decode_input = case_dir / "received" / "stream.glcv"
            receive_record = loopback_transfer(stream, decode_input, 100000)
            if not receive_record["verified"] or sha256(stream) != sha256(decode_input):
                raise ValueError("B4 v2 TCP loopback did not preserve the complete stream.")
        receiver_dir = case_dir / "receiver"
        decoder_summary_path = receiver_dir / "decode.json"
        if args.resume and decoder_summary_path.is_file():
            decoder_summary = json.loads(decoder_summary_path.read_text(encoding="utf-8"))
            decode_command_path = case_dir / "decode.command.json"
            decode_command = json.loads(decode_command_path.read_text(encoding="utf-8")) if decode_command_path.is_file() else None
            print(f"B4 decoder artifacts reused: {case_id}.", flush=True)
        else:
            decoder = [str(python), "-B", str(adapter), "decode", "--input", str(decode_input),
                       "--output", str(receiver_dir), "--repo", str(glc_repo), "--weights", str(weights),
                       "--ffmpeg", str(ffmpeg), "--device", "cuda"]
            decode_command = command_record(case_dir, "decode", decoder, REPO)
            decoder_summary = json.loads(decoder_summary_path.read_text(encoding="utf-8"))
        if (not decoder_summary["independent_decoder"] or not decoder_summary["encoder_modules_disabled"] or
                decoder_summary["source_media_reads"] or not decoder_summary["whole_file_checksum_verified"] or
                not decoder_summary["all_frame_checksums_match"] or
                decoder_summary["frame_checksum_verification"] != "ordered_aggregate" or
                len(decoder_summary["frames"]) != specification["frames"]):
            raise ValueError(f"Independent decoder integrity checks failed: {case_id}")
        for index, frame in enumerate(decoder_summary["frames"]):
            if not frame["sender_checksums_match"]:
                raise ValueError(f"B4 decoder checksum mismatch in {case_id}, frame {index + 1}")
        metrics = resolution_and_time_quality(baseline_source_dir, encoder_work / "source", receiver_dir / "frames",
                                              stride=specification["frame_stride"], reference_frames=250,
                                              width=encoder_summary["header"]["width"],
                                              height=encoder_summary["header"]["height"])
        file_bytes = decode_input.stat().st_size
        row = {
            **specification, "width": encoder_summary["header"]["width"],
            "height": encoder_summary["header"]["height"],
            "fps": encoder_summary["header"]["fps"], "frames": encoder_summary["header"]["frames"],
            "duration_seconds": encoder_summary["input"]["duration_seconds"],
            "container_version": decoder_summary["container_version"],
            "stream": str(decode_input), "file_bytes": file_bytes, "sha256": sha256(decode_input),
            "actual_bpp": file_bytes * 8 / (encoder_summary["header"]["width"] *
                                                encoder_summary["header"]["height"] *
                                                encoder_summary["header"]["frames"]),
            "container_overhead_bytes": encoder_summary["container_overhead_bytes"],
            "container_overhead_breakdown": encoder_summary["container_overhead_breakdown"],
            "input_mean_psnr_db": encoder_summary["mean_psnr_db"],
            "timeline_quality": metrics,
            "decode_verified": decoder_summary["all_frame_checksums_match"],
            "frame_checksum_verification": decoder_summary["frame_checksum_verification"],
            "source_media_reads": decoder_summary["source_media_reads"],
            "encode_seconds": encoder_summary["pipeline_seconds_before_preview"],
            "decode_seconds": decoder_summary["pipeline_seconds_before_preview"],
            "encode_peak_allocated_mib": encoder_summary["peak_allocated_mib"],
            "decode_peak_allocated_mib": decoder_summary["peak_allocated_mib"],
            "received_over_tcp_loopback": receive_record,
            "commands": {"encode": encode_command, "decode": decode_command},
        }
        row["transfer_estimate"] = transfer_estimate(file_bytes, row["sha256"])
        row["meets_20_second_target"] = {
            rate: row["transfer_estimate"]["ideal_seconds_by_kbps"][str(rate)] <= 20
            for rate in (20, 30, 40)
        }
        completed.append(row)
        status["cases"] = completed
        write_json(output / "pilot_status.json", status)
        print(f"B4 case complete: {case_id}, {file_bytes} B, "
              f"{row['transfer_estimate']['ideal_seconds_by_kbps']}", flush=True)
    preview_dir = output / "comparisons"
    preview_dir.mkdir(exist_ok=args.resume)
    cases_by_id = {row["id"]: row for row in completed}
    previews = []
    for case_id in ("fps_only_540p_12_5", "resolution_only_360p_25", "joint_360p_12_5"):
        candidate = cases_by_id[case_id]
        stream_parent = Path(candidate["stream"]).parent
        candidate_frames = (stream_parent.parent / "receiver" / "frames"
                            if stream_parent.name == "received"
                            else stream_parent / "receiver" / "frames")
        previews.append(create_comparison(
            ffmpeg, baseline_source_dir, baseline_decoded_dir,
            candidate_frames,
            preview_dir / f"{case_id}_comparison.mp4", candidate["label"], candidate["frame_stride"]))
    status.update({"status": "complete", "finished_at": datetime.now().astimezone().isoformat(),
                   "cases": completed, "comparison_previews": previews,
                   "experiment_limits": [
                       "same previously used development clip; not an independent evaluation set",
                       "q0, intra_period=64, reset_interval=32 held constant across new encodes",
                       "25fps source and 960x540 reference; no sound encoded",
                       "20/30/40 kbps timings are theoretical application-layer estimates, not physical acoustic measurements",
                       "the v2 header adds no entropy information and its neural payload is byte-preserved when repacking",
                       "the v2 sequence checksums are ordered SHA256 digests; independent decode passed on the combined-reduction case after TCP reception",
                   ]})
    write_json(output / "pilot_status.json", status)
    return status


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="Resume and validate completed artifacts in an interrupted pilot output.")
    args = parser.parse_args()
    output_existed = args.output.resolve().exists()
    try:
        result = run(args)
    except Exception as exc:
        status_path = args.output.resolve() / "pilot_status.json"
        if (not output_existed or args.resume) and status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                status.update({"status": "failed", "failed_at": datetime.now().astimezone().isoformat(),
                               "failure": {"type": type(exc).__name__, "message": str(exc)}})
                write_json(status_path, status)
            except Exception:
                pass
        raise
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                      "cases": [{"id": case["id"], "bytes": case["file_bytes"],
                                 "seconds": case["transfer_estimate"]["ideal_seconds_by_kbps"],
                                 "timeline_psnr": case["timeline_quality"]["mean_psnr_db_vs_960x540_25fps_timeline_after_lanczos_upscale"]}
                                for case in result["cases"]],
                      "comparison_previews": result["comparison_previews"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
