"""Compression statistics with explicit file and sampled-RGB baselines."""
from fractions import Fraction
import math


def compression_metrics(header, file_bytes, encoded=None):
    width, height, frames = (header[key] for key in ("width", "height", "frames"))
    fps = float(Fraction(str(header["fps"])))
    duration = frames / fps
    raw_bytes = width * height * 3 * frames
    result = {"width": width, "height": height, "frames": frames, "fps": fps,
              "duration_seconds": duration, "file_bytes": file_bytes,
              "video_kbps": file_bytes * 8 / duration / 1000,
              "raw_rgb24_bytes": raw_bytes, "raw_rgb24_ratio": raw_bytes / file_bytes,
              "source_file_bytes": None, "source_file_ratio": None,
              "source_file_reduction_percent": None, "whole_source_processed": None,
              "mean_psnr_db": None}
    if not encoded:
        return result
    source = encoded["input"]
    result["source_file_bytes"] = source["input_bytes"]
    result["source_file_ratio"] = source["input_bytes"] / file_bytes
    result["source_file_reduction_percent"] = (1 - file_bytes / source["input_bytes"]) * 100
    psnr = encoded.get("mean_psnr_db")
    if isinstance(psnr, (int, float)) and math.isfinite(psnr):
        result["mean_psnr_db"] = psnr
    plan = source.get("sampling_plan")
    if plan:
        source_duration = float(Fraction(plan["probe"]["duration"]))
        whole = abs(plan["start_seconds"]) < 1e-9 and plan["end_seconds"] >= source_duration - 1e-9
        result.update(whole_source_processed=whole,
                      selected_start_seconds=plan["start_seconds"],
                      selected_end_seconds=plan["end_seconds"],
                      source_duration_seconds=source_duration)
    return result


def timing_metrics(result, encode=None, decode=None):
    """Report recorded wall-clock endpoints; model acquisition is already included."""
    def seconds(value):
        return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None

    def mark(record, name):
        return seconds((record or {}).get("task_seconds", {}).get(name))

    sender = result.get("role") == "sender"
    remote = result.get("remote") or {}
    encoded = result.get("encoded_this_task") is True
    encode_seconds = mark(encode, "bitstream_closed_seconds") if encoded else (0 if sender else None)
    decode_seconds = mark(decode or (remote.get("timing") if sender else result.get("timing")),
                          "decode_verified_seconds")
    if sender and decode_seconds is None:
        decode_seconds = seconds(remote.get("durations", {}).get("decode_seconds"))
    prep = result.get("preprocessing")
    preprocess_seconds = seconds(prep.get("seconds")) if prep else (0 if sender else None)
    transfer = result.get("transfer") or {}
    return {
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "codec_total_seconds": encode_seconds + decode_seconds if encode_seconds is not None and decode_seconds is not None else None,
        "preprocess_seconds": preprocess_seconds,
        "transfer_seconds": seconds(transfer.get("elapsed_seconds" if sender else "payload_receive_seconds")),
        "task_total_seconds": seconds(result.get("service_seconds")),
        "encoded_this_task": encoded if sender else None,
        "codec_includes_model_acquisition": True,
        "codec_excludes_preprocessing_and_transfer": True,
    }
