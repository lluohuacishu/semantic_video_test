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
        if whole:
            result["source_file_ratio"] = source["input_bytes"] / file_bytes
            result["source_file_reduction_percent"] = (1 - file_bytes / source["input_bytes"]) * 100
    return result
