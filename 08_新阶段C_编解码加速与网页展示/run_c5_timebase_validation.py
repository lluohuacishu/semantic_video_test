"""Bounded C5-1 CUDA integration: legacy gate, main 10s, NOAA 10s and full."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import sys
import time

from c3_catalog import CATALOG, WORKSPACE, sha256
from c3_server import verify_result
from c4_config import load_settings, HERE, REPO
from c2_watch import supervise, inspect, utc, write
from c5_preflight import command


def run(root, runtime):
    root.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(CATALOG.read_text("utf-8"))
    main = WORKSPACE / manifest["assets"]["source.main"]["workspace_relative_path"]
    noaa = WORKSPACE / "02_实验数据/C_编解码加速与网页展示/C4_validation_20261020_r1/source/noaa_ex1202_octopus_original.mp4"
    reference = WORKSPACE / "02_实验数据/C_编解码加速与网页展示/C3_web_jobs/bf663b75ef624fdcbe829c31f33238a6/received.glcv"
    cases = [("legacy8", main, {"frames": 8}, 8),
             ("main10", main, {"sampling_mode": "time", "clip": {"mode": "first10"}}, 125),
             ("noaa10", noaa, {"sampling_mode": "time", "clip": {"mode": "first10"}}, 125),
             ("noaa_full", noaa, {"sampling_mode": "time", "clip": {"mode": "full"}}, None)]
    summary = {"started_utc": utc(), "status": "running", "cross_device_verified": False,
               "code_sha256": {name: sha256(HERE / name) for name in (
                   "c1_runtime.py", "c1_io.py", "c2_worker.py", "c5_timebase.py", "c5_compat.py")},
               "cases": []}
    write(root / "summary.json", summary)
    try:
        for name, source, options, frames in cases:
            folder = root / name
            folder.mkdir()
            started = time.monotonic()
            for role in ("encode", "decode"):
                task = {"id": name + "_" + role, "preset": "compact", "backend": "native",
                        "output_mode": "display", "profile": False,
                        "input": str(source if role == "encode" else folder / "received.glcv"),
                        "output": str(folder / role), **options}
                config = {k: str(v) for k, v in runtime.items() if k != "python"}
                config.update(role=role, device="cuda", process_folder=str(folder / (role + "_process")),
                              stop_file=str(root / "STOP_REQUESTED"), tasks=[task])
                config_path = folder / (role + "_config.json")
                write(config_path, config)
                outcome = supervise([str(runtime["python"]), "-B", str(HERE / "c2_worker.py"),
                                     "--config", str(config_path)], folder / (role + "_process"),
                                     REPO, root / "STOP_REQUESTED", poll_seconds=0.2)
                if outcome["returncode"] != 0 or inspect(folder / (role + "_process"))["observed_state"] != "completed":
                    raise RuntimeError(f"{name}/{role} failed; preserve its worker log")
                if role == "encode":
                    shutil.copyfile(folder / "encode/stream.glcv", folder / "received.glcv")
            enc, dec = verify_result(folder)
            if frames is not None and len(enc["frames"]) != frames:
                raise ValueError("Wrong output frame count")
            if name == "legacy8" and sha256(reference) != enc["file_sha256"]:
                raise ValueError("Legacy 8-frame stream changed from C4 reference")
            if name == "noaa_full" and (len(enc["frames"]) <= 125 or enc["input"]["duration_seconds"] < 27):
                raise ValueError("Full mode did not cover the entire NOAA video")
            previews = {}
            for role, path in (("source", folder / "encode/encoder/source_preview.mp4"),
                               ("decoded", folder / "decode/receiver/reconstruction_preview.mp4")):
                probe = json.loads(command([runtime["ffmpeg"].with_name("ffprobe.exe"), "-v", "error",
                    "-count_frames", "-select_streams", "v:0", "-show_entries",
                    "stream=width,height,nb_read_frames,avg_frame_rate,duration", "-of", "json", path]))["streams"][0]
                if int(probe["nb_read_frames"]) != len(enc["frames"]) or Fraction(probe["avg_frame_rate"]) != Fraction("12.5"):
                    raise ValueError("Preview timeline mismatch")
                previews[role] = probe
            row = {"id": name, "frames": len(enc["frames"]), "source_sha256": sha256(source),
                   "output_seconds": enc["input"]["duration_seconds"], "file_bytes": enc["file_bytes"],
                   "stream_sha256": enc["file_sha256"], "rgb_latent_equal": True,
                   "source_media_reads": dec["source_media_reads"], "previews": previews,
                   "wall_seconds": time.monotonic() - started,
                   "network_transfer_performed": False, "legacy_reference_equal": name == "legacy8"}
            summary["cases"].append(row)
            write(root / "summary.json", summary)
            print("C5_TIMEBASE_DONE " + json.dumps(row, ensure_ascii=False), flush=True)
        summary.update(status="passed", completed_utc=utc())
    except BaseException as exc:
        summary.update(status="failed", error=str(exc), failed_utc=utc())
        raise
    finally:
        write(root / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    run(args.output, load_settings(args.config)["runtime"])
