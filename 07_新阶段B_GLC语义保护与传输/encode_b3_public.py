"""Encode frozen public candidate clips with the unchanged B2 semantic pipeline."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from b2_common import ROOT, git, read, require, sha, write


def run(argv, log):
    with Path(log).open("x", encoding="utf-8") as handle:
        process = subprocess.Popen(list(map(str, argv)), stdout=handle, stderr=subprocess.STDOUT,
                                   env=os.environ.copy())
        code = process.wait()
    require(code == 0, f"Command failed with {code}: {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, default=Path(__file__).parent / "configs/b0_baseline.json")
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--q", type=int, default=0)
    parser.add_argument("--intra-period", type=int, default=64)
    parser.add_argument("--reset-interval", type=int, default=32)
    parser.add_argument("--roi-crf", type=float, default=32)
    args = parser.parse_args()
    baseline = read(args.baseline_config)
    require(not git("status", "--porcelain"), "Commit executed code before public candidate encoding")
    require(args.input.is_file() and 1 <= args.frames <= 10000, "Invalid input or frame count")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    deps = baseline["runtime_dependencies"]
    adapter = (ROOT / baseline["adapter"]["repository_path"]).resolve()
    require(sha(adapter) == baseline["adapter"]["sha256"], "GLC adapter changed")
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", FFMPEG_PATH=deps["ffmpeg"])
    sys.path.insert(0, str(ROOT / "01_主工程/semantic_mvp"))
    import numpy as np
    from PIL import Image
    import glc_layered
    import runtime
    from roi_layout import plan_halves
    from teacher import Teacher, regions

    glc = output / "glc"
    glc.mkdir()
    argv = [deps["glc_python"], "-B", adapter, "encode", "--input", args.input,
            "--output", glc / "stream.glcv", "--workdir", glc / "sender",
            "--frames", args.frames, "--start-frame", 0, "--width", 1280,
            "--q", args.q, "--intra-period", args.intra_period, "--reset-interval", args.reset_interval,
            "--repo", deps["glc_repo"], "--weights", deps["weights"], "--ffmpeg", deps["ffmpeg"], "--device", "cuda"]
    start = time.perf_counter()
    run(argv, output / "glc_encode.log")
    source_dir = glc / "source"
    require(source_dir.is_dir(), "GLC source frames missing")
    source_info = read(glc / "encode.json")["source"]
    require(source_info["processed_frames"] == args.frames and [source_info["width"], source_info["height"]] == [1280, 720], "GLC source geometry/count mismatch")
    fps = runtime.fps_string(source_info["fps"])
    frames = output / "private_frames"
    frames.mkdir()
    mapping = []
    for index in range(args.frames):
        source = source_dir / f"im{index + 1:05d}.png"
        destination = frames / f"{index:05d}.png"
        shutil.copyfile(source, destination)
        with Image.open(source) as image:
            rgb = np.asarray(image.convert("RGB"))
            mapping.append({"frame": index, "png_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                            "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
    write(output / "source_audit.json", {"role": "public_source_candidate_contiguous_window", "input": str(args.input.resolve()),
          "input_sha256": sha(args.input), "width": 1280, "height": 720, "fps": source_info["fps"],
          "frames": args.frames, "mapping": mapping, "ground_truth_read": False})

    records = [{"frame": i, "targets": {}} for i in range(args.frames)]
    profiles = {}
    mask_root = output / "private_masks"
    for target in ("fish", "diver"):
        teacher = Teacher(target)
        profiles[target] = teacher.profile
        for kind in ("raw", "kept"):
            (mask_root / target / kind).mkdir(parents=True)
        raw_pixels = []
        for index in range(args.frames):
            raw, confidence = teacher.mask(frames / f"{index:05d}.png")
            kept, components = regions(raw, confidence, 64)
            for kind, mask in (("raw", raw), ("kept", kept)):
                Image.fromarray(mask.astype(np.uint8) * 255).save(mask_root / target / kind / f"{index:05d}.png")
            records[index]["targets"][target] = {"pixels": int(kept.sum()), "regions": components}
            raw_pixels.append(int(raw.sum()))
        write(output / f"{target}_prediction_summary.json", {"profile": teacher.profile,
              "raw_mask_pixels": raw_pixels, "kept_mask_pixels": [r["targets"][target]["pixels"] for r in records],
              "ground_truth": None})
        del teacher
    merged = [{"regions": [region for value in row["targets"].values() for region in value["regions"]]} for row in records]
    patches = [{"file": f"roi_{i:04d}.mp4", **patch} for i, patch in enumerate(plan_halves(merged, 1280, 720, 32, 16))]
    streams = output / "sender_streams"
    streams.mkdir()
    shutil.copyfile(glc / "stream.glcv", streams / "base.glcv")
    for patch in patches:
        runtime.encode_h265(frames, streams / patch["file"], fps, args.roi_crf,
                            bbox=patch["box"], start=patch["start"], count=patch["frames"])
    semantics = {"meaning": "sender predictions; components are not verified object counts", "frames": records}
    (streams / "semantics.json.gz").write_bytes(gzip.compress(glc_layered.compact(semantics), mtime=0))
    manifest = {"profile": glc_layered.PROFILE, "version": 1, "mode": "video", "width": 1280, "height": 720,
                "frames": args.frames, "fps": fps, "base": {"file": "base.glcv", "codec": "glcv-lab-f32-v1"},
                "patches": patches, "targets": profiles, "semantics_file": "semantics.json.gz", "roi_layout": "halves",
                "compositor": "hard_replace_disjoint_rectangles_v1", "configuration": {"roi": {"layout": "halves",
                "segment_frames": 32, "padding": 16, "min_pixels": 64}, "roi_crf": args.roi_crf,
                "h265": "existing runtime.encode_h265; unchanged B2 parameters", "input_role": "public_candidate_diagnostic"}}
    packet = output / "message.scv"
    ledger = glc_layered.write_packet(streams, manifest, packet)
    write(output / "manifest.json", manifest)
    write(output / "encode.json", {"status": "B3_public_candidate_sender_complete", "packet": str(packet),
          "packet_sha256": sha(packet), "packet_bytes": packet.stat().st_size, "byte_ledger": ledger,
          "frames": args.frames, "fps": fps, "patches": len(patches), "source_predictions": {
          target: {"present_frames": sum(row["targets"][target]["pixels"] > 0 for row in records),
                   "component_instances": sum(len(row["targets"][target]["regions"]) for row in records)} for target in profiles},
          "elapsed_seconds": time.perf_counter() - start, "semantic_benefit": "not_evaluated",
          "ground_truth_used": False, "source_exposure_verified": False})
    print(f"Encoded {args.input.name}: {args.frames} frames, {packet.stat().st_size} packet bytes, {len(patches)} patches", flush=True)


if __name__ == "__main__":
    main()
