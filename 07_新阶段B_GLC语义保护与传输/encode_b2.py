"""B2 sender: frozen source predictions and existing H.265 ROI encoding."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time
from b2_common import HERE, ROOT, Journal, git, read, require, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "configs/b2_video91.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    baseline = read(args.config.parent / config["baseline_config"])
    frozen = read(args.config.parent / config["baseline_evidence"])
    require(Path(sys.executable).resolve() == Path(baseline["runtime_dependencies"]["segmentation_python"]).resolve(), "Use frozen segmentation Python")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "config.json", {"b2": config, "baseline": baseline})
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", FFMPEG_PATH=baseline["runtime_dependencies"]["ffmpeg"])
    sys.path.insert(0, str(ROOT / "01_主工程/semantic_mvp"))
    import numpy as np
    from PIL import Image
    import runtime
    from teacher import Teacher, regions
    from roi_layout import plan_halves
    import glc_layered
    journal = Journal(output)
    invocation = journal.begin([sys.executable, *sys.argv], cwd=str(Path.cwd()), code_commit=git("rev-parse", "HEAD"), pid=os.getpid())
    runtime.run = journal.run
    tick = time.perf_counter()
    try:
        require(config["roi"] == baseline["semantic_plan"]["roi"], "Frozen ROI parameters changed")
        require(sha(baseline["baseline_stream"]["path"]) == baseline["baseline_stream"]["sha256"], "Baseline changed")
        inputs_path = Path(baseline["source"]["a_inputs_manifest"])
        require(sha(inputs_path) == baseline["source"]["a_inputs_manifest_sha256"], "Input manifest changed")
        inputs = read(inputs_path)
        source = baseline["source"]
        n, w, h = source["frames"], source["width"], source["height"]
        require(len(inputs["records"]) == n, "Input count differs")
        frames = output / "private_frames"
        frames.mkdir()
        mapping = []
        for index, record in enumerate(inputs["records"]):
            original = inputs_path.parent / "source" / f"im{index + 1:05d}.png"
            require(sha(original) == record["png_sha256"], "Source PNG changed")
            with Image.open(original) as image:
                require(image.size == (w, h) and hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest() == record["rgb_sha256"], "Source RGB changed")
            destination = frames / f"{index:05d}.png"
            shutil.copyfile(original, destination)
            mapping.append({"frame": index, "source_frame": record["source_frame"], "png_sha256": record["png_sha256"], "rgb_sha256": record["rgb_sha256"]})
        write(output / "source_audit.json", {"role": source["role"], "source_manifest": str(inputs_path), "width": w, "height": h,
              "frames": n, "fps": source["fps"], "mapping": mapping, "ground_truth_read": False})
        code = [Path(__file__), HERE / "b2_common.py", ROOT / "01_主工程/semantic_mvp/glc_layered.py",
                ROOT / "01_主工程/semantic_mvp/runtime.py", ROOT / "01_主工程/semantic_mvp/teacher.py", ROOT / "01_主工程/semantic_mvp/roi_layout.py",
                ROOT / "01_主工程/adapters/common.py", ROOT / "01_主工程/adapters/semantic_teacher.py"]
        write(output / "code_hashes.json", {str(p): sha(p) for p in code})
        records = [{"frame": i, "targets": {}} for i in range(n)]
        profiles = {}
        mask_root = output / "private_masks"
        for target in config["targets"]:
            teacher = Teacher(target)
            profile = baseline["semantic_plan"]["target_profiles"][target]
            require(teacher.profile["model_sha256"] == profile["model_sha256"] and teacher.threshold == profile["threshold"], "Teacher differs from B0")
            profiles[target] = teacher.profile
            for kind in ("raw", "kept"):
                (mask_root / target / kind).mkdir(parents=True)
            raw_pixels = []
            for index in range(n):
                raw, confidence = teacher.mask(frames / f"{index:05d}.png")
                kept, components = regions(raw, confidence, config["roi"]["min_pixels"])
                for kind, mask in (("raw", raw), ("kept", kept)):
                    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_root / target / kind / f"{index:05d}.png")
                records[index]["targets"][target] = {"pixels": int(kept.sum()), "regions": components}
                raw_pixels.append(int(raw.sum()))
                if (index + 1) % 25 == 0:
                    print(f"B2 {target} predictions {index + 1}/{n}", flush=True)
                    write(output / "progress.json", {"phase": "source_predictions", "target": target, "completed_frames": index + 1})
            write(output / f"{target}_prediction_summary.json", {"profile": teacher.profile, "raw_mask_pixels": raw_pixels,
                  "kept_mask_pixels": [r["targets"][target]["pixels"] for r in records], "ground_truth": None})
            del teacher
        merged = [{"regions": [region for value in r["targets"].values() for region in value["regions"]]} for r in records]
        planned = plan_halves(merged, w, h, config["roi"]["segment_frames"], config["roi"]["padding"])
        patches = [{"file": f"roi_{index:04d}.mp4", **patch} for index, patch in enumerate(planned)]
        write(output / "roi_plan.json", patches)
        streams = output / "sender_streams"
        streams.mkdir()
        shutil.copyfile(baseline["baseline_stream"]["path"], streams / "base.glcv")
        for index, patch in enumerate(patches):
            runtime.encode_h265(frames, streams / patch["file"], runtime.fps_string(source["fps"]), config["roi_crf"],
                                bbox=patch["box"], start=patch["start"], count=patch["frames"])
            print(f"B2 ROI encoded {index + 1}/{len(patches)}: {patch['box']}", flush=True)
        semantics = {"meaning": "sender predictions; components are not verified object counts", "frames": records}
        (streams / "semantics.json.gz").write_bytes(gzip.compress(glc_layered.compact(semantics), mtime=0))
        manifest = {"profile": glc_layered.PROFILE, "version": 1, "mode": "video", "width": w, "height": h, "frames": n,
                    "fps": runtime.fps_string(source["fps"]), "base": {"file": "base.glcv", "codec": "glcv-lab-f32-v1"},
                    "patches": patches, "targets": profiles, "semantics_file": "semantics.json.gz", "roi_layout": "halves",
                    "compositor": config["compositor"], "configuration": {"roi": config["roi"], "roi_crf": config["roi_crf"],
                    "h265": "existing runtime.encode_h265: medium, zerolatency, P-only, keyint32, BT709 limited YUV420"}}
        packet = output / "message.scv"
        ledger = glc_layered.write_packet(streams, manifest, packet)
        write(output / "manifest.json", manifest)
        write(output / "encode.json", {"status": "B2_sender_complete", "packet": str(packet), "packet_sha256": sha(packet),
              "packet_bytes": packet.stat().st_size, "byte_ledger": ledger, "base_component_ledger": frozen["byte_ledger"],
              "frames": n, "patches": len(patches), "source_predictions": {target: {"present_frames": sum(r["targets"][target]["pixels"] > 0 for r in records),
              "component_instances": sum(len(r["targets"][target]["regions"]) for r in records)} for target in profiles},
              "average_patch_rectangle_fraction": sum(p["box"][2] * p["box"][3] * p["frames"] for p in patches) / (w * h * n),
              "elapsed_seconds": time.perf_counter() - tick, "semantic_benefit": "not_evaluated", "ground_truth_used": False})
        journal.finish(invocation, 0, time.perf_counter() - tick)
        print(f"B2 sender complete: {packet.stat().st_size} bytes, {len(patches)} ROI patches", flush=True)
    except BaseException as exc:
        write(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        journal.finish(invocation, 1, time.perf_counter() - tick)
        raise


if __name__ == "__main__":
    main()
