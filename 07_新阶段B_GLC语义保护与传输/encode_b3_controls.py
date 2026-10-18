"""Frozen byte-only searches; retain every trial and use existing codecs."""
from __future__ import annotations
import argparse
import copy
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from b2_common import HERE, ROOT, Journal, git, read, require, sha, write
from run_b0_b1 import child_env


def closest(rows, target, key):
    return min(rows, key=lambda r: (abs(r[key] - target), r[key]))


def next_integer(rows, target, parameter, size, low, high):
    """Use only observed brackets; increasing size for q, decreasing for CRF."""
    ordered = sorted(rows, key=lambda r: r[parameter])
    for a, b in zip(ordered, ordered[1:]):
        if (a[size] - target) * (b[size] - target) < 0:
            middle = (a[parameter] + b[parameter]) // 2
            if a[parameter] < middle < b[parameter]:
                return middle
    return None


def run_logged(journal, argv, logfile):
    row = journal.begin(argv, stdout=str(logfile), stderr=str(logfile))
    tick = time.perf_counter()
    with Path(logfile).open("x", encoding="utf-8") as handle:
        process = subprocess.Popen(list(map(str, argv)), stdout=handle, stderr=subprocess.STDOUT, env=child_env())
        row["pid"] = process.pid
        write(journal.path, journal.entries)
        code = process.wait()
    journal.finish(row, code, time.perf_counter() - tick)
    require(code == 0, f"Codec failed with {code}; see {logfile}")


def verify_source(directory, audit):
    import numpy as np
    from PIL import Image
    paths = sorted(directory.glob("*.png"))
    require(len(paths) == audit["frames"], "Extracted frame count mismatch")
    for i, path in enumerate(paths):
        require(path.name == f"im{i+1:05d}.png", "Extracted ordering mismatch")
        with Image.open(path) as image:
            require(image.size == (audit["width"], audit["height"]), "Source geometry mismatch")
            require(hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest() == audit["mapping"][i]["rgb_sha256"], "Lossless input RGB differs from B2")
    return {"frames": len(paths), "all_source_RGB_match_B2": True}


def glc_search(config, baseline, output, journal):
    deps = baseline["runtime_dependencies"]
    source = baseline["source"]
    adapter = ROOT / baseline["adapter"]["repository_path"]
    require(sha(adapter) == baseline["adapter"]["sha256"], "GLC adapter changed")
    base = Path(baseline["baseline_stream"]["path"])
    require(sha(base) == baseline["baseline_stream"]["sha256"], "Base changed")
    target = config["target_file_bytes"]
    rows = [{"q": 0, "file_bytes": base.stat().st_size, "packet": str(base), "sha256": sha(base), "reused_A_q0": True}]
    write(output / "input_identity.json", {"path": config["lossless_input"], "sha256": sha(config["lossless_input"]), "source_manifest_sha256": baseline["source"]["a_inputs_manifest_sha256"]})
    audit = read(Path(config["b2_run"]) / "sender/source_audit.json")
    q = config["glc_search"]["first_q"]
    reason = "trial_limit"
    for trial in range(config["glc_search"]["max_new_trials"]):
        directory = output / f"q{q:02d}"
        directory.mkdir()
        packet = directory / "stream.glcv"
        argv = [deps["glc_python"], "-B", adapter, "encode", "--input", config["lossless_input"], "--output", packet,
                "--workdir", directory / "sender", "--frames", source["frames"], "--start-frame", 0, "--width", source["width"],
                "--q", q, "--intra-period", baseline["codec"]["intra_period"], "--reset-interval", baseline["codec"]["reset_interval"],
                "--repo", deps["glc_repo"], "--weights", deps["weights"], "--ffmpeg", deps["ffmpeg"], "--device", "cuda"]
        print(f"GLC byte search: q={q}, full {source['frames']} frames", flush=True)
        run_logged(journal, list(map(str, argv)), directory / "encode.log")
        verified = verify_source(directory / "sender/source", audit)
        write(directory / "source_verification.json", verified)
        size = packet.stat().st_size
        rows.append({"q": q, "file_bytes": size, "packet": str(packet), "sha256": sha(packet), "source_verified": verified})
        write(output / "trials.json", rows)
        print(f"GLC q={q}: {size} bytes, difference={(size/target-1):.4%}", flush=True)
        if abs(size/target-1) <= config["budget_tolerance_fraction"]:
            reason = "first_within_tolerance"
            break
        q = next_integer(rows, target, "q", "file_bytes", 0, 63)
        if q is None:
            reason = "no_unmeasured_integer_in_observed_bracket"
            break
    chosen = closest(rows, target, "file_bytes")
    write(output / "selection.json", {"method": "full_frame_GLC", "selected": chosen, "target_file_bytes": target,
          "relative_difference": chosen["file_bytes"]/target-1, "matched": abs(chosen["file_bytes"]/target-1) <= config["budget_tolerance_fraction"],
          "stop_reason": reason, "selection_uses_quality_metrics": False, "trials": rows,
          "unmatched_interpretation": "bounded search only; no proof that all legal q values are unreachable"})


def geometry_search(config, baseline, output, journal):
    os.environ["FFMPEG_PATH"] = baseline["runtime_dependencies"]["ffmpeg"]
    sys.path.insert(0, str(ROOT / "01_主工程/semantic_mvp"))
    import runtime
    import glc_layered
    runtime.run = journal.run
    b2 = Path(config["b2_run"]) / "sender"
    original = read(b2 / "manifest.json")
    require(sha(b2 / "message.scv") == "c2ba3c7e6a22aab81d752bd7c5da909eacc03a5cfbdd3b657f16b1666b1fa049", "B2 packet changed")
    n, segment = original["frames"], config["geometry"]["segment_frames"]
    planned = []
    for start in range(0, n, segment):
        for part, box in zip(("left", "right"), config["geometry"]["boxes"]):
            planned.append({"file": f"roi_{len(planned):04d}.mp4", "start": start, "frames": min(segment, n-start), "box": box, "partition": part})
    write(output / "frozen_geometry.json", {"patches": planned, "uses_predictions_for_selection": False})
    target = config["target_supplement_bytes"]
    scale = round(1 / config["geometry"]["step"])
    rows, units = [], round(config["geometry"]["first_crf"]*scale)
    reason = "trial_limit"
    for trial in range(config["geometry"]["max_trials"]):
        crf = units/scale
        directory = output / f"crf{crf:06.3f}"
        streams = directory / "sender_streams"
        streams.mkdir(parents=True)
        for name in ("base.glcv", "semantics.json.gz"):
            shutil.copyfile(b2 / "sender_streams" / name, streams / name)
        manifest = copy.deepcopy(original)
        manifest.pop("files")
        manifest["patches"] = planned
        manifest["configuration"]["roi_crf"] = crf
        manifest["configuration"]["selection"] = "fixed_center_half_height_band; predictions transmitted but not used for geometry"
        print(f"Geometry byte search: CRF={crf}, {len(planned)} fixed patches", flush=True)
        for patch in planned:
            runtime.encode_h265(b2 / "private_frames", streams / patch["file"], original["fps"], crf,
                                bbox=patch["box"], start=patch["start"], count=patch["frames"])
        packet = directory / "message.scv"
        ledger = glc_layered.write_packet(streams, manifest, packet)
        write(directory / "manifest.json", manifest)
        write(directory / "encode.json", {"status": "B3_fixed_geometry_sender_complete", "byte_ledger": ledger, "packet_sha256": sha(packet)})
        row = {"crf_search_units": units, "crf_step": config["geometry"]["step"], "crf": crf, "file_bytes": packet.stat().st_size, "supplement_bytes": packet.stat().st_size-config["base_bytes"],
               "packet": str(packet), "sha256": sha(packet), "byte_ledger": ledger}
        rows.append(row)
        write(output / "trials.json", rows)
        print(f"Geometry CRF={crf}: {row['supplement_bytes']} supplement bytes, difference={(row['supplement_bytes']/target-1):.4%}", flush=True)
        if abs(row["supplement_bytes"]/target-1) <= config["budget_tolerance_fraction"]:
            reason = "first_within_tolerance"
            break
        if len(rows) == 1:
            units = round((config["geometry"]["max_crf"] if row["supplement_bytes"] > target else config["geometry"]["min_crf"])*scale)
        else:
            units = next_integer(rows, target, "crf_search_units", "supplement_bytes", 0, round(51*scale))
            if units is None:
                reason = "no_unmeasured_quarter_CRF_in_observed_bracket"
                break
    chosen = closest(rows, target, "supplement_bytes")
    write(output / "selection.json", {"method": "fixed_geometry", "selected": chosen, "target_supplement_bytes": target,
          "relative_supplement_difference": chosen["supplement_bytes"]/target-1,
          "relative_complete_file_difference": chosen["file_bytes"]/config["target_file_bytes"]-1,
          "matched": abs(chosen["supplement_bytes"]/target-1) <= config["budget_tolerance_fraction"],
          "stop_reason": reason, "selection_uses_quality_metrics": False, "trials": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["glc", "geometry"])
    parser.add_argument("--config", type=Path, default=HERE / "configs/b3_video91.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    baseline = read(args.config.parent / config["baseline_config"])
    require(not git("status", "--porcelain"), "Commit executed code before B3 encoding")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "config.json", {"b3": config, "baseline": baseline, "code_commit": git("rev-parse", "HEAD"), "script_sha256": sha(__file__)})
    journal = Journal(args.output)
    invocation = journal.begin([sys.executable, *sys.argv], code_commit=git("rev-parse", "HEAD"), pid=os.getpid())
    tick = time.perf_counter()
    try:
        (glc_search if args.method == "glc" else geometry_search)(config, baseline, args.output, journal)
        journal.finish(invocation, 0, time.perf_counter()-tick)
    except BaseException as exc:
        write(args.output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        journal.finish(invocation, 1, time.perf_counter()-tick)
        raise


if __name__ == "__main__":
    main()
