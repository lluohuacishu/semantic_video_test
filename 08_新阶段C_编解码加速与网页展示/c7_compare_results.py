"""Compare two saved C7 probe runs offline. NumPy only; no torch or inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

from c7_diagnostics import HASH_KEYS, sha256, write_json

STATE_KEYS = ("ref_latent", "ref_y")
TRACE_KEYS = ("compressed_sha256", "cdf_indexes_sha256", "symbols_sha256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def aggregate(records):
    """Mirror GLCVLAB2 ordered digests without importing a model installation."""
    result = {}
    for key in HASH_KEYS:
        digest = hashlib.sha256(b"GLCVLAB2/" + key.encode("ascii") + b"/ordered-frame-sha256/v1\0")
        for index, record in enumerate(records):
            require(valid_hash(record.get(key)), "Invalid frame checksum: " + key)
            digest.update(struct.pack(">I", index))
            digest.update(bytes.fromhex(record[key]))
        result[key] = digest.hexdigest()
    return result


def load_run(folder):
    folder = Path(folder).resolve()
    report_path = folder / "report.json"
    report = json.loads(report_path.read_text("utf-8-sig"))
    require(report.get("schema") == "c7-probe-v1", "Expected a C7 probe report: " + str(folder))
    require(report.get("state") in ("sample_passed", "failed", "cancelled"), "Probe run is not finished")
    require(report.get("backend") in ("native", "python"), "Missing entropy backend")
    require(valid_hash(report.get("input_sha256")), "Missing input stream SHA-256")
    header, reconstruction = report["header"], report["reconstruction"]
    for key in ("width", "height", "frames", "intra_period"):
        require(type(header.get(key)) is int and header[key] > 0, "Invalid header: " + key)
    require(header["frames"] <= 64, "Only bounded C7 probe results (at most 64 frames) are supported")
    require(reconstruction.get("header") == header
            and reconstruction.get("input_sha256") == report["input_sha256"], "Inconsistent report identity")
    records = reconstruction.get("frames")
    require(isinstance(records, list) and len(records) <= header["frames"], "Invalid frame records")
    require(reconstruction.get("completed_frames") == len(records)
            and reconstruction.get("total_frames") == header["frames"], "Frame counts differ")
    require(reconstruction.get("complete_sequence") == (len(records) == header["frames"]),
            "Inconsistent sequence completion")
    for index, record in enumerate(records):
        require(record.get("frame") == index + 1, "Frame records must be contiguous and ordered")
        require(record.get("expected", {}).get("frame") == index + 1, "Expected frame order differs")
        for key in HASH_KEYS:
            require(valid_hash(record.get("expected", {}).get(key)), "Missing expected frame hash")
        trace = record.get("entropy_trace")
        require(isinstance(trace, list), "Missing entropy trace")
        for part, row in enumerate(trace):
            require(row.get("part") == part and all(valid_hash(row.get(k)) for k in TRACE_KEYS),
                    "Invalid entropy trace")
    require(aggregate(records) == reconstruction.get("actual_sequence_checksums"),
            "Frame hashes do not match recorded sequence aggregate")
    expected = reconstruction.get("expected_sequence_checksums", {})
    require(all(valid_hash(expected.get(k)) for k in HASH_KEYS), "Missing expected sequence hashes")
    if len(records) == header["frames"]:
        require(aggregate([r["expected"] for r in records]) == expected, "Expected frame aggregate differs")
    log_path = folder / "frames.jsonl"
    if records:
        logged = [json.loads(line) for line in log_path.read_text("utf-8-sig").splitlines() if line.strip()]
        require(logged == records, "frames.jsonl differs from report.json")
    snapshots = {}
    for snapshot in report.get("snapshots", []):
        frame, name = snapshot.get("frame"), snapshot.get("file")
        require(type(frame) is int and 1 <= frame <= len(records) and frame not in snapshots,
                "Invalid or duplicate snapshot frame")
        require(isinstance(name, str) and Path(name).name == name and Path(name).suffix == ".npz",
                "Snapshot must be a local .npz filename")
        path = (folder / name).resolve()
        require(path.parent == folder, "Snapshot is outside the result folder")
        require(valid_hash(snapshot.get("sha256")) and sha256(path) == snapshot["sha256"],
                "Snapshot file checksum differs: " + name)
        snapshots[frame] = path
    return {"folder": str(folder), "report_sha256": sha256(report_path), "report": report,
            "records": records, "snapshots": snapshots}


def read_snapshot(run, frame):
    """Validate raw values against the per-frame hashes before measuring them."""
    header = run["report"]["header"]
    with np.load(run["snapshots"][frame], allow_pickle=False) as data:
        keys = set(data.files)
        is_intra = (frame - 1) % header["intra_period"] == 0
        expected = {"rgb", "ref_latent"} | (set() if is_intra else {"ref_y"})
        require(keys == expected and len(data.files) == len(keys), "Snapshot keys do not match frame type")
        values = {key: data[key] for key in keys}
    rgb = values["rgb"]
    require(rgb.dtype == np.uint8 and rgb.shape == (header["height"], header["width"], 3),
            "RGB must be uint8 H x W x 3 matching the stream")
    require(hashlib.sha256(rgb.tobytes()).hexdigest() == run["records"][frame - 1]["rgb_sha256"],
            "Snapshot RGB does not match the recorded frame hash")
    latent_hash = hashlib.sha256()
    for key in STATE_KEYS:
        latent_hash.update(key.encode())
        value = values.get(key)
        if value is not None:
            require(value.dtype.str == "<f4" and value.ndim == 4 and value.shape[0] == 1
                    and value.size > 0 and np.isfinite(value).all(),
                    "Reference state must be finite float32 NCHW: " + key)
            latent_hash.update(value.tobytes())
    require(latent_hash.hexdigest() == run["records"][frame - 1]["latent_sha256"],
            "Snapshot reference state does not match the recorded frame hash")
    return values


def array_metrics(left, right, *, rgb=False):
    require(left.shape == right.shape and left.dtype == right.dtype, "Snapshot shapes or dtypes differ")
    delta = left.astype(np.float64) - right.astype(np.float64)
    changed = left != right
    mse = float(np.mean(delta * delta))
    result = {"shape": list(left.shape), "dtype": str(left.dtype),
              "byte_equal": left.tobytes() == right.tobytes(),
              "numeric_equal": not bool(np.any(changed)), "elements": int(left.size),
              "different_elements": int(np.count_nonzero(changed)),
              "mae": float(np.mean(np.abs(delta))), "mse": mse,
              "max_abs_error": float(np.max(np.abs(delta)))}
    if rgb:
        pixels = np.any(changed, axis=2)
        result.update(different_pixels=int(np.count_nonzero(pixels)), pixels=int(pixels.size),
                      different_pixel_fraction=float(np.mean(pixels)),
                      different_channels_rgb=[int(n) for n in np.count_nonzero(changed, axis=(0, 1))],
                      psnr_db=10 * math.log10(255.0 ** 2 / mse) if mse else None,
                      psnr_infinite=(mse == 0), peak_value=255)
    return result


def compare_results(left_folder, right_folder):
    left, right = load_run(left_folder), load_run(right_folder)
    a, b = left["report"], right["report"]
    require(a["input_sha256"] == b["input_sha256"] and a["header"] == b["header"],
            "Both runs must use the exact same stream and header")
    require(a["backend"] == b["backend"], "Entropy backends differ; compare native/native or python/python")
    require(a["reconstruction"]["expected_sequence_checksums"] == b["reconstruction"]["expected_sequence_checksums"],
            "Expected sequence checksums differ")
    common = min(len(left["records"]), len(right["records"]))
    hashes, traces = [], []
    for index in range(common):
        x, y = left["records"][index], right["records"][index]
        require(x["expected"] == y["expected"], "Expected hashes differ at frame " + str(index + 1))
        hashes.append({"frame": index + 1, **{k: x[k] == y[k] for k in HASH_KEYS}})
        xt, yt = x["entropy_trace"], y["entropy_trace"]
        traces.append({"frame": index + 1, "available": bool(xt) and bool(yt),
                       "parts_left": len(xt), "parts_right": len(yt),
                       "matches": {k: len(xt) == len(yt) and all(p[k] == q[k] for p, q in zip(xt, yt))
                                   for k in TRACE_KEYS} if xt and yt else None})
    snapshot_frames = sorted(left["snapshots"].keys() & right["snapshots"].keys())
    # Validate even unpaired files. Missing paired data never becomes zero error.
    measurements = []
    for frame in sorted(left["snapshots"].keys() | right["snapshots"].keys()):
        x = read_snapshot(left, frame) if frame in left["snapshots"] else None
        y = read_snapshot(right, frame) if frame in right["snapshots"] else None
        if x is not None and y is not None:
            require(x.keys() == y.keys(), "Reference state presence differs")
            measurements.append({"frame": frame, "rgb": array_metrics(x["rgb"], y["rgb"], rgb=True),
                                 "reference_state": {k: array_metrics(x[k], y[k]) if k in x else None
                                                     for k in STATE_KEYS}})
    confounders = []
    for key in ("code_sha256", "probe_code_sha256", "native_library_sha256"):
        if key == "native_library_sha256" and a["backend"] != "native":
            continue
        if not a.get(key) or not b.get(key) or a[key] != b[key]:
            confounders.append(key + " missing or different")
    def describe(run):
        report = run["report"]
        return {"folder": run["folder"], "report_sha256": run["report_sha256"],
                "state": report["state"], "completed_frames": len(run["records"]),
                "snapshot_frames": sorted(run["snapshots"]),
                "receiver_arithmetic": report["reconstruction"].get("receiver_arithmetic"),
                "gpu_driver": report.get("gpu_driver")}
    first_differences = {k: next((r["frame"] for r in hashes if not r[k]), None) for k in HASH_KEYS}
    return {"schema": "c7-offline-comparison-v1",
            "state": "compared" if measurements else "insufficient_raw_snapshots",
            "input_sha256": a["input_sha256"], "header": a["header"], "backend": a["backend"],
            "left": describe(left), "right": describe(right), "common_recorded_frames": common,
            "snapshot_frames_compared": snapshot_frames,
            "common_frames_without_paired_snapshots": sorted(set(range(1, common + 1)) - set(snapshot_frames)),
            "first_hash_difference_between_runs": first_differences,
            "first_difference_has_paired_snapshot": {k: f in snapshot_frames if f is not None else None
                                                       for k, f in first_differences.items()},
            "full_sample_pixels_compared": len(snapshot_frames) == a["header"]["frames"],
            "hash_comparison": hashes, "entropy_comparison": traces, "measurements": measurements,
            "provenance_confounders": confounders,
            "metric_scope": "Raw decoded RGB differences between these two runs; not source-video quality.",
            "acceptance_note": "No tolerance or quality threshold is applied. Comparison does not approve compatibility.",
            "cross_device_accepted": False, "quality_accepted": False,
            "neural_inference_performed": False, "gpu_used": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True, help="First complete C7 results folder")
    parser.add_argument("--right", type=Path, required=True, help="Second complete C7 results folder")
    parser.add_argument("--output", type=Path, required=True, help="New comparison JSON file")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("Choose a new output file: " + str(args.output))
        report = compare_results(args.left, args.right)
        write_json(args.output, report)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("C7 comparison rejected: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"state": report["state"], "frames_compared": report["snapshot_frames_compared"],
                      "output": str(args.output.resolve()), "torch_imported": "torch" in sys.modules},
                     ensure_ascii=False))
    return 0 if report["measurements"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
