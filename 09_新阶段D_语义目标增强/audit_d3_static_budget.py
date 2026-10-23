"""D3 planning arithmetic only: read old JSON/stat sizes and the pure container module.

Does not open images, read bitstream payloads, load models, compress data, or run a codec.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    container_path = REPO / "04_外部依赖快照/semantic-video-training/scripts/glc_container.py"
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("d3_container_static", container_path)
    container = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(container)
    root = args.workspace / "02_实验数据/D_语义目标增强/D2_lowrate_20261022_r1"
    inputs = read_json(root / "inputs.json")
    old_gate = read_json(root / "rate_gate.json")
    envelope = struct.calcsize("<4sBBHHHIIIIHH")
    roi_record = struct.calcsize("<HHHHHHI")
    assert envelope == 32 and roi_record == 16
    rows = []
    source_files = [root / "inputs.json", root / "rate_gate.json"]
    for clip in inputs["clips"]:
        video_id = clip["video_id"]
        path = root / f"points/v{video_id:03d}_512x288_A_q00/encoder/encode.json"
        data = read_json(path)
        source_files.append(path)
        header = data["header"]
        assert (header["width"], header["height"], header["frames"], header["q"]) == (512, 288, 64, 0)
        actual = path.with_name("stream.glcv").stat().st_size
        assert actual == data["file_bytes"] == old_gate["anchors"][str(video_id)]["S0"]
        duration = Fraction(clip["duration_exact"])
        cap40 = int(Fraction(40000, 8) * duration)
        app_cap = min(cap40, 11 * actual // 10)
        assert app_cap == old_gate["anchors"][str(video_id)]["ceiling"]
        structural = container.container_overhead_breakdown(header, 2)
        z = sum(frame["z_bytes"] for frame in data["frames"])
        y = sum(frame["y_bytes"] for frame in data["frames"])
        assert structural["total"] + z + y == actual
        n = header["frames"]
        side = []
        for stride in [1, 8, 64]:
            records = (n + stride - 1) // stride
            metadata = envelope + 4 + roi_record * records
            side.append({"box_update_stride": stride, "records": records,
                         "metadata_bytes": metadata,
                         "remaining_payload_bytes_at_existing_base": app_cap - actual - metadata,
                         "note": "Proposed fixed binary layout arithmetic, not measured encoded ROI data; each block carries all frames."})
        lower_bounds = []
        for width, height in [(64, 64), (128, 96), (320, 180), (384, 216), (512, 288)]:
            h = dict(header, width=width, height=height)
            overhead = container.container_overhead_breakdown(h, 2)["total"]
            layouts = [container.frame_layout(h, index) for index in range(n)]
            fixed_z = sum(item[1] for item in layouts)
            min_y = sum(16 * item[2] for item in layouts)
            lower_bounds.append({"width": width, "height": height,
                                 "container_bytes": overhead, "fixed_z_bytes": fixed_z,
                                 "validator_min_y_bytes": min_y,
                                 "format_lower_bound_bytes": overhead + fixed_z + min_y,
                                 "claim": "Necessary size lower bound for this unchanged GLC v2 layout/header; not an achievable stream or codec performance bound."})
        rows.append({"video_id": video_id, "frames": n, "fps_exact": clip["fps"],
                     "duration_exact": str(duration), "S0_bytes": actual,
                     "rate40_cap_bytes": cap40, "growth_cap_bytes": 11 * actual // 10,
                     "application_cap_bytes": app_cap,
                     "spare_at_existing_q0_bytes": app_cap - actual,
                     "spare_bytes_per_frame": (app_cap - actual) / n,
                     "diagnostic80_cap_bytes": int(Fraction(80000, 8) * duration),
                     "diagnostic80_uses_application_growth_cap": False,
                     "old_stream_byte_split": {"container": structural, "z": z, "y": y},
                     "proposed_metadata_scenarios": side, "GLC_format_lower_bounds": lower_bounds})
    provenance = [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                  for path in source_files + [container_path, Path(__file__)]]
    result = {"schema": "d3-static-budget-v1", "status": "planning_arithmetic_only",
              "current_execution": {"new_encodes": 0, "new_decodes": 0, "new_semantic_inferences": 0,
                                    "training": False, "GPU_used": False, "new_payload_compression": False,
                                    "dataset_or_weight_integrity_scan": False},
              "proposed_layout": {"envelope_struct": "<4sBBHHHIIIIHH", "envelope_bytes": envelope,
                                  "record_struct": "<HHHHHHI", "record_bytes": roi_record, "file_crc32_bytes": 4},
              "clips": rows, "source_metadata_and_code_hashes": provenance}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    for row in rows:
        small = row["GLC_format_lower_bounds"][0]
        roi = row["GLC_format_lower_bounds"][1]
        print(f"{row['video_id']:03d}: S0={row['S0_bytes']}, cap={row['application_cap_bytes']}, "
              f"spare={row['spare_at_existing_q0_bytes']}, block8_meta={row['proposed_metadata_scenarios'][1]['metadata_bytes']}, "
              f"GLC64_floor={small['format_lower_bound_bytes']}, GLC128x96_floor={roi['format_lower_bound_bytes']}")


if __name__ == "__main__":
    main()
