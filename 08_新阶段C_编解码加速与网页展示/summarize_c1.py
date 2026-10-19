"""Read completed C1 records into a compact, reproducible report data artifact.

No GPU work; no missing validations or unfinished jobs enter the summary.
Pair deltas and final totals are calculated directly, never by adding ablations.
"""
import argparse
import json
from pathlib import Path

from c0_metrics import distribution
from run_c0 import read, sha

BOUNDARIES = ("evidence_ready_seconds", "offline_view_ready_seconds", "outer_call_wall_seconds")


def load_records(root):
    records = []
    for path in sorted(root.glob("*/timing.json")):
        validation_path = path.with_name("validation.json")
        if not validation_path.exists():
            continue
        value, check = read(path), read(validation_path)
        result = read(value["result_path"])
        record = {key: value.get(key) for key in (
            "run_id", "group", "pair", "variant", "role", "preset", "implementation", "output_mode",
            "png_level", "model_was_resident", "codec_acquire_seconds", "task_seconds", "outer_call_wall_seconds",
            "process_launch_to_evidence_ready_seconds", "process_launch_to_view_ready_seconds",
            "startup_before_first_task_seconds", "torch_threads_during_inference", "preview_feed_seconds")}
        record.update(evidence_path=str(path), timing_sha256=sha(path), result_sha256=sha(value["result_path"]),
                      validation_sha256=sha(validation_path), frames=len(result["frames"]),
                      frame_processing_seconds=result["frame_processing_seconds"],
                      peak_allocated_mib=result["peak_allocated_mib"], peak_reserved_mib=result["peak_reserved_mib"],
                      png_save_seconds=result.get("png_save_seconds"),
                      input_read_convert_hash_seconds=result.get("input_read_convert_hash_seconds"),
                      validation={key: check.get(key) for key in (
                          "all_rgb_and_latent_hashes_match_b4", "whole_stream_matches_b4", "input_rgb_matches_b4",
                          "stream_sha256", "stream_bytes", "prefix_payloads_match_b4", "independent_decoder",
                          "source_media_reads", "whole_file_checksum_verified", "png_count", "png_bytes")},
                      gpu_before=value.get("gpu_before"), gpu_after=value.get("gpu_after"))
        record["process_launch_to_task_return_seconds"] = value.get("process_launch_to_task_return_seconds")
        if "task_timer_origin_perf_counter" not in value:
            # Pre-fix measurements: launch->phase added an inner duration to
            # the outer call's start, omitting preparation before the inner
            # timer. Preserve raw files, publish only the exact outer boundary.
            record["legacy_cold_endpoint_note"] = "Original launch-to-evidence/view fields omitted pre-timer preparation; excluded here"
            if value.get("startup_before_first_task_seconds") is not None:
                record["process_launch_to_task_return_seconds"] = (
                    value["startup_before_first_task_seconds"] + value["outer_call_wall_seconds"])
            record["process_launch_to_evidence_ready_seconds"] = None
            record["process_launch_to_view_ready_seconds"] = None
            record["startup_before_first_task_seconds"] = None
        records.append(record)
    return records


def seconds(record, boundary):
    return record[boundary] if boundary == "outer_call_wall_seconds" else record["task_seconds"][boundary]


def delta(a, b, boundary):
    before, after = seconds(a, boundary), seconds(b, boundary)
    return {"before_id": a["run_id"], "after_id": b["run_id"],
            "before_seconds": before, "after_seconds": after,
            "saved_seconds": before - after, "reduction_percent": 100 * (before - after) / before,
            "speedup": before / after}


def grouped_delta(rows):
    return {key: distribution([r[key] for r in rows]) for key in (
        "before_seconds", "after_seconds", "saved_seconds", "reduction_percent", "speedup")}


def build(root, gates, supplements=()):
    root, gates = root.resolve(), gates.resolve()
    supplements = [source.resolve() for source in supplements]
    roots = [root, *supplements]
    records = [record for source in roots for record in load_records(source)]
    gate_records = load_records(gates)
    index = {row["run_id"]: row for row in records}
    if len(index) != len(records):
        raise ValueError("Duplicate validated run IDs across experiment roots")
    comparisons = []
    for mode in ("research", "display"):
        for role in ("encode", "decode"):
            for boundary in BOUNDARIES:
                rows = []
                for repeat in (1, 2, 3):
                    a, b = (f"{role}_{mode}_{repeat}_{impl}" for impl in ("legacy", "c1"))
                    if a in index and b in index:
                        rows.append({"repeat": repeat, **delta(index[a], index[b], boundary)})
                comparisons.append({"output_mode": mode, "role": role, "boundary": boundary,
                                    "pairs": rows, **grouped_delta(rows)})
    combined = []
    for mode in ("research", "display"):
        for boundary in BOUNDARIES:
            rows = []
            for repeat in (1, 2, 3):
                ids = [f"{role}_{mode}_{repeat}_{impl}" for role in ("encode", "decode") for impl in ("legacy", "c1")]
                if not all(i in index for i in ids):
                    continue
                before = sum(seconds(index[i], boundary) for i in (ids[0], ids[2]))
                after = sum(seconds(index[i], boundary) for i in (ids[1], ids[3]))
                rows.append({"repeat": repeat, "run_ids": ids, "before_seconds": before, "after_seconds": after,
                             "saved_seconds": before - after, "reduction_percent": 100 * (before - after) / before,
                             "speedup": before / after})
            combined.append({"output_mode": mode, "boundary": boundary, "pairs": rows, **grouped_delta(rows),
                             "note": "Sum of separately executed encode/decode task times; not a measured network/browser roundtrip"})
    attribution = []
    for role in ("encode", "decode"):
        for before_variant, after_variant, change in (("legacy", "memory_png6", "memory_rgb_and_raw_preview_at_png6"),
                                                     ("memory_png6", "c1", "png_level_6_to_1_after_memory_path")):
            for boundary in BOUNDARIES:
                rows = []
                for repeat in (1, 2, 3):
                    a, b = (f"{role}_research_{repeat}_{variant}" for variant in (before_variant, after_variant))
                    if a in index and b in index:
                        rows.append({"repeat": repeat, **delta(index[a], index[b], boundary)})
                attribution.append({"change": change, "role": role, "boundary": boundary,
                                    "output_mode": "research", "pairs": rows, **grouped_delta(rows)})
    residency = []
    for role in ("encode", "decode"):
        resident_command = root / f"{role}_resident_process/command.json"
        fresh_commands = [root / f"{role}_fresh_{i}_process/command.json" for i in (1, 2, 3)]
        ids = [f"{role}_{policy}_{i}" for policy in ("resident", "fresh") for i in (1, 2, 3)]
        if not resident_command.exists() or not all(p.exists() for p in fresh_commands) or not all(i in index for i in ids):
            continue
        commands = [read(p) for p in [resident_command, *fresh_commands]]
        if any(c["returncode"] != 0 for c in commands):
            continue
        resident_seconds = commands[0]["process_seconds"]
        fresh_seconds = sum(c["process_seconds"] for c in commands[1:])
        residency.append({"role": role, "tasks_per_policy": 3, "sequences_per_policy": 1,
                          "run_ids": ids, "output_mode": "display",
                          "resident_process_group_seconds": resident_seconds,
                          "fresh_process_group_seconds": fresh_seconds,
                          "group_saved_seconds": fresh_seconds - resident_seconds,
                          "average_saved_seconds_per_task": (fresh_seconds - resident_seconds) / 3,
                          "command_paths": [str(p) for p in [resident_command, *fresh_commands]],
                          "first_and_followup_seconds": {policy: [index[f"{role}_{policy}_{i}"]["task_seconds"]["offline_view_ready_seconds"] for i in (1, 2, 3)] for policy in ("resident", "fresh")},
                          "includes": "All imports, initializations, tasks, per-task GPU snapshots and process shutdown; external B4 validation excluded equally"})
    counts = {
        "formal_tasks": len(records), "gate_tasks": len(gate_records),
        "formal_encodes": sum(r["role"] == "encode" for r in records),
        "formal_decodes": sum(r["role"] == "decode" for r in records),
        "formal_rgb_latent_matches": sum(r["validation"]["all_rgb_and_latent_hashes_match_b4"] is True for r in records),
        "formal_whole_stream_matches": sum(r["validation"]["whole_stream_matches_b4"] is True for r in records),
    }
    return {"schema_version": 2, "scope": "C1 only; C2/C3/C4 not executed", "counts": counts,
            "manifest": read(root / "manifest.json"), "gate_manifest": read(gates / "manifest.json"),
            "supplement_manifests": [{"root": str(source), "manifest": read(source / "manifest.json")} for source in supplements],
            "records": records, "gate_records": gate_records, "comparisons": comparisons,
            "combined": combined, "attribution": attribution, "residency": residency,
            "unmeasured": ["independent_holdout_clip", "browser", "network", "real_acoustic_link", "GPU_kernel_attribution"],
            "interpretation": "Three pairs report median and range only. Ablation deltas must not replace measured final totals. PNG compression changes local evidence bytes, not GLC payload or reconstructed pixels."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, action="append", default=[],
                        help="Additional immutable recovery roots; duplicate validated IDs are rejected")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build(args.input, args.gates, args.supplement)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(args.output)
    print(value["counts"])
    for row in value["combined"]:
        print(row["output_mode"], row["boundary"], row["before_seconds"], row["after_seconds"], row["reduction_percent"])


if __name__ == "__main__":
    main()
