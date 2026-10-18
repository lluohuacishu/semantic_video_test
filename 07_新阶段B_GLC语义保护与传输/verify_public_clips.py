"""Audit frozen public windows, source hashes, and repeat-prepare RGB identity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from b2_common import require, sha, write


def frame_hashes(ffmpeg, path):
    result = subprocess.run([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-threads", "1",
                             "-i", str(path), "-map", "0:v:0", "-f", "framemd5", "-"],
                            capture_output=True, check=True)
    rows = [line for line in result.stdout.decode("ascii").splitlines() if line and not line.startswith("#")]
    values = [row.rsplit(",", 1)[-1].strip() for row in rows]
    require(len(values) > 0 and all(len(value) == 32 for value in values), f"Invalid decoded frame hashes: {path}")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--raw-audit", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--prepared-v2", type=Path, required=True)
    parser.add_argument("--prepared-v3", type=Path, required=True)
    parser.add_argument("--senders", type=Path, required=True, nargs="+")
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    audit = json.loads(args.raw_audit.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in audit["records"]}
    selected = {row["id"]: row for row in registry["selected_windows"]}
    sources = {row["id"]: row for row in registry["candidates"]}
    sender_records, senders = {}, {}
    for sender in args.senders:
        audit_path = sender / "source_audit.json"
        identity_path = sender / "run_identity.json"
        if not audit_path.is_file() or not identity_path.is_file():
            continue
        audit_row, identity = json.loads(audit_path.read_text(encoding="utf-8")), json.loads(identity_path.read_text(encoding="utf-8"))
        matches = [key for key, window in selected.items() if audit_row["frames"] == round(window["source_duration_seconds"] * window["output_fps"])
                   and Path(audit_row["input"]).name == f"{key}.mkv"]
        require(len(matches) == 1, f"Could not uniquely map public sender: {sender}")
        key = matches[0]
        require(sha(audit_row["input"]) == audit_row["input_sha256"] == identity["input_sha256"], "Frozen sender input hash mismatch")
        require(identity["training"] is False and identity["glc_quality"] == 0
                and identity["intra_period"] == 64 and identity["reset_interval"] == 32,
                "Frozen sender parameters changed")
        require(audit_row["ground_truth_read"] is False, "Ground truth entered the sender")
        senders[key] = {"directory": str(sender.resolve()), "input": audit_row["input"],
                        "input_sha256": audit_row["input_sha256"], "frames": audit_row["frames"],
                        "source_rgb_map_sha256": sha(sender / "source_audit.json"),
                        "run_identity_sha256": sha(identity_path), "ground_truth_read": False}
    require(set(senders) == set(selected), f"Expected all selected clips; found {sorted(senders)}")
    records = []
    for key in selected:
        raw = by_id[key]
        expected = sources[key]
        raw_path = args.raw_dir / expected["local_filename"]
        require(raw["status"] == "download_and_decode_passed" and raw["checks"]["sha1_matches_commons"]
                and raw["checks"]["size_matches_registry"], f"Downloaded source failed audit: {key}")
        require(raw_path.is_file() and sha(raw_path) == raw["sha256"]
                and raw["bytes"] == expected["expected_bytes"], f"Raw source bytes changed: {key}")
        expected_frames = round(selected[key]["source_duration_seconds"] * selected[key]["output_fps"])
        require(senders[key]["frames"] == expected_frames, f"Sender frame count differs from frozen window: {key}")
        variants = []
        for prepared in (args.prepared_v2, args.prepared_v3):
            record_path = prepared / "prepare.json"
            index = json.loads(record_path.read_text(encoding="utf-8"))
            record = next((row for row in index["records"] if row["id"] == key), None)
            if record is None:
                continue
            require(all(record["checks"].values()) and record["source_sha256"] == raw["sha256"],
                    f"Prepared window/source check failed: {key}")
            clip = Path(record["output"])
            require(clip.is_file() and sha(clip) == record["output_sha256"], f"Prepared clip hash changed: {key}")
            variants.append({"prepared_dir": str(prepared.resolve()), "path": str(clip), "sha256": sha(clip),
                             "rgb_frame_md5": frame_hashes(args.ffmpeg, clip)})
        require(any(Path(v["path"]).resolve() == Path(senders[key]["input"]).resolve() for v in variants),
                f"Sender input does not match a recorded prepared clip: {key}")
        if len(variants) == 2:
            require(variants[0]["rgb_frame_md5"] == variants[1]["rgb_frame_md5"],
                    f"Repeat preparation changed decoded source frames: {key}")
        records.append({"id": key, "source_page": expected["page"], "creator": expected["creator"],
                        "license": expected["license"], "license_url": expected["license_url"],
                        "source_sha1": raw["sha1"], "source_sha256": raw["sha256"],
                        "window": selected[key], "sender": senders[key], "prepared_variants": variants,
                        "prepared_repeats_same_decoded_RGB": len(variants) < 2 or
                        variants[0]["rgb_frame_md5"] == variants[1]["rgb_frame_md5"]})
    result = {"status": "passed", "registry": str(args.registry.resolve()), "raw_audit": str(args.raw_audit.resolve()),
              "records": records, "source_exposure_verified": False,
              "independent_capture_verified": False, "target_ground_truth": False,
              "candidate_only": True}
    write(args.output, result)
    print(json.dumps({"status": result["status"], "clips": len(records),
                      "repeat_preparation_identity_checked": sum(len(r["prepared_variants"]) == 2 for r in records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
