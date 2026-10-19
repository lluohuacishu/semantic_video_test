"""Select one accepted decode for diagnostics without another network trial."""
from __future__ import annotations

import argparse
from pathlib import Path

from b2_common import git, now, read, require, sha, write
from continue_b3_public import validate_receipt

SINGLE_DECODE = "B3_public_candidate_single_decode_input"
THREE_RATE = "B3_public_candidate_three_rate_engineering_passed"


def accepted_trial(sender, directory, source_id):
    sender, directory = Path(sender).resolve(), Path(directory).resolve()
    receiver = directory / "receiver"
    result = read(directory / "result.json")
    require(result["status"] == "passed", "Trial was not accepted")
    require(read(directory.parent / "environment.json")["source_id"] == source_id,
            "Trial source identity differs")
    _, transfer = validate_receipt(directory, sender / "message.scv")
    manifest = read(sender / "manifest.json")
    decoded = read(receiver / "decode.json")
    completion = read(receiver / "completion.json")
    require(completion["success"] is True and completion["pid"] == decoded["receiver_pid"]
            == result["receiver_pid"], "Independent decode incomplete or PID differs")
    require(read(receiver / "received_manifest.json") == manifest, "Received manifest changed")
    require(result["packet_sha256"] == decoded["packet_sha256"] == sha(sender / "message.scv"),
            "Accepted packet identity differs")
    require(result["frames"] == decoded["frames"] == manifest["frames"]
            == len(decoded["base_rgb_sha256"]) == len(decoded["enhanced_rgb_sha256"]),
            "Accepted frame count differs")
    require(result["enhanced_rgb_sequence_sha256"] == decoded["enhanced_rgb_sequence_sha256"],
            "Accepted RGB identity differs")
    require(result["application_bytes"] == transfer["application_bytes"]
            and result["elapsed_seconds"] == transfer["elapsed_seconds"], "Transfer measurement differs")
    require(result["playback_verified"] and result["outside_roi_unchanged"]
            and result["received_patch_matches_composite"]
            and result["external_source_or_cache_reads"] == 0, "Accepted decode checks failed")
    audit = read(receiver / "receiver_read_audit.json")
    require(not audit["denied_reads"] and not audit["external_source_or_sender_cache_reads"]
            and all(row["allowed"] for row in audit["subprocesses"]), "Receiver read audit failed")
    require(decoded["glc_encoder_modules_disabled"] and not decoded["glc_source_media_reads"],
            "Independent decode boundary failed")
    return result


def build_single_decode_input(sender, directory, source_id):
    sender, directory = Path(sender).resolve(), Path(directory).resolve()
    result = accepted_trial(sender, directory, source_id)
    return {
        "status": SINGLE_DECODE, "source_id": source_id, "sender": str(sender),
        "packet_sha256": result["packet_sha256"], "packet_bytes": (sender / "message.scv").stat().st_size,
        "frames": result["frames"], "selected_rate_kbps": result["rate_kbps"], "results": [result],
        "result_directories": {str(result["rate_kbps"]): str(directory)},
        "accepted_result_sha256": sha(directory / "result.json"),
        "created_at": now(), "code_commit": git("rev-parse", "HEAD"), "selector_sha256": sha(__file__),
        "new_transmission": False, "new_decode": False, "three_rate_engineering_claim": False,
        "time_estimates": {
            "kind": "calculated ideal application-byte send duration; not a new measurement",
            "application_bytes": result["application_bytes"],
            "seconds_by_kbps": {str(rate): 8 * result["application_bytes"] / (rate * 1000)
                                for rate in (20, 30, 40)},
            "excludes": ["connection setup", "TCP/IP and link bytes", "ACK", "retransmissions", "encoding and decoding"],
        },
    }


def select_diagnostic_input(summary_path):
    summary_path = Path(summary_path).resolve()
    summary = read(summary_path)
    if summary["status"] == THREE_RATE:
        require(len(summary["results"]) == 3
                and {row["rate_kbps"] for row in summary["results"]} == {20, 30, 40}
                and summary["all_rates_identical_enhanced_RGB"], "Three-rate evidence incomplete")
        rate = 40
    else:
        require(summary["status"] == SINGLE_DECODE, "Unsupported diagnostic input status")
        require(len(summary["results"]) == 1 and summary["three_rate_engineering_claim"] is False,
                "Single decode must not claim a three-rate pass")
        rate = summary["selected_rate_kbps"]
    directories = summary.get("result_directories", {
        str(r): str(summary_path.parent / f"rate_{r}kbps") for r in (20, 30, 40)})
    directory = Path(directories[str(rate)])
    result = accepted_trial(summary["sender"], directory, summary["source_id"])
    require(result == next(row for row in summary["results"] if row["rate_kbps"] == rate),
            "Selected accepted result changed")
    require(summary["packet_sha256"] == result["packet_sha256"]
            and summary["frames"] == result["frames"], "Diagnostic input identity differs")
    if summary["status"] == SINGLE_DECODE:
        require(sha(directory / "result.json") == summary["accepted_result_sha256"],
                "Accepted result file changed")
    return summary, directories, directory, rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sender", type=Path, required=True)
    parser.add_argument("--trial-directory", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Diagnostic input already exists")
    summary = build_single_decode_input(args.sender, args.trial_directory, args.source_id)
    write(args.output, summary)
    print(f"{args.source_id}: selected accepted decode; no transmission or decoding")


if __name__ == "__main__":
    main()
