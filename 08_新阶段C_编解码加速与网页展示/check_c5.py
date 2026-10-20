"""Audit preserved C5 local evidence without repeating GPU or throttled runs."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from c2_watch import write
from c5_server import inspect_stream


def read(path):
    return json.loads(path.read_text("utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence_root.resolve()
    folders = {key: root / value for key, value in {
        "timebase": "C5_timebase_20261020_r1", "endpoints": "C5_endpoints_20261020_r1",
        "performance": "C5_resident_20261020_r1", "stability": "C5_stability_20261020_r1",
        "resident_lifecycle": "C5_resident_lifecycle_20261020_r1",
        "bundle_validation": "C5_bundle_validation_20261020_r1", "browser": "C5_browser_20261020_r1"}.items()}
    data = {key: read(folder / "summary.json") for key, folder in folders.items()}
    for key in ("endpoints", "performance", "stability", "resident_lifecycle", "bundle_validation"):
        require(data[key]["status"] == "passed", key + " did not pass")
    stable = data["stability"]
    require(len(stable["successes"]) == 10, "Expected ten endpoint successes")
    audited = []
    for row in stable["successes"]:
        sender = folders["stability"] / "sender" / row["id"]
        receiver = folders["stability"] / "receiver" / row["receiver_id"]
        tx, rx = read(sender / "result.json"), read(receiver / "result.json")
        decode = read(receiver / "decode/receiver/decode.json")
        sent, received = inspect_stream(sender / "stream.glcv"), inspect_stream(receiver / "received.glcv")
        require(sent["sha256"] == received["sha256"] == tx["stream_sha256"] == rx["stream_sha256"], "Stream SHA differs")
        require(tx["verified"] and rx["verified"] and decode["whole_file_checksum_verified"]
                and decode["all_frame_checksums_match"] and decode["independent_decoder"], "Decode integrity failed")
        require(not decode["source_media_reads"], "Receiver read source media")
        require(decode["sequence_reconstruction_checksums"] == sent["integrity"]["sequence_reconstruction_checksums"],
                "Ordered RGB/latent checksum differs")
        require(sent["header"]["frames"] == row["frames"] == len(decode["frames"]), "Frame count differs")
        require(row["rate_mode"] == "max", "Unexpected rate-limited experiment")
        require((receiver / "decode/receiver/reconstruction_preview.mp4").is_file(), "Missing receiver MP4")
        audited.append({**row, "sha256": sent["sha256"], "offline_verified": True})
    require({r["stage"] for r in stable["cancellations"]} == {"encode", "receive", "decode"}, "Cancellation scope incomplete")
    require(all(r["status"] == "cancelled" for r in stable["cancellations"]), "Cancellation did not finish")
    life = data["resident_lifecycle"]
    for role in ("encode", "decode"):
        rows = [r for r in life["rows"] if r["role"] == role]
        require(len(rows) == 10 and all(r["verified"] for r in rows), "Resident sequence incomplete")
        require(len({r["pid"] for r in rows}) == 1 and not rows[0]["resident"]
                and all(r["resident"] for r in rows[1:]), "Resident reuse changed")
    require(all(r["verified"] and r["previous_pid"] != r["new_pid"] for r in life["recovery"]), "Resident recovery failed")
    require(all(s not in ("running", "orphan_running", "liveness_unknown")
                for block in (stable, life) for s in block["process_states"].values()), "Worker not terminated")
    bundle = data["bundle_validation"]
    require([r["frames"] for r in bundle["results"]] == [8, 125]
            and all(r["verified"] and not r["source_media_reads"] for r in bundle["results"]), "Bundle decode scope incomplete")
    browser = data["browser"]
    require(browser["receiver"]["currentTime"] > 0 and browser["receiver"]["duration"] == 10
            and browser["receiver"]["error"] is None and not browser["receiverLogs"]
            and not browser["senderLogs"], "Browser playback check failed")
    tests = {}
    for key, filename in (("B", "C5_regression_b.log"), ("C", "C5_regression_c_final_20261020.log")):
        log = (root / filename).read_text("utf-8-sig")
        match = re.search(r"Ran (\d+) tests in ([\d.]+)s", log)
        require(match is not None and re.search(r"^OK\s*$", log, re.M) and "skipped=" not in log, "Regression log failed: " + key)
        tests[key] = {"count": int(match[1]), "seconds": float(match[2]), "passed": True, "skipped": 0,
                      "log": filename, "sha256": hashlib.sha256((root / filename).read_bytes()).hexdigest()}
    summary = {"schema_version": 1, "as_of_local_date": "2026-10-20", "local_candidate_status": "passed",
               "final_two_machine_acceptance": "pending_receiver_machine", "two_physical_computers": False,
               "rate_limited_experiments_performed": False, "regression": tests, "stability": audited,
               "cancellations": stable["cancellations"], "resident_lifecycle": life,
               "performance": {k: v for k, v in data["performance"].items() if k not in ("rows", "pairs")},
               "bundle_validation": bundle, "browser": browser,
               "limits": ["4070 compatibility and real LAN untested", "fresh target Python installation untested",
                          "no browser 52/40-second speed claim", "no long-duration soak or host-RSS leak claim",
                          "60-second case is repeated NOAA at 320x180, not full-resolution native 60-second footage"],
               "evidence": {key: {"folder": folder.name,
                                   "summary_sha256": hashlib.sha256((folder / "summary.json").read_bytes()).hexdigest()}
                            for key, folder in folders.items()}}
    write(args.output, summary)
    print(json.dumps({"local_candidate_status": "passed", "audited_endpoint_successes": len(audited),
                      "resident_successes": len(life["rows"]), "regression_tests": sum(r["count"] for r in tests.values()),
                      "two_physical_computers": False}), flush=True)


if __name__ == "__main__":
    main()
