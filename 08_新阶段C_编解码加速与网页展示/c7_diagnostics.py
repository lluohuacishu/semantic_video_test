"""C7 evidence helpers. Standard library only; never weaken decoder checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HASH_KEYS = ("latent_sha256", "rgb_sha256")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def sequence_report(header, receiver_profile, expected, actual, records, stream_sha256):
    complete = len(records) == header["frames"]
    return {
        "schema": "c7-reconstruction-diagnostic-v1",
        "input_sha256": stream_sha256,
        "header": header,
        "sender_arithmetic": header["arithmetic"],
        "receiver_arithmetic": receiver_profile,
        "completed_frames": len(records), "total_frames": header["frames"],
        "complete_sequence": complete,
        "expected_sequence_checksums": expected,
        "actual_sequence_checksums": actual,
        # A prefix digest cannot be compared to a complete sequence digest.
        "sequence_matches": {k: actual[k] == expected[k] for k in HASH_KEYS} if complete else None,
        "mismatched_channels": [k for k in HASH_KEYS if actual[k] != expected[k]] if complete else None,
        "first_mismatch_frame": None,
        "first_mismatch_known": False,
        "frames": records,
        "cross_device_accepted": False,
    }


def verify_sequence(checksums, expected, diagnostic_path, *, header, receiver_profile,
                    records, stream_sha256):
    """Retain both computed values and frame evidence before propagating failure."""
    try:
        checksums.verify(expected)
    except ValueError:
        report = sequence_report(header, receiver_profile, expected, checksums.values(), records,
                                 stream_sha256)
        report["state"] = "failed"
        write_json(diagnostic_path, report)
        channels = ", ".join(report["mismatched_channels"] or [])
        raise ValueError("Ordered GLC frame checksum aggregate differs from the sender. "
                         f"Mismatched: {channels}. See {Path(diagnostic_path).name}") from None


def validate_reference(reference, stream_sha256, header, integrity, container):
    """Sidecar supplies hashes only, cryptographically bound to this v2 stream."""
    if (reference.get("schema") != "c7-frame-reference-v1"
            or reference.get("stream_sha256") != stream_sha256
            or reference.get("header") != header):
        raise ValueError("Reference does not belong to this exact stream")
    records = reference.get("frames")
    if not isinstance(records, list) or len(records) != header["frames"]:
        raise ValueError("Reference frame count differs")
    checksums = container.SequenceChecksums()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("frame") != index + 1:
            raise ValueError("Reference frame order differs")
        checksums.add(index, record)
    checksums.verify(integrity["sequence_reconstruction_checksums"])
    return records
