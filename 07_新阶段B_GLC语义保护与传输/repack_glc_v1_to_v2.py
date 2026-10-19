"""Repack an existing research GLC-v1 stream without running the neural model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ADAPTER = REPO / "04_外部依赖快照" / "semantic-video-training" / "scripts"
sys.path.insert(0, str(ADAPTER))

from glc_container import SequenceChecksums, StreamWriter, container_overhead_breakdown, read_stream


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def transfer_estimate(file_bytes: int, digest: str) -> dict:
    header = json.dumps({"bytes": file_bytes, "sha256": digest}, separators=(",", ":")).encode()
    application_bytes = file_bytes + 4 + len(header)
    return {
        "application_bytes_including_sender_header": application_bytes,
        "sender_header_bytes": 4 + len(header),
        "ideal_seconds_by_kbps": {
            str(rate): application_bytes * 8 / (rate * 1000) for rate in (20, 30, 40)
        },
        "scope": "theoretical sender application bytes including file/header; excludes TCP/IP, link overhead, ACK, retries, processing; no acoustic-link measurement",
    }


def repack(source: Path, destination: Path) -> dict:
    source, destination = source.resolve(), destination.resolve()
    report_path = destination.with_suffix(destination.suffix + ".repack.json")
    if source == destination or destination.exists() or report_path.exists():
        raise FileExistsError("Choose a new output stream and sidecar path.")
    header, frames, integrity = read_stream(source, with_integrity=True)
    if integrity["container_version"] != 1:
        raise ValueError("Only GLC research container v1 can be repacked by this tool.")
    compact_header = dict(header, container_version=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = StreamWriter(destination, compact_header, version=2)
    try:
        for frame in frames:
            writer.add(frame)
        writer.finish()
    finally:
        writer.close()
    checked_header, checked_frames, checked_integrity = read_stream(destination, with_integrity=True)
    if checked_header != compact_header or len(checked_frames) != len(frames):
        raise ValueError("Repacked header or frame count changed.")
    for original, checked in zip(frames, checked_frames, strict=True):
        if any(original[key] != checked[key] for key in ("kind", "z", "y")):
            raise ValueError("Repacked compressed frame payload changed.")
    aggregate = SequenceChecksums()
    for index, frame in enumerate(frames):
        aggregate.add(index, frame)
    aggregate.verify(checked_integrity["sequence_reconstruction_checksums"])
    with tempfile.TemporaryDirectory(prefix="b4_v1_roundtrip_", dir=destination.parent) as scratch:
        legacy_roundtrip = Path(scratch) / "roundtrip.glcv"
        legacy_writer = StreamWriter(legacy_roundtrip, header, version=1)
        try:
            for frame in frames:
                legacy_writer.add(frame)
            legacy_writer.finish()
        finally:
            legacy_writer.close()
        legacy_v1_byte_identical = (
            legacy_roundtrip.stat().st_size == source.stat().st_size and
            sha256(legacy_roundtrip) == sha256(source)
        )
    if not legacy_v1_byte_identical:
        raise ValueError("Rewriting the existing v1 stream changed its exact bytes.")
    payload_bytes = sum(len(frame["z"]) + sum(map(len, frame["y"])) for frame in frames)
    before = container_overhead_breakdown(header, 1)
    after = container_overhead_breakdown(compact_header, 2)
    if (source.stat().st_size - payload_bytes != before["total"] or
            destination.stat().st_size - payload_bytes != after["total"]):
        raise ValueError("Container byte accounting does not match the file.")
    output_bytes = destination.stat().st_size
    output_hash = sha256(destination)
    report = {
        "source_stream": str(source),
        "source_bytes": source.stat().st_size,
        "source_sha256": sha256(source),
        "source_container_overhead": before,
        "output_stream": str(destination),
        "output_bytes": output_bytes,
        "output_sha256": output_hash,
        "output_header": compact_header,
        "output_container_overhead": after,
        "entropy_and_codebook_payload_bytes": payload_bytes,
        "bytes_saved": source.stat().st_size - output_bytes,
        "frame_payloads_preserved": True,
        "reconstruction_checksum_aggregate_preserved": True,
        "legacy_v1_byte_identical_after_rewrite": legacy_v1_byte_identical,
        "container_checksum_verified_after_repack": checked_integrity["container_version"] == 2,
        "independent_decode_performed": False,
        "transfer_estimate": transfer_estimate(output_bytes, output_hash),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = repack(args.input, args.output)
    print(json.dumps({
        "output": result["output_stream"],
        "bytes_before": result["source_bytes"],
        "bytes_after": result["output_bytes"],
        "bytes_saved": result["bytes_saved"],
        "application_bytes": result["transfer_estimate"]["application_bytes_including_sender_header"],
        "ideal_seconds_by_kbps": result["transfer_estimate"]["ideal_seconds_by_kbps"],
        "report": str(args.output.resolve().with_suffix(args.output.suffix + ".repack.json")),
        "independent_decode_performed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
