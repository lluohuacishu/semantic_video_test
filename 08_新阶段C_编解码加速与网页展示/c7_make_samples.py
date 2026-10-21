"""Repackage verified prefixes of an existing stream, without neural encoding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from c7_diagnostics import HASH_KEYS, sha256, write_json


def make_samples(stream, encoder_record, output, counts, container):
    stream, encoder_record, output = map(Path, (stream, encoder_record, output))
    header, packets, integrity = container.read_stream(stream, with_integrity=True)
    if integrity["container_version"] != 2:
        raise ValueError("C7 samples require v2")
    encoded = json.loads(encoder_record.read_text("utf-8-sig"))
    source_hash = sha256(stream)
    if encoded.get("file_sha256") != source_hash or encoded.get("header") != header:
        raise ValueError("Encoder record does not belong to this stream")
    frames = encoded["frames"]
    if len(frames) != len(packets):
        raise ValueError("Encoder frame count differs")
    aggregate = container.SequenceChecksums()
    for index, frame in enumerate(frames):
        if frame.get("frame") != index + 1:
            raise ValueError("Encoder frame order differs")
        aggregate.add(index, frame)
    aggregate.verify(integrity["sequence_reconstruction_checksums"])
    counts = list(dict.fromkeys(counts))
    if not counts or any(type(n) is not int or not 1 <= n <= min(64, len(packets)) for n in counts):
        raise ValueError("Diagnostic samples must contain 1–64 existing frames")
    output.mkdir(parents=True, exist_ok=False)
    samples = []
    for count in counts:
        sample = output / f"prefix_{count:03d}.glcv"
        sample_header = {**header, "frames": count}
        writer = container.StreamWriter(sample, sample_header, version=2)
        try:
            for packet, frame in zip(packets[:count], frames[:count]):
                writer.add({**packet, **{key: frame[key] for key in HASH_KEYS}})
            writer.finish()
        finally:
            writer.close()
        checked_header, checked_packets, checked_integrity = container.read_stream(sample, with_integrity=True)
        if checked_header != sample_header or any(
                a[key] != b[key] for a, b in zip(checked_packets, packets[:count]) for key in ("kind", "z", "y")):
            raise ValueError("Repackaged sample changed compressed payload")
        reference = {"schema": "c7-frame-reference-v1", "source_stream_sha256": source_hash,
                     "stream_sha256": sha256(sample), "header": sample_header,
                     "frames": [{"frame": i + 1, **{key: frame[key] for key in HASH_KEYS}}
                                for i, frame in enumerate(frames[:count])]}
        from c7_diagnostics import validate_reference
        validate_reference(reference, sha256(sample), sample_header, checked_integrity, container)
        write_json(sample.with_suffix(".reference.json"), reference)
        samples.append({"filename": sample.name, "frames": count, "bytes": sample.stat().st_size,
                        "sha256": sha256(sample), "compressed_payload_unchanged": True})
    manifest = {"schema": "c7-prefix-samples-v1", "source_stream_sha256": source_hash,
                "encoder_record_sha256": sha256(encoder_record), "neural_encoding_performed": False,
                "gpu_used": False, "samples": samples}
    write_json(output / "samples.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--encoder-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[1, 8, 64])
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                           "04_外部依赖快照/semantic-video-training/scripts"))
    import glc_container
    print(json.dumps(make_samples(args.stream, args.encoder_record, args.output, args.frames,
                                  glc_container), ensure_ascii=False))


if __name__ == "__main__":
    main()
