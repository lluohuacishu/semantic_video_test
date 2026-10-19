"""Versioned local GLC research container; not an upstream interchange format."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import struct

MAGIC = b"GLCVLAB1"
MAGIC_COMPACT = b"GLCVLAB2"
MAX_BYTES = 256 * 1024 * 1024
MAX_HEADER = 8192


def validate_header(header):
    if not isinstance(header, dict):
        raise ValueError("GLC header must be an object.")
    ranges = {"width": (1, 4096), "height": (1, 2160), "frames": (1, 10000),
              "q": (0, 63), "intra_period": (1, 10000), "reset_interval": (0, 10000)}
    for key, (low, high) in ranges.items():
        value = header.get(key)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Invalid GLC header field: {key}")
    if header["width"] * header["height"] > 4096 * 2160:
        raise ValueError("Frame exceeds supported pixel count.")
    fps = header.get("fps")
    if type(fps) not in (int, float) or not math.isfinite(fps) or not 0 < fps <= 240:
        raise ValueError("Invalid frame rate.")
    for key in ("revision", "entropy", "table_sha256"):
        if not isinstance(header.get(key), str) or not header[key]:
            raise ValueError(f"Missing GLC header field: {key}")
    if not isinstance(header.get("weights"), dict) or not isinstance(header.get("arithmetic"), dict):
        raise ValueError("Missing model / arithmetic identifiers.")
    if ("container_version" in header and
            (type(header["container_version"]) is not int or header["container_version"] not in (1, 2))):
        raise ValueError("Unsupported GLC research container version.")


def frame_layout(header, index):
    is_i = index % header["intra_period"] == 0
    width = (header["width"] + 63) // 64 * 64
    height = (header["height"] + 63) // 64 * 64
    z_count = width * height // 4096 if is_i else 16
    parts = 4 if is_i else 2
    symbols_per_part = 256 * (height // 16) * (width // 16) // parts
    return (0 if is_i else 1), (z_count * 14 + 7) // 8, parts, symbols_per_part


def container_overhead_breakdown(header, version):
    """Count structural/header bytes separately from compressed z/y payload."""
    if version not in (1, 2):
        raise ValueError("Unsupported GLC research container version.")
    raw_header = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    frames = header["frames"]
    intra_frames = (frames + header["intra_period"] - 1) // header["intra_period"]
    inter_frames = frames - intra_frames
    result = {
        "magic_and_header_length": 12,
        "header_json": len(raw_header),
        "whole_file_sha256": 32,
        "y_stream_length_fields": 16 * intra_frames + 8 * inter_frames,
        "frame_length_fields": 4 * frames if version == 1 else 0,
        "frame_index_type_and_layout": 10 * frames if version == 1 else 0,
        "per_frame_reconstruction_hashes": 64 * frames if version == 1 else 0,
        "sequence_reconstruction_hashes": 64 if version == 2 else 0,
    }
    result["total"] = sum(result.values())
    return result


class SequenceChecksums:
    """Domain-separated aggregate of ordered per-frame latent/RGB hashes."""
    def __init__(self):
        self.count = 0
        self._digests = {
            key: hashlib.sha256(b"GLCVLAB2/" + key.encode("ascii") + b"/ordered-frame-sha256/v1\0")
            for key in ("latent_sha256", "rgb_sha256")
        }

    def add(self, index, frame):
        if index != self.count:
            raise ValueError("GLC sequence checksum frame order mismatch.")
        values = {}
        for key in self._digests:
            try:
                value = bytes.fromhex(frame[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Missing or invalid reconstruction checksum: {key}") from exc
            if len(value) != 32:
                raise ValueError(f"Invalid reconstruction checksum: {key}")
            values[key] = value
        for key, value in values.items():
            self._digests[key].update(struct.pack(">I", index))
            self._digests[key].update(value)
        self.count += 1

    def values(self):
        return {key: digest.hexdigest() for key, digest in self._digests.items()}

    def packed(self):
        return b"".join(bytes.fromhex(value) for value in self.values().values())

    def verify(self, expected):
        if self.values() != expected:
            raise ValueError("Ordered GLC frame checksum aggregate differs from the sender.")


def pack_frame(header, index, frame):
    kind, z_size, parts, symbols = frame_layout(header, index)
    if frame["kind"] != kind or len(frame["z"]) != z_size or len(frame["y"]) != parts:
        raise ValueError("Frame does not match the declared coding structure.")
    record = bytearray(struct.pack(">IBBI", index, kind, parts, z_size))
    record.extend(frame["z"])
    for stream in frame["y"]:
        if len(stream) < 16 or struct.unpack_from(">I", stream)[0] != symbols:
            raise ValueError("Unexpected entropy symbol count.")
        record.extend(struct.pack(">I", len(stream)))
        record.extend(stream)
    for key in ("latent_sha256", "rgb_sha256"):
        digest = bytes.fromhex(frame[key])
        if len(digest) != 32:
            raise ValueError("Invalid reconstruction checksum.")
        record.extend(digest)
    return struct.pack(">I", len(record)) + record


def pack_compact_frame(header, index, frame):
    """Pack only decoder-required payload and stream lengths for container v2."""
    kind, z_size, parts, symbols = frame_layout(header, index)
    if frame["kind"] != kind or len(frame["z"]) != z_size or len(frame["y"]) != parts:
        raise ValueError("Frame does not match the declared coding structure.")
    record = bytearray(frame["z"])
    for stream in frame["y"]:
        if len(stream) < 16 or struct.unpack_from(">I", stream)[0] != symbols:
            raise ValueError("Unexpected entropy symbol count.")
        record.extend(struct.pack(">I", len(stream)))
        record.extend(stream)
    return bytes(record)


class StreamWriter:
    def __init__(self, path, header, version=1):
        validate_header(header)
        if type(version) is not int or version not in (1, 2):
            raise ValueError("Unsupported GLC research container version.")
        if header.get("container_version", 1) != version:
            raise ValueError("Header and GLC container versions disagree.")
        if version == 2 and "container_version" not in header:
            raise ValueError("Compact streams must declare container_version=2.")
        raw_header = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if len(raw_header) > MAX_HEADER:
            raise ValueError("GLC header too large.")
        self.path = Path(path)
        self.part = self.path.with_name(self.path.name + ".part")
        if self.path.exists():
            raise FileExistsError(self.path)
        self.header, self.count, self.size, self.version = header, 0, 0, version
        self.digest = hashlib.sha256()
        self.sequence_checksums = SequenceChecksums() if version == 2 else None
        self.handle = self.part.open("xb")
        magic = MAGIC if version == 1 else MAGIC_COMPACT
        self._write(magic + struct.pack(">I", len(raw_header)) + raw_header)

    def _write(self, data):
        if self.size + len(data) + 32 > MAX_BYTES:
            raise ValueError("GLC stream exceeds 256 MiB.")
        self.handle.write(data)
        self.digest.update(data)
        self.size += len(data)

    def add(self, frame):
        if self.count >= self.header["frames"]:
            raise ValueError("Too many frames.")
        data = (pack_frame(self.header, self.count, frame) if self.version == 1
                else pack_compact_frame(self.header, self.count, frame))
        if self.sequence_checksums is not None:
            self.sequence_checksums.add(self.count, frame)
        self._write(data)
        self.count += 1
        return len(data)

    def finish(self):
        if self.count != self.header["frames"]:
            raise ValueError("Incomplete frame sequence.")
        if self.sequence_checksums is not None:
            self._write(self.sequence_checksums.packed())
        self.handle.write(self.digest.digest())
        self.handle.close()
        # The .part is on the same volume. Publishing a hard link is atomic and
        # fails if another process has already created the destination.
        os.link(self.part, self.path)
        self.part.unlink()

    def close(self):
        self.handle.close()


class Cursor:
    def __init__(self, data):
        self.data, self.position = data, 0

    def take(self, size):
        if size < 0 or self.position + size > len(self.data):
            raise ValueError("Truncated GLC container.")
        result = self.data[self.position:self.position + size]
        self.position += size
        return result

    def integer(self):
        return struct.unpack(">I", self.take(4))[0]

    def done(self):
        if self.position != len(self.data):
            raise ValueError("Unexpected trailing GLC bytes.")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate GLC header key.")
        result[key] = value
    return result


def read_stream(path, with_integrity=False):
    path = Path(path)
    if not 48 <= path.stat().st_size <= MAX_BYTES:
        raise ValueError("Invalid GLC file size.")
    data = path.read_bytes()
    if hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError("GLC file checksum failed.")
    cursor = Cursor(data[:-32])
    magic = cursor.take(8)
    if magic not in (MAGIC, MAGIC_COMPACT):
        raise ValueError("Unsupported GLC container version.")
    compact = magic == MAGIC_COMPACT
    header_size = cursor.integer()
    if not 1 <= header_size <= MAX_HEADER:
        raise ValueError("Invalid GLC header length.")
    header = json.loads(cursor.take(header_size), object_pairs_hook=_unique_object)
    validate_header(header)
    if header.get("container_version", 1) != (2 if compact else 1):
        raise ValueError("Header and GLC container versions disagree.")
    frames = []
    for index in range(header["frames"]):
        expected_kind, expected_z, expected_parts, symbols = frame_layout(header, index)
        record = cursor if compact else Cursor(cursor.take(cursor.integer()))
        if compact:
            frame = {"kind": expected_kind, "z": record.take(expected_z), "y": []}
        else:
            actual_index, kind, parts, z_size = struct.unpack(">IBBI", record.take(10))
            if (actual_index, kind, parts, z_size) != (index, expected_kind, expected_parts, expected_z):
                raise ValueError("Invalid GLC frame order or coding structure.")
            frame = {"kind": kind, "z": record.take(z_size), "y": []}
        for _ in range(expected_parts):
            stream = record.take(record.integer())
            if len(stream) < 16 or struct.unpack_from(">I", stream)[0] != symbols:
                raise ValueError("Invalid GLC entropy symbol count.")
            frame["y"].append(stream)
        if compact:
            frame["latent_sha256"] = None
            frame["rgb_sha256"] = None
        else:
            frame["latent_sha256"] = record.take(32).hex()
            frame["rgb_sha256"] = record.take(32).hex()
            record.done()
        frames.append(frame)
    sequence_checksums = None
    if compact:
        sequence_checksums = {
            "latent_sha256": cursor.take(32).hex(),
            "rgb_sha256": cursor.take(32).hex(),
        }
    cursor.done()
    if with_integrity:
        return header, frames, {
            "container_version": 2 if compact else 1,
            "container_content_digest": data[-32:].hex(),
            "sequence_reconstruction_checksums": sequence_checksums,
        }
    return header, frames
