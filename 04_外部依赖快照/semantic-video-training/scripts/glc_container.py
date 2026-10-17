"""Versioned local GLC research container; not an upstream interchange format."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import struct

MAGIC = b"GLCVLAB1"
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


def frame_layout(header, index):
    is_i = index % header["intra_period"] == 0
    width = (header["width"] + 63) // 64 * 64
    height = (header["height"] + 63) // 64 * 64
    z_count = width * height // 4096 if is_i else 16
    parts = 4 if is_i else 2
    symbols_per_part = 256 * (height // 16) * (width // 16) // parts
    return (0 if is_i else 1), (z_count * 14 + 7) // 8, parts, symbols_per_part


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


class StreamWriter:
    def __init__(self, path, header):
        validate_header(header)
        raw_header = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if len(raw_header) > MAX_HEADER:
            raise ValueError("GLC header too large.")
        self.path = Path(path)
        self.part = self.path.with_name(self.path.name + ".part")
        if self.path.exists():
            raise FileExistsError(self.path)
        self.header, self.count, self.size = header, 0, 0
        self.digest = hashlib.sha256()
        self.handle = self.part.open("xb")
        self._write(MAGIC + struct.pack(">I", len(raw_header)) + raw_header)

    def _write(self, data):
        if self.size + len(data) + 32 > MAX_BYTES:
            raise ValueError("GLC stream exceeds 256 MiB.")
        self.handle.write(data)
        self.digest.update(data)
        self.size += len(data)

    def add(self, frame):
        if self.count >= self.header["frames"]:
            raise ValueError("Too many frames.")
        data = pack_frame(self.header, self.count, frame)
        self._write(data)
        self.count += 1
        return len(data)

    def finish(self):
        if self.count != self.header["frames"]:
            raise ValueError("Incomplete frame sequence.")
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


def read_stream(path):
    path = Path(path)
    if not 48 <= path.stat().st_size <= MAX_BYTES:
        raise ValueError("Invalid GLC file size.")
    data = path.read_bytes()
    if hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError("GLC file checksum failed.")
    cursor = Cursor(data[:-32])
    if cursor.take(8) != MAGIC:
        raise ValueError("Unsupported GLC container version.")
    header_size = cursor.integer()
    if not 1 <= header_size <= MAX_HEADER:
        raise ValueError("Invalid GLC header length.")
    header = json.loads(cursor.take(header_size), object_pairs_hook=_unique_object)
    validate_header(header)
    frames = []
    for index in range(header["frames"]):
        record = Cursor(cursor.take(cursor.integer()))
        actual_index, kind, parts, z_size = struct.unpack(">IBBI", record.take(10))
        expected_kind, expected_z, expected_parts, symbols = frame_layout(header, index)
        if (actual_index, kind, parts, z_size) != (index, expected_kind, expected_parts, expected_z):
            raise ValueError("Invalid GLC frame order or coding structure.")
        frame = {"kind": kind, "z": record.take(z_size), "y": []}
        for _ in range(parts):
            stream = record.take(record.integer())
            if len(stream) < 16 or struct.unpack_from(">I", stream)[0] != symbols:
                raise ValueError("Invalid GLC entropy symbol count.")
            frame["y"].append(stream)
        frame["latent_sha256"] = record.take(32).hex()
        frame["rgb_sha256"] = record.take(32).hex()
        record.done()
        frames.append(frame)
    cursor.done()
    return header, frames
