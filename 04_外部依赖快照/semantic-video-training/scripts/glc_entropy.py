"""Bounded byte-rANS and 14-bit codebook packing for the local GLC stream.

The rANS recurrence follows Fabian Giesen's public-domain rans_byte.h.
This file format and Gaussian table quantization are local additions, not GLC's
unreleased bitstream format. No neural features are saved as floating-point files.
"""
from __future__ import annotations

from bisect import bisect_right
import hashlib
import math
import struct

import numpy as np
from scipy.special import ndtr

PRECISION = 16
TOTAL = 1 << PRECISION
LOWER = 1 << 23
MAX_SYMBOLS = 8_388_608
ENTROPY_ID = "python-byte-rans-gaussian128-v1"


def pack_indices(indices):
    values = np.asarray(indices).reshape(-1)
    if len(values) > MAX_SYMBOLS or not np.isfinite(values).all() or np.any(values != np.rint(values)):
        raise ValueError("Codebook indexes must be bounded integers.")
    if np.any(values < 0) or np.any(values >= 16384):
        raise ValueError("Codebook index must fit in 14 bits.")
    values = values.astype(np.int64)
    bits = (values[:, None] >> np.arange(13, -1, -1)) & 1
    return np.packbits(bits.astype(np.uint8).reshape(-1), bitorder="big").tobytes()


def unpack_indices(data, count):
    if type(count) is not int or not 0 <= count <= MAX_SYMBOLS:
        raise ValueError("Invalid codebook index count.")
    if len(data) != (count * 14 + 7) // 8:
        raise ValueError("Codebook byte count does not match image dimensions.")
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")
    if np.any(bits[count * 14:]):
        raise ValueError("Nonzero padding in codebook stream.")
    return bits[:count * 14].reshape(count, 14).astype(np.int64) @ (1 << np.arange(13, -1, -1))


def append_signed(buffer, value):
    zigzag = value * 2 if value >= 0 else -value * 2 - 1
    while zigzag >= 128:
        buffer.append((zigzag & 127) | 128)
        zigzag >>= 7
    buffer.append(zigzag)


def read_signed(data, position):
    value = 0
    for shift in range(0, 35, 7):
        if position >= len(data):
            raise ValueError("Truncated escape integer.")
        byte = data[position]
        position += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            if value > 0xFFFFFFFF or (shift and byte == 0):
                raise ValueError("Noncanonical or oversized escape integer.")
            return (value >> 1) ^ -(value & 1), position
    raise ValueError("Oversized escape integer.")


class GaussianRans:
    def __init__(self):
        self.scales = np.geomspace(0.11, 256.0, 128)
        self.log_min = math.log(float(self.scales[0]))
        self.log_step = (math.log(float(self.scales[-1])) - self.log_min) / 127
        self.tables = []
        self.radii = []
        digest = hashlib.sha256()
        for sigma in self.scales:
            radius = max(1, math.ceil(float(sigma) * 6))
            support = np.arange(-radius, radius + 1, dtype=np.float64)
            mass = ndtr((support + 0.5) / sigma) - ndtr((support - 0.5) / sigma)
            mass = np.r_[mass, max(0.0, 1.0 - float(mass.sum()))]
            mass /= mass.sum()
            # Reserve one count for every symbol, including the escape symbol.
            scaled = mass * (TOTAL - len(mass))
            floor = np.floor(scaled)
            frequencies = floor.astype(np.int64) + 1
            remaining = TOTAL - int(frequencies.sum())
            order = np.argsort(-(scaled - floor), kind="stable")
            frequencies[order[:remaining]] += 1
            cdf = np.r_[0, np.cumsum(frequencies)].astype("<u4")
            if int(cdf[-1]) != TOTAL or np.any(frequencies <= 0):
                raise RuntimeError("Invalid entropy table.")
            self.tables.append(tuple(int(x) for x in cdf))
            self.radii.append(radius)
            digest.update(struct.pack(">II", radius, len(cdf)))
            digest.update(cdf.tobytes())
        self.table_sha256 = digest.hexdigest()

    def indexes(self, scales):
        values = np.asarray(scales, dtype=np.float64).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite predicted Gaussian scale.")
        values = np.clip(values, self.scales[0], self.scales[-1])
        return np.rint((np.log(values) - self.log_min) / self.log_step).astype(np.int16)

    def _check_indexes(self, indexes):
        indexes = np.asarray(indexes).reshape(-1)
        if (len(indexes) > MAX_SYMBOLS or not np.isfinite(indexes).all()
                or np.any(indexes != np.rint(indexes))
                or np.any(indexes < 0) or np.any(indexes >= len(self.tables))):
            raise ValueError("Invalid entropy table indexes.")
        return indexes.astype(np.int64).tolist()

    def encode(self, symbols, indexes):
        indexes = self._check_indexes(indexes)
        values = np.asarray(symbols).reshape(-1)
        if len(values) != len(indexes) or not np.isfinite(values).all():
            raise ValueError("Symbols and probability indexes do not match.")
        if np.any(values < -(2**31)) or np.any(values >= 2**31) or np.any(values != np.rint(values)):
            raise ValueError("Entropy symbols must be signed 32-bit integers.")
        symbols = values.astype(np.int64).tolist()
        escapes = bytearray()
        escape_count = 0
        coded = []
        for value, table_index in zip(symbols, indexes):
            radius = self.radii[table_index]
            if -radius <= value <= radius:
                coded.append(value + radius)
            else:
                coded.append(2 * radius + 1)
                append_signed(escapes, value)
                escape_count += 1
        state = LOWER
        emitted = bytearray()
        for symbol, table_index in zip(reversed(coded), reversed(indexes)):
            cdf = self.tables[table_index]
            start, stop = cdf[symbol], cdf[symbol + 1]
            frequency = stop - start
            maximum = ((LOWER >> PRECISION) << 8) * frequency
            while state >= maximum:
                emitted.append(state & 255)
                state >>= 8
            state = ((state // frequency) << PRECISION) + state % frequency + start
        rans = struct.pack("<I", state) + bytes(reversed(emitted))
        return struct.pack(">III", len(symbols), len(rans), escape_count) + rans + escapes

    def decode(self, stream, indexes):
        indexes = self._check_indexes(indexes)
        if len(stream) < 16:
            raise ValueError("Truncated entropy stream.")
        count, rans_length, escape_count = struct.unpack_from(">III", stream)
        if count != len(indexes) or not 4 <= rans_length <= len(stream) - 12 or escape_count > count:
            raise ValueError("Invalid entropy stream lengths.")
        rans = memoryview(stream)[12:12 + rans_length]
        escapes = memoryview(stream)[12 + rans_length:]
        state = struct.unpack_from("<I", rans)[0]
        if not LOWER <= state < LOWER * 256:
            raise ValueError("Invalid initial rANS state.")
        position, escape_position, escape_used = 4, 0, 0
        output = np.empty(count, dtype=np.int32)
        for i, table_index in enumerate(indexes):
            cdf = self.tables[table_index]
            slot = state & (TOTAL - 1)
            symbol = bisect_right(cdf, slot) - 1
            start, stop = cdf[symbol], cdf[symbol + 1]
            state = (stop - start) * (state >> PRECISION) + slot - start
            while state < LOWER:
                if position >= len(rans):
                    raise ValueError("Truncated rANS payload.")
                state = (state << 8) | rans[position]
                position += 1
            radius = self.radii[table_index]
            if symbol == 2 * radius + 1:
                value, escape_position = read_signed(escapes, escape_position)
                escape_used += 1
                if -radius <= value <= radius:
                    raise ValueError("Noncanonical escape symbol.")
            else:
                value = symbol - radius
            output[i] = value
        if state != LOWER or position != len(rans) or escape_position != len(escapes) or escape_used != escape_count:
            raise ValueError("Entropy stream has unused bytes or an invalid terminal state.")
        return output
