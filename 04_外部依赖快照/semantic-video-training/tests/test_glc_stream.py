"""Entropy coding and malformed-container checks, without neural weights/GPU."""
from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from glc_entropy import GaussianRans, pack_indices, unpack_indices, read_signed
from glc_container import StreamWriter, read_stream, validate_header, frame_layout


class EntropyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coder = GaussianRans()

    def test_varied_probability_tables_and_extreme_escape_symbols(self):
        rng = np.random.default_rng(125)
        indexes = rng.integers(0, 128, 30000)
        values = np.rint(rng.normal(size=len(indexes)) * self.coder.scales[indexes] * 2).astype(np.int32)
        values[:4] = [-2**31, 2**31 - 1, -10000, 10000]
        stream = self.coder.encode(values, indexes)
        np.testing.assert_array_equal(self.coder.decode(stream, indexes), values)

    def test_empty_and_highly_predictable_streams(self):
        for count in (0, 1, 65536):
            values = np.zeros(count, dtype=np.int32)
            indexes = np.zeros(count, dtype=np.int16)
            stream = self.coder.encode(values, indexes)
            np.testing.assert_array_equal(self.coder.decode(stream, indexes), values)
            self.assertLess(len(stream), 64)

    def test_truncated_extra_and_bad_terminal_data_rejected(self):
        rng = np.random.default_rng(7)
        indexes = rng.integers(0, 128, 1200)
        values = rng.integers(-5000, 5000, 1200)
        stream = self.coder.encode(values, indexes)
        for malformed in (stream[:3], stream[:15], stream[:-1], stream + b"x",
                          stream[:12] + b"\0\0\0\0" + stream[16:]):
            with self.subTest(length=len(malformed)), self.assertRaises(ValueError):
                self.coder.decode(malformed, indexes)

    def test_invalid_symbols_indexes_and_scale_fail(self):
        for symbols, indexes in (([0.5], [1]), ([2**31], [1]), ([np.nan], [1]),
                                  ([1], [0.5]), ([1], [128]), ([1], [-1]), ([1, 2], [1])):
            with self.subTest(symbols=symbols, indexes=indexes), self.assertRaises(ValueError):
                self.coder.encode(symbols, indexes)
        with self.assertRaises(ValueError):
            self.coder.indexes([np.nan])
        with self.assertRaises(ValueError):
            self.coder.decode(self.coder.encode([1], [1]), [])

    def test_codebook_all_boundaries_and_padding(self):
        values = np.arange(16384)
        np.testing.assert_array_equal(unpack_indices(pack_indices(values), len(values)), values)
        for values in ([], [0], [16383], [0, 1, 16383]):
            np.testing.assert_array_equal(unpack_indices(pack_indices(values), len(values)), values)
        for values in ([0.1], [-1], [16384], [np.inf]):
            with self.assertRaises(ValueError):
                pack_indices(values)
        with self.assertRaises(ValueError):
            unpack_indices(b"\0\1", 1)
        with self.assertRaises(ValueError):
            unpack_indices(b"\0", 1)

    def test_malformed_escape_integer(self):
        for data in (b"\x80", b"\x80\0", b"\xff\xff\xff\xff\x7f"):
            with self.assertRaises(ValueError):
                read_signed(data, 0)


class ContainerTests(unittest.TestCase):
    def setUp(self):
        self.header = {"width": 63, "height": 61, "frames": 2, "fps": 5.0, "q": 63,
                       "intra_period": 32, "reset_interval": 32, "revision": "test",
                       "entropy": "test", "table_sha256": "test", "weights": {}, "arithmetic": {}}
        coder = GaussianRans()
        self.frames = []
        for index in range(2):
            kind, z_bytes, parts, symbols = frame_layout(self.header, index)
            self.frames.append({"kind": kind, "z": b"\0" * z_bytes,
                                "y": [coder.encode(np.zeros(symbols), np.zeros(symbols)) for _ in range(parts)],
                                "latent_sha256": "12" * 32, "rgb_sha256": "34" * 32})

    def write(self, path):
        writer = StreamWriter(path, self.header)
        try:
            for frame in self.frames:
                writer.add(frame)
            writer.finish()
        finally:
            writer.close()

    def test_container_roundtrip_and_existing_output_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.glcv"
            self.write(path)
            self.assertEqual(read_stream(path), (self.header, self.frames))
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                self.write(path)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(path.with_name(path.name + ".part").exists())

    def test_incomplete_stream_never_published(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.glcv"
            writer = StreamWriter(path, self.header)
            writer.add(self.frames[0])
            try:
                with self.assertRaises(ValueError):
                    writer.finish()
            finally:
                writer.close()
            self.assertFalse(path.exists())

    def test_header_resource_limits(self):
        for key, value in (("width", 0), ("width", 100000), ("height", True), ("frames", 10001),
                           ("fps", float("nan")), ("fps", 0), ("q", 64), ("intra_period", 0)):
            invalid = dict(self.header, **{key: value})
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_header(invalid)

    def test_corruption_and_truncation_detected_before_neural_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.glcv"
            self.write(path)
            original = path.read_bytes()
            damaged = bytearray(original)
            damaged[len(damaged) // 2] ^= 1
            for data in (original[:-1], bytes(damaged), original + b"x"):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    read_stream(path)

    def test_structural_errors_rejected_even_with_recomputed_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.glcv"
            self.write(path)
            original = path.read_bytes()[:-32]
            record_start = 12 + struct.unpack_from(">I", original, 8)[0] + 4
            variants = []
            wrong_order = bytearray(original)
            struct.pack_into(">I", wrong_order, record_start, 4)
            variants.append(bytes(wrong_order))
            oversized_header = bytearray(original)
            struct.pack_into(">I", oversized_header, 8, 0xFFFFFFFF)
            variants.append(bytes(oversized_header))
            variants.extend([original + b"extra", original[:record_start + 20], b"BADMAGIC" + original[8:]])
            for data in variants:
                path.write_bytes(data + hashlib.sha256(data).digest())
                with self.assertRaises(ValueError):
                    read_stream(path)


if __name__ == "__main__":
    unittest.main()
