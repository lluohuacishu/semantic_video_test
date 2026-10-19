import os
import struct
import unittest

import numpy as np

from c2_native import NativeGaussianRans, GaussianRans, MAX_SYMBOLS


@unittest.skipUnless(os.environ.get('GLC_RANS_DLL'), 'Set GLC_RANS_DLL to a freshly built, fingerprinted library')
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = GaussianRans()
        cls.native = NativeGaussianRans(os.environ['GLC_RANS_DLL'], cls.reference)

    def cross(self, values, indexes):
        python = self.reference.encode(values, indexes)
        native = self.native.encode(values, indexes)
        self.assertEqual(python, native)
        for backend in (self.reference, self.native):
            np.testing.assert_array_equal(backend.decode(python, indexes), np.asarray(values, dtype=np.int32))
            np.testing.assert_array_equal(backend.decode(native, indexes), np.asarray(values, dtype=np.int32))

    def rejection_parity(self, stream, indexes):
        results = []
        for backend in (self.reference, self.native):
            try:
                results.append(backend.decode(stream, indexes))
            except ValueError:
                results.append(None)
        self.assertEqual(results[0] is None, results[1] is None)
        if results[0] is not None:
            np.testing.assert_array_equal(*results)

    def test_same_tables_and_indexes(self):
        self.assertEqual(self.reference.table_sha256, self.native.table_sha256)
        scales = np.r_[-1, 0, np.geomspace(0.001, 1e5, 10000)]
        np.testing.assert_array_equal(self.reference.indexes(scales), self.native.indexes(scales))

    def test_empty_payload(self):
        self.cross([], [])

    def test_all_tables_boundaries_and_int32_escapes(self):
        for table, radius in enumerate(self.reference.radii):
            values = [-2**31, -radius-1, -radius, -1, 0, 1, radius, radius+1, 2**31-1]
            self.cross(values, [table] * len(values))

    def test_mixed_tables_random_roundtrips(self):
        rng = np.random.default_rng(20261019)
        for count in (1, 2, 127, 1024, 20000):
            indexes = rng.integers(0, 128, count, dtype=np.int16)
            values = rng.integers(-5000, 5001, count, dtype=np.int32)
            self.cross(values, indexes)

    def test_highly_compressible_and_worst_renormalization(self):
        for value in (0, 2, -2**31, 2**31-1):
            self.cross([value] * 4096, [0] * 4096)

    def test_noncontiguous_inputs(self):
        values = np.arange(-100, 100, dtype=np.int32)[::3]
        indexes = np.arange(400, dtype=np.int32)[::6][:len(values)] % 128
        self.cross(values, indexes)

    def test_invalid_indexes_and_symbols(self):
        for backend in (self.reference, self.native):
            for indexes in ([-1], [128], [1.5], [np.nan], [np.inf]):
                with self.assertRaises(ValueError):
                    backend.encode([0], indexes)
                with self.assertRaises(ValueError):
                    backend.decode(b'', indexes)
            for values in ([np.nan], [np.inf], [0.1], [-2**31-1], [2**31], []):
                with self.assertRaises(ValueError):
                    backend.encode(values, [0])
        with self.assertRaises(ValueError):
            self.native.encode([], np.zeros(MAX_SYMBOLS + 1, dtype=np.uint8))

    def test_every_truncation_and_header_corruption(self):
        indexes = np.arange(128, dtype=np.int16)
        data = self.reference.encode(np.arange(-64, 64), indexes)
        for cut in range(len(data)):
            self.rejection_parity(data[:cut], indexes)
        for field in (0, 4, 8, 12):
            for value in (0, 1, 2**23 - 1, 2**31, 2**32-1):
                changed = bytearray(data)
                struct.pack_into('<I' if field == 12 else '>I', changed, field, value)
                self.rejection_parity(changed, indexes)
        self.rejection_parity(data + b'\0', indexes)

    def test_noncanonical_escapes(self):
        data = self.reference.encode([2], [0])
        for suffix in (b'\0', b'\x84\0', b'\xff\xff\xff\xff\x10', b'\xff' * 6):
            damaged = data[:-1] + suffix
            for backend in (self.reference, self.native):
                with self.assertRaises(ValueError):
                    backend.decode(damaged, [0])

    def test_seeded_mutation_acceptance_and_output_match(self):
        rng = np.random.default_rng(519)
        indexes = rng.integers(0, 128, 256, dtype=np.int16)
        original = self.reference.encode(rng.integers(-2500, 2500, 256), indexes)
        for _ in range(1500):
            data = bytearray(original)
            position = int(rng.integers(0, len(data)))
            data[position] ^= 1 << int(rng.integers(0, 8))
            self.rejection_parity(data, indexes)


if __name__ == '__main__':
    unittest.main()
