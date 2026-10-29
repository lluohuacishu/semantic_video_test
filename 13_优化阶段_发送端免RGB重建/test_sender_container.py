"""Protocol failures must be rejected before any neural network is run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from sender_container import (MAGIC, PROFILE, ReferenceChecksums, SenderWriter,
                              read_sender_stream, validate_header)
from glc_container import SequenceChecksums, StreamWriter, frame_layout, read_stream


class SenderContainerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'sample.glcsnd'
        self.header = dict(width=64, height=64, frames=2, fps=25, q=0, intra_period=64,
                           reset_interval=32, revision='test', entropy='test', table_sha256='test',
                           weights={}, arithmetic={}, container_version=3,
                           integrity_profile=PROFILE, sender_rgb_reconstructed=False)

    def tearDown(self):
        self.temp.cleanup()

    def frame(self, index):
        kind, size, parts, symbols = frame_layout(self.header, index)
        return dict(kind=kind, z=b'z' * size,
                    y=[struct.pack('>I', symbols) + b'y' * 12 for _ in range(parts)],
                    latent_sha256=hashlib.sha256(str(index).encode()).hexdigest())

    def write(self):
        writer = SenderWriter(self.path, self.header)
        try:
            for index in range(self.header['frames']):
                writer.add(self.frame(index))
            writer.finish()
        finally:
            writer.close()

    def rewrite(self, body):
        self.path.write_bytes(body + hashlib.sha256(body).digest())

    def test_round_trip_and_reference_aggregate(self):
        self.write()
        header, packets, integrity = read_sender_stream(self.path)
        self.assertEqual(header, self.header)
        aggregate = ReferenceChecksums()
        for index, packet in enumerate(packets):
            self.assertEqual(packet['z'], self.frame(index)['z'])
            self.assertEqual(packet['y'], self.frame(index)['y'])
            self.assertIsNone(packet['rgb_sha256'])
            aggregate.add(index, self.frame(index))
        aggregate.verify(integrity['expected_reference_checksums'])
        self.assertFalse(integrity['sender_rgb_checksum_available'])

    def test_no_implicit_integrity_policy(self):
        for patch in [dict(integrity_profile=None), dict(sender_rgb_reconstructed=True),
                      dict(container_version=2), dict(container_version=True)]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_header(dict(self.header, **patch))

    def test_old_reader_refuses_new_format(self):
        self.write()
        with self.assertRaisesRegex(ValueError, 'Unsupported GLC'):
            read_stream(self.path)

    def test_old_writer_still_requires_rgb(self):
        header = dict(self.header, container_version=2)
        del header['integrity_profile']
        del header['sender_rgb_reconstructed']
        writer = StreamWriter(self.path, header, version=2)
        try:
            with self.assertRaisesRegex(ValueError, 'rgb_sha256'):
                writer.add(self.frame(0))
        finally:
            writer.close()

    def test_no_false_sender_rgb_hash(self):
        writer = SenderWriter(self.path, self.header)
        try:
            with self.assertRaisesRegex(ValueError, 'must not claim'):
                writer.add(dict(self.frame(0), rgb_sha256='0' * 64))
        finally:
            writer.close()

    def test_corruption_truncation_and_trailing_bytes(self):
        self.write()
        data = self.path.read_bytes()
        changed = bytearray(data)
        changed[len(changed)//2] ^= 1
        self.path.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            read_sender_stream(self.path)
        self.path.write_bytes(data[:-1])
        with self.assertRaises(ValueError):
            read_sender_stream(self.path)
        self.rewrite(data[:-32] + b'x')
        with self.assertRaisesRegex(ValueError, 'trailing'):
            read_sender_stream(self.path)

    def test_invalid_entropy_structure_with_valid_file_hash(self):
        self.write()
        data = bytearray(self.path.read_bytes()[:-32])
        header_size = struct.unpack_from('>I', data, 8)[0]
        _, z_size, _, _ = frame_layout(self.header, 0)
        offset = 12 + header_size + z_size + 4
        struct.pack_into('>I', data, offset, 1)
        self.rewrite(data)
        with self.assertRaisesRegex(ValueError, 'symbol count'):
            read_sender_stream(self.path)

    def test_duplicate_header_key(self):
        raw = json.dumps(self.header).encode()
        raw = b'{"frames":2,' + raw[1:]
        self.rewrite(MAGIC + struct.pack('>I', len(raw)) + raw)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            read_sender_stream(self.path)

    def test_reference_mismatch_and_frame_order(self):
        expected, actual = ReferenceChecksums(), ReferenceChecksums()
        expected.add(0, self.frame(0))
        with self.assertRaisesRegex(ValueError, 'order'):
            actual.add(1, self.frame(0))
        actual.add(0, self.frame(1))
        with self.assertRaisesRegex(ValueError, 'differ'):
            actual.verify(expected.values())

    def test_incomplete_sequence_and_no_overwrite(self):
        writer = SenderWriter(self.path, self.header)
        try:
            writer.add(self.frame(0))
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                writer.finish()
        finally:
            writer.close()
        self.path.with_name(self.path.name + '.part').unlink()
        self.write()
        with self.assertRaises(FileExistsError):
            SenderWriter(self.path, self.header)

    def test_old_rgb_aggregate_remains_strict(self):
        checks = SequenceChecksums()
        frame = dict(self.frame(0), rgb_sha256='0' * 64)
        checks.add(0, frame)
        expected = checks.values()
        expected['rgb_sha256'] = 'f' * 64
        with self.assertRaises(ValueError):
            checks.verify(expected)


if __name__ == '__main__':
    unittest.main()
