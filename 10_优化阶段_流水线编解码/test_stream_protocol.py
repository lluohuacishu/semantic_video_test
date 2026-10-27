import hashlib
import struct
import unittest
import zlib

import stream_protocol as p


class Fragmented:
    def __init__(self, raw):
        self.raw = raw

    def recv(self, count):
        value, self.raw = self.raw[:min(count, 7)], self.raw[min(count, 7):]
        return value


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.header = dict(width=640, height=360, fps=12.5, frames=66, q=0,
            intra_period=64, reset_interval=32, revision='fixture', entropy='fixture',
            table_sha256='fixture', weights={}, arithmetic={}, container_version=2)

    def packet(self, index):
        kind, z_size, parts, symbols = p.frame_layout(self.header, index)
        return dict(kind=kind, z=bytes(i % 256 for i in range(z_size)),
            y=[struct.pack('!IIII', symbols, 4, 0, 123 + j) for j in range(parts)],
            latent_sha256='ab' * 32, rgb_sha256='cd' * 32)

    def test_fragmented_packets_both_versions(self):
        for version in (1, 2):
            raw = p.encode_record(p.HEADER, 0, p.json_bytes(self.header), version)
            for i in (0, 1, 32, 64):
                raw += p.encode_record(p.FRAME, i, p.encode_frame(self.header, i, self.packet(i), version), version)
            conn = Fragmented(raw)
            self.assertEqual(p.decode_header(p.receive_expected(conn, p.HEADER, 0, version)[0]), self.header)
            for i in (0, 1, 32, 64):
                body, _ = p.receive_expected(conn, p.FRAME, i, version)
                got = p.decode_frame(self.header, i, body, version)
                for key in ('kind', 'z', 'y'):
                    self.assertEqual(got[key], self.packet(i)[key])

    def test_every_single_bit_in_frame_record_rejected(self):
        raw = p.encode_record(p.FRAME, 1, p.encode_frame(self.header, 1, self.packet(1)))
        for bit in range(len(raw) * 8):
            damaged = bytearray(raw)
            damaged[bit // 8] ^= 1 << (bit % 8)
            with self.assertRaises((ValueError, EOFError), msg=str(bit)):
                p.receive_expected(Fragmented(damaged), p.FRAME, 1)

    def test_control_hash(self):
        raw = bytearray(p.encode_record(p.END, 66, b'{}'))
        raw[-33] ^= 1
        with self.assertRaisesRegex(ValueError, 'integrity'):
            p.receive_record(Fragmented(raw))

    def test_invalid_length_rejected_before_body(self):
        prefix = p.COMPACT_PREFIX.pack(b'GSP2', p.FRAME, 0, p.MAX_RECORD + 1)
        raw = prefix + struct.pack('!I', zlib.crc32(prefix))
        with self.assertRaisesRegex(ValueError, 'header'):
            p.receive_record(Fragmented(raw))

    def test_all_truncated_positions(self):
        raw = p.encode_record(p.END, 66, b'{}')
        for cut in range(len(raw)):
            with self.assertRaises(EOFError):
                p.receive_record(Fragmented(raw[:cut]))

    def test_duplicate_and_reordered_frames(self):
        raw = p.encode_record(p.FRAME, 1, p.encode_frame(self.header, 1, self.packet(1)))
        for wanted in (0, 2):
            with self.assertRaisesRegex(ValueError, 'reordered'):
                p.receive_expected(Fragmented(raw), p.FRAME, wanted)

    def test_version_mismatch(self):
        raw = p.encode_record(p.END, 66, b'{}', version=1)
        with self.assertRaisesRegex(ValueError, 'version'):
            p.receive_record(Fragmented(raw), version=2)

    def test_missing_end(self):
        with self.assertRaises(EOFError):
            p.receive_expected(Fragmented(b''), p.END, 66)

    def test_trailer_count_hash_and_structure(self):
        transcript = hashlib.sha256(b'frames').hexdigest()
        value = dict(frames=66, frame_records_sha256=transcript, archive_sha256='ab' * 32)
        p.validate_trailer(value, 66, transcript)
        for update in (dict(frames=65), dict(frames=True), dict(frame_records_sha256='cd' * 32),
                       dict(archive_sha256='broken'), dict(extra=1)):
            with self.assertRaises(ValueError):
                p.validate_trailer({**value, **update}, 66, transcript)

    def test_compact_payload_excludes_diagnostic_hashes(self):
        packet = self.packet(1)
        a = p.encode_frame(self.header, 1, packet, 1)
        b = p.encode_frame(self.header, 1, packet, 2)
        self.assertEqual(len(a) - len(b), 78)
        got = p.decode_frame(self.header, 1, b, 2)
        self.assertIsNone(got['rgb_sha256'])
        self.assertIsNone(got['latent_sha256'])
        with self.assertRaises(ValueError):
            p.decode_frame(self.header, 1, b + b'garbage', 2)


if __name__ == '__main__':
    unittest.main()
