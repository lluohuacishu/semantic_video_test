"""Container-v1 compatibility and compact-v2 integrity tests."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

ADAPTER = (Path(__file__).resolve().parents[1] / "04_外部依赖快照" /
           "semantic-video-training" / "scripts")
sys.path.insert(0, str(ADAPTER))

from glc_container import (MAGIC, MAGIC_COMPACT, SequenceChecksums, StreamWriter,
                          container_overhead_breakdown, frame_layout, read_stream)
from glc_video_probe import extract_frames


def example_header(version=1):
    header = {
        "width": 64, "height": 64, "fps": 12.5, "frames": 2, "q": 0,
        "intra_period": 2, "reset_interval": 0, "revision": "r1",
        "entropy": "test-rans-v1", "table_sha256": "table",
        "arithmetic": {"device": "test"}, "weights": {"image": "a"},
    }
    if version == 2:
        header["container_version"] = 2
    return header


def example_frames(header):
    frames = []
    for index in range(header["frames"]):
        kind, z_size, parts, symbols = frame_layout(header, index)
        streams = [symbols.to_bytes(4, "big") + bytes([index + 1]) * 12
                   for _ in range(parts)]
        frames.append({"kind": kind, "z": bytes([index + 3]) * z_size, "y": streams,
                       "latent_sha256": hashlib.sha256(f"latent-{index}".encode()).hexdigest(),
                       "rgb_sha256": hashlib.sha256(f"rgb-{index}".encode()).hexdigest()})
    return frames


class CompactContainerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, name, version):
        header = example_header(version)
        frames = example_frames(header)
        writer = StreamWriter(self.root / name, header, version=version)
        try:
            for frame in frames:
                writer.add(frame)
            writer.finish()
        finally:
            writer.close()
        return header, frames, self.root / name

    def test_v1_round_trip_and_magic_stay_compatible(self):
        header, frames, path = self.write("legacy.glcv", 1)
        self.assertEqual(path.read_bytes()[:8], MAGIC)
        self.assertEqual(read_stream(path), (header, frames))

    def test_v2_round_trip_keeps_ordered_reconstruction_integrity(self):
        header, frames, path = self.write("compact.glcv", 2)
        self.assertEqual(path.read_bytes()[:8], MAGIC_COMPACT)
        decoded_header, packets, integrity = read_stream(path, with_integrity=True)
        self.assertEqual(decoded_header, header)
        self.assertEqual([(f["kind"], f["z"], f["y"]) for f in packets],
                         [(f["kind"], f["z"], f["y"]) for f in frames])
        self.assertTrue(all(f["rgb_sha256"] is None and f["latent_sha256"] is None for f in packets))
        aggregate = SequenceChecksums()
        for index, frame in enumerate(frames):
            aggregate.add(index, frame)
        self.assertEqual(integrity["sequence_reconstruction_checksums"], aggregate.values())
        self.assertEqual(integrity["container_content_digest"], hashlib.sha256(path.read_bytes()[:-32]).hexdigest())

    def test_v2_removes_per_frame_envelopes_but_keeps_file_integrity(self):
        _, _, legacy = self.write("legacy.glcv", 1)
        compact_header, _, compact = self.write("compact.glcv", 2)
        old = container_overhead_breakdown(example_header(1), 1)
        new = container_overhead_breakdown(compact_header, 2)
        self.assertEqual(old["per_frame_reconstruction_hashes"], 128)
        self.assertEqual(new["per_frame_reconstruction_hashes"], 0)
        self.assertEqual(new["whole_file_sha256"], 32)
        self.assertEqual(new["sequence_reconstruction_hashes"], 64)
        self.assertEqual(legacy.stat().st_size - compact.stat().st_size, 70)

    def test_v2_corruption_fails_whole_file_checksum(self):
        _, _, path = self.write("compact.glcv", 2)
        data = bytearray(path.read_bytes())
        data[20] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "checksum failed"):
            read_stream(path)

    def test_stride_samples_original_frame_indices_and_preserves_duration(self):
        class Frame:
            def __init__(self, index, width=6, height=4):
                self.index = index
                self.shape = (height, width, 3)

        class Reader:
            def __init__(self, count):
                self.index, self.count = 0, count

            def isOpened(self):
                return True

            def get(self, prop):
                return 25 if prop == 1 else self.count

            def grab(self):
                if self.index >= self.count:
                    return False
                self.index += 1
                return True

            def read(self):
                if self.index >= self.count:
                    return False, None
                frame = Frame(self.index)
                self.index += 1
                return True, frame

            def release(self):
                return None

        class Cv2:
            CAP_PROP_FPS, CAP_PROP_FRAME_COUNT, COLOR_BGR2RGB, INTER_AREA = 1, 2, 3, 4

            def VideoCapture(self, _path):
                return Reader(10)

            def resize(self, frame, size, interpolation=None):
                return Frame(frame.index, width=size[0], height=size[1])

            def cvtColor(self, frame, _code):
                return frame

        class Image:
            @staticmethod
            def fromarray(frame):
                class Output:
                    def save(self, path):
                        Path(path).write_text(str(frame.index), encoding="ascii")
                return Output()

        source_path = self.root / "synthetic.mkv"
        source_path.touch()
        (self.root / "extract").mkdir()
        result = extract_frames(
            SimpleNamespace(input=source_path, frames=5, start_frame=0, frame_stride=2, width=3),
            self.root / "extract", Cv2(), Image)
        self.assertEqual(result["sampled_source_frame_indices"], [0, 2, 4, 6, 8])
        self.assertEqual((result["width"], result["height"], result["fps"]), (3, 2, 12.5))
        self.assertEqual(result["duration_seconds"], 0.4)
        self.assertEqual([(self.root / "extract/source" / f"im{i:05d}.png").read_text("ascii")
                          for i in range(1, 6)], ["0", "2", "4", "6", "8"])

    def test_changed_frame_or_order_fails_sequence_checksum(self):
        frames = example_frames(example_header())
        expected = SequenceChecksums()
        for index, frame in enumerate(frames):
            expected.add(index, frame)
        for changed in ([frames[1], frames[0]],
                        [dict(frames[0], rgb_sha256="ff" * 32), frames[1]],
                        [dict(frames[0], latent_sha256="ee" * 32), frames[1]]):
            actual = SequenceChecksums()
            for index, frame in enumerate(changed):
                actual.add(index, frame)
            with self.assertRaisesRegex(ValueError, "aggregate differs"):
                actual.verify(expected.values())

    def test_v2_truncation_and_trailing_bytes_fail_after_file_hash_recalculation(self):
        _, _, path = self.write("compact.glcv", 2)
        body = path.read_bytes()[:-32]
        for altered in (body[:-1], body + b"extra"):
            path.write_bytes(altered + hashlib.sha256(altered).digest())
            with self.assertRaises(ValueError):
                read_stream(path)

    def test_header_version_mismatch_is_rejected_before_writing(self):
        with self.assertRaisesRegex(ValueError, "versions disagree"):
            StreamWriter(self.root / "bad.glcv", example_header(1), version=2)

    def test_quality_metric_compares_reduced_decode_to_actual_encoder_input_size(self):
        import numpy as np
        from PIL import Image

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from run_b4_pilot import resolution_and_time_quality

        source, encoder_input, decoded = (self.root / name for name in ("source", "encoder", "decoded"))
        for folder in (source, encoder_input, decoded):
            folder.mkdir()
        for index in range(1, 3):
            Image.new("RGB", (960, 540), (128, 128, 128)).save(source / f"im{index:05d}.png")
            Image.new("RGB", (640, 360), (128, 128, 128)).save(encoder_input / f"im{index:05d}.png")
            Image.new("RGB", (640, 360), (127, 127, 127)).save(decoded / f"im{index:05d}.png")

        result = resolution_and_time_quality(
            source, encoder_input, decoded, stride=1, reference_frames=2, width=640, height=360)
        self.assertEqual(result["frames_compared_at_reference_timeline"], 2)
        self.assertAlmostEqual(
            result["mean_psnr_db_vs_sampled_encoder_input_at_native_resolution"],
            20 * np.log10(255), places=5)


if __name__ == "__main__":
    unittest.main()
