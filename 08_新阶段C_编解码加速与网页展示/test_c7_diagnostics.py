"""No model inference: evidence binding, intact prefix payloads, and failure reports."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "04_外部依赖快照/semantic-video-training/scripts"))
import glc_container as container
from c7_diagnostics import sha256, sequence_report, validate_reference, verify_sequence, write_json
from c7_make_samples import make_samples


class C7Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.header = {"width": 64, "height": 64, "frames": 3, "fps": 15.0, "q": 0,
                       "intra_period": 64, "reset_interval": 32, "revision": "fixture",
                       "entropy": "fixture", "table_sha256": "fixture", "weights": {},
                       "arithmetic": {"device": "fixture"}, "container_version": 2}
        self.stream = self.root / "original.glcv"
        self.frames = []
        writer = container.StreamWriter(self.stream, self.header, version=2)
        try:
            for i in range(3):
                kind, z_size, parts, symbols = container.frame_layout(self.header, i)
                hashes = {k: hashlib.sha256(f"{i}:{k}".encode()).hexdigest()
                          for k in ("rgb_sha256", "latent_sha256")}
                writer.add({"kind": kind, "z": bytes([i]) * z_size,
                            "y": [struct.pack(">I", symbols) + bytes([i]) * 12] * parts, **hashes})
                self.frames.append({"frame": i + 1, **hashes})
            writer.finish()
        finally:
            writer.close()
        self.record = self.root / "encode.json"
        self.encoded = {"file_sha256": sha256(self.stream), "header": self.header, "frames": self.frames}
        write_json(self.record, self.encoded)
        self.integrity = container.read_stream(self.stream, with_integrity=True)[2]

    def test_samples_keep_exact_compressed_payload_and_bind_sidecar(self):
        result = make_samples(self.stream, self.record, self.root / "samples", [1, 3], container)
        original = container.read_stream(self.stream)[1]
        self.assertFalse(result["gpu_used"])
        for row in result["samples"]:
            path = self.root / "samples" / row["filename"]
            header, packets, integrity = container.read_stream(path, with_integrity=True)
            self.assertEqual(packets, original[:row["frames"]])
            reference = json.loads(path.with_suffix(".reference.json").read_text())
            records = validate_reference(reference, sha256(path), header, integrity, container)
            self.assertEqual(records, self.frames[:row["frames"]])

    def test_bad_encoder_record_is_rejected_before_creating_output(self):
        for change in ("file", "frame_hash", "order"):
            with self.subTest(change=change):
                record = json.loads(json.dumps(self.encoded))
                if change == "file": record["file_sha256"] = "0" * 64
                elif change == "frame_hash": record["frames"][1]["rgb_sha256"] = "0" * 64
                else: record["frames"].reverse()
                write_json(self.record, record)
                with self.assertRaises(ValueError):
                    make_samples(self.stream, self.record, self.root / change, [1], container)
                self.assertFalse((self.root / change).exists())

    def test_sample_rejects_overwrite_and_out_of_bounds(self):
        for counts in ([0], [4], [65], []):
            with self.assertRaises(ValueError):
                make_samples(self.stream, self.record, self.root / "invalid", counts, container)
        with self.assertRaises(FileExistsError):
            make_samples(self.stream, self.record, self.root, [1], container)

    def test_wrong_stream_and_altered_reference_are_rejected(self):
        reference = {"schema": "c7-frame-reference-v1", "stream_sha256": sha256(self.stream),
                     "header": self.header, "frames": self.frames}
        with self.assertRaisesRegex(ValueError, "exact stream"):
            validate_reference(reference, "0" * 64, self.header, self.integrity, container)
        reference["frames"][1]["latent_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "aggregate differs"):
            validate_reference(reference, sha256(self.stream), self.header, self.integrity, container)

    def test_mismatch_reports_specific_channel_without_allowing_success(self):
        checksums = container.SequenceChecksums()
        for i, frame in enumerate(self.frames): checksums.add(i, frame)
        expected = {**checksums.values(), "latent_sha256": "0" * 64}
        output = self.root / "failure_details.json"
        with self.assertRaisesRegex(ValueError, "aggregate differs"):
            verify_sequence(checksums, expected, output, header=self.header,
                            receiver_profile={"device": "another"}, records=self.frames,
                            stream_sha256=sha256(self.stream))
        report = json.loads(output.read_text())
        self.assertEqual(report["mismatched_channels"], ["latent_sha256"])
        self.assertTrue(report["sequence_matches"]["rgb_sha256"])
        self.assertFalse(report["first_mismatch_known"])
        self.assertFalse(report["cross_device_accepted"])
        self.assertEqual(report["frames"], self.frames)

    def test_prefix_digest_is_not_compared_to_full_sequence(self):
        report = sequence_report(self.header, {}, self.integrity["sequence_reconstruction_checksums"],
                                 container.SequenceChecksums().values(), [], sha256(self.stream))
        self.assertFalse(report["complete_sequence"])
        self.assertIsNone(report["sequence_matches"])
        self.assertIsNone(report["mismatched_channels"])

    def test_matching_aggregate_preserves_success_without_failure_file(self):
        checksums = container.SequenceChecksums()
        for i, frame in enumerate(self.frames): checksums.add(i, frame)
        output = self.root / "failure_details.json"
        verify_sequence(checksums, checksums.values(), output, header=self.header,
                        receiver_profile={}, records=self.frames, stream_sha256=sha256(self.stream))
        self.assertFalse(output.exists())

    def probe(self, name, fail_frame=None):
        import numpy as np
        from c1_runtime import receiver_audit
        from c7_decode_probe import run_probe
        frames, header = self.frames, self.header

        class Tensor:
            def detach(self): return self
            def cpu(self): return self
            def contiguous(self): return self
            def numpy(self): return np.zeros((1, 2, 2), dtype=np.float32)

        class Codec:
            def __init__(self, *args, **kwargs):
                self.profile = header["arithmetic"]
                self.entropy = SimpleNamespace(table_sha256="fixture", decode=lambda *a: np.array([1], dtype=np.int32))
                self.dpb = {"ref_latent": Tensor(), "ref_y": None}

            def frame(self, header, index, packet=None):
                self.entropy.decode(b"fixture", np.array([1], dtype=np.uint16))
                checked = dict(frames[index])
                if index + 1 == fail_frame: checked["latent_sha256"] = "0" * 64
                return np.zeros((64, 64, 3), dtype=np.uint8), checked

        fake_ref = SimpleNamespace(REVISION="fixture", ENTROPY_ID="fixture", ASSETS={}, NeuralCodec=Codec)
        fake_runtime = SimpleNamespace(ref=fake_ref, receiver_audit=receiver_audit)
        project = Path(__file__).resolve().parent.parent
        packets = container.read_stream(self.stream)[1]
        inputs = (project, {"repo": self.root, "weights": self.root}, container,
                  self.header, packets, self.integrity, self.frames)
        with patch.dict(sys.modules, {"c1_runtime": fake_runtime}), patch("c7_decode_probe.subprocess.run", side_effect=OSError):
            status = run_probe(inputs, self.stream, self.root / name, "python")
        return status, json.loads((self.root / name / "report.json").read_text(encoding="utf-8"))

    def test_probe_stops_at_first_divergence_and_retains_snapshots(self):
        status, report = self.probe("mismatch", fail_frame=2)
        self.assertEqual(status, 1)
        self.assertEqual(report["state"], "failed")
        self.assertEqual(report["reconstruction"]["first_mismatch_frame"], 2)
        self.assertEqual(report["reconstruction"]["completed_frames"], 2)
        self.assertIsNone(report["reconstruction"]["sequence_matches"])
        self.assertEqual(len(report["snapshots"]), 2)
        self.assertEqual(report["source_media_reads"], [])
        self.assertFalse(report["cross_device_accepted"])

    def test_probe_pass_does_not_claim_full_cross_device_acceptance(self):
        status, report = self.probe("pass")
        self.assertEqual(status, 0)
        self.assertEqual(report["state"], "sample_passed")
        self.assertTrue(all(report["reconstruction"]["sequence_matches"].values()))
        self.assertEqual(len(report["snapshots"]), 1)
        self.assertFalse(report["cross_device_accepted"])


if __name__ == "__main__":
    unittest.main()
