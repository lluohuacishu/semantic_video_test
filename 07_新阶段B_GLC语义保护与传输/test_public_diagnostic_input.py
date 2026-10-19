"""A single accepted decode is sufficient; incomplete evidence is not."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from b2_common import read, write
from public_diagnostic_input import build_single_decode_input, select_diagnostic_input, THREE_RATE


class DiagnosticInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.sender, self.trial = root / "sender", root / "network/rate_40kbps"
        self.receiver = self.trial / "receiver"
        self.sender.mkdir()
        self.receiver.mkdir(parents=True)
        payload = b"independent decoder input"
        digest = hashlib.sha256(payload).hexdigest()
        (self.sender / "message.scv").write_bytes(payload)
        (self.receiver / "message.scv").write_bytes(payload)
        write(self.trial.parent / "environment.json", {"source_id": "fixture"})
        manifest = {"frames": 1}
        write(self.sender / "manifest.json", manifest)
        write(self.receiver / "received_manifest.json", manifest)
        write(self.receiver / "receive.json", {"verified": True, "source": "actual TCP reception",
              "pid": 11, "bytes": len(payload), "sha256": digest})
        write(self.trial / "transfer.json", {"verified": True, "file_bytes": len(payload),
              "application_bytes": len(payload) + 96, "sha256": digest, "elapsed_seconds": 0.024})
        for name in ("transfer.csv", "send_schedule.csv"):
            (self.trial / name).write_text("preserved original trace", encoding="utf-8")
        write(self.receiver / "completion.json", {"success": True, "pid": 22})
        write(self.receiver / "decode.json", {"receiver_pid": 22, "packet_sha256": digest,
              "frames": 1, "base_rgb_sha256": ["base"], "enhanced_rgb_sha256": ["enhanced"],
              "enhanced_rgb_sequence_sha256": "sequence", "glc_encoder_modules_disabled": True,
              "glc_source_media_reads": []})
        write(self.receiver / "receiver_read_audit.json", {"denied_reads": [],
              "external_source_or_sender_cache_reads": [], "subprocesses": [{"allowed": True}]})
        write(self.trial / "result.json", {"status": "passed", "rate_kbps": 40,
              "application_bytes": len(payload) + 96, "elapsed_seconds": 0.024,
              "frames": 1, "packet_sha256": digest, "enhanced_rgb_sequence_sha256": "sequence",
              "receiver_pid": 22, "playback_verified": True, "outside_roi_unchanged": True,
              "received_patch_matches_composite": True, "external_source_or_cache_reads": 0})
        self.summary_path = root / "input.json"

    def prepare(self):
        value = build_single_decode_input(self.sender, self.trial, "fixture")
        write(self.summary_path, value)
        return value

    def test_one_accepted_decode_unlocks_diagnostics_without_other_rates(self):
        value = self.prepare()
        selected, _, directory, rate = select_diagnostic_input(self.summary_path)
        self.assertEqual(directory, self.trial)
        self.assertEqual(rate, 40)
        self.assertEqual(len(selected["results"]), 1)
        self.assertFalse(selected["three_rate_engineering_claim"])
        self.assertFalse(selected["new_transmission"])
        self.assertEqual(value["time_estimates"]["seconds_by_kbps"]["20"],
                         8 * selected["results"][0]["application_bytes"] / 20000)
        self.assertFalse((self.trial.parent / "rate_20kbps").exists())

    def test_completed_decoder_without_acceptance_is_rejected(self):
        result = read(self.trial / "result.json")
        result["status"] = "pending"
        write(self.trial / "result.json", result)
        with self.assertRaisesRegex(ValueError, "not accepted"):
            self.prepare()

    def test_tampered_packet_is_rejected(self):
        (self.receiver / "message.scv").write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.prepare()

    def test_missing_decode_completion_is_rejected(self):
        write(self.receiver / "completion.json", {"success": False, "pid": 22})
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.prepare()

    def test_wrong_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "source identity"):
            build_single_decode_input(self.sender, self.trial, "different_clip")

    def test_single_trial_cannot_masquerade_as_three_rate_summary(self):
        value = self.prepare()
        value["status"] = THREE_RATE
        write(self.summary_path, value)
        with self.assertRaisesRegex(ValueError, "Three-rate evidence incomplete"):
            select_diagnostic_input(self.summary_path)

    def test_changed_result_is_rejected_after_selection(self):
        self.prepare()
        result = read(self.trial / "result.json")
        result["review_note"] = "changed after selection"
        write(self.trial / "result.json", result)
        with self.assertRaisesRegex(ValueError, "accepted result changed"):
            select_diagnostic_input(self.summary_path)


if __name__ == "__main__":
    unittest.main()
