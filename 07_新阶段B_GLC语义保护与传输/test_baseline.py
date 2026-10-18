"""Focused checks for B1 evidence validity and source-read constraints."""
from __future__ import annotations
import csv
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest

from receiver_baseline import ReadPolicy
from run_b0_b1 import validate_schedule


class ReadGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "receiver"
        self.output.mkdir()
        self.model = self.root / "weights/model.pth.tar"
        self.config = {"shared_models": {str(self.model): "expected"}, "code_hashes": {},
                       "glc_repo": str(self.root / "decoder_code"), "ffmpeg": str(self.root / "ffmpeg.exe")}
        self.policy = ReadPolicy(self.output, self.config)

    def test_source_reads_and_readwrite_modes_are_denied(self):
        source = self.root / "source.png"
        for mode, flags in (("r", 0), ("rb", 0), ("r+b", os.O_RDWR), ("w+b", os.O_RDWR), (None, os.O_RDONLY), (None, os.O_RDWR)):
            with self.subTest(mode=mode, flags=flags):
                with self.assertRaises(PermissionError):
                    self.policy.hook("open", (os.fsencode(source), mode, flags))

    def test_model_allowlist_is_exact(self):
        self.assertEqual(self.policy.classify(self.model), "declared_shared_model")
        self.assertIsNone(self.policy.classify(self.model.with_name("unlisted.pth.tar")))
        self.assertIsNone(self.policy.classify(self.root / "decoder_code/cached_source.npz"))

    def test_received_file_and_outputs_allowed(self):
        self.assertEqual(self.policy.classify(self.output / "received.glcv"), "receiver_directory")
        self.policy.hook("open", (str(self.output / "frame.png"), "w+b", os.O_RDWR))
        self.assertEqual(len(self.policy.denied), 0)

    def test_subprocess_cannot_open_sender_media(self):
        with self.assertRaises(PermissionError):
            self.policy.hook("subprocess.Popen", (self.config["ffmpeg"], [self.config["ffmpeg"], "-i", str(self.root / "source.mp4"), str(self.output / "preview.mp4")], None, {}))


class PacingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def schedule(self, start, finish, count=100):
        path = self.root / "schedule.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("chunk_bytes", "cumulative_application_bytes", "scheduled_seconds", "send_started_seconds", "send_finished_seconds"))
            writer.writerow((count, count, count * 8 / 20000, start, finish))
        return path

    def test_early_start_cannot_be_hidden_by_late_finish(self):
        with self.assertRaisesRegex(ValueError, "before"):
            validate_schedule(self.schedule(0.01, 0.05), 20, 100, 1e-7)

    def test_missing_header_bytes_rejected(self):
        with self.assertRaisesRegex(ValueError, "omits"):
            validate_schedule(self.schedule(0.04, 0.05), 20, 200, 1e-7)

    def test_on_time_and_clock_precision_reported(self):
        result = validate_schedule(self.schedule(0.04, 0.05), 20, 100, 1e-7)
        self.assertEqual(result["maximum_early_seconds"], 0)

    def test_real_small_transfer_records_header_and_complete_payload(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_主工程/semantic_mvp"))
        from transport import send_file, receive_connection
        source, dest = self.root / "input.bin", self.root / "received.bin"
        source.write_bytes(bytes(range(256)) * 9)
        errors = []
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(5)
            def receive():
                try:
                    connection, _ = listener.accept()
                    with connection:
                        receive_connection(connection, dest)
                except Exception as exc:
                    errors.append(exc)
            worker = threading.Thread(target=receive, daemon=True)
            worker.start()
            result = send_file(source, "127.0.0.1", listener.getsockname()[1], 1000,
                               trace=self.root / "transfer.csv", schedule_trace=self.root / "calls.csv")
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
        self.assertFalse(errors)
        self.assertEqual(source.read_bytes(), dest.read_bytes())
        self.assertGreater(result["application_bytes"], result["file_bytes"])
        verified = validate_schedule(self.root / "calls.csv", 1000, result["application_bytes"], 1e-7)
        self.assertEqual(verified["calls_including_header"], 4)


if __name__ == "__main__":
    unittest.main()
