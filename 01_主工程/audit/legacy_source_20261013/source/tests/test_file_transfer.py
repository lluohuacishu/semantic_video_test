"""真实 loopback TCP 的限速、完整性和中断检查。"""
from __future__ import annotations

import hashlib
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from file_transfer import loopback_transfer, send_file


class TransferTests(unittest.TestCase):
    def test_real_tcp_transfer_respects_limit_including_header(self):
        with tempfile.TemporaryDirectory() as folder:
            source, received = Path(folder) / "发送文件.bin", Path(folder) / "收到文件.bin"
            source.write_bytes(bytes(range(256)) * 8)
            result = loopback_transfer(source, received, 40, trace=Path(folder) / "trace.csv")
            self.assertEqual(source.read_bytes(), received.read_bytes())
            self.assertEqual(result["sha256"], hashlib.sha256(received.read_bytes()).hexdigest())
            self.assertGreater(result["application_bytes"], result["file_bytes"])
            self.assertGreaterEqual(result["elapsed_seconds"], result["minimum_application_seconds"])
            self.assertLessEqual(result["average_application_kbps"], 40.001)

    def test_cancellation_never_publishes_complete_received_file(self):
        with tempfile.TemporaryDirectory() as folder:
            source, received = Path(folder) / "source.bin", Path(folder) / "received.bin"
            source.write_bytes(b"x" * 2048)
            cancel = threading.Event()
            timer = threading.Timer(0.15, cancel.set)
            timer.start()
            started = time.monotonic()
            try:
                with self.assertRaises(InterruptedError):
                    loopback_transfer(source, received, 1, event=cancel)
            finally:
                timer.cancel()
            self.assertFalse(received.exists())
            self.assertLess(time.monotonic() - started, 3)

    def test_existing_receiver_file_is_preserved_and_failure_is_not_cancellation(self):
        with tempfile.TemporaryDirectory() as folder:
            source, received = Path(folder) / "source.bin", Path(folder) / "received.bin"
            source.write_bytes(b"data")
            received.write_bytes(b"keep-existing")
            event = threading.Event()
            with self.assertRaises((ConnectionError, OSError)):
                loopback_transfer(source, received, 1000, event=event)
            self.assertEqual(received.read_bytes(), b"keep-existing")
            self.assertFalse(event.is_set())

    def test_invalid_rates_fail_before_connecting(self):
        for rate in (float("nan"), float("inf"), 0, -1):
            with self.assertRaises(ValueError):
                send_file(Path("missing.bin"), "127.0.0.1", 1, rate)


if __name__ == "__main__":
    unittest.main()
