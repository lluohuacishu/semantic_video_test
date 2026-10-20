"""No wall-clock low-rate experiments: pacing uses a virtual clock."""
import hashlib
from pathlib import Path
import socket
import tempfile
import threading
import unittest

from c5_transport import Pacer, acknowledge, file_info, read_header, receive_file, send_file, validate_header


class Clock:
    value = 0.
    def now(self):
        return self.value
    def sleep(self, seconds):
        self.value += seconds


class TransportTests(unittest.TestCase):
    def test_virtual_rate_counts_header_and_payload_without_burst(self):
        for mode, seconds in (("20", 4.), ("40", 2.)):
            clock = Clock()
            pacer = Pacer(mode, clock=clock.now, sleep=clock.sleep)
            for size in (300, 1024, 1024, 7652):
                pacer.wait(size)
                pacer.sent += size
                self.assertAlmostEqual(clock.value, pacer.sent * 8 / (int(mode) * 1000))
            self.assertAlmostEqual(clock.value, seconds)

    def test_max_never_sleeps(self):
        def fail(_):
            self.fail("Unlimited mode must not sleep")
        pacer = Pacer("max", sleep=fail)
        self.assertIsNone(pacer.wait(100000000))

    def test_cancel_before_pacing(self):
        event = threading.Event()
        event.set()
        for mode in ("20", "40", "max"):
            with self.assertRaises(InterruptedError):
                Pacer(mode, event).wait(1024)

    def test_invalid_modes(self):
        for mode in (20, None, "MAX", "0"):
            with self.assertRaises(ValueError):
                Pacer(mode)

    def test_max_tcp_roundtrip_and_ack(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source, output = folder / "input", folder / "output"
            source.write_bytes(bytes(range(256)) * 256)
            errors = []
            info = {}
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                listener.listen()
                def receiver():
                    try:
                        with listener.accept()[0] as connection:
                            header, _ = read_header(connection)
                            info.update(receive_file(connection, header, output))
                            acknowledge(connection, header)
                    except BaseException as exc:
                        errors.append(exc)
                thread = threading.Thread(target=receiver)
                thread.start()
                result = send_file(source, "127.0.0.1", listener.getsockname()[1],
                                   {"id": "a" * 32, "ticket": "b" * 40, "rate_mode": "max"},
                                   trace=folder / "trace.csv")
                thread.join(5)
                self.assertFalse(thread.is_alive())
            self.assertFalse(errors)
            self.assertEqual(source.read_bytes(), output.read_bytes())
            self.assertEqual(result["sha256"], info["sha256"])
            self.assertEqual(result["application_bytes"], result["file_bytes"] + result["header_bytes"])
            self.assertIsNone(result["minimum_application_seconds"])

    def test_corruption_and_duplicate_preserve_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "received.glcv"
            header = {"version": 2, "job_id": "a" * 32, "ticket": "b" * 40,
                      "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest(), "rate_mode": "max"}
            left, right = socket.socketpair()
            try:
                left.sendall(b"xyz")
                with self.assertRaises(ValueError):
                    receive_file(right, header, output)
                self.assertFalse(output.exists())
                self.assertEqual(output.with_suffix(".glcv.part").read_bytes(), b"xyz")
                with self.assertRaises(FileExistsError):
                    receive_file(right, header, output)
            finally:
                left.close()
                right.close()


if __name__ == "__main__":
    unittest.main()
