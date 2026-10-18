"""Guard recovery provenance: completed TCP bytes must survive unchanged."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from b2_common import read, write
from continue_b3_public import copy_receipt, validate_receipt


class PublicRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.old, self.new, self.packet = root / "old", root / "new", root / "sender.scv"
        (self.old / "receiver").mkdir(parents=True)
        payload = b"complete received packet\x00\xff"
        self.packet.write_bytes(payload)
        (self.old / "receiver/message.scv").write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        write(self.old / "receiver/receive.json", {"verified": True, "source": "actual TCP reception",
              "pid": 1234, "bytes": len(payload), "sha256": digest})
        write(self.old / "transfer.json", {"verified": True, "file_bytes": len(payload),
              "application_bytes": len(payload) + 96, "sha256": digest})
        for name in ("transfer.csv", "send_schedule.csv"):
            (self.old / name).write_bytes(b"original,trace\r\n1,2\r\n")

    def test_receipt_copy_preserves_source_and_exact_traces(self):
        snapshots = {p.relative_to(self.old): p.read_bytes() for p in self.old.rglob("*") if p.is_file()}
        copy_receipt(self.old, self.new, self.packet)
        for relative, payload in snapshots.items():
            self.assertEqual((self.old / relative).read_bytes(), payload)
            self.assertEqual((self.new / relative).read_bytes(), payload)
        self.assertFalse(read(self.new / "recovery_provenance.json")["retransmission"])

    def test_tampered_received_bytes_refused(self):
        (self.old / "receiver/message.scv").write_bytes(b"different data")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_receipt(self.old, self.packet)

    def test_unacknowledged_transfer_refused(self):
        transfer = read(self.old / "transfer.json")
        transfer["verified"] = False
        write(self.old / "transfer.json", transfer)
        with self.assertRaisesRegex(ValueError, "Unverified"):
            validate_receipt(self.old, self.packet)

    def test_missing_original_schedule_refused(self):
        (self.old / "send_schedule.csv").unlink()
        with self.assertRaisesRegex(ValueError, "traces missing"):
            validate_receipt(self.old, self.packet)


if __name__ == "__main__":
    unittest.main()
