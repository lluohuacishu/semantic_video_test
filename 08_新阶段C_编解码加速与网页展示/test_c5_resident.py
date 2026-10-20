"""Supervisor mailbox tests with a fake compute worker; no neural inference."""
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from c5_resident import ResidentService, read_json, write


class ResidentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.service = ResidentService("decode", {"python": "unused"}, self.root)
        self.launches = 0
        self.seen = []
        self.patch_supervise = patch("c5_resident.supervise", self.worker)
        self.patch_inspect = patch("c5_resident.inspect", return_value={"observed_state": "completed"})
        self.patch_supervise.start()
        self.patch_inspect.start()

    def tearDown(self):
        self.service.close()
        self.patch_supervise.stop()
        self.patch_inspect.stop()
        self.temp.cleanup()

    def worker(self, argv, folder, cwd, stop_file, poll_seconds):
        self.launches += 1
        mailbox = folder.parent / "mailbox"
        seen = set()
        while not stop_file.exists():
            for path in mailbox.glob("*.request.json"):
                if path.name in seen:
                    continue
                seen.add(path.name)
                task = read_json(path)["task"]
                self.seen.append(task["id"])
                if task.get("hang"):
                    continue
                result = {"state": "failed", "message": "injected failure"} if task.get("fail") else {"state": "done"}
                write(path.with_name(path.name.replace(".request.json", ".response.json")), result)
                if task.get("fail"):
                    return {"returncode": 1}
            time.sleep(.01)
        return {"returncode": 0}

    def folder(self, name):
        path = self.root / name
        path.mkdir()
        return path

    def test_lazy_single_worker_multiple_tasks(self):
        self.assertEqual(0, self.launches)
        for name in ("one", "two", "three"):
            self.service.execute({"id": name}, self.folder(name))
        self.assertEqual(1, self.launches)
        self.assertEqual(["one", "two", "three"], self.seen)
        self.service.close()
        self.assertIsNone(self.service.future)

    def test_failure_discards_worker_then_next_task_recovers(self):
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            self.service.execute({"id": "bad", "fail": True}, self.folder("bad"))
        self.service.execute({"id": "good"}, self.folder("good"))
        self.assertEqual(2, self.launches)

    def test_cancellation_reaps_worker_before_next_dispatch(self):
        folder = self.folder("cancel")
        def cancel():
            while not self.seen:
                time.sleep(.01)
            (folder / "STOP_REQUESTED").touch()
        thread = threading.Thread(target=cancel)
        thread.start()
        with self.assertRaises(InterruptedError):
            self.service.execute({"id": "cancel", "hang": True}, folder)
        thread.join()
        self.assertIsNone(self.service.future)
        self.service.execute({"id": "after"}, self.folder("after"))
        self.assertEqual(2, self.launches)

    def test_uncertain_worker_blocks_reuse(self):
        self.service.execute({"id": "one"}, self.folder("one"))
        (self.service.folder / "STOP_REQUESTED").touch()
        with patch("c5_resident.inspect", return_value={"observed_state": "liveness_unknown"}):
            with self.assertRaisesRegex(RuntimeError, "uncertain"):
                self.service.reap()
            self.assertIsNotNone(self.service.future)


if __name__ == "__main__":
    unittest.main()
