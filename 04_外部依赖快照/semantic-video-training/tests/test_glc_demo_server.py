"""GLC 网页的任务隔离、上传取消、重跑和媒体访问检查；无需 GPU。"""
from __future__ import annotations

import http.client
from contextlib import closing
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glc_demo_server import GLCApp, GLCServer, GLCStore, Handler, parse_options


class FakeApp(GLCApp):
    def initialize(self):
        self.model_info = {"state": "ready", "message": "test", "gpu": "test"}

    def run(self, identifier, source):
        if self.store.events[identifier].is_set():
            self.store.update(identifier, status="cancelled", phase="stopped")
            return
        folder = self.result_folder(identifier) / "sender"
        folder.mkdir(parents=True)
        (folder / "source_preview.mp4").write_bytes(bytes(range(100)))
        self.store.update(identifier, status="done", phase="test completed", input_bytes=source.read_bytes().hex())


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.app = FakeApp(self.temporary.name)
        self.app.examples = {}
        self.server = GLCServer(("127.0.0.1", 0), self.app)
        self.port = self.server.server_address[1]
        self.worker = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        self.worker.start()
        deadline = time.monotonic() + 2
        while self.app.model_info["state"] != "ready" and time.monotonic() < deadline:
            time.sleep(0.01)

    def tearDown(self):
        self.server.shutdown()
        self.worker.join(timeout=2)
        self.server.server_close()
        self.app.close()
        self.temporary.cleanup()

    def request(self, method, path, body=None, headers=None):
        with closing(http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)) as connection:
            connection.request(method, path, body, headers or {})
            response = connection.getresponse()
            data = response.read()
            return response.status, data, dict(response.getheaders())

    def reserve(self, name="水下.MP4", size=8, **options):
        status, body, _ = self.request("POST", "/api/jobs/reserve", json.dumps({
            "filename": name, "bytes": size, "options": options}).encode(), {"Content-Type": "application/json"})
        return status, json.loads(body)

    def await_done(self, identifier):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            job = self.app.store.get(identifier)
            if job["status"] == "done":
                return job
            time.sleep(0.01)
        self.fail(str(self.app.store.get(identifier)))

    def completed(self):
        status, job = self.reserve()
        self.assertEqual(status, 202)
        status, _, _ = self.request("POST", f"/api/jobs/{job['id']}/source", b"testdata")
        self.assertEqual(status, 202)
        return self.await_done(job["id"])

    def test_registered_upload_isolates_original_name_and_bytes(self):
        job = self.completed()
        self.assertEqual(job["filename"], "水下.MP4")
        self.assertEqual(job["input_bytes"], b"testdata".hex())
        source = Path(self.temporary.name) / job["id"] / "source.mp4"
        self.assertEqual(source.read_bytes(), b"testdata")
        status, body, _ = self.request("GET", f"/api/runs/{job['id']}")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "done")

    def test_cancel_before_file_upload_never_queues_job(self):
        _, job = self.reserve()
        status, body, _ = self.request("POST", f"/api/jobs/{job['id']}/cancel")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "cancelled")
        status, _, _ = self.request("POST", f"/api/jobs/{job['id']}/source", b"testdata")
        self.assertEqual(status, 400)
        self.assertEqual(self.app.store.get(job["id"])["status"], "cancelled")
        self.assertFalse((Path(self.temporary.name) / job["id"] / "source.mp4").exists())

    def test_wrong_upload_length_cannot_replace_registered_file(self):
        _, job = self.reserve(size=8)
        status, _, _ = self.request("POST", f"/api/jobs/{job['id']}/source", b"short")
        self.assertEqual(status, 400)
        self.assertEqual(self.app.store.get(job["id"])["status"], "uploading")
        self.assertFalse((Path(self.temporary.name) / job["id"] / "source.mp4").exists())

    def test_duplicate_upload_and_rerun_preserve_original(self):
        first = self.completed()
        status, _, _ = self.request("POST", f"/api/jobs/{first['id']}/source", b"changed!")
        self.assertEqual(status, 400)
        self.assertEqual(self.app.store.get(first["id"])["status"], "done")
        status, body, _ = self.request("POST", f"/api/jobs/{first['id']}/rerun", b'{"q":0,"kbps":40}')
        self.assertEqual(status, 202)
        second = self.await_done(json.loads(body)["id"])
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(second["input_bytes"], first["input_bytes"])
        self.assertEqual(second["options"]["q"], 0)
        self.assertEqual(self.app.store.get(first["id"])["options"]["q"], 63)

    def test_video_range_head_and_download(self):
        job = self.completed()
        path = f"/media/{job['id']}/input.mp4?download=1"
        status, body, headers = self.request("GET", path, headers={"Range": "bytes=10-19"})
        self.assertEqual((status, body), (206, bytes(range(10, 20))))
        self.assertEqual(headers["Content-Range"], "bytes 10-19/100")
        self.assertIn("attachment", headers["Content-Disposition"])
        status, body, headers = self.request("HEAD", path)
        self.assertEqual((status, body, headers["Content-Length"]), (200, b"", "100"))
        self.assertEqual(self.request("GET", path, headers={"Range": "bytes=999-"})[0], 416)

    def test_lost_upload_response_does_not_fail_submitted_job(self):
        _, job = self.reserve()
        original = Handler.json
        completed = threading.Event()
        original_run = self.app.run

        def run_and_signal(identifier, source):
            original_run(identifier, source)
            completed.set()

        def disconnected_response(handler, data, status=200):
            if handler.path.endswith("/source"):
                self.assertTrue(completed.wait(2))
                raise BrokenPipeError("browser closed after uploading")
            return original(handler, data, status)

        with patch.object(self.app, "run", run_and_signal), patch.object(Handler, "json", disconnected_response):
            with self.assertRaises(http.client.RemoteDisconnected):
                self.request("POST", f"/api/jobs/{job['id']}/source", b"testdata")
        self.assertEqual(self.app.store.get(job["id"])["status"], "done")

    def test_frame_pagination(self):
        job = self.completed()
        encoded = [{"frame": i, "type": "I" if i == 1 else "P", "record_bytes": 123,
                    "psnr_db": 30.0, "seconds": 1.0} for i in range(1, 61)]
        decoded = [{"frame": i, "seconds": 0.5} for i in range(1, 61)]
        path = Path(self.temporary.name) / job["id"] / "result/summary.json"
        path.write_text(json.dumps({"results": [{"encode": {"frames": encoded}, "decode": {"frames": decoded}}]}), encoding="utf-8")
        status, body, _ = self.request("GET", f"/api/runs/{job['id']}/frames?offset=50")
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual((data["total"], len(data["rows"]), data["rows"][0]["frame"]), (60, 10, 51))
        self.assertEqual(self.request("GET", f"/api/runs/{job['id']}/frames?offset=-1")[0], 400)

    def test_local_origin_and_media_allowlist(self):
        self.assertEqual(self.request("GET", "/api/config", headers={"Host": "untrusted.example"})[0], 403)
        self.assertEqual(self.request("POST", "/api/jobs/reserve", b"{}", {"Origin": "https://untrusted.example"})[0], 403)
        job = self.completed()
        for path in (f"/media/{job['id']}/../../README.md", f"/media/{job['id']}/source.mp4", "/.venv-glc/pyvenv.cfg"):
            self.assertEqual(self.request("GET", path)[0], 404)

    def test_invalid_file_metadata_and_queue_limit(self):
        for name, size in (("x.png", 8), ("x.exe", 8), ("x.mp4", 0), ("x.mp4", 2**31), (42, 8)):
            self.assertEqual(self.reserve(name, size)[0], 400)
        jobs = [self.reserve()[1] for _ in range(3)]
        self.assertEqual(self.reserve()[0], 400)
        self.request("POST", f"/api/jobs/{jobs[0]['id']}/cancel")
        self.assertEqual(self.reserve()[0], 202)


class StateTests(unittest.TestCase):
    def test_restart_preserves_finished_and_marks_pending_interrupted(self):
        with tempfile.TemporaryDirectory() as folder:
            store = GLCStore(folder)
            first = store.create("a.mp4", "video", parse_options({}))
            second = store.create("b.avi", "video", parse_options({}))
            store.update(first["id"], status="done", summary={"verified": True})
            store.update(second["id"], status="running")
            restarted = GLCStore(folder)
            self.assertEqual(restarted.get(first["id"])["summary"], {"verified": True})
            self.assertEqual(restarted.get(second["id"])["status"], "interrupted")

    def test_invalid_options(self):
        for query in ({"q": ["64"]}, {"q": ["1", "2"]}, {"kbps": ["nan"]}, {"kbps": ["0"]},
                      {"width": ["8192"]}, {"frames": ["0"]}, {"start_frame": ["-1"]},
                      {"intra_period": ["0"]}, {"reset_interval": ["-2"]}):
            with self.subTest(query=query), self.assertRaises(ValueError):
                parse_options(query)


if __name__ == "__main__":
    unittest.main()
