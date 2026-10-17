"""演示服务的上传隔离、任务恢复与视频 Range 回归检查。"""
from __future__ import annotations

import http.client
import json
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_server import DemoServer, JobStore, byte_range, filename_info, parse_options


class StoreTests(unittest.TestCase):
    def test_restart_preserves_finished_and_marks_unfinished(self):
        with tempfile.TemporaryDirectory() as folder:
            store = JobStore(folder)
            first = store.create("图片.jpg", "image", parse_options({}))
            second = store.create("图片.jpg", "image", parse_options({}))
            self.assertNotEqual(first["id"], second["id"])
            store.update(first["id"], status="done", summary={"example": 1})
            store = JobStore(folder)
            self.assertEqual(store.get(first["id"])["summary"], {"example": 1})
            self.assertEqual(store.get(second["id"])["status"], "interrupted")
            with self.assertRaises(KeyError):
                store.get("../job.json")

    def test_options_and_untrusted_names(self):
        self.assertEqual(filename_info(r"C:\假路径\水下图片.JPG"), ("水下图片.JPG", ".jpg", "image"))
        for value in ("file.exe", "", "file\x00.jpg"):
            with self.assertRaises(ValueError):
                filename_info(value)
        for query in ({"alpha": ["nan"]}, {"size": ["8192"]}, {"max_frames": ["-1"]},
                      {"save_masks": ["yes"]}, {"size": ["256", "512"]},
                      {"mode": ["other"]}, {"codec_quality": ["2"]}, {"transfer_kbps": ["nan"]}):
            with self.assertRaises(ValueError):
                parse_options(query)
        self.assertEqual(parse_options({"mode": ["semantic_codec"]})["mode"], "semantic_codec")
        for query in ({"mode": ["semantic_codec"], "size": ["512"]},
                      {"mode": ["semantic_codec"], "codec_quality": ["1"]}):
            with self.assertRaises(ValueError):
                parse_options(query)

    def test_browser_byte_ranges(self):
        self.assertEqual(byte_range("bytes=0-9", 20), (0, 9))
        self.assertEqual(byte_range("bytes=-5", 20), (15, 19))
        self.assertEqual(byte_range("bytes=10-", 20), (10, 19))
        self.assertEqual(byte_range("bytes=10-30", 20), (10, 19))
        for header in ("bytes=20-", "bytes=9-2", "bytes=-0", "bytes=", "bytes=0-1,3-4"):
            with self.assertRaises(ValueError):
                byte_range(header, 20)


class FakeApp:
    def __init__(self, directory):
        self.store = JobStore(directory)
        self.model_info = {"state": "ready"}
        self.ffmpeg = None
        self.executor = ThreadPoolExecutor(max_workers=1)

    def run(self, identifier, source):
        self.store.update(identifier, status="done")


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.app = FakeApp(cls.directory.name)
        cls.server = DemoServer(("127.0.0.1", 0), cls.app)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)
        cls.app.executor.shutdown(wait=True)
        cls.directory.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_unicode_raw_upload_isolated_and_exact(self):
        query = urlencode({"filename": "水下 图片.jpg"})
        body = bytes(range(256)) * 1024
        ids = []
        for _ in range(2):
            status, _, payload = self.request("POST", "/api/jobs?" + query, body)
            self.assertEqual(status, 202)
            identifier = json.loads(payload)["id"]
            ids.append(identifier)
            self.assertEqual((Path(self.directory.name) / identifier / "source.jpg").read_bytes(), body)
        self.assertNotEqual(*ids)

    def test_invalid_upload_and_other_origin_rejected(self):
        for path, headers, expected in (
            ("/api/jobs?filename=bad.exe", {}, 400),
            ("/api/jobs?filename=test.jpg&alpha=nan", {}, 400),
            ("/api/jobs?filename=test.jpg", {"Origin": "https://example.org"}, 403),
            ("/api/jobs?filename=test.jpg", {"Host": "example.org"}, 403),
        ):
            status, _, _ = self.request("POST", path, b"x", headers)
            self.assertEqual(status, expected)

    def test_codec_mode_rejects_video_and_preserves_codec_options(self):
        self.assertEqual(self.request("POST", "/api/jobs?filename=test.mp4&mode=image_codec", b"x")[0], 400)
        self.assertEqual(self.request("POST", "/api/jobs?filename=test.mp4&mode=semantic_codec", b"x")[0], 400)
        query = urlencode({"filename": "图片.jpg", "mode": "image_codec", "codec_quality": 5, "transfer_kbps": 40})
        status, _, payload = self.request("POST", "/api/jobs?" + query, b"image-placeholder")
        self.assertEqual(status, 202)
        job = json.loads(payload)
        self.assertEqual(job["options"]["codec_quality"], 5)
        self.assertEqual(job["options"]["transfer_kbps"], 40)

    def test_media_ranges_and_private_files(self):
        job = self.app.store.create("range.jpg", "image", parse_options({}))
        identifier = job["id"]
        path = Path(self.directory.name) / identifier / "result" / "input.mp4"
        path.write_bytes(b"0123456789abcdef")
        route = f"/media/{identifier}/input.mp4"
        self.assertEqual(self.request("GET", route)[0], 404)
        self.app.store.update(identifier, status="done")
        status, headers, body = self.request("GET", route, headers={"Range": "bytes=2-5"})
        self.assertEqual((status, headers["Content-Range"], body), (206, "bytes 2-5/16", b"2345"))
        self.assertEqual(headers["Content-Type"], "video/mp4")
        self.assertEqual(self.request("HEAD", route)[2], b"")
        self.assertEqual(self.request("GET", route, headers={"Range": "bytes=99-"})[0], 416)
        for suffix in ("job.json", "../job.json", "%2e%2e%2fjob.json", "source.jpg"):
            self.assertEqual(self.request("GET", f"/media/{identifier}/{suffix}")[0], 404)

    def test_cancel_sets_worker_event(self):
        job = self.app.store.create("cancel.jpg", "image", parse_options({}))
        identifier = job["id"]
        status, _, body = self.request("POST", f"/api/jobs/{identifier}/cancel", b"")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "cancelled")
        self.assertTrue(self.app.store.events[identifier].is_set())

    def test_rerun_uses_selected_history_source(self):
        job = self.app.store.create("历史图片.jpg", "image", parse_options({}))
        identifier = job["id"]
        source = Path(self.directory.name) / identifier / "source.jpg"
        source.write_bytes(b"original-history-image")
        self.app.store.update(identifier, status="done")
        status, _, data = self.request("POST", f"/api/jobs/{identifier}/rerun", json.dumps({"size": 384, "save_masks": True}))
        self.assertEqual(status, 202)
        child = json.loads(data)
        self.assertEqual(child["parent_id"], identifier)
        self.assertEqual(child["options"]["size"], 384)
        self.assertTrue(child["options"]["save_masks"])
        self.assertNotEqual(child["id"], identifier)
        self.assertEqual((Path(self.directory.name) / child["id"] / "source.jpg").read_bytes(), source.read_bytes())


if __name__ == "__main__":
    unittest.main()
