from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from serve_viewer import VideoHandler


class QuietHandler(VideoHandler):
    def log_message(self, *args):
        pass


class RangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.payload = bytes(range(256)) * 8
        Path(cls.temp.name, 'video.mp4').write_bytes(cls.payload)
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=cls.temp.name))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}/video.mp4'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join(); cls.temp.cleanup()

    def test_whole_file(self):
        with urlopen(self.url) as r:
            self.assertEqual(r.status, 200)
            self.assertEqual(r.headers['Accept-Ranges'], 'bytes')
            self.assertEqual(r.read(), self.payload)

    def test_middle_and_suffix(self):
        for requested, start, end in [('bytes=103-219', 103, 219), ('bytes=-17', 2031, 2047), ('bytes=2040-', 2040, 2047)]:
            with urlopen(Request(self.url, headers={'Range': requested})) as r:
                self.assertEqual(r.status, 206)
                self.assertEqual(r.headers['Content-Range'], f'bytes {start}-{end}/2048')
                self.assertEqual(r.read(), self.payload[start:end + 1])

    def test_bad_ranges(self):
        for value in ('bytes=4000-', 'bytes=20-10', 'bytes=-0', 'bytes=0-1,3-4'):
            with self.assertRaises(HTTPError) as ctx:
                urlopen(Request(self.url, headers={'Range': value}))
            self.assertEqual(ctx.exception.code, 416)
            ctx.exception.close()

    def test_head_range_has_no_body(self):
        with urlopen(Request(self.url, method='HEAD', headers={'Range': 'bytes=0-15'})) as r:
            self.assertEqual(r.status, 206)
            self.assertEqual(r.headers['Content-Length'], '16')
            self.assertEqual(r.read(), b'')


if __name__ == '__main__':
    unittest.main()
