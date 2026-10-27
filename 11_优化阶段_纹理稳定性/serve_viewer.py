"""Local viewer for completed experiment artifacts."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.request import urlopen
import webbrowser


class VideoHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.byte_range = None
        path = Path(self.translate_path(self.path))
        requested = self.headers.get('Range')
        if not requested or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested.strip())
        try:
            if not match or not any(match.groups()) or not size:
                raise ValueError('Invalid range')
            left, right = match.groups()
            if left:
                start, end = int(left), min(int(right), size - 1) if right else size - 1
            else:
                if int(right) <= 0: raise ValueError('Invalid suffix')
                start, end = max(0, size - int(right)), size - 1
            if start > end or start >= size: raise ValueError('Outside file')
        except ValueError:
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return None
        source = path.open('rb')
        source.seek(start)
        self.byte_range = (start, end)
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Last-Modified', self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        return source

    def end_headers(self):
        self.send_header('Accept-Ranges', 'bytes')
        super().end_headers()

    def copyfile(self, source, outputfile):
        if self.byte_range is None:
            return super().copyfile(source, outputfile)
        remaining = self.byte_range[1] - self.byte_range[0] + 1
        while remaining:
            chunk = source.read(min(65536, remaining))
            if not chunk: break
            outputfile.write(chunk)
            remaining -= len(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8780)
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args(); root = args.root.resolve()
    if not (root / 'index.html').is_file() or not (root / 'summary.json').is_file():
        raise FileNotFoundError('Completed viewer not found')
    expected = json.loads((root / 'summary.json').read_text('utf-8'))
    url = f'http://127.0.0.1:{args.port}/'
    if args.open:
        def ready():
            try:
                with urlopen(url + 'summary.json', timeout=1) as response:
                    return json.load(response) == expected
            except (OSError, ValueError):
                return False
        if not ready():
            command = [sys.executable, str(Path(__file__).resolve()), '--root', str(root), '--port', str(args.port)]
            with (root / 'viewer.log').open('ab') as log:
                child = subprocess.Popen(command, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                    start_new_session=os.name != 'nt')
            deadline = time.monotonic() + 10
            while not ready():
                if child.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError('Viewer could not start; inspect viewer.log or port conflict')
                time.sleep(0.15)
        webbrowser.open(url)
    else:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(VideoHandler, directory=str(root)))
        print(url, flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
