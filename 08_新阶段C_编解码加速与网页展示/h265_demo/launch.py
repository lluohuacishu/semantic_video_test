import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler
import webbrowser

here = Path(__file__).resolve().parent
workspace = here.parents[4]
store = workspace / "02_实验数据" / "C_编解码加速与网页展示" / "H265_web_jobs"
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8777)
parser.add_argument("--no-browser", action="store_true")
args = parser.parse_args()
url = f"http://127.0.0.1:{args.port}/"
opener = build_opener(ProxyHandler({}))


def ready():
    try:
        with opener.open(url + "api/jobs", timeout=2) as response:
            return json.load(response).get("app") == "h265-web"
    except (OSError, ValueError):
        return False


if not ready():
    with socket.socket() as connection:
        connection.settimeout(2)
        if connection.connect_ex(("127.0.0.1", args.port)) == 0:
            raise SystemExit(f"Port {args.port} is already occupied.")
    store.mkdir(parents=True, exist_ok=True)
    with (store / "server.log").open("a", encoding="utf-8") as out, (store / "server-error.log").open("a", encoding="utf-8") as err:
        process = subprocess.Popen([sys.executable, str(here / "server.py"), "--port", str(args.port)], cwd=here,
            env={**os.environ, "PYTHONUTF8": "1"}, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    deadline = time.monotonic() + 45
    while not ready():
        if process.poll() is not None:
            raise SystemExit(f"Server failed. See {store / 'server-error.log'}")
        if time.monotonic() > deadline:
            raise SystemExit(f"Server did not become ready. See {store / 'server-error.log'}")
        time.sleep(.3)
if not args.no_browser:
    webbrowser.open(url)
print(f"H.265: {url}")
