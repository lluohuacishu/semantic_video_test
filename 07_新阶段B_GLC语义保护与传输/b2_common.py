"""Small durable command journal shared by B2 experiment scripts."""
from __future__ import annotations
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def now():
    return datetime.now().astimezone().isoformat()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, encoding="utf-8").strip()


class Journal:
    def __init__(self, directory, authorize=None):
        self.directory = Path(directory)
        self.path = self.directory / "commands.json"
        self.entries = []
        self.authorize = authorize

    def begin(self, argv, **extra):
        row = {"argv": list(map(str, argv)), "started_at": now(), "status": "started", **extra}
        self.entries.append(row)
        write(self.path, self.entries)
        return row

    def finish(self, row, returncode, seconds):
        row.update(returncode=returncode, seconds=seconds, finished_at=now(), status="completed" if returncode == 0 else "failed")
        write(self.path, self.entries)

    def run(self, argv):
        argv = list(map(str, argv))
        if self.authorize:
            self.authorize(argv)
        row = self.begin(argv)
        tick = time.perf_counter()
        result = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace")
        stem = f"command_{len(self.entries):03d}"
        stdout, stderr = self.directory / f"{stem}_stdout.txt", self.directory / f"{stem}_stderr.txt"
        stdout.write_text(result.stdout, encoding="utf-8")
        stderr.write_text(result.stderr, encoding="utf-8")
        row.update(stdout=str(stdout), stderr=str(stderr))
        self.finish(row, result.returncode, time.perf_counter() - tick)
        require(result.returncode == 0, f"Command failed; see {stderr}")
        return result.stdout
