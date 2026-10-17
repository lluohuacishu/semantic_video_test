"""Read an official public ZIP by HTTP Range; download selected entries only."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import time
import urllib.request
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/sfishtrack_official"
META_URL = "https://zenodo.org/api/records/20617668"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class RangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.bytes_received = 0
        self.ranges = []
        self.cache_start, self.cache = 0, b""

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0: raise ValueError("Negative seek")
        return self.position

    def read(self, size=-1):
        if size < 0: size = self.size - self.position
        size = min(size, self.size - self.position)
        if size == 0: return b""
        if size > 1024**3: raise ValueError("Refusing a range larger than 1 GiB")
        start = self.position
        if self.cache and self.cache_start<=start and start+size<=self.cache_start+len(self.cache):
            offset = start-self.cache_start
            self.position += size
            return self.cache[offset:offset+size]
        fetched_size = min(max(4096,size) if size<4096 else size,self.size-start)
        end = start+fetched_size-1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}", "Cache-Control": "no-cache", "User-Agent": "semantic-video-research/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            content_range = response.headers.get("Content-Range")
            if response.status != 206 or content_range != f"bytes {start}-{end}/{self.size}":
                raise RuntimeError(f"Server must honor exact partial read: {response.status}, {content_range}, expected {start}-{end}")
            data = response.read(fetched_size+1)
        if len(data) != fetched_size: raise RuntimeError("Incomplete range response")
        self.bytes_received += len(data)
        self.ranges.append({"start": start, "end": end, "bytes": len(data)})
        if size<4096: self.cache_start,self.cache=start,data
        self.position += size
        return data[:size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", nargs="*", default=[])
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--reuse", action="store_true", help="Reuse only files matching published ZIP CRC and length")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(META_URL, timeout=30) as response:
        metadata = json.load(response)
    write_json(DATA / "zenodo_metadata.json", metadata)
    file = next(f for f in metadata["files"] if f["key"] == "SFISHTRACK.zip")
    remote = RangeFile(file["links"]["self"], file["size"])
    with zipfile.ZipFile(remote) as archive:
        entries = [{"name": z.filename, "bytes": z.file_size, "compressed_bytes": z.compress_size,
                    "crc32": f"{z.CRC:08x}", "offset": z.header_offset, "directory": z.is_dir()} for z in archive.infolist()]
        write_json(DATA / "remote_entries.json", entries)
        print(json.dumps({"archive_bytes": remote.size, "entries": len(entries), "videos": [z for z in entries if Path(z["name"]).suffix.lower() in {".mp4", ".avi", ".mov"}], "annotation_examples": [z for z in entries if z["name"].endswith(".json")][:8]}, indent=2), flush=True)
        downloaded = []
        for name in args.entry:
            info = archive.getinfo(name)
            target = (DATA / "selected" / name).resolve()
            if not target.is_relative_to((DATA / "selected").resolve()): raise ValueError("Archive path escapes data directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                payload = target.read_bytes()
                if not args.reuse or len(payload)!=info.file_size or zlib.crc32(payload)!=info.CRC: raise FileExistsError(target)
            else:
                payload = archive.read(info)  # ZIP CRC32 checked by zipfile.
                target.write_bytes(payload)
            downloaded.append({"entry": name, "path": str(target), "bytes": len(payload), "zip_crc32_verified": True,
                               "sha256": hashlib.sha256(payload).hexdigest()})
            print(f"saved {name}: {len(payload)} bytes", flush=True)
    audit = {"official_record": "https://zenodo.org/records/20617668", "source": remote.url,
             "archive_bytes": remote.size, "archive_published_checksum": file["checksum"],
             "archive_checksum_checked": False, "reason": "Only selected entries fetched; full archive not downloaded",
             "license": metadata["metadata"].get("license"), "downloaded": downloaded,
             "http_bytes_received": remote.bytes_received, "ranges": remote.ranges}
    log = DATA / f"fetch_{time.time_ns()}.json"
    write_json(log, audit)
    print(f"HTTP bytes received: {remote.bytes_received}; audit {log}", flush=True)


if __name__ == "__main__": main()
