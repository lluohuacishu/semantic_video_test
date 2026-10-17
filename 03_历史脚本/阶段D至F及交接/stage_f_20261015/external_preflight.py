"""Prospective F3 source audit: reserve one public archive before looking at RGB."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import urllib.request
import zipfile

WORKSPACE = Path(r"C:\Users\lenovo\Documents\ChatGPT\语义通信")
PROJECT = Path(r"C:\Users\lenovo\Desktop\works\work\semantic-video-mlvc")
RUN = PROJECT/"runs/semantic_mvp_stage_f_20261015/step3"
OUT = WORKSPACE/"outputs/semantic_roi_phase_f/step3_frozen_external"
DATA = PROJECT/"data/phase_f_external_20261015"


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", "utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AuditedRange(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.transferred = 0
        self.log = []
        self.cache_start, self.cache = 0, b""

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        if self.position < 0:
            raise ValueError("Negative position")
        return self.position

    def read(self, count=-1):
        count = self.size-self.position if count < 0 else min(count, self.size-self.position)
        if not count:
            return b""
        if count > 64*1024**2:
            raise ValueError("Refuse a read larger than 64 MiB")
        start = self.position
        if self.cache and self.cache_start <= start and start+count <= self.cache_start+len(self.cache):
            offset = start-self.cache_start
            self.position += count
            return self.cache[offset:offset+count]
        fetch = min(max(4096, count), self.size-start)
        end = start+fetch-1
        last_error = None
        for attempt in range(3):
            # Use a range-specific cache key to avoid a proxy reusing another range.
            join = "&" if "?" in self.url else "?"
            url = self.url+join+f"stage_f_range={start}_{end}"
            request = urllib.request.Request(url, headers={
                "Range": f"bytes={start}-{end}", "Cache-Control": "no-cache",
                "User-Agent": "semantic-video-research/1.0"})
            try:
                with urllib.request.urlopen(request, timeout=25) as response:
                    content_range = response.headers.get("Content-Range")
                    if response.status != 206 or content_range != f"bytes {start}-{end}/{self.size}":
                        raise RuntimeError(f"Exact partial read required: {response.status}, {content_range}")
                    payload = response.read(fetch+1)
                if len(payload) != fetch:
                    raise RuntimeError("Short partial response")
                self.log.append({"start": start, "end": end, "received": len(payload),
                                 "range_verified": True})
                self.transferred += len(payload)
                self.cache_start, self.cache = start, payload
                self.position += count
                return payload[:count]
            except Exception as error:
                last_error = error
                print("RANGE_RETRY", start, end, attempt+1, repr(error), flush=True)
        raise last_error


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    roles = read(PROJECT/"data/phase_c_20261013/protocol.json")
    meta = read(PROJECT/"audit/uvmulti_figshare_metadata.json")
    archive = next(f for f in meta["files"] if f["name"] == "video7.zip")
    protocol = {
        "date": "2026-10-15", "stage": "F", "step": 3,
        "source_selection": "the only publicly listed archive previously reserved as video7; not selected by visual or semantic performance",
        "reason": "local experiment roles cover all currently prepared continuous videos; this archive was reserved before any phase-C training",
        "archive": archive,
        "actual_inner_video_and_split_must_be_verified": True,
        "clip_rule": "first 50 genuinely consecutive original sequence frames, preserving true indices; no scene or GT-score selection",
        "budget_rule": "100000 bytes complete packet, 2% tolerance; byte-only CRF search; same continuous RGB for both methods",
        "baseline": "offline H265 medium, GOP32, B4; actual full MP4 charged",
        "frozen_E": "existing model, threshold0.9, min-component64, padding16, window32, halves, background scale0.5 CRF42, P-only ROI",
        "no_training_or_method_selection_on_new_clip": True,
        "known_experiment_roles": roles["experiment_roles"],
        "capture_level_independence": "not yet verified; separate video identity alone is insufficient",
        "source_audit_before_rgb_and_gt_metrics": True,
        "script_sha256": digest(__file__),
        "existing_data_protocol_sha256": digest(PROJECT/"data/phase_c_20261013/protocol.json"),
    }
    path = RUN/"source_preflight_protocol.json"
    if path.exists():
        assert read(path) == protocol
    else:
        write(path, protocol)
    write(OUT/"新视频预先选择约定.json", protocol)
    live_path = DATA/"figshare_metadata_live.json"
    if not live_path.exists():
        try:
            request = urllib.request.Request("https://api.figshare.com/v2/articles/30105151",
                                             headers={"User-Agent": "semantic-video-research/1.0"})
            with urllib.request.urlopen(request, timeout=25) as response:
                live = json.load(response)
            write(live_path, live)
        except Exception as error:
            write(DATA/"metadata_live_failure.json", {"error": repr(error), "cached_metadata_retained": True})
            print("LIVE_METADATA_FAILED", repr(error), flush=True)
    if live_path.exists():
        live = read(live_path)
        current = next(f for f in live["files"] if f["name"] == archive["name"])
        assert current["size"] == archive["size"] and current["supplied_md5"] == archive["supplied_md5"]
        print("LIVE_METADATA_MATCH", current["name"], current["size"], flush=True)
    index_path = DATA/"remote_index.json"
    reader = AuditedRange(archive["download_url"], archive["size"])
    if not index_path.exists():
        with zipfile.ZipFile(reader) as remote:
            entries = [{"name": z.filename, "bytes": z.file_size, "compressed_bytes": z.compress_size,
                        "crc32": format(z.CRC, "08x"), "offset": z.header_offset,
                        "is_dir": z.is_dir()} for z in remote.infolist()]
        write(index_path, entries)
        write(DATA/"index_http_audit.json", {
            "archive": archive, "http_bytes": reader.transferred, "requests": reader.log,
            "whole_archive_md5_checked": False})
    entries = read(index_path)
    types = {}
    for row in entries:
        if row["is_dir"]: continue
        parts = row["name"].replace("\\", "/").split("/")
        key = "/".join(parts[:-1])
        types[key] = types.get(key, 0)+1
    write(OUT/"新视频目录预检.json", {
        "archive": archive["name"], "entries": len(entries), "directory_counts": types,
        "archive_checksum_verified": False, "rgb_not_scored": True})
    print("INDEX_COMPLETE", json.dumps({"entries": len(entries), "directory_counts": types}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
