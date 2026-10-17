"""Check the remaining publisher masks for positive-case availability, never model scores."""
from __future__ import annotations
import json
from pathlib import Path
import zipfile
import zlib
import numpy as np
from PIL import Image
from external_preflight import RUN, OUT, DATA, read, write, digest, AuditedRange


def main():
    entries = read(DATA/"remote_index.json")
    names = sorted(r["name"] for r in entries if
                   r["name"].startswith("video6/mask/") and Path(r["name"]).stem.isdigit())
    rule = {
        "date": "2026-10-15", "stage": "F", "purpose": "supplement coverage of positive targets after the fixed first50 clip has only empty sparse GT",
        "original_negative_case_preserved": True,
        "eligibility": "first publisher fish-positive anchor >=50 in ascending real-frame order",
        "clip_if_available": "50 consecutive frames; start=max(50,anchor-24), capped at609; disjoint from0..49",
        "not_chosen_using_model_probabilities_encoding_quality_or_task_gain": True,
        "same_video": "video6; an additional clip, not a second independent video",
        "if_none": "record no suitable positive case in this video; do not call positive protection validated",
        "script_sha256": digest(__file__),
    }
    destination = RUN/"positive_availability_protocol.json"
    if destination.exists(): assert read(destination) == rule
    else: write(destination, rule)
    meta = read(DATA/"figshare_metadata_live.json")
    archive_meta = next(f for f in meta["files"] if f["name"] == "video7.zip")
    reader = AuditedRange(archive_meta["download_url"], archive_meta["size"])
    with zipfile.ZipFile(reader) as remote:
        ordered = sorted(remote.infolist(), key=lambda z:z.header_offset)
        next_offset = {a.filename:b.header_offset for a,b in zip(ordered,ordered[1:])}
        start = min(remote.getinfo(name).header_offset for name in names)
        last = max((remote.getinfo(name) for name in names), key=lambda z:z.header_offset)
        reader.seek(start); reader.read(next_offset[last.filename]-start)
        rows = []
        for name in names:
            path = DATA/"raw"/name
            if not path.exists():
                path.write_bytes(remote.read(name))
            info = remote.getinfo(name)
            payload = path.read_bytes()
            assert len(payload) == info.file_size and zlib.crc32(payload) == info.CRC
            colors = np.asarray(Image.open(path).convert("RGB"))
            fish = np.all(colors == [0,0,128], axis=2)
            rows.append({"source_frame": int(path.stem), "fish_pixels": int(fish.sum()),
                         "published_mask": str(path), "mask_sha256": digest(path)})
    positives = [row for row in rows if row["source_frame"] >= 50 and row["fish_pixels"] > 0]
    result = {
        "selection_rule": rule, "gt_masks": rows, "available_positive_anchors": len(positives),
        "first_eligible": positives[0] if positives else None,
        "http_bytes": reader.transferred, "requests": reader.log,
        "model_inferences_or_changes": 0,
    }
    write(OUT/"同视频正样本可用性检查.json", result)
    print("POSITIVE_AVAILABILITY", json.dumps({
        "masks": len(rows), "positive_anchors": len(positives),
        "first_eligible": result["first_eligible"], "http_bytes": reader.transferred},
        ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
