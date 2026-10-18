"""Read VDD-C public ZIP indexes by HTTP ranges without downloading image bodies."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request
import zipfile

from b2_common import git, sha, write

HERE = Path(__file__).resolve().parent
ADAPTERS = HERE.parent / "01_主工程/adapters"
sys.path.insert(0, str(ADAPTERS))
from fetch_sfishtrack import RangeFile

ITEM_API = "https://conservancy.umn.edu/server/api/core/items/3b926c46-c328-40d8-841f-069ef8ab1ce2"
HANDLE = "https://hdl.handle.net/11299/219383"
IMAGES_URL = "https://conservancy.umn.edu/server/api/core/bitstreams/34ef37fe-6145-4931-af1e-068be0380dfa/content"
IMAGES_BYTES = 8_196_894_778
LABELS_URL = "https://conservancy.umn.edu/server/api/core/bitstreams/7d68ed44-9d8a-4877-9fb9-0c37a3e80da5/content"
LABELS_BYTES = 29_176_091
README_URL = "https://conservancy.umn.edu/server/api/core/bitstreams/d039ad1b-e21f-42ba-b67d-57fa42b4a400/content"


def central_index(url, size):
    remote = RangeFile(url, size)
    with zipfile.ZipFile(remote) as archive:
        names = [info.filename for info in archive.infolist()]
        return names, remote


def image_sequences(names):
    groups = collections.defaultdict(list)
    for name in names:
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        parts = name.split("/")
        if len(parts) < 3:
            continue
        match = re.search(r"(\d+)$", parts[-1].rsplit(".", 1)[0])
        if match:
            groups[parts[1]].append(int(match.group(1)))
    rows = []
    for sequence, values in groups.items():
        values = sorted(set(values))
        runs = []
        if values:
            start = previous = values[0]
            for value in values[1:]:
                if value != previous + 1:
                    if previous - start + 1 >= 200:
                        runs.append([start, previous, previous - start + 1])
                    start = value
                previous = value
            if previous - start + 1 >= 200:
                runs.append([start, previous, previous - start + 1])
        rows.append({"sequence": sequence, "image_count": len(values), "min_frame": min(values) if values else None,
                     "max_frame": max(values) if values else None, "contiguous_runs_ge_200": runs})
    return sorted(rows, key=lambda row: (-max([run[2] for run in row["contiguous_runs_ge_200"]] or [0]), row["sequence"]))


def yolo_val_sequences(names):
    rows = collections.Counter()
    for name in names:
        parts = name.split("/")
        if len(parts) != 3 or parts[1] != "val" or not name.endswith(".txt"):
            continue
        rows[parts[2].rsplit("_", 1)[0]] += 1
    return dict(sorted(rows.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    image_names, image_remote = central_index(IMAGES_URL, IMAGES_BYTES)
    label_names, label_remote = central_index(LABELS_URL, LABELS_BYTES)
    with urllib.request.urlopen(README_URL, timeout=45) as response:
        readme = response.read()
    readme_path = args.output / "VDD-C_README.txt"
    readme_path.write_bytes(readme)
    rows = image_sequences(image_names)
    result = {
        "scope": "public ZIP central-directory audit; no image or original-video member bodies downloaded",
        "code_commit": git("rev-parse", "HEAD"),
        "official_item_api": ITEM_API,
        "official_handle": HANDLE,
        "images_zip": {"url": IMAGES_URL, "bytes": IMAGES_BYTES, "entry_count": len(image_names),
                        "http_range_bytes": image_remote.bytes_received, "ranges": image_remote.ranges,
                        "sequence_count": len(rows), "sequences_with_contiguous_200": sum(bool(r["contiguous_runs_ge_200"]) for r in rows),
                        "sequences": rows},
        "labels_zip": {"url": LABELS_URL, "bytes": LABELS_BYTES, "entry_count": len(label_names),
                        "http_range_bytes": label_remote.bytes_received, "ranges": label_remote.ranges,
                        "yolo_val_sequence_frame_counts": yolo_val_sequences(label_names)},
        "readme": {"url": README_URL, "path": str(readme_path), "bytes": len(readme), "sha256": hashlib.sha256(readme).hexdigest()},
        "candidate_sequence_level_holdouts": ["barbados_scuba_002_B", "barbados_scuba_003_C", "pool_swimmer_003_A"],
        "candidate_status": "sequential_video_derived_images_only",
        "eligibility_decision": "Do not call these continuous videos: public package omits original video and removes blurry/occluded frames; recording-level independence and original timestamps are not supplied.",
        "next_requirement": "Obtain permitted original-video or timestamp-complete source, then verify three recording groups before B3 independent run.",
    }
    write(args.output / "vddc_source_audit.json", result)
    print(json.dumps({"sequence_count": len(rows), "sequences_with_contiguous_200": result["images_zip"]["sequences_with_contiguous_200"],
                      "candidate_val_sequences": result["candidate_sequence_level_holdouts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
