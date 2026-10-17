"""Fetch a small prospective video6 clip, preserving sparse GT and true frame IDs."""
from __future__ import annotations
import hashlib
import io
from pathlib import Path
import zipfile
import zlib
import numpy as np
from PIL import Image
from external_preflight import WORKSPACE, PROJECT, RUN, OUT, DATA, read, write, digest, AuditedRange


def main():
    metadata = read(DATA/"figshare_metadata_live.json")
    archive_meta = next(f for f in metadata["files"] if f["name"] == "video7.zip")
    old_protocol = read(PROJECT/"data/phase_c_20261013/protocol.json")
    old_manifest = read(PROJECT/"data/phase_c_20261013/manifest.json")
    assert "video6" not in old_protocol["experiment_roles"]
    assert "video6" in old_protocol["official_splits"]["train"]
    source_protocol = {
        "stage": "F", "step": 3, "date": "2026-10-15",
        "actual_video": "video6", "archive_label": "video7.zip",
        "archive_name_is_misleading": True, "official_split": "train",
        "project_role": "new holdout diagnostic video; not official test split",
        "selection": "outer video directory video6, first 50 consecutive original sequence frames 0..49; fixed before RGB inspection and metric calculation",
        "not_chosen_by_number_of_fish_or_model_performance": True,
        "previous_local_model_training_videos": [
            v["video"] for v in old_manifest["videos"] if v["experiment_role"] == "train"],
        "previous_local_development_videos": [
            v["video"] for v in old_manifest["videos"] if v["experiment_role"] != "train"],
        "capture_independence": "unknown; no recording metadata for video6 yet",
        "source_fps": "25/1 nominal, matching prior experiments; capture timestamp unverified",
        "gt_color": [0, 0, 128],
        "sparse_gt_policy": "only published masks at matching original frame indices; 46 unlabeled frames are not background",
        "code_sha256": digest(__file__),
    }
    target = RUN/"actual_video_protocol.json"
    if target.exists():
        assert read(target) == source_protocol
    else:
        write(target, source_protocol)
    write(OUT/"真实视频选择与划分说明.json", source_protocol)
    entries = read(DATA/"remote_index.json")
    files = [r for r in entries if not r["is_dir"]]
    by_name = {r["name"]: r for r in files}
    groups = {
        "sequence": [f"video6/sequence/{i:05d}.jpg" for i in range(50)],
        "images": [f"video6/images/{i:05d}.jpg" for i in (0, 15, 30, 45)],
        "mask": [f"video6/mask/{i:05d}.png" for i in (0, 15, 30, 45)],
    }
    assert all(name in by_name for names in groups.values() for name in names)
    logs = []
    reader = AuditedRange(archive_meta["download_url"], archive_meta["size"])
    with zipfile.ZipFile(reader) as remote:
        ordered = sorted(remote.infolist(), key=lambda z: z.header_offset)
        next_offsets = {a.filename: b.header_offset for a, b in zip(ordered, ordered[1:])}
        for group, names in groups.items():
            missing = [name for name in names if not (DATA/"raw"/name).exists()]
            if missing:
                start = min(remote.getinfo(name).header_offset for name in missing)
                last = max((remote.getinfo(name) for name in missing), key=lambda z: z.header_offset)
                end = next_offsets[last.filename]
                assert end-start < 32*1024**2
                print("PREFETCH", group, "bytes", end-start, flush=True)
                reader.seek(start)
                prefetched = reader.read(end-start)
                assert len(prefetched) == end-start
            for name in names:
                path = (DATA/"raw"/name).resolve()
                assert path.is_relative_to((DATA/"raw").resolve())
                info = remote.getinfo(name)
                if path.exists():
                    payload = path.read_bytes()
                else:
                    payload = remote.read(name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(payload)
                assert len(payload) == info.file_size and zlib.crc32(payload) == info.CRC
                logs.append({"entry": name, "path": str(path), "bytes": len(payload),
                             "sha256": digest(path), "zip_entry_crc_verified": True})
            write(DATA/"entry_download_audit.json", {
                "archive": archive_meta, "entries": logs, "http_bytes_this_run": reader.transferred,
                "requests_this_run": reader.log, "whole_archive_checksum_checked": False})
            print("FETCHED", group, len(names), flush=True)
    frozen_hashes = {h for v in old_manifest["videos"] for h in v["source_frame_rgb_hashes"]}
    hashes = []
    for index in range(50):
        path = DATA/f"raw/video6/sequence/{index:05d}.jpg"
        image = np.asarray(Image.open(path).convert("RGB"))
        assert image.shape == (720, 1280, 3)
        value = hashlib.sha256(image.tobytes()).hexdigest()
        assert value not in frozen_hashes, "New RGB duplicates an existing experiment frame"
        hashes.append(value)
    labels = []
    for index in (0, 15, 30, 45):
        rgb_path = DATA/f"raw/video6/sequence/{index:05d}.jpg"
        annotation_rgb = DATA/f"raw/video6/images/{index:05d}.jpg"
        mask_path = DATA/f"raw/video6/mask/{index:05d}.png"
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        assert np.array_equal(rgb, np.asarray(Image.open(annotation_rgb).convert("RGB")))
        gt_rgb = np.asarray(Image.open(mask_path).convert("RGB"))
        assert gt_rgb.shape == rgb.shape
        fish = np.all(gt_rgb == [0, 0, 128], axis=2)
        fish_path = DATA/f"fish_masks/{index:05d}.png"
        fish_path.parent.mkdir(exist_ok=True)
        Image.fromarray(fish.astype(np.uint8)*255).save(fish_path)
        labels.append({
            "source_frame": index, "local_frame": index, "published_mask": str(mask_path),
            "published_mask_sha256": digest(mask_path), "fish_mask": str(fish_path),
            "fish_mask_sha256": digest(fish_path), "annotated_rgb_equals_sequence": True,
            "gt_available": True, "fish_pixels": int(fish.sum()),
        })
    result = {
        **source_protocol,
        "sequence_dir": str(DATA/"raw/video6/sequence"), "source_frame_indices": list(range(50)),
        "frames": 50, "width": 1280, "height": 720, "fps": "25/1",
        "fps_basis": "user-declared nominal rate; capture timestamps not available",
        "source_rgb_sha256": hashes, "sparse_gt": labels,
        "exact_rgb_duplicates_with_all_5_previous_experiment_videos": 0,
        "previous_rgb_hashes_checked": len(frozen_hashes),
        "http_bytes_this_run": reader.transferred,
        "all_entry_crc_checked": True, "all_annotated_images_equal_sequence": True,
        "full_archive_md5_checked": False,
    }
    write(DATA/"manifest.json", result)
    write(OUT/"新视频数据核验.json", result)
    print("DATA_COMPLETE", {"video": "video6", "frames": 50, "http_bytes": reader.transferred,
                            "gt_frames": [r["source_frame"] for r in labels],
                            "gt_fish_pixels": [r["fish_pixels"] for r in labels]}, flush=True)


if __name__ == "__main__":
    main()
