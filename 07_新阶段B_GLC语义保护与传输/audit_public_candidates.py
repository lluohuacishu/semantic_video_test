"""Audit downloaded public-video candidates without running the semantic models."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_json(argv):
    result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", check=True)
    return json.loads(result.stdout)


def ffprobe(ffprobe_path: Path, path: Path):
    return run_json([str(ffprobe_path), "-v", "error", "-count_frames", "-select_streams", "v:0",
                     "-show_entries", "format=duration,size,format_name:stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_read_frames,duration",
                     "-of", "json", str(path)])


def scene_cuts(ffmpeg_path: Path, path: Path, threshold: float):
    command = [str(ffmpeg_path), "-hide_banner", "-nostdin", "-i", str(path), "-vf",
               f"select='gt(scene,{threshold})',showinfo", "-an", "-f", "null", "-"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", check=True)
    return [float(v) for v in re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", result.stderr)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, required=True)
    parser.add_argument("--scene-threshold", type=float, default=0.35)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    records = []
    for candidate in registry["candidates"]:
        path = (args.media_dir / candidate["local_filename"]).resolve()
        if not path.is_file():
            records.append({"id": candidate["id"], "status": "missing", "path": str(path)})
            continue
        probe = ffprobe(args.ffprobe, path)
        video = probe["streams"][0]
        record = {
            "id": candidate["id"], "path": str(path), "status": "downloaded",
            "bytes": path.stat().st_size, "sha1": digest(path, "sha1"),
            "sha256": digest(path, "sha256"), "probe": probe,
            "scene_cut_threshold": args.scene_threshold,
            "scene_cut_pts_seconds": scene_cuts(args.ffmpeg, path, args.scene_threshold),
        }
        record["checks"] = {
            "size_matches_registry": record["bytes"] == candidate["expected_bytes"],
            "sha1_matches_commons": record["sha1"] == candidate["commons_sha1"],
            "dimensions_match_registry": [video.get("width"), video.get("height")] == candidate["expected_size"],
            "duration_at_least_minimum": float(probe["format"]["duration"]) >= registry["selection_protocol"]["minimum_duration_seconds"],
            "video_frames_decoded": int(video.get("nb_read_frames", "0")) > 0,
        }
        record["status"] = "download_and_decode_passed" if all(record["checks"].values()) else "metadata_or_decode_mismatch"
        records.append(record)
    result = {
        "registry": str(args.registry.resolve()), "media_dir": str(args.media_dir.resolve()),
        "ffmpeg": str(args.ffmpeg.resolve()), "ffprobe": str(args.ffprobe.resolve()),
        "scene_cut_threshold": args.scene_threshold, "records": records,
        "status": "audit_complete_no_model_run",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "records": len(records),
                      "passed": sum(r["status"] == "download_and_decode_passed" for r in records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
