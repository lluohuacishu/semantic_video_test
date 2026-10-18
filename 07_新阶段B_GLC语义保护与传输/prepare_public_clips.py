"""Create reproducible lossless, fixed-rate clips from frozen public source windows."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def probe(ffprobe: Path, path: Path):
    result = subprocess.run([
        str(ffprobe), "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "format=duration,size,format_name:stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_read_frames,duration",
        "-of", "json", str(path)], check=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in registry["candidates"]}
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for window in registry["selected_windows"]:
        candidate = by_id[window["id"]]
        source = (args.raw_dir / candidate["local_filename"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        output = args.output / f"{window['id']}.mkv"
        width = int(window["output_width"])
        fps = int(window["output_fps"])
        duration = float(window["source_duration_seconds"])
        command = [str(args.ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin",
                   "-ss", str(window["source_start_seconds"]), "-t", str(duration),
                   "-i", str(source), "-an", "-vf", f"fps={fps},scale={width}:-2:flags=lanczos",
                   "-c:v", "ffv1", "-level", "3", "-coder", "1", "-pix_fmt", "bgr0",
                   "-r", str(fps), "-y", str(output)]
        subprocess.run(command, check=True)
        info = probe(args.ffprobe, output)
        video = info["streams"][0]
        records.append({
            "id": window["id"], "source": str(source), "source_sha256": sha(source),
            "source_start_seconds": window["source_start_seconds"],
            "source_duration_seconds": window["source_duration_seconds"],
            "output": str(output), "output_sha256": sha(output), "output_bytes": output.stat().st_size,
            "command": command, "probe": info,
            "checks": {
                "width": video.get("width") == width,
                "height": video.get("height") == 720,
                "fps": video.get("avg_frame_rate") == f"{fps}/1",
                "frames": int(video.get("nb_read_frames", "0")) == round(duration * fps),
                "duration_at_least_ten_seconds": float(info["format"]["duration"]) >= 10,
            },
            "targets": window["expected_targets"], "cut_guard": window["cut_guard"],
        })
    result = {"status": "prepared_lossless_candidate_clips", "registry": str(args.registry.resolve()),
              "records": records, "model_run": False,
              "qualification": "candidate diagnostics only; source exposure and independent capture certification remain open"}
    (args.output / "prepare.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "clips": len(records),
                      "passed": sum(all(r["checks"].values()) for r in records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
