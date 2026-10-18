"""Inventory local continuous videos and prior use; do not invent independence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from b2_common import HERE, ROOT, Journal, git, read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = read(HERE/"configs/b0_baseline.json")
    data = Path("C:/Users/lenovo/Desktop/works/work/semantic-video-mlvc/data")
    official = data/"sfishtrack_official"
    files = official/"selected/SFISHTRACK"
    journal = Journal(args.output)
    invocation = journal.begin([sys.executable, *sys.argv], code_commit=git("rev-parse", "HEAD"))
    started = time.perf_counter()
    groups_path = ROOT/"01_主工程/data/recording_groups_review.json"
    groups = read(groups_path)
    previous = read(ROOT/"01_主工程/data/phase_b_annotation_corpus/manifest.json")
    prior = {s["name"]: s for s in previous["sequences"]}
    rows = []
    for video in sorted((files/"Videos").glob("*.mp4")):
        number = int(video.stem.split("_")[1])
        metadata_path = files/"metadata"/f"video_{number:03d}.json"
        metadata = read(metadata_path)
        probe = json.loads(journal.run([Path(baseline["runtime_dependencies"]["ffmpeg"]).with_name("ffprobe.exe"), "-v", "error", "-select_streams", "v:0",
                                        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration", "-of", "json", video]))["streams"][0]
        history = prior.get(f"sfishtrack_{number:03d}")
        group = next((day for day, entries in groups["normalized_groups"].items() if any(v["video_id"] == number for v in entries)), None)
        rows.append({"video_id": number, "path": str(video), "bytes": video.stat().st_size, "sha256": sha(video), "probe": probe,
                     "metadata": metadata, "metadata_sha256": sha(metadata_path), "normalized_recording_date": group,
                     "at_least_10_seconds": float(probe["duration"]) >= 10,
                     "prior_use": "historical evaluation-only annotated sequence; no weight updates per manifest" if history else "historical alignment audit; not unused",
                     "historical_role": history["role"] if history else None,
                     "new_unused_holdout": False,
                     "independence_notes": "Different recording dates distinguish 001/027/054; 028 shares 054 date/site/camera. Date normalization retains publisher ambiguity.",
                     "diver_coverage_verified": False})
    teacher = ROOT/"01_主工程/runs/phase_c_20261013/segformer_adaptation_preserve8"
    uv = read(ROOT/"01_主工程/data/phase_c_20261013/protocol.json")
    result = {"scope": "local-source and historical-protocol audit; no new external download or visual target annotation",
              "code_commit": git("rev-parse", "HEAD"), "videos": rows,
              "source_record_snapshot": {"path": str(official/"zenodo_metadata.json"), "sha256": sha(official/"zenodo_metadata.json"), "official_record": "https://zenodo.org/records/20617668"},
              "evidence_files": {str(p): sha(p) for p in [groups_path, ROOT/"01_主工程/data/phase_b_annotation_corpus/manifest.json", teacher/"config.json", teacher/"result.json"]},
              "teacher_config": read(teacher/"config.json"), "teacher_result": read(teacher/"result.json"),
              "uvmulti_historical_roles": uv["experiment_roles"], "uvmulti_capture_independence_verified": uv["capture_level_independence_verified"],
              "uvmulti_misnamed_video7": "Actual extracted directory video6, official train, already used in F/A; not a new test scene",
              "eligible_new_unused_independent_clips": 0,
              "historical_external_reuse_candidates": [1,27,54],
              "remaining_gaps": ["No three newly unused independent clips certified", "No verified continuous diver scene", "SFISHTRACK027 lasts 7.106667s, below preferred 10s", "Do not concatenate sparse annotated stills as continuous video"],
              "permitted_next_use": "001/027/054 can be separately labeled historical external evaluation reuse after freezing contiguous MP4 frame spans; not first-use holdout and not diver validation"}
    write(args.output/"materials_audit.json", result)
    journal.finish(invocation, 0, time.perf_counter()-started)
    print(json.dumps({"local_videos": len(rows), "new_unused_holdouts_certified": 0, "historical_external_reuse_candidates": [1,27,54]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
