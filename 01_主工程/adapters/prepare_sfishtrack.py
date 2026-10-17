"""Build video-level evaluation corpus with exact timestamp-to-COCO-frame mapping."""
from __future__ import annotations
from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from common import ROOT, FFMPEG, FFPROBE, probe, sha256, save_json

DATA = ROOT / "data/sfishtrack_official/selected/SFISHTRACK"
CORPUS = ROOT / "data/phase_b_corpus"
CORPUS.mkdir(exist_ok=False)
records = []
for sequence_id in [1, 27, 54]:
    name = f"sfishtrack_{sequence_id:03d}"
    source = DATA / f"Videos/video_{sequence_id:03d}.mp4"
    annotation_file = DATA / f"Annotations/video{sequence_id:03d}.json"
    data = json.loads(annotation_file.read_text("utf-8"))
    if data["categories"] != [{"id": 1, "name": "fish"}]: raise ValueError("Unexpected COCO categories")
    info = probe(source)
    frame_info = json.loads(subprocess.check_output([str(FFPROBE), "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(source)]))["frames"]
    times = np.asarray([float(f["best_effort_timestamp_time"]) for f in frame_info])
    frame_rows = {int(r["frame_id"]):r for r in data["images"]}
    if len(frame_info) != int(info["nb_frames"]) or not np.all(np.diff(times)>0): raise ValueError("Invalid source timestamp order/count")
    duration = min(10.0, float(info["duration"]))
    count = math.floor(duration*5+1e-7)
    selected = [int(np.argmin(np.abs(times-(idx/5)))) for idx in range(count)]
    if len(set(selected)) != count or any(idx not in frame_rows for idx in selected): raise ValueError("Sampled frame not annotated or duplicated")
    run = CORPUS / name
    (run / "reference").mkdir(parents=True)
    (run / "ground_truth_fish").mkdir()
    expression = "+".join(f"eq(n,{idx})" for idx in selected)
    command = [str(FFMPEG), "-v", "error", "-i", str(source), "-vf", f"select='{expression}',scale=1280:720:flags=lanczos", "-an", "-fps_mode", "passthrough", "-start_number", "0", str(run / "reference/%05d.png")]
    subprocess.run(command, check=True)
    if len(list((run / "reference").glob("*.png"))) != count: raise ValueError("Frame extraction count differs")
    annotations = defaultdict(list)
    for annotation in data["annotations"]: annotations[annotation["image_id"]].append(annotation)
    mapping = []
    for idx, frame_idx in enumerate(selected):
        frame = frame_rows[frame_idx]
        if (frame["width"], frame["height"]) != (int(info["width"]),int(info["height"])): raise ValueError("Label/source resolution mismatch")
        union = np.zeros((frame["height"], frame["width"]), dtype=bool)
        instances = annotations[frame["id"]]
        for instance in instances:
            if instance["category_id"] != 1 or instance.get("iscrowd",0): raise ValueError("Unsupported category/crowd")
            rle = dict(instance["segmentation"])
            if isinstance(rle["counts"],str): rle["counts"] = rle["counts"].encode("ascii")
            mask = coco_mask.decode(rle).astype(bool)
            if mask.shape != union.shape or abs(int(mask.sum())-instance["area"]) > 1: raise ValueError("COCO RLE area/dimensions invalid")
            union |= mask
        resized = Image.fromarray(union.astype(np.uint8)).resize((1280,720), Image.Resampling.NEAREST)
        gt_path = run / "ground_truth_fish" / f"{idx:05d}.png"
        resized.save(gt_path)
        mapping.append({"sample_index": idx, "source_frame_index": frame_idx, "source_timestamp": float(times[frame_idx]),
                        "playback_timestamp": idx/5, "coco_image_id": frame["id"], "coco_file_name": frame["file_name"], "instances": len(instances),
                        "source_fish_pixels": int(union.sum()), "scaled_fish_pixels": int(np.asarray(resized).sum()),
                        "reference_sha256": sha256(run / "reference" / f"{idx:05d}.png"), "ground_truth_sha256": sha256(gt_path)})
    # Provided JPEG and frame-zero decode must depict the same aligned content.
    original_jpeg = DATA / f"Frames/video_{sequence_id:03d}/frame_000000.jpg"
    aligned_jpeg = Image.open(original_jpeg).convert("RGB").resize((1280,720),Image.Resampling.LANCZOS)
    reference = Image.open(run / "reference/00000.png").convert("RGB")
    delta = (np.asarray(reference).astype(np.float64)-np.asarray(aligned_jpeg).astype(np.float64))
    mse = float((delta*delta).mean())
    jpeg_psnr = None if mse==0 else float(10*np.log10(255**2/mse))
    if jpeg_psnr is not None and jpeg_psnr < 25: raise ValueError(f"Provided frame/MP4 content may not align: PSNR {jpeg_psnr}")
    record = {"name": name, "role": "evaluation_only", "source": str(source), "source_sha256": sha256(source),
              "annotation_source": str(annotation_file), "annotation_sha256": sha256(annotation_file),
              "metadata": json.loads((DATA / f"metadata/video_{sequence_id:03d}.json").read_text("utf-8")),
              "recording_group": json.loads((DATA / f"metadata/video_{sequence_id:03d}.json").read_text("utf-8"))["Date of recording"],
              "native_source_probe": info, "reference_dir": str(run / "reference"), "ground_truth_dir": str(run / "ground_truth_fish"),
              "label_type": "fish_binary_0_1", "width": 1280, "height": 720, "fps": "5/1", "frame_count": count,
              "duration_seconds": count/5, "sampling": "nearest actual PTS to t=0,0.2,... for first min(10s,full duration); no quality-based selection",
              "original_jpeg_alignment_psnr_db": jpeg_psnr, "mask_conversion": "pycocotools RLE decode; validate area; union fish instances; nearest resize", "frames": mapping,
              "official_record": "https://zenodo.org/records/20617668", "license": "CC-BY-4.0", "extraction_command": command}
    save_json(run / "manifest.json", record)
    records.append(record)
    print(f"prepared {name}: {count} frames, recording {record['recording_group']}, JPEG alignment PSNR {jpeg_psnr:.2f}dB", flush=True)
if len({r["recording_group"] for r in records}) != 3: raise ValueError("Selected clips share recording dates")
save_json(CORPUS / "manifest.json", {"purpose": "Frozen teacher + codec evaluation; all three videos reserved from training", "selection": "IDs 1,27,54 evenly spaced through 54-video archive, selected before model/codec scores", "sequences": records})
