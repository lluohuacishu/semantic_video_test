"""Use publisher's annotation RGB frames to avoid uncertain MP4 frame mapping."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from common import ROOT, FFMPEG, probe, sha256, save_json

DATA = ROOT / "data/sfishtrack_official/selected/SFISHTRACK"
CORPUS = ROOT / "data/phase_b_annotation_corpus"
CORPUS.mkdir(exist_ok=False)
selection = json.loads((ROOT / "data/annotation_frame_selection.json").read_text("utf-8"))
records = []
for selected in selection["sequences"]:
    sequence_id = selected["video_id"]
    name = f"sfishtrack_{sequence_id:03d}"
    annotation_file = DATA / f"Annotations/video{sequence_id:03d}.json"
    data = json.loads(annotation_file.read_text("utf-8"))
    if len(data["categories"])!=1 or data["categories"][0]["id"]!=1 or data["categories"][0]["name"] not in {"fish","object"}:
        raise ValueError("Unsupported category mapping")
    by_id = {r["id"]:r for r in data["images"]}
    annotations = defaultdict(list)
    for item in data["annotations"]: annotations[item["image_id"]].append(item)
    run = CORPUS / name
    (run / "reference").mkdir(parents=True)
    (run / "ground_truth_fish").mkdir()
    mapping = []
    for idx, frame_selection in enumerate(selected["frames"]):
        frame = by_id[frame_selection["coco_image_id"]]
        image_file = ROOT / "data/sfishtrack_official/selected" / frame_selection["archive_entry"]
        image = Image.open(image_file).convert("RGB")
        if image.size!=(frame["width"],frame["height"]): raise ValueError("Annotated RGB frame dimensions differ")
        union = np.zeros((frame["height"],frame["width"]),dtype=bool)
        instances = annotations[frame["id"]]
        for instance in instances:
            if instance["category_id"]!=1 or instance.get("iscrowd",0): raise ValueError("Unsupported crowd/category")
            rle = dict(instance["segmentation"])
            if isinstance(rle["counts"],str): rle["counts"]=rle["counts"].encode("ascii")
            mask = coco_mask.decode(rle).astype(bool)
            if mask.shape!=union.shape or abs(int(mask.sum())-instance["area"])>1: raise ValueError("RLE area/dimensions invalid")
            union |= mask
        reference_file = run / "reference" / f"{idx:05d}.png"
        image.resize((1280,720),Image.Resampling.LANCZOS).save(reference_file)
        mask_file = run / "ground_truth_fish" / f"{idx:05d}.png"
        resized = Image.fromarray(union.astype(np.uint8)).resize((1280,720),Image.Resampling.NEAREST)
        resized.save(mask_file)
        mapping.append({"sample_index":idx, **frame_selection, "source_filename_numeric_index":int(Path(frame_selection["annotation_file_name"]).stem),
                        "source_image_sha256":sha256(image_file), "instances":len(instances),
                        "source_fish_pixels":int(union.sum()),"scaled_fish_pixels":int(np.asarray(resized).sum()),
                        "reference_sha256":sha256(reference_file),"ground_truth_sha256":sha256(mask_file)})
    # Lossless source video has an explicitly controlled playback clock.
    source = run / "reference_lossless.mkv"
    command = [str(FFMPEG),"-v","error","-framerate","5/1","-start_number","0","-i",str(run / "reference/%05d.png"),
               "-an","-c:v","ffv1","-level","3","-pix_fmt","bgr0",str(source)]
    subprocess.run(command,check=True)
    process = subprocess.Popen([str(FFMPEG),"-v","error","-i",str(source),"-an","-fps_mode","passthrough","-pix_fmt","rgb24","-f","rawvideo","pipe:1"],stdout=subprocess.PIPE)
    frame_bytes = 1280*720*3
    for idx, frame in enumerate(mapping):
        pixels = process.stdout.read(frame_bytes)
        expected = np.asarray(Image.open(run / "reference" / f"{idx:05d}.png").convert("RGB"))
        if len(pixels)!=frame_bytes or hashlib.sha256(pixels).digest()!=hashlib.sha256(expected.tobytes()).digest():
            process.kill()
            raise ValueError("Lossless video changed reference RGB pixels")
    if process.stdout.read(1) or process.wait()!=0: raise ValueError("Lossless reference count/decode mismatch")
    metadata = json.loads((DATA / f"metadata/video_{sequence_id:03d}.json").read_text("utf-8"))
    record = {"name":name,"role":"evaluation_only_no_weight_updates","source":str(source),"source_sha256":sha256(source),
              "original_mp4_for_context":str(DATA / f"Videos/video_{sequence_id:03d}.mp4"),"annotation_source":str(annotation_file),
              "annotation_sha256":sha256(annotation_file),"source_categories":data["categories"],
              "category_mapping":"category_id 1 is fish; generic 'object' alias follows publisher's single-fish-class definition; no eight-class truth inferred",
              "metadata":metadata,"recording_group":metadata["Date of recording"],"reference_dir":str(run / "reference"),
              "ground_truth_dir":str(run / "ground_truth_fish"),"label_type":"fish_binary_0_1","width":1280,"height":720,
              "fps":"5/1","frame_count":len(mapping),"duration_seconds":len(mapping)/5,"annotation_stride":selected["annotation_stride"],
              "timing":"controlled nominal 5fps playback; original MP4 PTS not used to align GT or assert capture cadence",
              "max_selected_source_filename_gap":max(np.diff([m["source_filename_numeric_index"] for m in mapping])).item(),
              "input_temporal_limitation":"Publisher's annotated-frame sequence may have omissions; preserving its order does not establish dense, equally timed original footage",
              "lossless_reference_all_frames_pixel_exact":True,"frames":mapping,"official_record":"https://zenodo.org/records/20617668",
              "license":"CC-BY-4.0","lossless_reference_command":command}
    save_json(run / "manifest.json",record)
    records.append(record)
    print(f"prepared {name}: {len(mapping)} GT-aligned frames; date {record['recording_group']}; lossless roundtrip exact",flush=True)
if len({r["recording_group"] for r in records})!=3: raise ValueError("Recording groups not distinct")
save_json(CORPUS / "manifest.json",{"purpose":"All three sequences reserved for evaluation; no classifier/codec updates", "input_kind":"publisher annotated RGB frame sequences + pixel-exact lossless caches",
          "selection_policy":selection["policy"],"sequences":records})
