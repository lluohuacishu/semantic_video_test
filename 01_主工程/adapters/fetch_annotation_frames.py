"""Select before model scoring; use exactly the RGB frames associated with masks."""
import json
from pathlib import Path
import subprocess
import sys
from fractions import Fraction
from common import ROOT, probe, save_json

DATA = ROOT / "data/sfishtrack_official/selected/SFISHTRACK"
selection = []
entries = []
for video_id in [1,27,54]:
    annotation = json.loads((DATA / f"Annotations/video{video_id:03d}.json").read_text("utf-8"))
    info = probe(DATA / f"Videos/video_{video_id:03d}.mp4")
    stride = max(1,round(float(Fraction(info["avg_frame_rate"]))/5))
    frames = sorted(annotation["images"],key=lambda r:int(r["frame_id"]))[::stride][:50]
    selected = []
    for frame in frames:
        filename = f"frame_{int(frame['frame_id']):06d}.jpg" if video_id==1 else frame["file_name"]
        entry = f"SFISHTRACK/Frames/video_{video_id:03d}/{filename}"
        entries.append(entry)
        selected.append({"coco_image_id":frame["id"],"annotation_frame_id":frame["frame_id"],"annotation_file_name":frame["file_name"],"archive_entry":entry})
    selection.append({"video_id":video_id,"native_mp4_probe_for_context_only":info,"annotation_stride":stride,"frame_count":len(selected),"frames":selected})
save_json(ROOT / "data/annotation_frame_selection.json",{"policy":"Video IDs 1,27,54 chosen before semantic scores. Use publisher's annotated RGB image frames, not uncertain MP4 mapping. Stride 6/3/6, up to 50 frames, fixed nominal 5fps playback; no claim of original capture timing.","sequences":selection})
subprocess.run([sys.executable,str(ROOT / "adapters/fetch_sfishtrack.py"),"--reuse","--entry",*entries],cwd=ROOT,check=True)
