"""Resolve annotation ordinal vs physical video index using provided frame pixels."""
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw
from common import ROOT, FFMPEG, probe, save_json

DATA = ROOT / "data/sfishtrack_official/selected/SFISHTRACK"
run = ROOT / "runs/sfishtrack_alignment_v2"
run.mkdir(exist_ok=False)
data = json.loads((DATA / "Annotations/video027.json").read_text("utf-8"))
source = DATA / "Videos/video_027.mp4"
raw = subprocess.check_output([str(FFMPEG),"-v","error","-i",str(source),"-an","-vf","scale=480:270:flags=bilinear","-fps_mode","passthrough","-pix_fmt","rgb24","-f","rawvideo","pipe:1"])
video = np.frombuffer(raw,dtype=np.uint8).reshape(-1,270,480,3).astype(np.float32)
if len(video) != int(probe(source)["nb_frames"]): raise ValueError("Decoded frames duplicated or omitted")
matches = []
for filename in ["000002.png","000030.png","000059.png"]:
    ref = np.asarray(Image.open(DATA / f"Frames/video_027/{filename}").convert("RGB").resize((480,270),Image.Resampling.BILINEAR),dtype=np.float32)
    frame = next(f for f in data["images"] if f["file_name"]==filename)
    boxes = [a["bbox"] for a in data["annotations"] if a["image_id"]==frame["id"]]
    if not boxes: raise ValueError("Alignment sentinel has no foreground")
    x0 = max(0,int(min(b[0] for b in boxes)/4)-4)
    y0 = max(0,int(min(b[1] for b in boxes)/4)-4)
    x1 = min(480,int(max(b[0]+b[2] for b in boxes)/4)+5)
    y1 = min(270,int(max(b[1]+b[3] for b in boxes)/4)+5)
    mse = ((video-ref)**2).mean(axis=(1,2,3))
    roi_mse = ((video[:,y0:y1,x0:x1]-ref[y0:y1,x0:x1])**2).mean(axis=(1,2,3))
    ranked = np.argsort(roi_mse)[:5]
    match = {"provided_filename":filename,"annotation_frame_id":frame["frame_id"],"nominal_filename_index":int(Path(filename).stem),
             "best_source_frame_index":int(ranked[0]),"best_global_psnr_db":float(10*np.log10(255**2/mse[ranked[0]])),
             "candidates":[{"source_frame_index":int(idx),"roi_mse":float(roi_mse[idx]),"global_mse":float(mse[idx])} for idx in ranked]}
    matches.append(match)
    print(json.dumps(match,indent=2),flush=True)
offsets = [m["best_source_frame_index"]-m["nominal_filename_index"] for m in matches]
save_json(run / "result.json",{"method":"Provided frame pixel matching, weighted foreground bbox; all video frames searched; three temporal sentinels",
                                 "matches":matches,"filename_to_video_offsets":offsets,"constant_offset":offsets[0] if len(set(offsets))==1 else None})
