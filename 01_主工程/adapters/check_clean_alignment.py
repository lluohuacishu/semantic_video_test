"""Verify first/middle/end annotation image numbering against physical MP4 frames."""
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
from common import ROOT, FFMPEG, save_json

data_root = ROOT / "data/sfishtrack_official/selected/SFISHTRACK"
output = ROOT / "runs/sfishtrack_clean_alignment"
output.mkdir(exist_ok=False)
all_results = []
for video_id in [1,28,54]:
    annotation = json.loads((data_root / f"Annotations/video{video_id:03d}.json").read_text("utf-8"))
    source = data_root / f"Videos/video_{video_id:03d}.mp4"
    sentinels = [0,148,294] if video_id==1 else [0,150,294]
    candidates = sorted(set(n for idx in sentinels for n in range(max(0,idx-5),idx+6)))
    expression = "+".join(f"eq(n,{n})" for n in candidates)
    raw = subprocess.check_output([str(FFMPEG),"-v","error","-i",str(source),"-an","-vf",f"select='{expression}',scale=480:270:flags=bilinear","-fps_mode","passthrough","-pix_fmt","rgb24","-f","rawvideo","pipe:1"])
    video = np.frombuffer(raw,dtype=np.uint8).reshape(-1,270,480,3).astype(np.float32)
    if len(video)!=len(candidates): raise ValueError("Candidate frames outside video coverage")
    for frame_idx in sentinels:
        frame = next(row for row in annotation["images"] if int(row["frame_id"])==frame_idx)
        filename = f"frame_{frame_idx:06d}.jpg" if video_id==1 else f"{frame_idx:06d}.png"
        provided = data_root / f"Frames/video_{video_id:03d}/{filename}"
        ref = np.asarray(Image.open(provided).convert("RGB").resize((480,270),Image.Resampling.BILINEAR),dtype=np.float32)
        boxes = [a["bbox"] for a in annotation["annotations"] if a["image_id"]==frame["id"]]
        if boxes:
            x0,y0=max(0,int(min(b[0] for b in boxes)/4)-5),max(0,int(min(b[1] for b in boxes)/4)-5)
            x1,y1=min(480,int(max(b[0]+b[2] for b in boxes)/4)+6),min(270,int(max(b[1]+b[3] for b in boxes)/4)+6)
        else: x0,y0,x1,y1=0,0,480,270
        mse=((video-ref)**2).mean(axis=(1,2,3))
        roi=((video[:,y0:y1,x0:x1]-ref[y0:y1,x0:x1])**2).mean(axis=(1,2,3))
        selected=candidates.index(frame_idx)
        best=int(np.argmin(roi))
        psnr=float(10*np.log10(255**2/mse[selected]))
        record={"video_id":video_id,"frame_id":frame_idx,"provided_image":str(provided),"nominal_global_psnr_db":psnr,
                "nominal_roi_mse":float(roi[selected]),"best_candidate_frame_id":candidates[best],"best_roi_mse":float(roi[best]),
                "nominal_within_noise":float(roi[selected])<=float(roi[best])*1.05+1.0,"category_names":annotation["categories"]}
        all_results.append(record)
        save_json(output / "result.json",{"checks":all_results})
        print(json.dumps(record),flush=True)
        if psnr<25 or not record["nominal_within_noise"]: raise ValueError(f"Annotation pixels disagree with nominal video frame: {record}")
save_json(output / "result.json",{"all_sentinels_passed":True,"checks":all_results,
         "quarantined_video":27,"reason":"Annotation image numeric index and physical video index have nonconstant offset; no supervised scoring permitted for this clip"})
