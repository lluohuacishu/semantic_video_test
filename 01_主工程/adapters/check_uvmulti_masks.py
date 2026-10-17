"""Check only small published masks, not entire video archives or model updates."""
import io
import json
from pathlib import Path
import urllib.request
import zipfile
import numpy as np
from PIL import Image
from common import ROOT, save_json, sha256
from fetch_sfishtrack import RangeFile

folder=ROOT / "data/uvmulti_preflight"
metadata=json.loads((ROOT / "audit/uvmulti_figshare_metadata.json").read_text("utf-8"))
files={f["name"]:f for f in metadata["files"]}
records=[]
for video in [91,96]:
    file=files[f"video{video}.zip"]
    remote=RangeFile(file["download_url"],file["size"])
    with zipfile.ZipFile(remote) as archive:
        masks=sorted([z for z in archive.infolist() if f"video{video}/mask/" in z.filename and z.filename.endswith(".png")],key=lambda z:z.filename)
        for idx in sorted({0,len(masks)//2,len(masks)-1}):
            info=masks[idx]
            payload=archive.read(info)
            target=folder / "mask_samples" / info.filename
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(payload)
            rgb=np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
            fish=np.all(rgb==np.asarray([0,0,128]),axis=2)
            colors,counts=np.unique(rgb.reshape(-1,3),axis=0,return_counts=True)
            records.append({"video":video,"entry":info.filename,"size_wh":[rgb.shape[1],rgb.shape[0]],"fish_pixels":int(fish.sum()),
                            "colors":[{"rgb":color.tolist(),"pixels":int(count)} for color,count in zip(colors,counts,strict=True)],
                            "crc32_verified":True,"sha256":sha256(target)})
save_json(folder / "mask_availability.json",{"role":"Prospective source inspection only, no formal split or training", "fish_color_from_published_table":[0,0,128],"records":records,
         "limitation":"Only first/middle/last masks sampled; does not establish complete label quality or recording independence"})
print(json.dumps(records,indent=2))
