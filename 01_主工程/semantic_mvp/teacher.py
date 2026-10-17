"""Frozen sender/evaluation models; no training, no access to GT at encoding."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from runtime import ROOT, sha256

sys.path.insert(0,str(ROOT/"adapters"))
from common import configure, torch
from semantic_teacher import load_teacher, predict, CHECKPOINT, EXPECTED_SHA

FISH_CONFIG=ROOT/"runs/phase_c_20261013/fixed_teacher.json"

class Teacher:
    def __init__(self,target,device="cuda"):
        self.device=device
        configure(device)
        self.model,state=load_teacher(device)
        self.target=target
        if target=="fish":
            import json
            fixed=json.loads(FISH_CONFIG.read_text("utf-8"))
            if fixed["architecture"]!="preserved_8class" or fixed["inference"]!="tiles512":
                raise ValueError("Fish teacher profile changed")
            if sha256(fixed["checkpoint"])!=fixed["checkpoint_sha256"]:
                raise ValueError("Fish teacher hash mismatch")
            checkpoint=torch.load(fixed["checkpoint"],map_location="cpu",weights_only=True)
            self.model.load_state_dict(checkpoint["model"],strict=True)
            self.threshold=float(fixed["threshold"])
            self.profile={"target":"fish","class_id":6,"label":"鱼类","model_sha256":fixed["checkpoint_sha256"],"inference":"512 tiles, stride384, mean overlapping logits","threshold":self.threshold,"reliability":"limited; training video131 demo does not establish generalization"}
        else:
            self.threshold=.7
            self.profile={"target":"diver","class_id":1,"label":"潜水员","model_sha256":EXPECTED_SHA,"inference":"long edge 256, letterbox, argmax class1 and confidence>=0.7","threshold":self.threshold,"reliability":"demo has no manual GT; consistency only"}
        self.model.eval().requires_grad_(False)

    def mask(self,path):
        with torch.inference_mode():
            if self.target=="diver":
                labels,conf,_=predict(self.model,path,256,self.device)
                return (labels==1)&(conf>=self.threshold),conf
            rgb=np.asarray(Image.open(path).convert("RGB")).copy()
            h,w=rgb.shape[:2]
            total=torch.zeros((1,8,h,w),device=self.device)
            counts=torch.zeros((1,1,h,w),device=self.device)
            xs=sorted(set(list(range(0,max(1,w-512),384))+[max(0,w-512)]))
            ys=sorted(set(list(range(0,max(1,h-512),384))+[max(0,h-512)]))
            for y in ys:
                for x in xs:
                    crop=torch.from_numpy(rgb[y:y+512,x:x+512].copy()).permute(2,0,1)[None].float().to(self.device)/255
                    total[:,:,y:y+512,x:x+512]+=self.model(crop)
                    counts[:,:,y:y+512,x:x+512]+=1
            prob=(total/counts).softmax(1)[0,6].cpu().numpy()
            return prob>=self.threshold,prob

def regions(mask,confidence,min_pixels=64):
    components,n=ndimage.label(mask,structure=np.ones((3,3),dtype=np.uint8))
    sizes=np.bincount(components.ravel())
    kept=np.zeros_like(mask,dtype=bool)
    objects=[]
    for label,sl in enumerate(ndimage.find_objects(components),1):
        if sl is None or sizes[label]<min_pixels:
            continue
        ys,xs=sl
        local=components[sl]==label
        kept[sl] |= local
        objects.append({"box":[xs.start,ys.start,xs.stop-xs.start,ys.stop-ys.start],"pixels":int(sizes[label]),"confidence":round(float(confidence[sl][local].mean()),4)})
    # Regions are connected segmentation components, not verified object counts.
    return kept,sorted(objects,key=lambda o:o["pixels"],reverse=True)
