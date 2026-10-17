"""Preserve the eight-class competition; adapt decoder on train GT only."""
import json
import time
import numpy as np
from PIL import Image
from scipy import ndimage
from torch import nn
from experiment_common import *
from semantic_teacher import load_teacher,fit_image
from probe_teacher_scale import full_resolution_probability

SEED=20261015
THRESHOLDS=[.1,.25,.5,.75,.9]


def calibration(model,videos):
    model.eval()
    totals={t:[0,0,0] for t in THRESHOLDS}
    for video in videos:
        for entry in video["labels"]:
            probability,_=full_resolution_probability(model,entry["rgb"],binary=False)
            gt=np.asarray(Image.open(entry["fish_mask"]))>0
            for t in THRESHOLDS:
                pred=probability>=t
                counts=[int((pred&gt).sum()),int((pred&~gt).sum()),int((~pred&gt).sum())]
                totals[t]=[a+b for a,b in zip(totals[t],counts,strict=True)]
    rows=[{"threshold":t,**binary_metrics(*counts)} for t,counts in totals.items()]
    return rows,max(rows,key=lambda x:x["fish_iou"] or 0)


def main():
    configure();rng_seed(SEED)
    folder=RUN/"segformer_adaptation_preserve8";folder.mkdir(parents=True,exist_ok=False)
    m=manifest();training=[v for v in m["videos"] if v["experiment_role"]=="train"];dev=[v for v in m["videos"] if v["experiment_role"]=="development_validation"]
    entries=[x for v in training for x in v["labels"]];positives=[x for x in entries if x["fish_pixels"]>0]
    model,_=load_teacher();model.network.decode_head.requires_grad_(True)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    config={"variant":"preserve8_decoder_only","reason":"naive two-class mean background initialization weakened existing class competition; first adaptation and scale diagnostics retained as negative results",
            "steps":1024,"seed":SEED,"lr":1e-4,"encoder":"frozen pretrained SUIM","batchnorm_statistics":"frozen pretrained","loss":"binary CE on FV-vs-logsumexp(other7), fish weight8, +0.1 Dice",
            "sampling":"25% positive fish crop512,25% random crop512,50% full-frame letterbox512","evaluation":"native-scale 512 tiles stride384, mean overlapping logits",
            "selection":"original video96 only, checkpoints0/256/512/768/1024 and thresholds0.1/0.25/0.5/0.75/0.9; official video108 and codec reconstructions not used","test_used":False}
    save_json(folder/"config.json",config)
    rows,best=calibration(model,dev);history=[{"step":0,"rows":rows}];score=best["fish_iou"];best_step=0;best_threshold=best["threshold"]
    torch.save({"model":model.state_dict(),"config":config,"selected_step":0,"threshold":best_threshold},folder/"best.pt")
    rng=np.random.default_rng(SEED);start=time.perf_counter()
    with (folder/"steps.jsonl").open("x",encoding="utf-8") as log:
        for step in range(1,1025):
            guided=step%4==0;entry=(positives if guided else entries)[int(rng.integers(len(positives if guided else entries)))]
            image=Image.open(entry["rgb"]).convert("RGB");mask=Image.open(entry["fish_mask"]).convert("L")
            if step%4 in (0,1):
                if guided:
                    binary=np.asarray(mask)>0;cc,count=ndimage.label(binary,np.ones((3,3),bool));obj=ndimage.find_objects(cc)[int(rng.integers(count))]
                    left=int(np.clip((obj[1].start+obj[1].stop)//2-256+int(rng.integers(-64,65)),0,768));top=int(np.clip((obj[0].start+obj[0].stop)//2-256+int(rng.integers(-64,65)),0,208))
                else:left=int(rng.integers(769));top=int(rng.integers(209))
                bounds=(left,top,left+512,top+512);image=image.crop(bounds);mask=mask.crop(bounds)
            if rng.integers(2):image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT);mask=mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            tensor,labels,_=fit_image(image,512,mask);tensor=tensor[None].cuda();labels=labels[None].long().cuda()
            model.train();model.network.segformer.eval()
            for module in model.modules():
                if isinstance(module,nn.BatchNorm2d):module.eval()
            optimizer.zero_grad(set_to_none=True);rng_seed(SEED+step)
            logits=model(tensor)
            bg=torch.logsumexp(logits[:,[0,1,2,3,4,5,7]],dim=1);z=logits[:,6]-bg
            gt=(labels==1).float();valid=(labels!=255).float()
            ce=((F.softplus(z)*(1-gt)+8*F.softplus(-z)*gt)*valid).sum()/((1+7*gt)*valid).sum()
            prob=torch.sigmoid(z);dice=1-(2*(prob*gt*valid).sum()+1)/((prob*valid).sum()+(gt*valid).sum()+1)
            loss=ce+.1*dice
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite teacher loss")
            loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);optimizer.step()
            log.write(json.dumps({"step":step,"frame":entry["frame_index"],"rgb":entry["rgb"],"guided":guided,"loss":loss.item(),"gradient_norm":norm.item()})+"\n");log.flush()
            if step%128==0:print("PRESERVE8",step,loss.item(),flush=True)
            if step%256==0:
                rows,best=calibration(model,dev);history.append({"step":step,"rows":rows});save_json(folder/"validation_history.json",history)
                torch.save({"model":model.state_dict(),"config":config,"step":step},folder/f"step_{step}.pt")
                if best["fish_iou"]>score:
                    score=best["fish_iou"];best_step=step;best_threshold=best["threshold"]
                    torch.save({"model":model.state_dict(),"config":config,"selected_step":step,"threshold":best_threshold},folder/"best.pt")
                print("PRESERVE8_VALIDATION",step,best,flush=True)
    save_json(folder/"result.json",{"success":True,"steps":1024,"best_step":best_step,"best_dev_fish_iou":score,"best_threshold":best_threshold,"checkpoint_sha256":sha256(folder/"best.pt"),"seconds":time.perf_counter()-start})
    fixed={"architecture":"preserved_8class","checkpoint":str(folder/"best.pt"),"checkpoint_sha256":sha256(folder/"best.pt"),"inference":"tiles512","threshold":best_threshold,"selected_step":best_step,
           "selection_data":"original official-train video96 held out for development","official_val108_used_for_selection":False,"codec_task_scores_used_for_selection":False,"development_original_fish_iou":score,
           "reliability":"limited" if score<.3 else "development performance acceptable; cross-video validation still pending","frozen_before_codec_task_evaluation":True}
    save_json(RUN/"fixed_teacher.json",fixed)
    print("FIXED_TEACHER_OK",fixed,flush=True)


if __name__=="__main__":main()
