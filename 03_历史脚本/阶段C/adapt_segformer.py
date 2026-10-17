"""Independent binary fish adaptation; select only on original development RGB."""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
from PIL import Image
from scipy import ndimage
from experiment_common import *
from semantic_teacher import load_teacher, fit_image, FrozenSegFormer
from torch import nn

SIZE=512
SEED=20261014


def deterministic_binary_ce(logits,labels):
    valid=(labels!=255).float();gt=(labels==1).float()
    z=logits[:,1]-logits[:,0]
    return ((F.softplus(z)*(1-gt)+4*F.softplus(-z)*gt)*valid).sum()/((1+3*gt)*valid).sum()


def binary_model(device="cuda",checkpoint=None):
    model,_=load_teacher(device)
    old=model.network.decode_head.classifier
    new=nn.Conv2d(old.in_channels,2,kernel_size=1).to(device)
    with torch.no_grad():
        bg=[i for i in range(8) if i!=6]
        new.weight[0].copy_(old.weight[bg].mean(0));new.weight[1].copy_(old.weight[6])
        new.bias[0].copy_(old.bias[bg].mean());new.bias[1].copy_(old.bias[6])
    model.network.decode_head.classifier=new
    model.network.config.num_labels=2
    model.network.config.id2label={0:"non_fish",1:"fish"}
    model.network.config.label2id={"non_fish":0,"fish":1}
    if checkpoint:
        state=torch.load(checkpoint,map_location="cpu",weights_only=True)
        model.load_state_dict(state["model"],strict=True)
    return model


def predict_fish(model,path,size=SIZE):
    image=Image.open(path).convert("RGB")
    tensor,_,g=fit_image(image,size)
    with torch.inference_mode():
        logits=model(tensor[None].cuda())
        logits=logits[:,:,g["top"]:g["top"]+g["scaled_height"],g["left"]:g["left"]+g["scaled_width"]]
        logits=F.interpolate(logits,size=(g["height"],g["width"]),mode="bilinear",align_corners=False)
        return (logits.argmax(1)[0].cpu().numpy()==1)


def evaluate(model,videos):
    model.eval()
    result=[]
    for video in videos:
        tp=fp=fn=0
        for entry in video["labels"]:
            pred=predict_fish(model,entry["rgb"])
            gt=np.asarray(Image.open(entry["fish_mask"]))>0
            tp+=int((pred&gt).sum());fp+=int((pred&~gt).sum());fn+=int((~pred&gt).sum())
        result.append({"video":video["video"],"labelled_frames":len(video["labels"]),**binary_metrics(tp,fp,fn)})
    valid=[r["fish_iou"] for r in result if r["fish_iou"] is not None]
    return {"per_video":result,"macro_fish_iou":float(np.mean(valid)) if valid else None}


def train(steps):
    configure();rng_seed(SEED)
    output=RUN/"segformer_adaptation"
    output.mkdir(parents=True,exist_ok=False)
    m=manifest();training=[v for v in m["videos"] if v["experiment_role"]=="train"]
    development=[v for v in m["videos"] if v["experiment_role"]=="development_validation"]
    entries=[(v,x) for v in training for x in v["labels"]]
    positives=[(v,x) for v,x in entries if x["fish_pixels"]>0]
    if not positives:raise ValueError("No positive fish GT")
    config={"task":"binary fish / non-fish; UVMulti colors mapped explicitly","initialization":"legacy SUIM SegFormer-B0 encoder and decoder, binary classifier initialized from fish row and mean non-fish rows",
            "initial_legacy_weight_sha256":"ff00cb439a4af290dc3efde8ae76d4acfad797784483b669ab6db79aaf5b1476","steps":steps,"seed":SEED,"lr":2e-5,"optimizer":"AdamW","input_size":SIZE,
            "loss":"cross-entropy class weights [1,4] + 0.5 fish Dice","sampling":"50% fish-directed crop, 25% random crop, 25% full image; train video GT only",
            "selection":"macro per-video fish IoU on ORIGINAL development images every128 steps; no reconstructed codec images or final test used",
            "dataset_manifest_sha256":sha256(DATA/"manifest.json"),"development_validation_is_official_validation":False,"test_used":False}
    save_json(output/"config.json",config)
    model=binary_model().requires_grad_(True)
    original=evaluate(model,development)
    save_json(output/"initial_binary_evaluation.json",original)
    best_score=original["macro_fish_iou"]
    best_step=0
    torch.save({"model":model.state_dict(),"config":config,"selected_step":0},output/"best.pt")
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    rng=np.random.default_rng(SEED)
    logs=[];history=[{"step":0,**original}]
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (output/"steps.jsonl").open("x",encoding="utf-8") as log:
        for step in range(1,steps+1):
            guided=step%4<2
            choices=positives if guided else entries
            video,entry=choices[int(rng.integers(len(choices)))]
            image=Image.open(entry["rgb"]).convert("RGB");mask=Image.open(entry["fish_mask"]).convert("L")
            if step%4!=3:
                crop=int(rng.choice([256,384,512]))
                if guided:
                    binary=np.asarray(mask)>0
                    cc,count=ndimage.label(binary,np.ones((3,3),bool));obj=ndimage.find_objects(cc)[int(rng.integers(count))]
                    cx=(obj[1].start+obj[1].stop)//2+int(rng.integers(-crop//6,crop//6+1));cy=(obj[0].start+obj[0].stop)//2+int(rng.integers(-crop//6,crop//6+1))
                    left=int(np.clip(cx-crop//2,0,image.width-crop));top=int(np.clip(cy-crop//2,0,image.height-crop))
                else:left=int(rng.integers(image.width-crop+1));top=int(rng.integers(image.height-crop+1))
                bounds=(left,top,left+crop,top+crop);image=image.crop(bounds);mask=mask.crop(bounds)
            if rng.integers(2):image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT);mask=mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            tensor,labels,_=fit_image(image,SIZE,mask)
            tensor=tensor[None].cuda();labels=labels[None].long().cuda()
            model.train();optimizer.zero_grad(set_to_none=True);rng_seed(SEED+step)
            logits=model(tensor)
            ce=deterministic_binary_ce(logits,labels)
            valid=labels!=255;gt=(labels==1).float();prob=logits.softmax(1)[:,1]
            dice=1-(2*(prob*gt*valid).sum()+1)/((prob*valid).sum()+(gt*valid).sum()+1)
            loss=ce+.5*dice
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite segmentation loss")
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            record={"step":step,"video":video["video"],"frame":entry["frame_index"],"fish_guided":guided,"loss":loss.item(),"ce":ce.item(),"dice":dice.item(),"gradient_norm":norm.item(),"peak_allocated_bytes":torch.cuda.max_memory_allocated()}
            log.write(json.dumps(record,ensure_ascii=False)+"\n");log.flush();logs.append(record)
            if step%32==0:print("SEGFORMER",step,"loss",loss.item(),"peakMiB",torch.cuda.max_memory_allocated()/2**20,flush=True)
            if step%128==0 or step==steps:
                measured=evaluate(model,development);history.append({"step":step,**measured})
                save_json(output/"validation_history.json",history)
                torch.save({"model":model.state_dict(),"config":config,"selected_step":step},output/f"step_{step}.pt")
                if measured["macro_fish_iou"]>best_score:
                    best_score=measured["macro_fish_iou"];best_step=step
                    torch.save({"model":model.state_dict(),"config":config,"selected_step":step},output/"best.pt")
                print("SEGFORMER_VALIDATION",step,measured,flush=True)
    save_json(output/"result.json",{"success":True,"steps":steps,"selected_step":best_step,"best_macro_original_fish_iou":best_score,"selected_checkpoint_sha256":sha256(output/"best.pt"),"seconds":time.perf_counter()-start,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"test_used":False,"teacher_fixed_before_codec_task_evaluation":True})
    model.eval().requires_grad_(False)
    save_json(output/"fixed_original_evaluation.json",evaluate(model if best_step==steps else binary_model(checkpoint=output/"best.pt"),development))
    print("SEGFORMER_ADAPTATION_OK",best_step,best_score,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--steps",type=int,default=512);args=parser.parse_args();train(args.steps)
