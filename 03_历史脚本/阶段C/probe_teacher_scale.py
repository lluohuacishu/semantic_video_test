"""Development-only resolution/threshold diagnostic before any codec task score."""
import json
import numpy as np
from PIL import Image
from experiment_common import *
from adapt_segformer import binary_model
from semantic_teacher import load_teacher


def full_resolution_probability(model,path,binary=True):
    rgb=np.asarray(Image.open(path).convert("RGB")).copy();height,width=rgb.shape[:2]
    accumulation=torch.zeros((1,2 if binary else 8,height,width),device="cuda")
    counts=torch.zeros((1,1,height,width),device="cuda")
    xs=sorted(set(list(range(0,max(1,width-512),384))+[max(0,width-512)]))
    ys=sorted(set(list(range(0,max(1,height-512),384))+[max(0,height-512)]))
    with torch.inference_mode():
        for y in ys:
            for x in xs:
                patch=torch.from_numpy(rgb[y:y+512,x:x+512].copy()).permute(2,0,1)[None].float().cuda()/255
                logits=model(patch)
                accumulation[:,:,y:y+512,x:x+512]+=logits
                counts[:,:,y:y+512,x:x+512]+=1
        logits=accumulation/counts
        probability=logits.softmax(1)[:,1 if binary else 6]
        winner=logits.argmax(1)
        return probability[0].cpu().numpy(),winner[0].cpu().numpy()


def main():
    configure()
    dev=[v for v in manifest()["videos"] if v["experiment_role"]=="development_validation"]
    records=[]
    for name,model,binary in [("adapted_binary",binary_model(checkpoint=RUN/"segformer_adaptation/best.pt"),True),("legacy_8class",load_teacher()[0],False)]:
        model.eval().requires_grad_(False)
        thresholds=[.5,.75,.9,.95,.99] if binary else [.1,.25,.5,.75,.9]
        totals={t:[0,0,0] for t in thresholds};argmax_total=[0,0,0]
        for video in dev:
            for entry in video["labels"]:
                probability,winner=full_resolution_probability(model,entry["rgb"],binary)
                gt=np.asarray(Image.open(entry["fish_mask"]))>0
                for threshold in thresholds:
                    pred=probability>=threshold
                    counts=[int((pred&gt).sum()),int((pred&~gt).sum()),int((~pred&gt).sum())]
                    totals[threshold]=[a+b for a,b in zip(totals[threshold],counts,strict=True)]
                pred=winner==(1 if binary else 6)
                counts=[int((pred&gt).sum()),int((pred&~gt).sum()),int((~pred&gt).sum())]
                argmax_total=[a+b for a,b in zip(argmax_total,counts,strict=True)]
        for threshold,counts in totals.items():records.append({"model":name,"input":"512 tiles, stride384, uniform logit averaging","threshold":threshold,**binary_metrics(*counts)})
        records.append({"model":name,"input":"512 tiles, stride384, uniform logit averaging","threshold":"argmax",**binary_metrics(*argmax_total)})
        del model;torch.cuda.empty_cache()
    save_json(RUN/"teacher_scale_diagnostic.json",{"development_only":True,"official_validation_used":False,"codec_task_scores_seen":False,"records":records})
    print(json.dumps(records,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
