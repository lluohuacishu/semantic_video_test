"""Verify the MVP and compare actual decoded images using frozen perception."""
from __future__ import annotations
import argparse
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from runtime import *

def psnr(error,pixels):
    return float(10*math.log10(255**2*pixels/error)) if error and pixels else None

def counts(pred,truth):
    return np.array([np.count_nonzero(pred&truth),np.count_nonzero(pred&~truth),np.count_nonzero(~pred&truth)],dtype=np.int64)

def tasks(c):
    tp,fp,fn=(int(x) for x in c)
    return {"tp":tp,"fp":fp,"fn":fn,"iou":tp/(tp+fp+fn) if tp+fp+fn else None,"recall":tp/(tp+fn) if tp+fn else None,"precision":tp/(tp+fp) if tp+fp else None}

def baseline_by_bytes(frames,output,fps,target,n):
    output.mkdir()
    candidates={}
    low,high=0.,51.
    for step in range(12):
        crf=round((low+high)/2,3)
        file=output/f"crf_{crf:g}.mp4"
        encode_h265(frames,file,fps,crf,count=n,offline=True)
        candidates[crf]={"file":str(file),"bytes":file.stat().st_size,"crf":crf}
        if abs(file.stat().st_size-target)/target<=.005:
            break
        if file.stat().st_size>target:
            low=crf
        else:
            high=crf
    selected=min(candidates.values(),key=lambda x:abs(x["bytes"]-target))
    selected["relative_complete_file_difference"]=(selected["bytes"]-target)/target
    selected["selection"]="complete file bytes only; before computing task/quality metrics"
    selected["profile"]="libx265 medium, GOP32, B4, offline; no semantic metadata"
    decode_png(selected["file"],output/"rgb")
    write_json(output/"selection.json",{"target_full_semantic_packet_bytes":target,"selected":selected,"candidates":list(candidates.values())})
    return selected

def validate(args):
    root=Path(args.run).resolve()
    output=fresh(root/"validation")
    sender=root/"sender";receiver=root/"receiver"
    encoded=json.loads((sender/"encode.json").read_text("utf-8"))
    end=json.loads((root/"end_to_end.json").read_text("utf-8"))
    decoded=json.loads((receiver/"decode.json").read_text("utf-8"))
    audit=json.loads((receiver/"receiver_read_audit.json").read_text("utf-8"))
    source=json.loads((sender/"source_audit.json").read_text("utf-8"))
    assert end["separate_sender_receiver_processes"] and end["packet_sha256_equal"]
    assert sha256(sender/"message.scv")==sha256(receiver/"message.scv")==decoded["file_sha256"]
    assert decoded["frames"]==source["frames"]==encoded["frames"]
    assert all(x["allowed"] for x in audit["python_reads"])
    with zipfile.ZipFile(sender/"message.scv") as archive:
        semantic=json.loads(gzip.decompress(archive.read("semantics.json.gz")))
    assert semantic==json.loads((receiver/"target_information.json").read_text("utf-8"))
    assert sum(encoded[x] for x in ["context_bytes","roi_stream_bytes","semantic_bytes","manifest_and_zip_bytes"])==encoded["file_bytes"]
    # Byte matching never sees GT, source predictions or reconstructed quality.
    baseline=baseline_by_bytes(sender/"private_frames",output/"matched_h265",source["fps"],encoded["file_bytes"],source["frames"])
    from teacher import Teacher
    teacher=Teacher(encoded["target"])
    labels=[]
    if encoded["target"]=="fish":
        manifest=json.loads((ROOT/"data/phase_c_20261013/manifest.json").read_text("utf-8"))
        video=next(x for x in manifest["videos"] if x["video"]=="video131")
        real_indices=source["source_frame_indices"]
        lookup={idx:local for local,idx in enumerate(real_indices)}
        labels=[{**x,"local":lookup[x["frame_index"]]} for x in video["labels"] if x["frame_index"] in lookup]
    per_stream=[]
    references=sender/"private_frames"
    views=[("original",references,None),("context_only",receiver/"context_rgb",encoded["context_bytes"]),("semantic_video",receiver/"rgb",encoded["file_bytes"]),("matched_offline_h265",output/"matched_h265/rgb",baseline["bytes"])]
    source_masks=[]
    for idx,path in enumerate(sorted(references.glob("*.png"))):
        mask,_=teacher.mask(path);source_masks.append(mask)
    for name,directory,file_bytes in views:
        sse=roi_sse=bg_sse=0.
        samples=roi_samples=bg_samples=0
        agreement=np.zeros(3,dtype=np.int64)
        gt_total=np.zeros(3,dtype=np.int64)
        gt_sse=gt_bg_sse=0.;gt_samples=gt_bg_samples=0
        gt_by_index={x["local"]:x for x in labels}
        frame_metrics=[]
        for idx,path in enumerate(sorted(directory.glob("*.png"))):
            raw=np.asarray(Image.open(references/f"{idx:05d}.png").convert("RGB"),dtype=np.float64)
            image=np.asarray(Image.open(path).convert("RGB"),dtype=np.float64)
            error=(raw-image)**2
            roi=source_masks[idx]
            sse+=float(error.sum());samples+=error.size
            roi_sse+=float(error[roi].sum());roi_samples+=int(roi.sum())*3
            bg_sse+=float(error[~roi].sum());bg_samples+=int((~roi).sum())*3
            pred=source_masks[idx] if name=="original" else teacher.mask(path)[0]
            c=counts(pred,roi);agreement+=c
            item={"frame":idx,"source_prediction_consistency":tasks(c)}
            if idx in gt_by_index:
                gt=np.asarray(Image.open(gt_by_index[idx]["fish_mask"]))>0
                cgt=counts(pred,gt);gt_total+=cgt
                gt_sse+=float(error[gt].sum());gt_samples+=int(gt.sum())*3
                gt_bg_sse+=float(error[~gt].sum());gt_bg_samples+=int((~gt).sum())*3
                item["fish_gt"]=tasks(cgt)
            frame_metrics.append(item)
        row={"name":name,"file_bytes":file_bytes,"global_rgb_psnr_db":psnr(sse,samples),"source_predicted_target_rgb_psnr_db":psnr(roi_sse,roi_samples),"background_rgb_psnr_db":psnr(bg_sse,bg_samples),"source_prediction_consistency":tasks(agreement),"receiver_task_predictions_are_independently_computed":True}
        if labels:
            row.update({"fish_gt":tasks(gt_total),"gt_fish_rgb_psnr_db":psnr(gt_sse,gt_samples),"gt_background_rgb_psnr_db":psnr(gt_bg_sse,gt_bg_samples),"gt_frames":len(labels)})
        per_stream.append(row)
        write_json(output/f"{name}_per_frame.json",frame_metrics)
        say="VALIDATED "+name+" "+json.dumps(row,ensure_ascii=False)
        print(say,flush=True)
    result={"success":True,"end_to_end_verified":True,"semantic_metadata_exact":True,"complete_byte_accounting_verified":True,"all_receiver_python_media_reads_local":True,"teacher":teacher.profile,"source":source,"baseline":baseline,"rows":per_stream,"gt_limitation":"video131 was a training video for the fixed fish model; 4 sparse GT frames in this engineering demo; no independent research conclusion" if labels else "no manual GT; source-prediction consistency is not task accuracy"}
    write_json(output/"results.json",result)
    make_comparison(root,views,labels)
    return result

def make_comparison(root,views,labels):
    idx=max(labels,key=lambda x:x["fish_pixels"])["local"] if labels else 25
    w,h=Image.open(views[0][1]/f"{idx:05d}.png").size
    canvas=Image.new("RGB",(w*2,h*2+88),(15,28,40))
    pen=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc",25)
    except OSError:font=ImageFont.load_default()
    names=["原始输入","只恢复低清背景","语义分层恢复","近似完整文件预算 H.265（B帧）"]
    for k,(name,directory,_) in enumerate(views):
        x=(k%2)*w;y=(k//2)*(h+44)
        pen.text((x+14,y+8),names[k],font=font,fill=(230,242,250))
        canvas.paste(Image.open(directory/f"{idx:05d}.png"),(x,y+44))
    canvas.resize((1920,round(canvas.height*1920/canvas.width))).save(root/"validation/恢复画面对比.jpg",quality=95)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--run",required=True)
    validate(parser.parse_args())
