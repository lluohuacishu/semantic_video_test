"""Frozen inputs and preregistered controls for the existing stage-D prototype."""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np
from PIL import Image
from scipy import ndimage

PROJECT=Path(r"C:\Users\lenovo\Desktop\works\work\semantic-video-mlvc")
WORKSPACE=Path(r"C:\Users\lenovo\Documents\ChatGPT\语义通信")
OUT=WORKSPACE/"outputs/roi_control_20261015"
RUN=PROJECT/"runs/semantic_mvp_roi_control_20261015"
OLD=PROJECT/"runs/semantic_mvp_20261014_fish"
DIRECTION=WORKSPACE/"现状总结与后续方向_2026-10-15.md"
sys.path.insert(0,str(PROJECT/"semantic_mvp"))
from runtime import sha256,write_json,encode_h265,decode_png,probe,preview,FFMPEG,FFPROBE,run
from system import roi_box,deliver,install_read_guard

def read(path):return json.loads(Path(path).read_text("utf-8"))

def frozen():
    if not DIRECTION.is_file():raise RuntimeError("Direction document must be written before experiment work")
    with zipfile.ZipFile(OLD/"receiver/message.scv") as archive:
        manifest=json.loads(archive.read("manifest.json"))
        semantic_bytes=archive.read("semantics.json.gz")
        semantics=json.loads(gzip.decompress(semantic_bytes))
    source=read(OLD/"sender/source_audit.json")
    data=read(PROJECT/"data/phase_c_20261013/manifest.json")
    video=next(x for x in data["videos"] if x["video"]=="video131")
    labels=[x for x in video["labels"] if 420<=x["frame_index"]<470]
    assert source["source_frame_indices"]==list(range(420,470))
    assert len(semantics["frames"])==source["frames"]==50 and len(labels)==4
    return manifest,semantics,semantic_bytes,source,labels

def plan(records,window,w=1280,h=720):
    return [{"start":start,"frames":len(records[start:start+window]),"box":box} for start in range(0,len(records),window) if (box:=roi_box(records[start:start+window],w,h,16)) is not None]

def geometry(records,window,labels):
    patches=plan(records,window)
    n=len(records);h=720;w=1280
    by_frame={i:p["box"] for p in patches for i in range(p["start"],p["start"]+p["frames"])}
    frame_rows=[];component_rows=[]
    covered=total=0
    for label in labels:
        idx=label["frame_index"]-420
        gt=np.asarray(Image.open(label["fish_mask"]))>0
        enhanced=np.zeros_like(gt)
        if idx in by_frame:
            x,y,pw,ph=by_frame[idx];enhanced[y:y+ph,x:x+pw]=True
        pixels=int(gt.sum());inside=int((gt&enhanced).sum())
        total+=pixels;covered+=inside
        frame_rows.append({"local_frame":idx,"source_frame":label["frame_index"],"gt_pixels":pixels,"covered_gt_pixels":inside,"gt_coverage":inside/pixels if pixels else None,"roi_rectangle_fraction":float(enhanced.mean())})
        components,k=ndimage.label(gt,structure=np.ones((3,3),dtype=np.uint8))
        for component in range(1,k+1):
            mask=components==component;area=int(mask.sum());hit=int((mask&enhanced).sum())
            component_rows.append({"source_frame":label["frame_index"],"component":component,"area":area,"covered_pixels":hit,"coverage":hit/area,"size_group":"le_256" if area<=256 else "257_1024" if area<=1024 else "gt_1024"})
    positives=[x for x in frame_rows if x["gt_pixels"]]
    def comp_summary(rows):
        return {"components":len(rows),"mean_coverage":float(np.mean([x["coverage"] for x in rows])) if rows else None,"high_coverage_fraction":sum(x["coverage"]>=.95 for x in rows)/len(rows) if rows else None}
    return {"window":window,"patches":patches,"segments":len(patches),"mean_rectangle_fraction":sum(p["box"][2]*p["box"][3]*p["frames"] for p in patches)/(n*w*h),"pooled_gt_pixel_coverage":covered/total,"positive_gt_frames":len(positives),"positive_frame_hit_fraction":sum(x["covered_gt_pixels"]>0 for x in positives)/len(positives),"positive_frame_mean_coverage":float(np.mean([x["gt_coverage"] for x in positives])),"positive_frame_high_coverage_fraction":sum(x["gt_coverage"]>=.95 for x in positives)/len(positives),"min_positive_frame_coverage":min(x["gt_coverage"] for x in positives),"frames":frame_rows,"components":component_rows,"component_summary":comp_summary(component_rows),"tiny_component_summary":comp_summary([x for x in component_rows if x["area"]<=256]),"scope":"seen training clip; four sparse publisher GT; components are not verified fish instances"}

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    m,s,b,source,labels=frozen()
    assert sha256(PROJECT/"runs/phase_c_20261013/segformer_adaptation_preserve8/best.pt")==m["teacher"]["model_sha256"]
    refs=OLD/"sender/private_frames"
    for i,expected in enumerate(source["rgb_sha256"]):
        pixels=np.asarray(Image.open(refs/f"{i:05d}.png").convert("RGB"))
        assert hashlib.sha256(pixels.tobytes()).hexdigest()==expected
    for label in labels:
        mask=np.asarray(Image.open(label["fish_mask"]))
        assert mask.shape==(720,1280) and np.isin(mask,[0,1]).all() and int(mask.sum())==label["fish_pixels"]
    protocol={"stage":"D controlled diagnosis; not a new stage","date":"2026-10-15","direction_document":str(DIRECTION),"direction_sha256":sha256(DIRECTION),"direction_written_before_experiment":True,"target":"restore video and protect important objects","reference_dir":str(refs),"source":source,"teacher":m["teacher"],"semantic_gzip_sha256":hashlib.sha256(b).hexdigest(),"fixed_predictions_reused":True,"gt_frames":[{"frame":x["frame_index"],"pixels":x["fish_pixels"],"file":x["fish_mask"],"sha256":sha256(x["fish_mask"])} for x in labels],"groups":{"A":{"roi":"fixed predictions","window":32},"B":{"roi":"same fixed predictions","window":4},"C":{"roi":"continuous verified human/publisher boxes","window":32,"status":"waiting for 46 frames human verification"},"D":{"roi":"same continuous verified boxes","window":4,"status":"waiting for 46 frames human verification"}},"target_complete_bytes":100000,"tolerance":.02,"selection":"absolute complete file byte difference only; no task scores","background":{"scale":.5,"crf":42},"roi_padding":16,"roi_min_component_pixels":64,"roi_h265":"medium, P-only GOP32, zerolatency, BT709 limited YUV420","baseline":"native full-frame H265 MP4, medium, B4, GOP32, same color configuration","transport":"independent process, 20 decimal kbps, real TCP loopback","gt_high_coverage_threshold":.95,"tiny_component_threshold_pixels":256,"no_new_training":True,"fixed_crf_diagnostic":32,"limitations":["Known training video; not independent test","Only four GT frames for task accuracy; other 46 unknown","GT box annotations will not become pixel segmentation GT","Cached sender perception excludes perception time from new encoder timing"]}
    write_json(RUN/"protocol.json",protocol);write_json(OUT/"实验协议.json",protocol)
    geometries=[geometry(s["frames"],window,labels) for window in [1,4,8,16,32]]
    write_json(OUT/"逐帧与小目标覆盖复核.json",geometries)
    kit=OUT/"连续人工ROI核对";images=kit/"frames";images.mkdir(parents=True,exist_ok=True)
    known={x["frame_index"]:x for x in labels};annotations=[]
    for idx in range(420,470):
        image=Path(source["source"])/f"{idx:05d}.jpg"
        shutil.copy2(image,images/image.name)
        record={"source_frame":idx,"local_frame":idx-420,"image":f"frames/{idx:05d}.jpg","width":1280,"height":720,"status":"unreviewed","no_fish":None,"boxes":[],"source":"missing; prediction is not GT"}
        if idx in known:
            mask=np.asarray(Image.open(known[idx]["fish_mask"]))>0
            comps,k=ndimage.label(mask,structure=np.ones((3,3),dtype=np.uint8))
            boxes=[]
            for sl in ndimage.find_objects(comps):
                if sl is not None:
                    yy,xx=sl;boxes.append([xx.start,yy.start,xx.stop-xx.start,yy.stop-yy.start])
            record.update({"status":"publisher_verified","no_fish":not boxes,"boxes":boxes,"source":"publisher semantic mask converted to component bounding boxes; not instance IDs"})
        annotations.append(record)
    template={"dataset":"video131_00420_00469","purpose":"oracle ROI diagnosis; no pixel segmentation GT inferred from boxes","frames":annotations,"confirmed_frames":4,"unknown_frames":46}
    if not (kit/"annotations.json").exists():write_json(kit/"annotations.json",template)
    write_json(kit/"publisher_template.json",template)
    (kit/"标注须知.md").write_text("# 连续ROI人工核对\n\n50帧中4帧来自发布者掩码，其余46帧默认未核对。请逐帧检查原图中的全部鱼，画出目标框，并明确确认本帧标注完成或无鱼。模型预测、插值和传播不自动成为GT。目标框用于ROI控制；正式分割IoU仍只使用已有发布者像素掩码。全部帧明确核对后才启动人工ROI组C/D。\n",encoding="utf-8")
    print(json.dumps({"protocol":str(RUN/"protocol.json"),"direction_sha256":protocol["direction_sha256"],"verified_original_frames":50,"verified_gt_frames":4,"annotation_kit":str(kit),"unknown_frames":46},ensure_ascii=False),flush=True)

if __name__=="__main__":prepare()
