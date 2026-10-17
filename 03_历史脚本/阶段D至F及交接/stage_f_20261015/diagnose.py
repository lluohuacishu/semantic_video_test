"""Stage F, step 1: frozen-model probability and ROI failure diagnosis only."""
from __future__ import annotations
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
from PIL import Image
from scipy import ndimage

WORKSPACE=Path(r"C:\Users\lenovo\Documents\ChatGPT\语义通信")
PROJECT=Path(r"C:\Users\lenovo\Desktop\works\work\semantic-video-mlvc")
OUT=WORKSPACE/"outputs/semantic_roi_phase_f/step1_probability_diagnosis"
RUN=PROJECT/"runs/semantic_mvp_stage_f_20261015/step1"
OLD=PROJECT/"runs/semantic_mvp_20261014_fish"
CONTROL=PROJECT/"runs/semantic_mvp_roi_control_20261015"
STAGE_E=PROJECT/"runs/semantic_mvp_stage_e_20261015"
FRAMES=[435,450,465]
sys.path.insert(0,str(PROJECT/"semantic_mvp"))
from runtime import sha256,write_json
from system import roi_box
from roi_layout import plan_halves
from teacher import Teacher,regions

def read(path):return json.loads(Path(path).read_text("utf-8"))
def say(*values):print(*values,flush=True)

def mask_for_boxes(boxes,w=1280,h=720):
    result=np.zeros((h,w),bool)
    for x,y,bw,bh in boxes:result[y:y+bh,x:x+bw]=True
    return result

def frame_boxes(plans,local):
    return [p["box"] for p in plans if p["start"]<=local<p["start"]+p["frames"]]

def mean_area(plans):return sum(p["box"][2]*p["box"][3]*p["frames"] for p in plans)/(50*1280*720)

def probability_stats(prob,mask):
    v=prob[mask]
    return {"pixels":int(mask.sum()),"min":float(v.min()),"max":float(v.max()),"mean":float(v.mean()),"p10":float(np.quantile(v,.1)),"p50":float(np.quantile(v,.5)),"p90":float(np.quantile(v,.9)),"p95":float(np.quantile(v,.95)),"p99":float(np.quantile(v,.99)),"pixels_at_threshold":int((v>=.9).sum())}

def image_counts(mask,gt):
    return {"tp":int((mask&gt).sum()),"fp":int((mask&~gt).sum()),"fn":int((~mask&gt).sum())}

def component_records(raw,prob,gt):
    labels,k=ndimage.label(raw,structure=np.ones((3,3),np.uint8))
    sizes=np.bincount(labels.ravel());distance=ndimage.distance_transform_edt(~gt)
    rows=[]
    for ident,slices in enumerate(ndimage.find_objects(labels),1):
        if slices is None:continue
        yy,xx=slices;component=labels[slices]==ident
        box=[xx.start,yy.start,xx.stop-xx.start,yy.stop-yy.start]
        padded=roi_box([{"regions":[{"box":box}]}],1280,720,16)
        pmask=mask_for_boxes([padded])
        rows.append({"id":ident,"pixels":int(sizes[ident]),"box":box,"confidence":float(prob[slices][component].mean()),"kept":bool(sizes[ident]>=64),"gt_intersection":int((gt[slices]&component).sum()),"min_gt_pixel_distance":float(distance[slices][component].min()),"padded_box_16":padded,"gt_pixels_inside_padded_box":int((pmask&gt).sum())})
    return rows

def main():
    OUT.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    model_config=read(PROJECT/"runs/phase_c_20261013/fixed_teacher.json")
    assert sha256(model_config["checkpoint"])==model_config["checkpoint_sha256"]
    with zipfile.ZipFile(OLD/"receiver/message.scv") as z:semantics=json.loads(gzip.decompress(z.read("semantics.json.gz")))
    with zipfile.ZipFile(read(STAGE_E/"selection.json")["file"]) as z:manifest=json.loads(z.read("manifest.json"))
    plans=plan_halves(semantics["frames"],1280,720,32,16)
    assert [(p["start"],p["frames"],p["box"]) for p in plans]==[(p["start"],p["frames"],p["box"]) for p in manifest["patches"]]
    references={x["source_frame"]:x for r in read(WORKSPACE/"outputs/roi_control_20261015/实际码流与GT任务评价.json") if r["group"]=="original" for x in r["per_gt_frame"]}
    e_references={x["source_frame"]:x for x in read(WORKSPACE/"outputs/semantic_roi_phase_e/阶段E真实任务评价.json")["per_gt_frame"]}
    gt_records={r["frame"]:r for r in read(CONTROL/"protocol.json")["gt_frames"]}
    source=read(OLD/"sender/source_audit.json");e_decode=read(STAGE_E/"network/receiver/decode.json")
    jobs=[]
    for frame in FRAMES:
        assert sha256(gt_records[frame]["file"])==gt_records[frame]["sha256"]
        for name,path,expected in (("original",OLD/f"sender/private_frames/{frame-420:05d}.png",source["rgb_sha256"][frame-420]),("received_E",STAGE_E/f"network/receiver/rgb/{frame-420:05d}.png",e_decode["decoded_rgb_sha256"][frame-420])):
            pixels=np.asarray(Image.open(path).convert("RGB"));assert hashlib.sha256(pixels.tobytes()).hexdigest()==expected
            jobs.append({"frame":frame,"condition":name,"image":str(path),"file_sha256":sha256(path),"rgb_sha256":expected,"gt":gt_records[frame]})
    frozen_files=[Path(__file__),PROJECT/"semantic_mvp/teacher.py",PROJECT/"semantic_mvp/roi_layout.py",PROJECT/"semantic_mvp/system.py",PROJECT/"runs/phase_c_20261013/fixed_teacher.json"]
    protocol={"stage":"F","step":1,"scope":"three original and three received-E images; fixed model and thresholds; diagnosis only","frames":FRAMES,"jobs":jobs,"model":model_config,"code_sha256":{str(p):sha256(p) for p in frozen_files},"threshold":.9,"min_pixels":64,"padding":16,"window":32,"layout":"frozen stage E left/right halves","same_frame_small_component_check":"geometry only: add all rejected 0.9 components from this frame, leave every other frame unchanged; no new encoding","no_training":True,"no_parameter_search":True,"no_new_video_test":True,"human_boxes_not_modified":True}
    write_json(RUN/"protocol.json",protocol);write_json(OUT/"步骤1冻结协议.json",protocol)
    model=Teacher("fish");assert model.threshold==.9
    results=[]
    for job in jobs:
        frame=job["frame"];name=job["condition"];started=time.perf_counter()
        raw,prob=model.mask(job["image"])
        assert prob.shape==(720,1280) and np.isfinite(prob).all()
        kept,objects=regions(raw,prob,64)
        gt=np.asarray(Image.open(job["gt"]["file"]))>0
        counts=image_counts(raw,gt)
        expected=(references if name=="original" else e_references)[frame]["metrics"]
        assert counts=={k:expected[k] for k in ("tp","fp","fn")},"Frozen inference failed to reproduce previous result"
        raw_components=component_records(raw,prob,gt)
        if name=="original":
            known=semantics["frames"][frame-420]["regions"]
            assert [(o["box"],o["pixels"]) for o in objects]==[(o["box"],o["pixels"]) for o in known]
        all_objects=[{"box":c["box"],"pixels":c["pixels"],"confidence":c["confidence"]} for c in raw_components]
        before=plan_halves([{"regions":all_objects}],1280,720,32,16)
        after=plan_halves([{"regions":objects}],1280,720,32,16)
        before_mask=mask_for_boxes([p["box"] for p in before]);after_mask=mask_for_boxes([p["box"] for p in after])
        actual_mask=mask_for_boxes(frame_boxes(plans,frame-420))
        candidate_records=copy.deepcopy(semantics["frames"])
        candidate_records[frame-420]["regions"]=all_objects
        hypothetical=plan_halves(candidate_records,1280,720,32,16)
        hypothetical_mask=mask_for_boxes(frame_boxes(hypothetical,frame-420))
        gt_components,k=ndimage.label(gt,structure=np.ones((3,3),np.uint8));targets=[]
        for ident in range(1,k+1):
            target=gt_components==ident;yy,xx=np.where(target);pixels=int(target.sum())
            local=[]
            for c in raw_components:
                cmask=mask_for_boxes([c["padded_box_16"]])
                overlap=int((cmask&target).sum())
                if overlap:
                    local.append({"id":c["id"],"pixels":c["pixels"],"kept":c["kept"],"box":c["box"],"target_pixels_inside_padded_box":overlap})
            targets.append({"component":ident,"pixels":pixels,"box":[int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)],"probability":probability_stats(prob,target),"threshold_gt_recall":int((raw&target).sum())/pixels,"filtered_gt_recall":int((kept&target).sum())/pixels,"same_frame_all_09_component_roi_coverage":int((before_mask&target).sum())/pixels,"same_frame_kept_roi_coverage":int((after_mask&target).sum())/pixels,"actual_stage_e_window_roi_coverage":int((actual_mask&target).sum())/pixels,"window_if_this_frame_rejected_components_restored":int((hypothetical_mask&target).sum())/pixels if name=="original" else None,"components_whose_individual_16px_padding_touches_target":local})
        saved=RUN/f"probabilities/{name}_{frame:05d}.npz";saved.parent.mkdir(exist_ok=True)
        np.savez_compressed(saved,fish_probability=prob,threshold_mask=raw,filtered_mask=kept)
        result={"frame":frame,"condition":name,"image":job["image"],"gt_file":job["gt"]["file"],"whole_frame_counts":counts,"raw_mask_pixels":int(raw.sum()),"filtered_mask_pixels":int(kept.sum()),"raw_components":raw_components,"retained_regions":objects,"targets":targets,"same_frame_all_boxes":[p["box"] for p in before],"same_frame_kept_boxes":[p["box"] for p in after],"actual_stage_e_window_boxes":frame_boxes(plans,frame-420),"actual_average_roi_area":mean_area(plans),"hypothetical_average_roi_area_restoring_this_frame_only":mean_area(hypothetical) if name=="original" else None,"arrays":str(saved),"arrays_sha256":sha256(saved),"inference_and_statistics_seconds":time.perf_counter()-started,"reproduced_existing_threshold_metrics":True}
        results.append(result);write_json(OUT/"概率与逐步骤诊断.json",results)
        say("DIAGNOSED",frame,name,json.dumps({"counts":counts,"raw_pixels":int(raw.sum()),"kept_pixels":int(kept.sum()),"targets":targets},ensure_ascii=False))
    assert sha256(model_config["checkpoint"])==model_config["checkpoint_sha256"]
    assert all(sha256(p)==v for p,v in protocol["code_sha256"].items())
    write_json(OUT/"步骤1数据核验.json",{"all_six_inferences_reproduce_previous_tp_fp_fn":True,"all_three_original_kept_regions_reproduce_sender_records":True,"original_and_received_rgb_hashes_verified":True,"pixel_gt_hashes_verified":True,"model_and_production_code_unchanged":True,"no_training_or_parameter_search":True,"npz_probability_maps_saved":6,"stage_f":"in_progress","step1":"data_complete_report_pending"})
    say("F_STEP1_DATA_COMPLETE",str(OUT))

if __name__=="__main__":main()
