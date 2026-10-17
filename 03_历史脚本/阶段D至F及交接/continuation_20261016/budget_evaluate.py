"""Evaluate every preselected allocation from actual independently received RGB."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
from PIL import Image
from scipy import ndimage
from budget_experiment import (WORKSPACE,PROJECT,OLD_RUN,OLD_OUT,DATA,RUN,OUT,SOURCE,VARIANTS,
                               read,check,sha256,write_json,originals)
from probability_diagnosis import exact_pr, stats
from teacher import Teacher

PROB_RUN = PROJECT/"runs/semantic_mvp_probability_review_20261016"


def psnr(sse,samples):
    if not samples or not sse:
        return None
    return float(10*np.log10(255**2*samples/sse))


def task_metrics(counts):
    tp,fp,fn = [int(x) for x in counts]
    return {"tp":tp,"fp":fp,"fn":fn,"iou":tp/(tp+fp+fn) if tp+fp+fn else None,
            "recall":tp/(tp+fn) if tp+fn else None,"precision":tp/(tp+fp) if tp+fp else None}


def main():
    protocol=check()
    locked=read(RUN/"selections_locked.json")
    selected={"E42":read(OLD_RUN/"E/selection.json"),**locked["selections"],
              "H265":read(OLD_RUN/"H265/selection.json")}
    directories={"E42":OLD_RUN/"E/network/receiver", "H265":OLD_RUN/"H265/network/receiver",
                 **{v:RUN/v/"network/receiver" for v in VARIANTS}}
    data=read(DATA)
    labels={r["local_frame"]:r for r in data["sparse_gt"]}
    gt={i:np.asarray(Image.open(label["fish_mask"]))>0 for i,label in labels.items()}
    for i,label in labels.items():
        assert sha256(label["fish_mask"]) == label["fish_mask_sha256"]
        assert int(gt[i].sum()) == label["fish_pixels"]
    rings={i:(ndimage.maximum_filter(mask,size=65,mode="constant",cval=False)&~mask)
           for i,mask in gt.items()}
    _,base,semantic=originals()
    decoder_hashes={}
    for name, folder in directories.items():
        assert (folder.parent/"end_to_end.json").is_file()
        decoder=read(folder/"decode.json")
        assert decoder["frames"] == 50
        assert decoder["file_sha256"] == selected[name]["sha256"]
        packet=folder/("video.mp4" if name=="H265" else "message.scv")
        assert sha256(packet) == selected[name]["sha256"]
        assert all(row["allowed"] for row in read(folder/"receiver_read_audit.json")["python_reads"])
        decoder_hashes[name]=decoder["decoded_rgb_sha256"]
        if name != "H265":
            with zipfile.ZipFile(packet) as archive:
                manifest=json.loads(archive.read("manifest.json"))
                assert manifest["patches"] == base["patches"]
                assert archive.read("semantics.json.gz") == semantic
    original_probability_rows=read(WORKSPACE/"outputs/semantic_review_20261016/逐帧逐区域概率.json")
    old_probability_lookup={(r["source_frame"],r["condition"]):r for r in original_probability_rows}
    teacher=Teacher("fish")
    assert teacher.threshold==.9
    sums={c:{"global_sse":0,"global_samples":0,"outside_roi_sse":0,"outside_roi_samples":0,
              "inside_roi_sse":0,"inside_roi_samples":0,"fish_sse":0,"fish_samples":0,
              "background_sse":0,"background_samples":0,"ring_sse":0,"ring_samples":0}
          for c in selected}
    frame_rows=[]
    gt_rows=[]
    counts={c:np.zeros(3,dtype=np.int64) for c in selected}
    prob_values={c:[] for c in (*selected,"original")}
    truth_values=[]
    roi_area=0
    gt_covered=0
    for i in range(50):
        original=np.asarray(Image.open(SOURCE/f"{i:05d}.png").convert("RGB"))
        assert hashlib.sha256(original.tobytes()).hexdigest() == data["source_rgb_sha256"][i]
        roi=np.zeros((720,1280),dtype=bool)
        for patch in base["patches"]:
            if patch["start"]<=i<patch["start"]+patch["frames"]:
                x,y,w,h=patch["box"]
                assert not roi[y:y+h,x:x+w].any()
                roi[y:y+h,x:x+w]=True
        roi_area+=int(roi.sum())
        if i in labels:
            gt_covered+=int((gt[i]&roi).sum())
            real=labels[i]["source_frame"]
            old=old_probability_lookup[(real,"original")]
            assert sha256(old["array"]) == old["array_sha256"]
            with np.load(old["array"]) as cache:
                p=cache[cache.files[0]]
            prob_values["original"].append(p.reshape(-1))
            truth_values.append(gt[i].reshape(-1))
        for condition,folder in directories.items():
            path=folder/"rgb"/f"{i:05d}.png"
            image=np.asarray(Image.open(path).convert("RGB"))
            assert image.shape==original.shape==(720,1280,3)
            assert hashlib.sha256(image.tobytes()).hexdigest()==decoder_hashes[condition][i]
            diff=original.astype(np.int16)-image.astype(np.int16)
            error=np.square(diff,dtype=np.int32)
            acc=sums[condition]
            acc["global_sse"]+=int(error.sum(dtype=np.int64))
            acc["global_samples"]+=error.size
            for key, mask in (("outside_roi",~roi),("inside_roi",roi)):
                acc[f"{key}_sse"]+=int(error[mask].sum(dtype=np.int64))
                acc[f"{key}_samples"]+=int(mask.sum())*3
            frame_rows.append({"condition":condition,"local_frame":i,"source_frame":data["source_frame_indices"][i],
                               "global_psnr_db":psnr(int(error.sum(dtype=np.int64)),error.size),
                               "gt_available":i in labels,"roi_fraction":float(roi.mean())})
            if i not in labels:
                continue
            mask=gt[i]
            for key, selector in (("fish",mask),("background",~mask),("ring",rings[i])):
                acc[f"{key}_sse"]+=int(error[selector].sum(dtype=np.int64))
                acc[f"{key}_samples"]+=int(selector.sum())*3
            real=labels[i]["source_frame"]
            if condition in {"E42","H265"}:
                name="E" if condition=="E42" else "H265"
                old=old_probability_lookup[(real,name)]
                assert sha256(old["array"])==old["array_sha256"]
                with np.load(old["array"]) as archive:
                    probability=archive[archive.files[0]].copy()
                probability_path=Path(old["array"])
                inference_seconds=0
            else:
                cache_folder=RUN/condition/"evaluation_probabilities"
                cache_folder.mkdir(exist_ok=True)
                probability_path=cache_folder/f"{real:05d}.npz"
                meta_path=probability_path.with_suffix(".json")
                if meta_path.exists():
                    meta=read(meta_path)
                    assert meta["received_png_sha256"]==sha256(path)
                    assert meta["checkpoint_sha256"]==protocol["fixed_teacher"]["model_sha256"]
                    assert meta["npz_sha256"]==sha256(probability_path)
                    with np.load(probability_path) as archive:
                        probability=archive["probability"].copy()
                    inference_seconds=0
                else:
                    started=time.perf_counter()
                    _,probability=teacher.mask(path)
                    inference_seconds=time.perf_counter()-started
                    np.savez_compressed(probability_path,probability=probability.astype(np.float32))
                    write_json(meta_path,{"received_png_sha256":sha256(path),
                                         "checkpoint_sha256":protocol["fixed_teacher"]["model_sha256"],
                                         "npz_sha256":sha256(probability_path),"inference_seconds":inference_seconds})
            assert probability.shape==(720,1280) and np.isfinite(probability).all()
            predicted=probability>=.9
            totals=np.array([(predicted&mask).sum(),(predicted&~mask).sum(),(~predicted&mask).sum()],dtype=np.int64)
            counts[condition]+=totals
            if condition in {"E42","H265"}:
                assert all(int(totals[k])==old["fixed_09"][metric] for k,metric in enumerate(("tp","fp","fn")))
            components,n=ndimage.label(mask,structure=np.ones((3,3),dtype=bool))
            per_region=[]
            for component in range(1,n+1):
                area=components==component
                per_region.append({"component":component,"GT_pixels":int(area.sum()),
                                   "GT_roi_coverage":float(roi[area].mean()),
                                   "fish_psnr_db":psnr(int(error[area].sum(dtype=np.int64)),int(area.sum())*3),
                                   "GT_probability":stats(probability[area]),
                                   "model_GT_recall_09":float(predicted[area].mean())})
            gt_rows.append({"condition":condition,"source_frame":real,"local_frame":i,
                            "task_fixed_09":task_metrics(totals),"GT_probability":stats(probability[mask]),
                            "probability_array":str(probability_path),"probability_array_sha256":sha256(probability_path),
                            "GT_fish_psnr_db":psnr(int(error[mask].sum(dtype=np.int64)),int(mask.sum())*3),
                            "GT_background_ring32_psnr_db":psnr(int(error[rings[i]].sum(dtype=np.int64)),int(rings[i].sum())*3),
                            "inference_seconds":inference_seconds,"components":per_region})
            prob_values[condition].append(probability.reshape(-1))
            print("BUDGET_GT_EVAL",condition,real,totals.tolist(),flush=True)
    truth=np.concatenate(truth_values)
    assert int(truth.sum())==2010
    results=[]
    curves={}
    for condition,choice in selected.items():
        values=np.concatenate(prob_values[condition])
        curve=exact_pr(values,truth)
        curves[condition]=curve
        acc=sums[condition]
        metrics={key+"_psnr_db":psnr(acc[key+"_sse"],acc[key+"_samples"])
                 for key in ("global","outside_roi","inside_roi","fish","background","ring")}
        network=read(directories[condition].parent/"end_to_end.json")
        row={"condition":condition,"complete_file_bytes":choice["file_bytes"],
             "relative_to_fixed_target":choice["file_bytes"]/protocol["target_complete_file_bytes"]-1,
             "parts":choice.get("parts",{}),"background_crf":42. if condition=="E42" else choice.get("background_crf"),
             "roi_crf":31. if condition=="E42" else choice.get("roi_crf"),
             "H265_offline_crf":choice.get("crf") if condition=="H265" else None,
             **metrics,"task_fixed_09_three_labels":task_metrics(counts[condition]),
             "GT_probability":stats(values[truth]),"pixel_AP_three_labels":curve["AP"],
             "transfer_seconds":network["transfer_seconds"],"application_transmit_bytes":network["application_transmit_bytes"],
             "network_is_archived_previous_run":condition in {"E42","H265"},"receiver_rgb":str(directories[condition]/"rgb"),
             "preview_video":str(directories[condition]/"reconstructed.mp4")}
        results.append(row)
    original_values=np.concatenate(prob_values["original"])
    curves["original"]=exact_pr(original_values,truth)
    reference=read(OLD_OUT/"新视频实测评价.json")
    for condition,old_name in (("E42","E"),("H265","H265")):
        actual=next(r for r in results if r["condition"]==condition)
        previous=next(r for r in reference["rows"] if r["condition"]==old_name)
        assert abs(actual["global_psnr_db"]-previous["global_rgb_psnr_all50_db"])<1e-8
        assert abs(actual["fish_psnr_db"]-previous["gt_fish_rgb_psnr_db"])<1e-8
    assert gt_covered==1940
    write_json(OUT/"预算对照实测结果.json",{
        "scope":"video6 source50..99; three existing GT frames60/75/90, all variants retained",
        "target_complete_file_bytes":protocol["target_complete_file_bytes"],"rows":results,
        "fixed_roi_mean_frame_fraction":roi_area/(50*720*1280),"fixed_GT_coverage":gt_covered/2010,
        "all_new_selection_completed_before_evaluation":True,
        "human_ratings_new_variants":None,"global_is_all50_GT_metrics_only3":True,
        "unlabeled_frames":47,"no_negative_GT_frame_in_this_clip":True,
        "original_GT_probability":stats(original_values[truth]),
        "original_task_fixed_09_three_labels":curves["original"]["fixed_09"],
        "source_is_development_after_diagnostics":True})
    write_json(OUT/"预算对照逐标注帧与连通域.json",gt_rows)
    write_json(OUT/"预算对照逐帧画质.json",frame_rows)
    write_json(OUT/"三标注帧PR诊断.json",curves)
    check()
    write_json(OUT/"预算对照核验.json",{
        "all4_conditions_received_RGB_hashes_verified":200,"new_TCP_receivers":2,
        "new_receiver_frames":100,"baseline_pixel_quality_exactly_reproduced":True,
        "baseline_fixed09_counts_exactly_reproduced":True,"same_patches_and_semantic_bytes":True,
        "GT_coverage_unchanged":1940,"GT_pixels":2010,"production_code_model_protocol_unchanged":True,
        "all_new_byte_selections_locked_before_scores":True,"human_scores_created_for_new_variants":0})
    print("BUDGET_EVALUATION_COMPLETE",json.dumps([{k:v for k,v in row.items() if k in {
        "condition","complete_file_bytes","global_psnr_db","fish_psnr_db","ring_psnr_db",
        "task_fixed_09_three_labels"}} for row in results],ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
