"""Fixed-model follow-up: original/received probability response, no new codec or threshold."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
from scipy import ndimage

WORKSPACE = Path(r"C:\Users\lenovo\Documents\ChatGPT\语义通信")
PROJECT = Path(r"C:\Users\lenovo\Desktop\works\work\semantic-video-mlvc")
F_RUN = PROJECT/"runs/semantic_mvp_stage_f_20261015/step3"
F_OUT = WORKSPACE/"outputs/semantic_roi_phase_f/step3_frozen_external"
F_DATA = PROJECT/"data/phase_f_external_20261015"
RUN = PROJECT/"runs/semantic_mvp_probability_review_20261016"
OUT = WORKSPACE/"outputs/semantic_review_20261016"
sys.path.insert(0, str(PROJECT/"semantic_mvp"))
from runtime import sha256, write_json
from teacher import Teacher

BINS = [0., .1, .25, .5, .75, .85, .9, .95, 1.]


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    quantiles = np.quantile(values,[.05,.1,.25,.5,.75,.9,.95,.99])
    histogram, _ = np.histogram(values, bins=BINS)
    return {
        "pixels": int(values.size), "min": float(values.min()), "max": float(values.max()),
        "mean": float(values.mean()),
        **{k:float(v) for k,v in zip(("p05","p10","p25","p50","p75","p90","p95","p99"),quantiles)},
        "ge_09_pixels": int((values >= .9).sum()),
        "ge_09_fraction": float((values >= .9).mean()),
        "near_085_to_09_pixels": int(((values >= .85) & (values < .9)).sum()),
        "near_085_to_09_fraction": float(((values >= .85) & (values < .9)).mean()),
        "lt_075_fraction": float((values < .75).mean()),
        "histogram_bins": BINS, "histogram": histogram.tolist(),
    }


def metrics(tp,fp,fn):
    return {"tp":int(tp),"fp":int(fp),"fn":int(fn),
            "precision":float(tp/(tp+fp)) if tp+fp else None,
            "recall":float(tp/(tp+fn)) if tp+fn else None,
            "iou":float(tp/(tp+fp+fn)) if tp+fp+fn else None}


def exact_pr(scores, truth):
    """Exact step AP at score-tie group ends; retain only PR vertices affecting the curve."""
    scores = np.asarray(scores,dtype=np.float32).ravel()
    truth = np.asarray(truth,dtype=bool).ravel()
    assert scores.size == truth.size and np.isfinite(scores).all()
    positive = int(truth.sum()); negative = len(truth)-positive
    work = scores >= .9
    fixed = metrics(int((work&truth).sum()),int((work&~truth).sum()),int((~work&truth).sum()))
    if not positive:
        return {"positive_pixels":0,"negative_pixels":negative,"AP":None,
                "fixed_09":fixed,"curve":{"threshold":[],"precision":[],"recall":[]}}
    order = np.argsort(scores,kind="stable")[::-1]
    ranked = scores[order]
    tp_cumulative = np.cumsum(truth[order],dtype=np.int64)
    ends = np.flatnonzero(np.r_[ranked[:-1] != ranked[1:], True])
    tp = tp_cumulative[ends]
    delta = np.diff(np.r_[0,tp])
    precision = tp/(ends+1)
    event_indices = np.flatnonzero(delta)
    ap = float(np.sum((delta[event_indices]/positive)*precision[event_indices]))
    # Background-only groups have fixed recall and decreasing precision; their
    # endpoints immediately before each positive group preserve those segments.
    keep = np.unique(np.r_[0,len(ends)-1,event_indices,np.maximum(0,event_indices-1)])
    curve = {
        "threshold": [None]+ranked[ends[keep]].astype(float).tolist(),
        "precision": [1.]+precision[keep].astype(float).tolist(),
        "recall": [0.]+(tp[keep]/positive).astype(float).tolist(),
    }
    assert 0 <= ap <= 1 and int(tp[-1]) == positive
    return {"positive_pixels":positive,"negative_pixels":negative,
            "positive_prevalence":positive/len(truth),"AP":ap,
            "fixed_09":fixed,"score_tie_groups":len(ends),
            "curve_storage":"all recall-changing groups, preceding groups, endpoints; AP uses every group",
            "curve":curve}


def jobs():
    result = []
    for case,root,output,data_dir in (
        ("initial_0_49",F_RUN,F_OUT,F_DATA),
        ("positive_50_99",F_RUN/"positive_case",F_OUT/"positive_case",F_DATA/"positive_case"),
    ):
        data = read(data_dir/"manifest.json")
        historical = {(r["condition"],r["source_frame"]):r for r in read(output/"逐标注帧分割评价.json")}
        source = read(root/"initial_crf32/source_audit.json")
        decode = {name:read(root/name/"network/receiver/decode.json") for name in ("E","H265")}
        for label in data["sparse_gt"]:
            local = label["local_frame"]; real = label["source_frame"]
            for condition in ("original","E","H265"):
                path = (root/f"initial_crf32/private_frames/{local:05d}.png" if condition == "original"
                        else root/f"{condition}/network/receiver/rgb/{local:05d}.png")
                expected_rgb = source["rgb_sha256"][local] if condition == "original" else decode[condition]["decoded_rgb_sha256"][local]
                result.append({
                    "case":case,"source_frame":real,"local_frame":local,"condition":condition,
                    "image":str(path),"image_file_sha256":sha256(path),"rgb_sha256":expected_rgb,
                    "fish_mask":label["fish_mask"],"fish_mask_sha256":label["fish_mask_sha256"],
                    "expected_fixed_09":{key:historical[(condition,real)]["metrics_raw_09"][key]
                                         for key in ("tp","fp","fn")},
                })
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True); RUN.mkdir(parents=True,exist_ok=True)
    all_jobs = jobs()
    assert len(all_jobs) == 21
    fixed = read(PROJECT/"runs/phase_c_20261013/fixed_teacher.json")
    frozen_files = [
        Path(__file__),PROJECT/"semantic_mvp/teacher.py",PROJECT/"semantic_mvp/system.py",
        PROJECT/"semantic_mvp/roi_layout.py",PROJECT/"semantic_mvp/runtime.py",
        PROJECT/"runs/phase_c_20261013/fixed_teacher.json",Path(fixed["checkpoint"]),
        WORKSPACE/"阶段F_首轮总结与后续方向_2026-10-15.md",
    ]
    assert sha256(fixed["checkpoint"]) == fixed["checkpoint_sha256"]
    previous = read(F_RUN/"experiment_protocol.json")
    for p,v in previous["input_sha256"].items():
        if Path(p).name in {"teacher.py","system.py","roi_layout.py","runtime.py","best.pt","fixed_teacher.json"}:
            assert sha256(p) == v
    protocol = {
        "date":"2026-10-16","scope":"follow-up to completed F; probability and human-review preparation, no newly named stage",
        "video":"video6 previously reported; one video, seven sparse annotated frames, three conditions",
        "jobs":all_jobs,"model":fixed,"threshold_unchanged":.9,
        "diagnostic_near_threshold_interval":[.85,.9],
        "histogram_bins":BINS,
        "probability_statistics":"GT fish pixels pooled, per frame and per8-connected component; background separately",
        "PR":"all7 publisher-labeled frames, including4 empty fish masks, all fish/background pixels; no93 unlabeled frames",
        "AP":"exact step-weighted precision on every recall increment, grouping equal float32 scores; not trapezoidal AUC or object mAP",
        "diagnostic_score_sweep_not_threshold_selection":True,
        "fixed_09_historical_counts_must_reproduce":True,
        "zero_training_encodes_transfers_and_method_changes":True,
        "file_sha256":{str(p):sha256(p) for p in frozen_files},
    }
    target = RUN/"protocol.json"
    if target.exists(): assert read(target) == protocol
    else: write_json(target,protocol)
    write_json(OUT/"概率诊断冻结协议.json",protocol)
    # Small independently known AP cases check exact tie handling, not model behavior.
    assert abs(exact_pr([.9,.8,.8,.1],[1,0,1,0])["AP"]-5/6) < 1e-12
    assert exact_pr([.9,.1],[1,0])["AP"] == 1.
    assert exact_pr([.9,.1],[0,1])["AP"] == .5
    assert exact_pr([.9,.1],[0,0])["AP"] is None
    model = Teacher("fish")
    assert model.threshold == .9
    records = []
    for job in all_jobs:
        real,condition = job["source_frame"],job["condition"]
        path = RUN/f"probabilities/{condition}_{real:05d}.npz"
        assert sha256(job["image"]) == job["image_file_sha256"]
        rgb = np.asarray(Image.open(job["image"]).convert("RGB"))
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == job["rgb_sha256"]
        assert sha256(job["fish_mask"]) == job["fish_mask_sha256"]
        gt = np.asarray(Image.open(job["fish_mask"])) > 0
        start = time.perf_counter()
        if path.exists():
            with np.load(path) as archive:
                probability = archive["fish_probability"]
        else:
            mask,probability = model.mask(job["image"])
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,fish_probability=probability)
        assert probability.shape == gt.shape == (720,1280)
        assert probability.dtype == np.float32 and np.isfinite(probability).all()
        assert 0 <= probability.min() <= probability.max() <= 1
        predicted = probability >= .9
        fixed_counts = {"tp":int((predicted&gt).sum()),"fp":int((predicted&~gt).sum()),
                        "fn":int((~predicted&gt).sum())}
        assert fixed_counts == job["expected_fixed_09"],(real,condition,fixed_counts,job["expected_fixed_09"])
        labels,n = ndimage.label(gt,np.ones((3,3),dtype=np.uint8))
        components = []
        for ident in range(1,n+1):
            target = labels == ident
            yy,xx = np.where(target)
            components.append({
                "component":ident,"fish_pixels":int(target.sum()),
                "box":[int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)],
                "fish_probability":stats(probability[target]),
            })
        row = {
            **job,"array":str(path),"array_sha256":sha256(path),
            "GT_fish_probability":stats(probability[gt]),
            "background_probability":stats(probability[~gt]),
            "fixed_09":metrics(**fixed_counts),"components":components,
            "compute_or_cache_read_seconds":time.perf_counter()-start,
        }
        records.append(row); write_json(OUT/"逐帧逐区域概率.json",records)
        print("PROBABILITY",real,condition,json.dumps({
            "GT":row["GT_fish_probability"],"fixed_09":row["fixed_09"]},ensure_ascii=False),flush=True)
    curves = {}; summaries = {}
    for condition in ("original","E","H265"):
        selected = [r for r in records if r["condition"] == condition]
        probabilities = []
        masks = []
        for r in selected:
            with np.load(r["array"]) as archive: probabilities.append(archive["fish_probability"].ravel())
            masks.append((np.asarray(Image.open(r["fish_mask"]))>0).ravel())
        scores,truth = np.concatenate(probabilities),np.concatenate(masks)
        curve = exact_pr(scores,truth)
        pooled_fish = stats(scores[truth])
        summaries[condition] = {
            "GT_fish_probability":pooled_fish,
            "seven_GT_frames_fixed_09":curve["fixed_09"],
            "positive_pixel_prevalence":curve["positive_prevalence"],
            "pixel_AP_seven_frames":curve["AP"],
            "total_labeled_pixels":scores.size,"GT_fish_pixels":int(truth.sum()),
            "equal_component_mean_probability":float(np.mean([
                c["fish_probability"]["mean"] for r in selected for c in r["components"]])),
        }
        curves[condition] = curve
    write_json(OUT/"七标注帧PR曲线.json",curves)
    write_json(OUT/"概率诊断汇总.json",summaries)
    assert all(sha256(path) == value for path,value in protocol["file_sha256"].items())
    write_json(OUT/"概率诊断核验.json",{
        "date":"2026-10-16","maps":21,"historical_fixed_09_counts_reproduced":21,
        "source_and_received_RGB_hashes_verified":True,"all_GT_hashes_verified":True,
        "exact_AP_tie_and_empty_GT_checks_passed":True,"frozen_model_and_production_code_unchanged":True,
        "new_model_training":0,"new_video_encodes":0,"new_TCP_transfers":0,
        "human_reviews_completed":0,"status":"probability_data_complete_report_pending",
    })
    print("PROBABILITY_DIAGNOSIS_COMPLETE",json.dumps(summaries,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
