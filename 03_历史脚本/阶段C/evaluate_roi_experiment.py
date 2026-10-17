"""Complete files, Q selected by bytes only, cold decoder, per-video GT evaluation."""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
from experiment_common import *
from common import FFMPEG, FFPROBE
import mlvc_file as native

CODECS=RUN/"codecs"
Q_ANCHORS=[8,24,40]
I_WEIGHT=PROJECT/"checkpoints/DMCI-6.0-mirror.pth.tar"


def eval_videos():return [v for v in manifest()["videos"] if v["experiment_role"]!="train"]


def sources():
    records=[]
    for v in eval_videos():
        folder=RUN/"evaluation_sources"/v["video"]
        refs=folder/"reference";refs.mkdir(parents=True,exist_ok=True)
        for idx in range(v["first_index"],v["last_index"]+1):
            target=refs/f"{idx-v['first_index']:05d}.png"
            if not target.exists():Image.open(Path(v["sequence_dir"])/f"{idx:05d}.jpg").convert("RGB").save(target)
        rate=Fraction(v["evaluation_fps"]).limit_denominator(10000)
        fps=f"{rate.numerator}/{rate.denominator}"
        source=folder/"lossless_reference.mkv"
        if not source.exists():subprocess.run([str(FFMPEG),"-v","error","-framerate",fps,"-i",str(refs/"%05d.png"),"-c:v","ffv1","-level","3","-pix_fmt","bgr0",str(source)],check=True)
        verify_file=folder/"lossless_roundtrip.json"
        if not verify_file.exists():
            command=[str(FFMPEG),"-v","error","-i",str(source),"-fps_mode","passthrough","-f","rawvideo","-pix_fmt","rgb24","pipe:1"]
            with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
                for idx,expected in enumerate(v["source_frame_rgb_hashes"]):
                    payload=proc.stdout.read(1280*720*3)
                    if len(payload)!=1280*720*3 or hashlib.sha256(payload).hexdigest()!=expected:raise ValueError(f"Lossless RGB roundtrip failed {v['video']}/{idx}")
                if proc.stdout.read(1):raise ValueError("Unexpected extra frames")
                if proc.wait()!=0:raise RuntimeError(proc.stderr.read().decode())
            save_json(verify_file,{"all_rgb_frames_exact":True,"frames":v["frames"],"source_sha256":sha256(source)})
        records.append({"video":v["video"],"reference_dir":str(refs),"source":str(source),"fps":fps,"frames":v["frames"],"official_split":v["official_split"],"experiment_role":v["experiment_role"],"fps_basis":v["fps_basis"]})
        print("SOURCE_OK",v["video"],v["frames"],flush=True)
    save_json(RUN/"evaluation_sources.json",records)


def branch_weights():
    return {"M_RD":RUN/"training/M_RD/checkpoint_512.ckpt","M_ROI4":RUN/"training/M_ROI4/checkpoint_512.ckpt","M_ROI16":RUN/"training/M_ROI16/checkpoint_512.ckpt","M_PRE":PROJECT/"checkpoints/mlvc-s-psnr-v1.ckpt"}


def encode_all():
    configure()
    inputs=json.loads((RUN/"evaluation_sources.json").read_text("utf-8"))
    selections=[];search_log=[]
    weights=branch_weights()
    native_loader=native.load_models
    for branch in ("M_RD","M_ROI4","M_ROI16","M_PRE"):
        if branch!="M_PRE":
            result=json.loads((RUN/"training"/branch/"result.json").read_text("utf-8"))
            if not result["success"] or result["steps"]!=512:raise ValueError("Unfinished codec training")
        weight=weights[branch]
        p,i,config=load_models(weight,I_WEIGHT,entropy=True)
        def cached_loader(requested_weight,requested_i_weight,device="cuda",entropy=True):
            if Path(requested_weight).resolve()!=weight.resolve() or Path(requested_i_weight).resolve()!=I_WEIGHT.resolve() or device!="cuda":raise ValueError("Encoder model cache key mismatch")
            return p,i,config
        native.load_models=cached_loader
        try:
            for source in inputs:
                video=source["video"]
                cache={}
                def encode(q):
                    if q in cache:return cache[q]
                    folder=CODECS/video/branch/f"q{q:02d}"
                    args=SimpleNamespace(source=source["source"],run=str(folder),weight=str(weight),i_weight=str(I_WEIGHT),q_index=q,intra_period=32,frames=None,width=None,height=None,reference_dir=source["reference_dir"],fps=source["fps"],device="cuda")
                    if not (folder/"encode.json").exists():native.encode(args)
                    result=json.loads((folder/"encode.json").read_text("utf-8"))
                    if sha256(folder/"video.mlvcv")!=result["file_sha256"]:raise ValueError("Encoded file hash mismatch")
                    result.update({"branch":branch,"video":video,"q_index":q,"run":str(folder),"weight":str(weight)})
                    cache[q]=result
                    return result
                for anchor in Q_ANCHORS:
                    if branch in ("M_RD","M_PRE"):
                        chosen=encode(anchor)
                        target=chosen["file_bytes"] if branch=="M_RD" else next(r["file_bytes"] for r in selections if r["branch"]=="M_RD" and r["video"]==video and r["anchor"]==anchor)
                        candidates=[chosen]
                    else:
                        target=next(r["file_bytes"] for r in selections if r["branch"]=="M_RD" and r["video"]==video and r["anchor"]==anchor)
                        candidates=[encode(anchor)]
                        low,high=0,63
                        if candidates[-1]["file_bytes"]<target:low=anchor+1
                        else:high=anchor-1
                        for iteration in range(8):
                            best=min(candidates,key=lambda r:abs(r["file_bytes"]-target))
                            if abs(best["file_bytes"]-target)/target<=.01 or low>high:break
                            if iteration==0:q=int(np.clip(anchor+(1 if best["file_bytes"]<target else -1),low,high))
                            else:q=(low+high)//2
                            result=encode(q);candidates.append(result)
                            if result["file_bytes"]<target:low=q+1
                            else:high=q-1
                        chosen=min(candidates,key=lambda r:abs(r["file_bytes"]-target))
                    selection={k:chosen[k] for k in ("branch","video","q_index","run","weight","file_bytes","payload_bytes","container_bytes","i_bytes","p_bytes","file_sha256")}
                    selection.update({"anchor":anchor,"target_M_RD_bytes":target,"relative_file_difference":(chosen["file_bytes"]-target)/target,"matched_within_2_percent":abs(chosen["file_bytes"]-target)/target<=.02,"selection_metric":"absolute complete file byte difference only; no GT/PSNR/IoU used"})
                    selections.append(selection)
                    search_log.append({"branch":branch,"video":video,"anchor":anchor,"target_bytes":target,"candidates":[{"q":c["q_index"],"bytes":c["file_bytes"]} for c in candidates],"selected_q":chosen["q_index"]})
                    save_json(RUN/"file_selections.json",selections);save_json(RUN/"file_size_search.json",search_log)
                    print("FILE_SELECTED",selection,flush=True)
        finally:native.load_models=native_loader
        del p,i;torch.cuda.empty_cache()
    print("REAL_ENCODING_OK",len(selections),flush=True)


def invoke(command,log):
    with Path(log).open("w",encoding="utf-8") as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f"Subprocess failed: {log}")


def decode_all():
    selections=json.loads((RUN/"file_selections.json").read_text("utf-8"))
    checked=set()
    python=str(PROJECT/".venv/Scripts/python.exe")
    script=str(PROJECT/"adapters/mlvc_file.py")
    for row in selections:
        folder=Path(row["run"])
        if str(folder) in checked:continue
        receiver=folder/"receiver"
        if not (receiver/"decode.json").exists():
            invoke([python,script,"decode","--file",str(folder/"video.mlvcv"),"--output",str(receiver),"--weight",row["weight"],"--i-weight",str(I_WEIGHT)],folder/"cold_decode.log")
        if not (folder/"verification.json").exists():invoke([python,script,"verify","--run",str(folder),"--decoded",str(receiver)],folder/"verification.log")
        verified=json.loads((folder/"verification.json").read_text("utf-8"))
        if not verified["independent_decode_exact_rgb_and_state"]:raise AssertionError("Cold decoder verification missing")
        checked.add(str(folder));print("COLD_DECODE_OK",row["video"],row["branch"],row["q_index"],verified["verified_frames"],flush=True)
    save_json(RUN/"cold_decoder_summary.json",{"all_selected_files_verified":True,"unique_files":len(checked),"rows":len(selections),"decoded_without_original_RGB_or_GT":True})


def psnr(error,samples):return float(10*np.log10(255**2*samples/error)) if error else None


def task_evaluation():
    configure("cuda")
    # Explicitly configured and hashed teacher is required before reading reconstruction task scores.
    fixed=json.loads((RUN/"fixed_teacher.json").read_text("utf-8"))
    from probe_teacher_scale import full_resolution_probability
    from adapt_segformer import binary_model,predict_fish
    from semantic_teacher import load_teacher
    model=binary_model(checkpoint=fixed["checkpoint"]) if fixed["architecture"]=="binary" else load_teacher()[0]
    if fixed["architecture"]=="preserved_8class":
        state=torch.load(fixed["checkpoint"],map_location="cpu",weights_only=True);model.load_state_dict(state["model"],strict=True)
    if sha256(fixed["checkpoint"])!=fixed["checkpoint_sha256"]:raise ValueError("Evaluation teacher changed")
    model.eval().requires_grad_(False)
    def prediction(path):
        if fixed["inference"]=="tiles512":
            probability,winner=full_resolution_probability(model,path,binary=fixed["architecture"]=="binary")
            return winner==(1 if fixed["architecture"]=="binary" else 6) if fixed["threshold"]=="argmax" else probability>=fixed["threshold"]
        return predict_fish(model,path,size=512)
    original=[];results=[];per_frame=[]
    videos={v["video"]:v for v in eval_videos()}
    for video in videos.values():
        tp=fp=fn=0
        for label in video["labels"]:
            pred=prediction(label["rgb"]);gt=np.asarray(Image.open(label["fish_mask"]))>0
            tp+=int((pred&gt).sum());fp+=int((pred&~gt).sum());fn+=int((~pred&gt).sum())
        original.append({"video":video["video"],"official_split":video["official_split"],"labelled_frames":len(video["labels"]),**binary_metrics(tp,fp,fn)})
    save_json(RUN/"original_task_metrics.json",{"teacher":fixed,"rows":original})
    for row in json.loads((RUN/"file_selections.json").read_text("utf-8")):
        folder=Path(row["run"]);v=videos[row["video"]]
        if not (folder/"verification.json").exists():
            print("WAIT_FOR_VERIFIED_COLD_FILE",row["video"],row["branch"],row["q_index"],flush=True)
            wait_start=time.monotonic()
            while not (folder/"verification.json").exists():
                if time.monotonic()-wait_start>3600:raise TimeoutError("Cold decoder file not verified within one hour")
                time.sleep(3)
        quality=json.loads((folder/"verification.json").read_text("utf-8"))
        tp=fp=fn=0;roi_sse=bg_sse=0.;roi_samples=bg_samples=0
        for label in v["labels"]:
            path=folder/"receiver/rgb"/f'{label["frame_index"]-v["first_index"]:05d}.png'
            gt=np.asarray(Image.open(label["fish_mask"]))>0;pred=prediction(path)
            counts=[int((pred&gt).sum()),int((pred&~gt).sum()),int((~pred&gt).sum())];tp+=counts[0];fp+=counts[1];fn+=counts[2]
            a=np.asarray(Image.open(label["rgb"]).convert("RGB"),dtype=np.float64);b=np.asarray(Image.open(path).convert("RGB"),dtype=np.float64)
            error=(a-b)**2;roi_sse+=float(error[gt].sum());bg_sse+=float(error[~gt].sum());roi_samples+=int(gt.sum())*3;bg_samples+=int((~gt).sum())*3
            per_frame.append({"video":row["video"],"branch":row["branch"],"anchor":row["anchor"],"q_index":row["q_index"],"frame":label["frame_index"],**binary_metrics(*counts)})
        result={**row,"official_split":v["official_split"],"experiment_role":v["experiment_role"],"frames":v["frames"],"gt_frames":len(v["labels"]),"gt_positive_frames":v["annotated_frames_with_fish"],
                "global_rgb_psnr_db":quality["rgb_psnr_global_db"],"gt_fish_rgb_psnr_db":psnr(roi_sse,roi_samples),"gt_background_rgb_psnr_db":psnr(bg_sse,bg_samples),
                "independent_decode_verified":True,"teacher_sha256":fixed["checkpoint_sha256"],**binary_metrics(tp,fp,fn)}
        results.append(result);save_json(RUN/"task_quality_results.json",{"teacher":fixed,"original":original,"rows":results});save_json(RUN/"per_frame_task_metrics.json",per_frame)
        print("TASK_METRICS",row["video"],row["branch"],row["anchor"],result["fish_iou"],result["global_rgb_psnr_db"],flush=True)
    print("EVALUATION_OK",len(results),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=["sources","encode","decode","semantics"]);args=parser.parse_args()
    {"sources":sources,"encode":encode_all,"decode":decode_all,"semantics":task_evaluation}[args.mode]()
