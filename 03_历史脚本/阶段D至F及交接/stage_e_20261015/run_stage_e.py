"""One bounded stage-E automatic ROI experiment. No training or GT-guided encoding."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import numpy as np
from PIL import Image
from scipy import ndimage

WORKSPACE=Path(r"C:\Users\lenovo\Documents\ChatGPT\语义通信")
PROJECT=Path(r"C:\Users\lenovo\Desktop\works\work\semantic-video-mlvc")
RUN=PROJECT/"runs/semantic_mvp_stage_e_20261015"
OUT=WORKSPACE/"outputs/semantic_roi_phase_e"
OLD=PROJECT/"runs/semantic_mvp_20261014_fish"
CONTROL=PROJECT/"runs/semantic_mvp_roi_control_20261015"
INTENT=WORKSPACE/"阶段E_执行约定_2026-10-15.md"
SYSTEM=PROJECT/"semantic_mvp/system.py"
sys.path.insert(0,str(PROJECT/"semantic_mvp"))
from runtime import sha256,write_json,encode_h265,FFPROBE,run
from system import deliver
from roi_layout import MULTI_PROFILE,plan_halves

def read(path):return json.loads(Path(path).read_text("utf-8"))
def say(*args):print(*args,flush=True)

def inputs():
    with zipfile.ZipFile(OLD/"receiver/message.scv") as z:
        original=json.loads(z.read("manifest.json"))
        semantics=z.read("semantics.json.gz")
    return original,json.loads(gzip.decompress(semantics)),semantics,read(OLD/"sender/source_audit.json")

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);RUN.mkdir(exist_ok=True)
    original,semantics,semantic_bytes,source=inputs()
    assert source["source_frame_indices"]==list(range(420,470))
    for i,expected in enumerate(source["rgb_sha256"]):
        assert hashlib.sha256(np.asarray(Image.open(OLD/f"sender/private_frames/{i:05d}.png").convert("RGB")).tobytes()).hexdigest()==expected
    assert sha256(PROJECT/"runs/phase_c_20261013/segformer_adaptation_preserve8/best.pt")==original["teacher"]["model_sha256"]
    plans=plan_halves(semantics["frames"],1280,720,32,16)
    protocol={"stage":"E","goal":"restore video; semantics protect targets","intent_document":str(INTENT),"intent_sha256":sha256(INTENT),"method":"automatic fixed left/right spatial partitions, 32-frame windows, at most 2 disjoint ROI layers","model_and_threshold":original["teacher"],"source":source,"semantic_gzip_sha256":hashlib.sha256(semantic_bytes).hexdigest(),"all_predictions_reused_unchanged":True,"encoding_reads_manual_roi":False,"encoding_reads_pixel_gt":False,"plans":plans,"target_bytes":100000,"tolerance":.02,"selection":"byte difference only, up to 12 CRFs, stop within 1% if possible","background":"cached same 640x360 CRF42 stream, charged in every packet","roi":"medium P-only GOP32, padding16, min-component64, BT709 limited YUV420","new_profile":MULTI_PROFILE,"baselines":{"A":read(CONTROL/"A/selection.json"),"H265":read(CONTROL/"H265/selection.json")},"gt_evaluation":"unchanged four publisher pixel masks, after encoding selection only","human_boxes":"offline coverage diagnostics only","cached_inference_excluded_from_candidate_timing":True,"code_sha256":{p.name:sha256(p) for p in [SYSTEM,PROJECT/"semantic_mvp/roi_layout.py",Path(__file__)]},"scope":"one known training clip, one budget, no new model or training; stop after report"}
    target=RUN/"protocol.json"
    if target.exists():
        assert read(target)==protocol,"Protocol/code changed after preparation; preserve old run"
    else:write_json(target,protocol)
    write_json(OUT/"阶段E冻结协议.json",protocol)
    say("PREPARED",len(plans),"ROI streams",protocol["intent_sha256"])

def candidate(crf):
    folder=RUN/"candidates"/f"crf_{crf:.3f}"
    if (folder/"candidate.json").exists():return read(folder/"candidate.json")
    folder.mkdir(parents=True,exist_ok=False);streams=folder/"streams";streams.mkdir()
    started=time.perf_counter()
    original,semantics,semantic_bytes,source=inputs()
    with zipfile.ZipFile(OLD/"receiver/message.scv") as z:(streams/"context.mp4").write_bytes(z.read("context.mp4"))
    assert sha256(streams/"context.mp4")==original["files"]["context.mp4"]["sha256"]
    (streams/"semantics.json.gz").write_bytes(semantic_bytes)
    plans=plan_halves(semantics["frames"],1280,720,32,16)
    def encode_one(job):
        i,patch=job;name=f"roi_{i:04d}.mp4"
        command=encode_h265(OLD/"sender/private_frames",streams/name,source["fps"],crf,bbox=patch["box"],start=patch["start"],count=patch["frames"])
        return {**patch,"file":name},command
    with ThreadPoolExecutor(max_workers=2) as pool:encoded=list(pool.map(encode_one,enumerate(plans)))
    files={p.name:{"bytes":p.stat().st_size,"sha256":sha256(p)} for p in sorted(streams.iterdir())}
    config={**original["configuration"],"roi_crf":crf,"segment_frames":32,"roi_layout":"halves"}
    manifest={"profile":MULTI_PROFILE,"mode":"video","roi_layout":"halves","width":1280,"height":720,"frames":50,"fps":source["fps"],"fps_basis":source["fps_basis"],"context":original["context"],"patches":[x[0] for x in encoded],"semantics_file":"semantics.json.gz","teacher":original["teacher"],"files":files,"configuration":config,"experiment":{"stage":"E","roi_source":"unchanged automatic predictions only","selection":"complete file bytes only"}}
    packet=folder/"message.scv"
    with zipfile.ZipFile(packet,"x",compression=zipfile.ZIP_STORED) as z:
        z.writestr("manifest.json",json.dumps(manifest,ensure_ascii=False,separators=(",",":")))
        for name in files:z.write(streams/name,name)
    write_json(folder/"manifest.json",manifest)
    parts={"context":files["context.mp4"]["bytes"],"roi":sum(files[p["file"]]["bytes"] for p in manifest["patches"]),"semantic":len(semantic_bytes),"container":packet.stat().st_size-sum(x["bytes"] for x in files.values())}
    result={"group":"E","crf":crf,"file":str(packet),"file_bytes":packet.stat().st_size,"sha256":sha256(packet),"parts":parts,"relative_budget_difference":(packet.stat().st_size-100000)/100000,"candidate_encode_seconds":time.perf_counter()-started,"timing_excludes_cached_prediction_and_background":True,"commands":[x[1] for x in encoded]}
    write_json(folder/"candidate.json",result);say("CANDIDATE",crf,result["file_bytes"])
    return result

def select():
    target=RUN/"selection.json"
    if target.exists():return read(target)
    tested=[candidate(32.)];low,high=(32.,51.) if tested[0]["file_bytes"]>100000 else (0.,32.)
    for _ in range(11):
        best=min(tested,key=lambda r:abs(r["file_bytes"]-100000))
        if abs(best["file_bytes"]-100000)<=1000:break
        crf=round((low+high)/2,3)
        if crf in {r["crf"] for r in tested}:break
        result=candidate(crf);tested.append(result)
        if result["file_bytes"]>100000:low=crf
        else:high=crf
        write_json(RUN/"search.json",tested)
    best=min(tested,key=lambda r:abs(r["file_bytes"]-100000))
    result={**best,"matched_within_2_percent":abs(best["file_bytes"]-100000)<=2000,"target_bytes":100000,"selection_metric":"complete file bytes only; no GT metrics","fixed_crf32_diagnostic":tested[0]}
    write_json(target,result);write_json(RUN/"search.json",tested);write_json(OUT/"阶段E码流选择.json",result)
    say("SELECTED",result["file_bytes"],result["crf"],result["matched_within_2_percent"])
    return result

def audit(selected):
    directory=Path(selected["file"]).parent
    m=read(directory/"manifest.json");rows=[]
    for name,entry in m["files"].items():
        if not name.endswith(".mp4"):continue
        value=json.loads(run([FFPROBE,"-v","error","-select_streams","v:0","-show_entries","frame=pict_type","-of","json",directory/"streams"/name]))
        from collections import Counter
        rows.append({"file":name,"bytes":entry["bytes"],"frame_types":dict(Counter(x["pict_type"] for x in value["frames"]))})
    write_json(RUN/"stream_audit.json",rows)

def cli_checks():
    # An old complete packet must reconstruct the exact same RGB as before.
    legacy=RUN/"legacy_decode"
    if not (legacy/"decode.json").exists():
        run([sys.executable,SYSTEM,"decode","--input",read(CONTROL/"A/selection.json")["file"],"--output",legacy])
    old=read(CONTROL/"A/network/receiver/decode.json");new=read(legacy/"decode.json")
    assert new["decoded_rgb_sha256"]==old["decoded_rgb_sha256"]
    # Run the public encode/decode interface with genuinely fresh predictions.
    fresh=RUN/"fresh_cli"
    if not (fresh/"sender/encode.json").exists():
        _,_,_,source=inputs()
        command=[sys.executable,SYSTEM,"encode","--input",source["source"],"--first","420","--frames","5","--fps","25","--mode","video","--target","fish","--roi-layout","halves","--output",fresh/"sender"]
        output=run(command);(RUN/"fresh_cli.log").write_text(output,"utf-8")
        write_json(RUN/"fresh_cli_command.json",[str(x) for x in command])
    if not (fresh/"receiver/decode.json").exists():run([sys.executable,SYSTEM,"decode","--input",fresh/"sender/message.scv","--output",fresh/"receiver"])
    assert read(fresh/"sender/encode.json")["profile"]==MULTI_PROFILE
    assert read(fresh/"receiver/decode.json")["frames"]==5
    write_json(OUT/"旧包与公开CLI验证.json",{"legacy_v1_50_rgb_frame_hashes_identical":True,"fresh_five_frame_inference_encode_decode":True,"fresh_profile":MULTI_PROFILE,"fresh_sender":read(fresh/"sender/encode.json"),"fresh_receiver":read(fresh/"receiver/decode.json"),"fresh_cli_not_a_budget_experiment":True})

def evaluate():
    from teacher import Teacher
    data=read(PROJECT/"data/phase_c_20261013/manifest.json")
    labels={r["frame_index"]-420:r for v in data["videos"] if v["video"]=="video131" for r in v["labels"] if 420<=r["frame_index"]<470}
    model=Teacher("fish");totals=np.zeros(3,dtype=np.int64);reference_totals=np.zeros(3,dtype=np.int64)
    per=[];sse=fish_sse=bg_sse=0.;samples=fish_samples=bg_samples=0
    def stats(values):
        tp,fp,fn=map(int,values)
        return {"tp":tp,"fp":fp,"fn":fn,"iou":tp/(tp+fp+fn) if tp+fp+fn else None,"recall":tp/(tp+fn) if tp+fn else None,"precision":tp/(tp+fp) if tp+fp else None}
    for idx in range(50):
        raw_path=OLD/f"sender/private_frames/{idx:05d}.png"
        rec_path=RUN/f"network/receiver/rgb/{idx:05d}.png"
        raw=np.asarray(Image.open(raw_path).convert("RGB"),dtype=np.float64)
        rec=np.asarray(Image.open(rec_path).convert("RGB"),dtype=np.float64)
        error=(raw-rec)**2;sse+=float(error.sum());samples+=error.size
        if idx not in labels:continue
        gt=np.asarray(Image.open(labels[idx]["fish_mask"]))>0
        pred,_=model.mask(rec_path);ref_pred,_=model.mask(raw_path)
        counts=[int((pred&gt).sum()),int((pred&~gt).sum()),int((~pred&gt).sum())]
        reference_totals+=np.array([int((ref_pred&gt).sum()),int((ref_pred&~gt).sum()),int((~ref_pred&gt).sum())])
        totals+=counts;fish_sse+=float(error[gt].sum());fish_samples+=int(gt.sum())*3
        bg_sse+=float(error[~gt].sum());bg_samples+=int((~gt).sum())*3
        cm,k=ndimage.label(gt,structure=np.ones((3,3),dtype=np.uint8));components=[]
        for c in range(1,k+1):
            mask=cm==c;area=int(mask.sum());components.append({"component":c,"pixels":area,"size_group":"le_256" if area<=256 else "257_1024","recall":int((pred&mask).sum())/area})
        per.append({"source_frame":idx+420,"gt_pixels":int(gt.sum()),"metrics":stats(counts),"components":components})
    assert list(reference_totals)==[471,373,954]
    positive=[r for r in per if r["gt_pixels"]]
    psnr=lambda s,n:float(10*np.log10(255**2*n/s))
    result={"group":"E","gt":stats(totals),"global_rgb_psnr_db":psnr(sse,samples),"gt_fish_rgb_psnr_db":psnr(fish_sse,fish_samples),"gt_background_rgb_psnr_db":psnr(bg_sse,bg_samples),"positive_frame_mean_iou":float(np.mean([r["metrics"]["iou"] for r in positive])),"positive_frame_mean_recall":float(np.mean([r["metrics"]["recall"] for r in positive])),"positive_frame_task_hit_fraction":sum(r["metrics"]["tp"]>0 for r in positive)/len(positive),"per_gt_frame":per,"teacher":model.profile,"reference_task_counts_reproduced":[471,373,954],"evaluation_only_four_sparse_gt":True}
    write_json(OUT/"阶段E真实任务评价.json",result);say("TASK",json.dumps(result["gt"]),result["global_rgb_psnr_db"])

def main():
    prepare();selected=select();audit(selected)
    if not (RUN/"network/end_to_end.json").exists():
        network=RUN/"network";network.mkdir()
        deliver(Path(selected["file"]),network,20)
    cli_checks();evaluate()
    write_json(OUT/"阶段E实验状态.json",{"controlled_experiment":"complete","training":False,"new_model":False,"report_pending":True,"scope":"one clip, one budget, one automatic organization change"})
    say("STAGE_E_EXPERIMENT_COMPLETE",str(OUT))

if __name__=="__main__":main()
