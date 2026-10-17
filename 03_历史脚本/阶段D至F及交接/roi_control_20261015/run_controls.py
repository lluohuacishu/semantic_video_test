"""A/B real-video controls after the direction MD was saved; C/D await real GT."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from control_common import *
from transport import send_file,receive_connection

TARGET=100000
TOLERANCE=.02
SCRIPT=Path(__file__).resolve()
GROUP_WINDOWS={"A":32,"B":4,"C":32,"D":4}

def roi_records(group,semantic_frames):
    if group not in {"C","D"}:return semantic_frames,None
    snapshot=RUN/"manual_roi_snapshot.json"
    manual_protocol=read(RUN/"manual_protocol.json")
    assert sha256(snapshot)==manual_protocol["annotation_snapshot_sha256"]
    annotation=read(snapshot)
    assert [row["source_frame"] for row in annotation["frames"]]==list(range(420,470))
    assert all(row["status"] in {"human_reviewed","publisher_verified"} for row in annotation["frames"])
    return [{"frame":row["local_frame"],"regions":[{"box":box} for box in row["boxes"]]} for row in annotation["frames"]],manual_protocol["annotation_snapshot_sha256"]

def say(*values):print(*values,flush=True)

def context_stream():
    folder=RUN/"shared";folder.mkdir(exist_ok=True)
    file=folder/"context.mp4"
    if not file.exists():
        with zipfile.ZipFile(OLD/"receiver/message.scv") as archive:file.write_bytes(archive.read("context.mp4"))
    m,*_=frozen()
    assert sha256(file)==m["files"]["context.mp4"]["sha256"]
    return file

def candidate(group,crf):
    folder=RUN/group/"candidates"/f"crf_{crf:.3f}"
    if (folder/"candidate.json").exists():return read(folder/"candidate.json")
    folder.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    original,semantics,semantic_bytes,source,_=frozen()
    refs=OLD/"sender/private_frames"
    commands=[]
    if group=="H265":
        file=folder/"video.mp4"
        commands.append(encode_h265(refs,file,source["fps"],crf,count=50,offline=True))
        parts={"full_frame_mp4":file.stat().st_size}
        profile="native full-frame H265 MP4; medium B4; no semantic metadata"
    else:
        window=GROUP_WINDOWS[group]
        streams=folder/"streams";streams.mkdir()
        shutil.copy2(context_stream(),streams/"context.mp4")
        (streams/"semantics.json.gz").write_bytes(semantic_bytes)
        records,annotation_sha=roi_records(group,semantics["frames"])
        patches=plan(records,window)
        def one(job):
            index,patch=job
            name=f"roi_{index:04d}.mp4";path=streams/name
            if group=="A" and crf==32:
                with zipfile.ZipFile(OLD/"receiver/message.scv") as archive:path.write_bytes(archive.read(name))
                command=["reuse archived fixed-CRF32 ROI stream with unchanged source/profile",name]
            else:
                command=encode_h265(refs,path,source["fps"],crf,bbox=patch["box"],start=patch["start"],count=patch["frames"])
            return {**patch,"file":name},command
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=list(pool.map(one,enumerate(patches)))
        patches=[x[0] for x in jobs];commands.extend(x[1] for x in jobs)
        files={p.name:{"bytes":p.stat().st_size,"sha256":sha256(p)} for p in sorted(streams.iterdir())}
        config=copy.deepcopy(original["configuration"])
        config.update({"segment_frames":window,"roi_crf":crf})
        manifest={"profile":original["profile"],"mode":"video","width":1280,"height":720,"frames":50,"fps":source["fps"],"context":original["context"],"patches":patches,"semantics_file":"semantics.json.gz","teacher":original["teacher"],"files":files,"configuration":config,"fps_basis":source["fps_basis"],"experiment":{"group":group,"predictions":"same frozen cached sender predictions","budget_matching":"complete bytes only"}}
        if annotation_sha:
            manifest["experiment"].update({"roi_source":"human-reviewed continuous boxes; diagnostic only","annotation_snapshot_sha256":annotation_sha,"semantic_messages":"unchanged frozen model predictions; not human labels"})
        file=folder/"message.scv"
        with zipfile.ZipFile(file,"x",compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("manifest.json",json.dumps(manifest,ensure_ascii=False,separators=(",",":")))
            for name in files:archive.write(streams/name,name)
        payload=sum(x["bytes"] for x in files.values())
        parts={"context":files["context.mp4"]["bytes"],"roi":sum(files[p["file"]]["bytes"] for p in patches),"semantic":len(semantic_bytes),"container":file.stat().st_size-payload}
        write_json(folder/"manifest.json",manifest)
        source_name="human-reviewed diagnostic ROI" if annotation_sha else "predicted ROI"
        profile=f"{source_name}, window{window}; fixed background CRF42, ROI CRF{crf}; medium P-only"
    record={"group":group,"crf":crf,"file":str(file),"file_bytes":file.stat().st_size,"sha256":sha256(file),"relative_budget_difference":(file.stat().st_size-TARGET)/TARGET,"parts":parts,"profile":profile,"candidate_encode_seconds":time.perf_counter()-started,"shared_context_and_predictions_cached":True,"commands":commands}
    write_json(folder/"candidate.json",record)
    say("CANDIDATE",group,f"crf={crf:.3f}",f"bytes={record['file_bytes']}",f"difference={record['relative_budget_difference']:.3%}")
    return record

def select(group):
    folder=RUN/group;folder.mkdir(exist_ok=True)
    if (folder/"selection.json").exists():return read(folder/"selection.json")
    tested=[candidate(group,32.)]
    low,high=0.,51.
    if tested[0]["file_bytes"]>TARGET:low=32.
    else:high=32.
    for _ in range(11):
        best=min(tested,key=lambda x:abs(x["file_bytes"]-TARGET))
        if abs(best["file_bytes"]-TARGET)/TARGET<=.01:break
        q=round((low+high)/2,3)
        if q in {x["crf"] for x in tested}:break
        result=candidate(group,q);tested.append(result)
        if result["file_bytes"]>TARGET:low=q
        else:high=q
        write_json(folder/"search.json",tested)
    best=min(tested,key=lambda x:abs(x["file_bytes"]-TARGET))
    selected={**best,"target_bytes":TARGET,"matched_within_2_percent":abs(best["file_bytes"]-TARGET)/TARGET<=TOLERANCE,"selection_metric":"complete file byte difference only; no GT/quality metrics","tested_crfs":[x["crf"] for x in tested],"fixed_crf32_diagnostic":tested[0]}
    write_json(folder/"search.json",tested);write_json(folder/"selection.json",selected)
    say("SELECTED",group,best["crf"],best["file_bytes"],selected["matched_within_2_percent"])
    return selected

def frame_types(file):
    data=json.loads(run([FFPROBE,"-v","error","-select_streams","v:0","-show_entries","frame=pict_type","-of","json",file]))
    return dict(Counter(x["pict_type"] for x in data["frames"]))

def stream_audit(selected):
    if selected["group"]=="H265":
        return {"streams":[{"file":"native full-frame MP4","bytes":selected["file_bytes"],"frames_by_type":frame_types(selected["file"])}]}
    folder=Path(selected["file"]).parent
    manifest=read(folder/"manifest.json")
    rows=[]
    for name,entry in manifest["files"].items():
        if name.endswith(".mp4"):
            rows.append({"file":name,"bytes":entry["bytes"],"frames_by_type":frame_types(folder/"streams"/name)})
    total=Counter()
    for row in rows:total.update(row["frames_by_type"])
    return {"streams":rows,"total_stream_frames_by_type":dict(total),"unique_output_frames":50,"note":"ROI and background stream frame counts include repeated spatial layers, not additional source frames"}

def receive_plain(args):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    audit=install_read_guard(output)
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1",0));listener.listen(1);listener.settimeout(120)
        write_json(args.ready,{"host":"127.0.0.1","port":listener.getsockname()[1],"pid":os.getpid()})
        connection,_=listener.accept()
        with connection:receipt=receive_connection(connection,output/"received.mp4")
    started=time.perf_counter()
    info=probe(output/"received.mp4")
    assert (info["width"],info["height"],int(info["nb_read_frames"]))==(1280,720,50)
    command=decode_png(output/"received.mp4",output/"rgb")
    preview(output/"rgb",output/"reconstructed.mp4",info["avg_frame_rate"])
    result={"success":True,"pid":os.getpid(),"frames":50,"file_sha256":sha256(output/"received.mp4"),"file_bytes":(output/"received.mp4").stat().st_size,"inputs":"received MP4 only; no source, GT, semantic model or sender reference","decode_and_preview_seconds":time.perf_counter()-started,"command":command,"receipt":receipt}
    write_json(output/"decode.json",result)
    write_json(output/"receiver_read_audit.json",{"guard_installed":True,"python_reads":audit,"guard_scope":"Python media/model reads and explicit FFmpeg inputs; not an OS sandbox"})

def plain_transfer(selected,network):
    network.mkdir()
    ready=network/"ready.json";log=network/"receiver_process.log"
    command=[sys.executable,str(SCRIPT),"receive_plain","--output",str(network/"receiver"),"--ready",str(ready)]
    with log.open("w",encoding="utf-8") as handle:
        child=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+30
            while not ready.exists():
                if child.poll() is not None or time.monotonic()>deadline:raise RuntimeError(f"Baseline receiver startup failed: {log}")
                time.sleep(.1)
            addr=read(ready)
            say("TCP_TRANSFER","H265",selected["file_bytes"],"20kbps")
            transfer=send_file(Path(selected["file"]),addr["host"],addr["port"],20,trace=network/"transfer.csv")
            if child.wait(timeout=120):raise RuntimeError(f"Baseline receiver failed: {log}")
        finally:
            if child.poll() is None:child.terminate();child.wait(timeout=10)
    decoder=read(network/"receiver/decode.json")
    assert decoder["file_sha256"]==selected["sha256"]
    result={"success":True,"target":"fish","frames":50,"file_bytes":selected["file_bytes"],"application_transmit_bytes":transfer["application_bytes"],"transfer_seconds":transfer["elapsed_seconds"],"separate_sender_receiver_processes":decoder["pid"]!=os.getpid(),"packet_sha256_equal":True,"receiver_pid":decoder["pid"],"link":"real TCP loopback, 20 decimal kbps; no physical channel test"}
    write_json(network/"transfer.json",transfer);write_json(network/"end_to_end.json",result)
    return result

def transmit(selected):
    network=RUN/selected["group"]/"network"
    if (network/"end_to_end.json").exists():return read(network/"end_to_end.json")
    if selected["group"]=="H265":return plain_transfer(selected,network)
    network.mkdir()
    return deliver(Path(selected["file"]),network,20)

def metric_counts(pred,gt):
    tp=int((pred&gt).sum());fp=int((pred&~gt).sum());fn=int((~pred&gt).sum())
    return [tp,fp,fn]

def task_summary(counts):
    tp,fp,fn=map(int,counts)
    return {"tp":tp,"fp":fp,"fn":fn,"iou":tp/(tp+fp+fn) if tp+fp+fn else None,"recall":tp/(tp+fn) if tp+fn else None,"precision":tp/(tp+fp) if tp+fp else None}

def psnr(sse,samples):return float(10*np.log10(255**2*samples/sse)) if sse and samples else None

def evaluate(selections,destination=None):
    from teacher import Teacher
    model=Teacher("fish")
    _,_,_,source,labels=frozen()
    lookup={x["frame_index"]-420:x for x in labels}
    refs=OLD/"sender/private_frames"
    streams=[("original",refs,None),*[(g,RUN/g/"network/receiver/rgb",selections[g]) for g in selections]]
    results=[]
    for group,directory,selected in streams:
        total=np.zeros(3,dtype=np.int64);per=[];sse=roi_sse=bg_sse=0.;samples=roi_samples=bg_samples=0
        paths=sorted(directory.glob("*.png"));assert len(paths)==50
        for idx,path in enumerate(paths):
            raw=np.asarray(Image.open(refs/f"{idx:05d}.png").convert("RGB"),dtype=np.float64)
            rec=np.asarray(Image.open(path).convert("RGB"),dtype=np.float64)
            assert raw.shape==rec.shape==(720,1280,3)
            error=(raw-rec)**2;sse+=float(error.sum());samples+=error.size
            if idx in lookup:
                gt=np.asarray(Image.open(lookup[idx]["fish_mask"]))>0
                pred,_=model.mask(path)
                c=metric_counts(pred,gt);total+=c
                roi_sse+=float(error[gt].sum());roi_samples+=int(gt.sum())*3
                bg_sse+=float(error[~gt].sum());bg_samples+=int((~gt).sum())*3
                component_rows=[];cm,k=ndimage.label(gt,structure=np.ones((3,3),dtype=np.uint8))
                for j in range(1,k+1):
                    component=cm==j;area=int(component.sum());hits=int((pred&component).sum())
                    component_rows.append({"component":j,"area":area,"size_group":"le_256" if area<=256 else "257_1024","task_recall":hits/area})
                per.append({"source_frame":idx+420,"gt_pixels":int(gt.sum()),"metrics":task_summary(c),"components":component_rows})
        positive=[x for x in per if x["gt_pixels"]]
        result={"group":group,"file_bytes":selected["file_bytes"] if selected else None,"matched_within_2_percent":selected["matched_within_2_percent"] if selected else None,"gt":task_summary(total),"gt_frames":len(per),"positive_gt_frames":len(positive),"positive_frame_task_hit_fraction":sum(x["metrics"]["tp"]>0 for x in positive)/len(positive),"positive_frame_mean_iou":float(np.mean([x["metrics"]["iou"] for x in positive])),"positive_frame_mean_recall":float(np.mean([x["metrics"]["recall"] for x in positive])),"global_rgb_psnr_db":psnr(sse,samples),"gt_fish_rgb_psnr_db":psnr(roi_sse,roi_samples),"gt_background_rgb_psnr_db":psnr(bg_sse,bg_samples),"per_gt_frame":per,"teacher":model.profile,"task_predictions_independently_computed_from_received_video":True}
        if group=="original":assert result["gt"]["tp"]==471 and result["gt"]["fp"]==373 and result["gt"]["fn"]==954
        results.append(result);write_json(destination or OUT/"实际码流与GT任务评价.json",results)
        say("TASK",group,json.dumps({k:result[k] for k in ["file_bytes","gt","positive_frame_task_hit_fraction","positive_frame_mean_iou","global_rgb_psnr_db"]},ensure_ascii=False))
    return results

def main_run():
    assert (RUN/"protocol.json").exists(),"Run preparation only after direction MD is saved"
    assert read(RUN/"protocol.json")["direction_sha256"]==sha256(DIRECTION)
    selections={}
    for group in ["A","B","H265"]:
        selected=select(group);selections[group]=selected
        write_json(RUN/group/"stream_audit.json",stream_audit(selected))
        transmit(selected)
    write_json(OUT/"完整文件预算选择.json",selections)
    results=evaluate(selections)
    write_json(OUT/"实验状态.json",{"A":"completed","B":"completed","H265":"completed","C":"waiting for continuous verified human/publisher ROI","D":"waiting for continuous verified human/publisher ROI","new_training":False,"positive_or_negative_results_saved":True,"direction_sha256":sha256(DIRECTION),"scope":"one known training clip, one budget, four sparse GT; no generalization claim"})
    say("CONTROL_RUN_COMPLETE",str(OUT))

def main_manual():
    supplement=OUT/"人工ROI补充";supplement.mkdir(exist_ok=True)
    assert sha256(DIRECTION)==read(RUN/"protocol.json")["direction_sha256"]
    selections={}
    for group in ["C","D"]:
        selected=select(group);selections[group]=selected
        write_json(RUN/group/"stream_audit.json",stream_audit(selected))
        transmit(selected)
    write_json(supplement/"人工ROI完整文件预算选择.json",selections)
    evaluate(selections,supplement/"原图与人工ROI任务评价.json")
    write_json(supplement/"人工ROI实验状态.json",{"C":"completed","D":"completed","new_training":False,"annotation_snapshot_sha256":sha256(RUN/"manual_roi_snapshot.json"),"evaluation_gt":"unchanged four publisher pixel masks; human boxes control ROI only","scope":"one known training clip and one budget; diagnostic human ROI, not deployed performance"})
    say("MANUAL_CONTROL_RUN_COMPLETE",str(supplement))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=["run","run_manual","receive_plain"]);parser.add_argument("--output");parser.add_argument("--ready");args=parser.parse_args()
    if args.mode=="receive_plain":receive_plain(args)
    elif args.mode=="run_manual":main_manual()
    else:main_run()
