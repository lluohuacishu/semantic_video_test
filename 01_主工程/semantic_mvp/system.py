"""Minimal semantic video communication: sender -> TCP -> independent receiver.

Semantic boxes and high-resolution ROI streams are transmitted with a low-res
context stream. The receiver uses only the received packet and standard FFmpeg.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import zipfile
from fractions import Fraction
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from runtime import *
from transport import send_file, receive_connection
from roi_layout import MULTI_PROFILE,plan_halves,spatial_overlap

SCRIPT=Path(__file__).resolve()
MAX_BYTES=256*1024*1024

def say(text):
    print(text,flush=True)

def prepare_source(args,output):
    frames=output/"private_frames"
    frames.mkdir()
    source=Path(args.input).resolve()
    if source.is_dir():
        files=sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"})
        selected=files[args.first:args.first+args.frames]
        if not selected or len(selected)!=args.frames:
            raise ValueError("Insufficient source frames")
        indices=[]
        for idx,path in enumerate(selected):
            if not path.stem.isdigit():
                raise ValueError("Image-sequence names must contain their real frame indices")
            indices.append(int(path.stem))
            Image.open(path).convert("RGB").save(frames/f"{idx:05d}.png",compress_level=1)
        if any(b!=a+1 for a,b in zip(indices,indices[1:])):
            raise ValueError("Nonconsecutive frames cannot be treated as a video")
        if args.fps is None:
            raise ValueError("Image sequences require an explicit nominal --fps")
        fps=fps_string(args.fps)
        provenance={"type":"consecutive_image_sequence","source":str(source),"source_frame_indices":indices,"fps_basis":"user-declared nominal rate; capture timestamps not available"}
    elif source.is_file():
        if args.first:
            raise ValueError("Video input starts from its first frame; --first is for image sequences")
        data=probe(source)
        fps=fps_string(args.fps or data["avg_frame_rate"])
        command=[FFMPEG,"-v","error","-i",source,"-map","0:v:0","-an","-fps_mode","passthrough","-frames:v",str(args.frames),"-start_number","0","-compression_level","1",frames/"%05d.png"]
        run(command)
        provenance={"type":"video","source":str(source),"source_sha256":sha256(source),"fps_basis":"declared override" if args.fps else "video avg_frame_rate; v1 profile assumes CFR"}
    else:
        raise FileNotFoundError(source)
    files=sorted(frames.glob("*.png"))
    if not files:
        raise ValueError("No RGB frames extracted")
    width,height=Image.open(files[0]).size
    if min(width,height)<32 or max(width,height)>4096 or width%2 or height%2:
        raise ValueError("Use even image dimensions from 32 through 4096")
    if any(Image.open(p).size!=(width,height) for p in files):
        raise ValueError("Sequence frame dimensions differ")
    provenance.update({"frames":len(files),"width":width,"height":height,"fps":fps,"rgb_sha256":[hashlib.sha256(np.asarray(Image.open(p).convert("RGB")).tobytes()).hexdigest() for p in files]})
    write_json(output/"source_audit.json",provenance)
    return frames,provenance

def roi_box(records,width,height,padding):
    boxes=[obj["box"] for record in records for obj in record["regions"]]
    if not boxes:
        return None
    x=max(0,min(b[0] for b in boxes)-padding)//2*2
    y=max(0,min(b[1] for b in boxes)-padding)//2*2
    right=min(width,math.ceil((max(b[0]+b[2] for b in boxes)+padding)/2)*2)
    bottom=min(height,math.ceil((max(b[1]+b[3] for b in boxes)+padding)/2)*2)
    right=min(width,max(right,x+16));bottom=min(height,max(bottom,y+16))
    x=min(x,right-16);y=min(y,bottom-16)
    return [x,y,right-x,bottom-y]

def encode(args):
    require_runtime()
    output=fresh(args.output)
    started=time.perf_counter()
    frames,source=prepare_source(args,output)
    width,height,n,fps=source["width"],source["height"],source["frames"],source["fps"]
    from teacher import Teacher,regions
    teacher=Teacher(args.target,args.device)
    masks=output/"private_target_masks";masks.mkdir()
    records=[]
    say(f"识别 {n} 帧中的 {teacher.profile['label']}…")
    for idx,path in enumerate(sorted(frames.glob("*.png"))):
        mask,confidence=teacher.mask(path)
        mask,components=regions(mask,confidence,args.min_pixels)
        Image.fromarray(mask.astype(np.uint8)*255).save(masks/f"{idx:05d}.png")
        records.append({"frame":idx,"pixels":int(mask.sum()),"regions":components})
        if (idx+1)%10==0 or idx==n-1:
            say(f"语义提取 {idx+1}/{n}")
    streams=output/"sender_streams";streams.mkdir()
    base_width=max(16,round(width*args.background_scale/2)*2)
    base_height=max(16,round(height*args.background_scale/2)*2)
    commands=[]
    if args.mode=="video":
        commands.append(encode_h265(frames,streams/"context.mp4",fps,args.background_crf,scale=(base_width,base_height),count=n))
    layout=getattr(args,"roi_layout","single")
    profile=MULTI_PROFILE if layout=="halves" and args.mode=="video" else PROFILE
    planned=plan_halves(records,width,height,args.segment_frames,args.padding) if layout=="halves" else [{"start":start,"frames":min(args.segment_frames,n-start),"box":box} for start in range(0,n,args.segment_frames) if (box:=roi_box(records[start:start+args.segment_frames],width,height,args.padding))]
    patches=[]
    for patch in planned if args.mode=="video" else []:
        start,count,box=patch["start"],patch["frames"],patch["box"]
        name=f"roi_{len(patches):04d}.mp4"
        commands.append(encode_h265(frames,streams/name,fps,args.roi_crf,bbox=box,start=start,count=count))
        patches.append({"file":name,**patch})
    semantics={"target":args.target,"class_id":teacher.profile["class_id"],"label":teacher.profile["label"],"frames":records,"meaning":"sender model predictions; regions are connected components, not verified physical object counts"}
    semantic_bytes=gzip.compress(json.dumps(semantics,separators=(",",":"),ensure_ascii=False).encode("utf-8"),mtime=0)
    (streams/"semantics.json.gz").write_bytes(semantic_bytes)
    file_info={p.name:{"bytes":p.stat().st_size,"sha256":sha256(p)} for p in sorted(streams.iterdir())}
    manifest={"profile":PROFILE,"mode":args.mode,"width":width,"height":height,"frames":n,"fps":fps,"context":{"file":"context.mp4","width":base_width,"height":base_height} if args.mode=="video" else None,"patches":patches,"semantics_file":"semantics.json.gz","teacher":teacher.profile,"files":file_info,"configuration":{"background_crf":args.background_crf,"roi_crf":args.roi_crf,"background_scale":args.background_scale,"segment_frames":args.segment_frames,"padding":args.padding,"min_pixels":args.min_pixels,"h265":"medium, P-only, keyint32, BT709 limited YUV420"},"fps_basis":source["fps_basis"]}
    manifest["profile"]=profile
    if profile==MULTI_PROFILE:manifest["roi_layout"]="halves"
    manifest["configuration"]["roi_layout"]=layout
    packet=output/"message.scv"
    with zipfile.ZipFile(packet,"x",compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json",json.dumps(manifest,ensure_ascii=False,separators=(",",":")).encode("utf-8"))
        for name in file_info:
            archive.write(streams/name,name)
    if packet.stat().st_size>MAX_BYTES:
        raise ValueError("Packet exceeds the 256MiB v1 transfer limit")
    payload=sum(v["bytes"] for v in file_info.values())
    summary={"success":True,"profile":PROFILE,"mode":args.mode,"message":str(packet),"file_bytes":packet.stat().st_size,"file_sha256":sha256(packet),"context_bytes":file_info.get("context.mp4",{}).get("bytes",0),"roi_stream_bytes":sum(file_info[p["file"]]["bytes"] for p in patches),"semantic_bytes":len(semantic_bytes),"manifest_and_zip_bytes":packet.stat().st_size-payload,"frames":n,"duration_seconds":n/float(Fraction(fps)),"target":args.target,"target_present_frames":sum(bool(x["regions"]) for x in records),"roi_segments":len(patches),"average_roi_rectangle_fraction":sum(p["box"][2]*p["box"][3]*p["frames"] for p in patches)/(width*height*n),"encode_and_inference_seconds":time.perf_counter()-started,"source_audit":"source_audit.json","commands":commands}
    summary["profile"]=profile;summary["roi_layout"]=layout
    write_json(output/"manifest.json",manifest);write_json(output/"encode.json",summary)
    say(f"发送包完成：{summary['file_bytes']} B，目标出现 {summary['target_present_frames']}/{n} 帧，高清区域 {len(patches)} 段")
    return summary

def install_read_guard(output):
    allowed=Path(output).resolve()
    accessed=[]
    media={".png",".jpg",".jpeg",".mp4",".avi",".mkv",".npz",".npy",".pt",".ckpt",".scv",".gz"}
    def hook(event,args):
        if event!="open" or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        path=Path(os.fsdecode(args[0])).resolve()
        mode=args[1]
        flags=args[2]
        reading=(isinstance(mode,str) and ("r" in mode or "+" in mode)) or (mode is None and isinstance(flags,int) and flags & os.O_ACCMODE != os.O_WRONLY)
        if reading and path.suffix.lower() in media:
            permitted=path.is_relative_to(allowed)
            accessed.append({"path":str(path),"allowed":permitted})
            if not permitted:
                raise PermissionError(f"Independent receiver cannot read external media/models: {path}")
    sys.addaudithook(hook)
    return accessed

def unpack_packet(packet,output):
    from glc_layered import PROFILE as GLC_PROFILE, packet_profile, unpack_packet as unpack_glc
    if packet_profile(packet)==GLC_PROFILE:
        return unpack_glc(packet,output)
    directory=output/"received_streams";directory.mkdir()
    with zipfile.ZipFile(packet,"r") as archive:
        entries=archive.infolist()
        if len(entries)>1000 or len({x.filename for x in entries})!=len(entries):
            raise ValueError("Invalid/duplicate packet entries")
        if any(not re.fullmatch(r"manifest\.json|context\.mp4|semantics\.json\.gz|roi_[0-9]{4}\.mp4",x.filename) or x.file_size>MAX_BYTES for x in entries):
            raise ValueError("Unsupported packet member")
        if sum(x.file_size for x in entries)>MAX_BYTES or archive.getinfo("manifest.json").file_size>2*1024*1024:
            raise ValueError("Packet content exceeds limits")
        manifest=json.loads(archive.read("manifest.json"))
        if manifest.get("profile") not in {PROFILE,MULTI_PROFILE}:
            raise ValueError("Unsupported communication profile")
        expected={"manifest.json",*manifest["files"]}
        if expected!={x.filename for x in entries}:
            raise ValueError("Manifest/member mismatch")
        for name,record in manifest["files"].items():
            payload=archive.read(name)
            if len(payload)!=record["bytes"] or hashlib.sha256(payload).hexdigest()!=record["sha256"]:
                raise ValueError(f"Packet hash failed: {name}")
            (directory/name).write_bytes(payload)
    for key in ("width","height","frames"):
        if type(manifest[key]) is not int:
            raise ValueError("Noninteger geometry/count")
    w,h,n=manifest["width"],manifest["height"],manifest["frames"]
    if not 32<=min(w,h) or max(w,h)>4096 or not 1<=n<=10000 or w%2 or h%2:
        raise ValueError("Invalid frame geometry/count")
    manifest["fps"]=fps_string(manifest["fps"])
    mode=manifest.get("mode","video")
    if mode not in {"video","information"} or manifest["semantics_file"]!="semantics.json.gz":
        raise ValueError("Invalid named streams")
    if mode=="video" and (not manifest["context"] or manifest["context"]["file"]!="context.mp4"):
        raise ValueError("Video mode requires a context stream")
    if mode=="information" and (manifest["context"] is not None or manifest["patches"] or set(manifest["files"])!={"semantics.json.gz"}):
        raise ValueError("Information mode cannot include video streams")
    multi=manifest["profile"]==MULTI_PROFILE
    if multi and (mode!="video" or manifest.get("roi_layout")!="halves"):
        raise ValueError("Unsupported v2 ROI layout or mode")
    covered=set();per_frame_boxes={}
    for patch in manifest["patches"]:
        x,y,pw,ph=patch["box"]
        start,count=patch["start"],patch["frames"]
        if any(type(k) is not int for k in [x,y,pw,ph,start,count]) or min(x,y,start)<0 or min(pw,ph,count)<1 or x+pw>w or y+ph>h or start+count>n:
            raise ValueError("Invalid ROI geometry/count")
        if not re.fullmatch(r"roi_[0-9]{4}\.mp4",patch["file"]) or patch["file"] not in manifest["files"]:
            raise ValueError("Invalid ROI stream")
        if multi:
            midpoint=(w//4)*2
            if not (x+pw<=midpoint or x>=midpoint):
                raise ValueError("ROI crosses its spatial partition")
            if pw%2 or ph%2 or x%2 or y%2 or min(pw,ph)<16:
                raise ValueError("Invalid v2 ROI alignment")
            for frame in range(start,start+count):
                prior=per_frame_boxes.setdefault(frame,[])
                if len(prior)>=2 or any(spatial_overlap(patch["box"],box) for box in prior):
                    raise ValueError("Overlapping ROI spatial regions or too many layers")
                prior.append(patch["box"])
        elif covered.intersection(range(start,start+count)):
            raise ValueError("Overlapping ROI temporal segments")
        covered.update(range(start,start+count))
    compressed=(directory/"semantics.json.gz").read_bytes()
    import io
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as reader:
        semantic_payload=reader.read(16*1024*1024+1)
    if len(semantic_payload)>16*1024*1024:
        raise ValueError("Semantic metadata exceeds limits")
    semantics=json.loads(semantic_payload)
    if len(semantics["frames"])!=n or semantics["target"]!=manifest["teacher"]["target"]:
        raise ValueError("Semantic frame/target mismatch")
    for idx,record in enumerate(semantics["frames"]):
        if record["frame"]!=idx:
            raise ValueError("Semantic frame order mismatch")
        for region in record["regions"]:
            x,y,bw,bh=region["box"]
            if any(type(v) is not int for v in [x,y,bw,bh]) or min(x,y)<0 or min(bw,bh)<1 or x+bw>w or y+bh>h:
                raise ValueError("Semantic box outside frame")
    return manifest,semantics,directory

def decode_packet(packet,output,*,glc_deployment=None,run_command=None,authorize=None):
    from glc_layered import PROFILE as GLC_PROFILE, packet_profile
    if packet_profile(packet)==GLC_PROFILE:
        if glc_deployment is None or run_command is None or authorize is None:
            raise ValueError("GLC profile requires declared shared deployment and audited receiver_b2.py entrypoint")
        from glc_layered_decoder import decode as decode_glc
        return decode_glc(packet,output,glc_deployment,run_command,authorize)
    require_runtime()
    started=time.perf_counter()
    manifest,semantics,streams=unpack_packet(packet,output)
    w,h,n,fps=manifest["width"],manifest["height"],manifest["frames"],manifest["fps"]
    mode=manifest.get("mode","video")
    commands=[]
    if mode=="video":
        base=streams/"context.mp4"
        bp=probe(base)
        if [bp["width"],bp["height"],int(bp["nb_read_frames"])]!=[manifest["context"]["width"],manifest["context"]["height"],n] or fps_string(bp["avg_frame_rate"])!=fps:
            raise ValueError("Context stream does not match protocol")
        commands.append(decode_png(base,output/"context_rgb",size=(w,h)))
    patch_files={}
    for idx,patch in enumerate(manifest["patches"]):
        stream=streams/patch["file"]
        pp=probe(stream)
        if [pp["width"],pp["height"],int(pp["nb_read_frames"])]!=[patch["box"][2],patch["box"][3],patch["frames"]] or fps_string(pp["avg_frame_rate"])!=fps:
            raise ValueError("ROI stream does not match protocol")
        directory=output/f"roi_rgb_{idx:04d}"
        commands.append(decode_png(stream,directory))
        for local in range(patch["frames"]):
            patch_files.setdefault(patch["start"]+local,[]).append((directory/f"{local:05d}.png",patch["box"]))
    frames=output/"rgb";frames.mkdir()
    overlays=output/"overlay_rgb";overlays.mkdir()
    frame_hashes=[]
    try: font=ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc",max(14,round(h/40)))
    except OSError: font=ImageFont.load_default()
    for idx in range(n):
        image=Image.open(output/"context_rgb"/f"{idx:05d}.png").convert("RGB") if mode=="video" else Image.new("RGB",(w,h),(18,35,49))
        for path,box in patch_files.get(idx,[]):
            image.paste(Image.open(path).convert("RGB"),tuple(box[:2]))
        frame_hashes.append(hashlib.sha256(np.asarray(image).tobytes()).hexdigest())
        image.save(frames/f"{idx:05d}.png",compress_level=1)
        drawing=image.copy();pen=ImageDraw.Draw(drawing)
        for region in semantics["frames"][idx]["regions"]:
            x,y,bw,bh=region["box"]
            pen.rectangle((x,y,x+bw-1,y+bh-1),outline=(48,235,158),width=3)
            pen.text((x,max(0,y-25)),f"{semantics['label']} {region['confidence']:.2f}",font=font,fill=(48,235,158),stroke_width=1,stroke_fill=(0,0,0))
        drawing.save(overlays/f"{idx:05d}.png",compress_level=1)
    preview(frames,output/"reconstructed.mp4",fps)
    preview(overlays,output/"semantic_overlay.mp4",fps)
    best=max(range(n),key=lambda i:semantics["frames"][i]["pixels"])
    Image.open(overlays/f"{best:05d}.png").save(output/"poster.jpg",quality=93)
    Image.open(frames/f"{best:05d}.png").save(output/"reconstructed_poster.jpg",quality=93)
    write_json(output/"received_manifest.json",manifest)
    write_json(output/"target_information.json",semantics)
    result={"success":True,"profile":PROFILE,"mode":mode,"independent_process_pid":os.getpid(),"inputs":"received packet only; standard FFmpeg; no source, GT, teacher or sender reference state","file_bytes":packet.stat().st_size,"file_sha256":sha256(packet),"frames":n,"width":w,"height":h,"fps":fps,"fps_basis":manifest["fps_basis"],"target":semantics["target"],"target_present_frames":sum(bool(x["regions"]) for x in semantics["frames"]),"roi_segments":len(manifest["patches"]),"decoded_rgb_sha256":frame_hashes,"decode_and_preview_seconds":time.perf_counter()-started,"commands":commands,"semantic_results_are":"sender predictions transmitted as data; overlay is not receiver-side detection accuracy","preview_files_are":"generated after reception; not transmitted"}
    result["profile"]=manifest["profile"];result["roi_layout"]=manifest.get("roi_layout","single")
    write_json(output/"decode.json",result)
    make_page(output,result,semantics)
    say(f"独立接收完成：恢复 {n} 帧，接收到 {result['target_present_frames']} 帧目标信息")
    return result

def make_page(output,result,semantics):
    label=html.escape(semantics["label"])
    information=result.get("mode","video")=="information"
    visual_label="目标分布（从收到的坐标绘制）" if information else "恢复视频与发送端目标信息"
    scene_label="目标分布画布（未传视频画面）" if information else "恢复视频"
    fps_note="名义播放帧率，拍摄时间戳尚未确认" if "nominal" in result["fps_basis"] else "采用视频声明的播放帧率"
    media=f'<div><h3>{visual_label}</h3><video controls preload="metadata" poster="poster.jpg" src="semantic_overlay.mp4"></video></div>'
    if not information:
        media=f'<div><h3>{scene_label}</h3><video controls preload="metadata" poster="reconstructed_poster.jpg" src="reconstructed.mp4"></video></div>'+media
    transfer=json.loads((output/"transfer.json").read_text("utf-8")) if (output/"transfer.json").exists() else None
    timing=f"{transfer['elapsed_seconds']:.2f} 秒 / {transfer['limit_kbps']:g} kbps" if transfer else "从本地通信包解码"
    rows="".join(f"<tr><td>{r['frame']}</td><td>{'有' if r['regions'] else '未检出'}</td><td>{len(r['regions'])}</td><td>{r['pixels']}</td></tr>" for r in semantics["frames"])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>语义通信原型 · 接收结果</title>
<style>body{{margin:0;background:#101a25;color:#eaf1f7;font:16px/1.65 "Microsoft YaHei",sans-serif}}main{{max-width:1120px;margin:40px auto;padding:0 24px}}h1{{font-size:28px;margin-bottom:8px}}p{{color:#b9cad9}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card{{padding:18px;background:#1c2b3b;border-radius:12px;flex:1;min-width:170px}}strong{{font-size:24px;color:#67e7bb}}.media{{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin:28px 0}}video{{width:100%;border-radius:10px;background:#000}}a{{color:#67e7bb}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #304456}}details{{margin-top:24px}}@media(max-width:750px){{.media{{grid-template-columns:1fr}}}}</style>
<main><p>发送端识别 → 目标区域编码 → TCP 限速传输 → 独立接收端恢复</p><h1>语义通信原型已完成接收</h1>
<p>目标：{label}。{'本次只传类别、位置和置信度，接收端按坐标绘制目标分布。' if information else '接收端直接从收到的通信包恢复视频和目标信息。'}</p><div class="cards">
<div class="card">完整通信包<br><strong>{result['file_bytes']/1024:.2f} KiB</strong></div>
<div class="card">传输耗时<br><strong>{html.escape(timing)}</strong></div>
<div class="card">{'目标记录帧' if information else '恢复帧数'}<br><strong>{result['frames']}</strong></div>
<div class="card">目标出现帧<br><strong>{result['target_present_frames']} / {result['frames']}</strong></div></div>
<div class="media">{media}</div>
<p><a href="message.scv" download>下载完整通信包</a> · <a href="target_information.json">逐帧目标信息</a> · <a href="decode.json">独立解码记录</a> · <a href="transfer.json">传输记录</a></p>
<p>这是语义通信工程原型。绿色框来自发送端模型预测；连接区域数量不代表真实物体数量。当前采用本机 TCP 限速链路，{fps_note}。</p>
<details><summary>查看逐帧目标信息</summary><table><tr><th>帧号（从0开始）</th><th>{label}</th><th>连接区域</th><th>目标像素</th></tr>{rows}</table></details></main></html>'''
    (output/"查看结果.html").write_text(page,encoding="utf-8")

def listen(args):
    require_runtime()
    output=fresh(args.output)
    audit=install_read_guard(output)
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as listener:
        listener.bind((args.host,args.port));listener.listen(1);listener.settimeout(120)
        host,port=listener.getsockname()
        if args.ready:
            write_json(args.ready,{"host":host,"port":port,"pid":os.getpid()})
        say(f"接收端等待通信包：{host}:{port}")
        connection,_=listener.accept()
        with connection:
            transfer=receive_connection(connection,output/"message.scv")
        write_json(output/"receive.json",transfer)
    decode_packet(output/"message.scv",output)
    write_json(output/"receiver_read_audit.json",{"guard_installed":True,"all_external_media_and_models_denied":True,"python_reads":audit,"ffmpeg_inputs":"only received_streams and decoded frames inside receiver directory; commands in decode.json","guard_scope":"Python file reads and explicit FFmpeg arguments; not an OS sandbox"})

def information_packet(args):
    """Reuse sender predictions without any RGB/model input, and send them alone."""
    output=fresh(args.output)
    manifest,semantics,streams=unpack_packet(Path(args.input),output)
    manifest["mode"]="information";manifest["context"]=None;manifest["patches"]=[]
    manifest["profile"]=PROFILE;manifest.pop("roi_layout",None)
    manifest["files"]={"semantics.json.gz":manifest["files"]["semantics.json.gz"]}
    packet=output/"message.scv"
    with zipfile.ZipFile(packet,"x",compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json",json.dumps(manifest,ensure_ascii=False,separators=(",",":")).encode("utf-8"))
        archive.write(streams/"semantics.json.gz","semantics.json.gz")
    # The unpacked RGB streams are not included in the outgoing information packet.
    result={"success":True,"mode":"information","message":str(packet),"file_bytes":packet.stat().st_size,"file_sha256":sha256(packet),"semantic_bytes":manifest["files"]["semantics.json.gz"]["bytes"],"source_prediction_packet_sha256":sha256(args.input),"target":semantics["target"],"frames":manifest["frames"],"target_present_frames":sum(bool(x["regions"]) for x in semantics["frames"])}
    write_json(output/"encode.json",result);write_json(output/"manifest.json",manifest)
    say(json.dumps(result,ensure_ascii=False,indent=2))
    return result

def demo(args):
    output=fresh(args.output or ROOT/"runs"/f"semantic_mvp_{time.strftime('%Y%m%d_%H%M%S')}")
    if args.input:
        source=args.input;first=args.first;fps=args.fps
    elif args.target=="fish":
        source=ROOT/"data/uvmulti_downloads/video131/sequence";first=420;fps="25/1"
    else:
        source=Path(r"C:\Users\lenovo\Desktop\临时\语义\test5_5fps_reconstructed.avi");first=0;fps=None
    encode_args=argparse.Namespace(**vars(args));encode_args.input=str(source);encode_args.first=first;encode_args.fps=fps;encode_args.output=str(output/"sender")
    summary=encode(encode_args)
    return deliver(Path(summary["message"]),output,args.kbps,args.open)

def deliver(packet,output,kbps,open_page=False):
    """A real socket transfer to a separately spawned receive/decode process."""
    ready=output/"receiver_ready.json"
    log=output/"receiver_process.log"
    command=[sys.executable,str(SCRIPT),"listen","--host","127.0.0.1","--port","0","--output",str(output/"receiver"),"--ready",str(ready)]
    with log.open("w",encoding="utf-8") as handle:
        receiver=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,cwd=SCRIPT.parent)
        try:
            deadline=time.monotonic()+30
            while not ready.exists():
                if receiver.poll() is not None or time.monotonic()>deadline:
                    raise RuntimeError(f"Receiver startup failed: {log}")
                time.sleep(.1)
            network=json.loads(ready.read_text("utf-8"))
            say(f"独立接收进程 {network['pid']}，开始以 {kbps:g} kbps 发送…")
            transfer=send_file(packet,network["host"],network["port"],kbps,trace=output/"transfer.csv")
            write_json(output/"transfer.json",transfer)
            if receiver.wait(timeout=300):
                raise RuntimeError(f"Receiver decode failed: {log}")
        finally:
            if receiver.poll() is None:
                receiver.terminate();receiver.wait(timeout=10)
    received=output/"receiver"
    decode=json.loads((received/"decode.json").read_text("utf-8"))
    if sha256(received/"message.scv")!=sha256(packet):
        raise AssertionError("End-to-end packet/frame verification failed")
    write_json(received/"transfer.json",transfer)
    make_page(received,decode,json.loads((received/"target_information.json").read_text("utf-8")))
    result={"success":True,"mode":decode.get("mode","video"),"target":decode["target"],"output":str(output),"sender_pid":os.getpid(),"receiver_pid":decode["independent_process_pid"],"separate_sender_receiver_processes":os.getpid()!=decode["independent_process_pid"],"packet_sha256_equal":True,"file_bytes":packet.stat().st_size,"application_transmit_bytes":transfer["application_bytes"],"transfer_seconds":transfer["elapsed_seconds"],"frames":decode["frames"],"target_present_frames":decode["target_present_frames"],"result_page":str(received/"查看结果.html"),"link":"TCP loopback, rate limited; no physical acoustic channel","batch_operation":True,"source_is_copied_not_moved":True}
    write_json(output/"end_to_end.json",result)
    say(json.dumps(result,ensure_ascii=False,indent=2))
    if open_page:
        os.startfile(result["result_page"])
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ("encode","demo"):
        p=sub.add_parser(name)
        p.add_argument("--input",required=name=="encode")
        p.add_argument("--output",required=name=="encode")
        p.add_argument("--target",choices=["fish","diver"],default="fish")
        p.add_argument("--mode",choices=["video","information"],default="video")
        p.add_argument("--frames",type=int,default=50)
        p.add_argument("--first",type=int,default=0)
        p.add_argument("--fps")
        p.add_argument("--device",choices=["cuda","cpu"],default="cuda")
        p.add_argument("--background-scale",type=float,default=.5)
        p.add_argument("--background-crf",type=float,default=42)
        p.add_argument("--roi-crf",type=float,default=32)
        p.add_argument("--segment-frames",type=int,default=32)
        p.add_argument("--roi-layout",choices=["single","halves"],default="single")
        p.add_argument("--padding",type=int,default=16)
        p.add_argument("--min-pixels",type=int,default=64)
        if name=="demo":
            p.add_argument("--kbps",type=float,default=20)
            p.add_argument("--open",action="store_true")
    p=sub.add_parser("listen")
    p.add_argument("--host",default="127.0.0.1");p.add_argument("--port",type=int,default=8772)
    p.add_argument("--output",required=True);p.add_argument("--ready")
    p=sub.add_parser("decode")
    p.add_argument("--input",required=True);p.add_argument("--output",required=True)
    p=sub.add_parser("send")
    p.add_argument("--input",required=True);p.add_argument("--host",default="127.0.0.1")
    p.add_argument("--port",type=int,default=8772);p.add_argument("--kbps",type=float,default=20)
    p=sub.add_parser("information")
    p.add_argument("--input",required=True);p.add_argument("--output",required=True)
    p=sub.add_parser("transfer")
    p.add_argument("--input",required=True);p.add_argument("--output",required=True)
    p.add_argument("--kbps",type=float,default=20);p.add_argument("--open",action="store_true")
    args=parser.parse_args()
    if args.command in {"encode","demo"}:
        if not 1<=args.frames<=10000 or args.first<0 or not 0<args.background_scale<=1 or not 0<=args.background_crf<=51 or not 0<=args.roi_crf<=51 or not 1<=args.segment_frames<=256 or args.padding<0 or args.min_pixels<1:
            parser.error("Invalid source/quality/ROI configuration")
    if args.command=="encode":encode(args)
    elif args.command=="demo":demo(args)
    elif args.command=="listen":listen(args)
    elif args.command=="send":say(json.dumps(send_file(Path(args.input),args.host,args.port,args.kbps),ensure_ascii=False,indent=2))
    elif args.command=="information":information_packet(args)
    elif args.command=="transfer":deliver(Path(args.input).resolve(),fresh(args.output),args.kbps,args.open)
    else:
        output=fresh(args.output)
        destination=output/"message.scv"
        with Path(args.input).open("rb") as source,destination.open("xb") as target:
            import shutil
            shutil.copyfileobj(source,target)
        audit=install_read_guard(output)
        decode_packet(destination,output)
        write_json(output/"receiver_read_audit.json",{"guard_installed":True,"python_reads":audit,"initial_input":"explicit received package, copied into receiver directory before guard"})

if __name__=="__main__":main()
