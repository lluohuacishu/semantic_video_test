"""GLC priority round 2: reuse real codec adapters, bounded searches, no training."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image

def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p, x): Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
def sha(p):
    with Path(p).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()
def rgb(p):
    with Image.open(p) as im: return np.asarray(im.convert("RGB"))
def run(cmd, log, cwd=None, timeout=1800):
    started=time.perf_counter(); last=""
    with log.open("w",encoding="utf-8") as f:
        p=subprocess.Popen([str(v) for v in cmd],stdout=f,stderr=subprocess.STDOUT,cwd=cwd)
        while True:
            try: code=p.wait(timeout=20); break
            except subprocess.TimeoutExpired:
                lines=log.read_text(encoding="utf-8",errors="replace").splitlines()
                progress=next((l for l in reversed(lines) if l.startswith(("encode ","decode "))),"")
                if progress and progress!=last: print(f"{log.parent.name}: {progress}",flush=True);last=progress
                if time.perf_counter()-started>timeout:p.kill();p.wait();raise TimeoutError(str(log))
    record={"command":[str(v) for v in cmd],"cwd":str(cwd) if cwd else None,"seconds":time.perf_counter()-started,"returncode":code}
    save(log.with_suffix(log.suffix+".json"),record)
    if code:raise RuntimeError(log.read_text(encoding="utf-8",errors="replace")[-3500:])
    return record
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

p=argparse.ArgumentParser()
p.add_argument("phase",choices=["prepare","glc","standards","evaluate"])
p.add_argument("--clip",choices=["short","long108","long91","long24"],required=True)
p.add_argument("--project",type=Path,required=True)
p.add_argument("--legacy",type=Path,required=True)
p.add_argument("--previous",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--ffmpeg",type=Path,required=True)
a=p.parse_args();sys.stdout.reconfigure(encoding="utf-8")
root=a.output.resolve(); base=root/a.clip;base.mkdir(parents=True,exist_ok=True)
specs={"short":(6,50,50,25),"long108":(108,0,250,25),"long91":(91,0,250,15),"long24":(24,0,250,25)}
video,start,count,fps=specs[a.clip]
raw=(a.project/"data/phase_f_external_20261015/raw" if video==6 else a.project/"data/phase_c_20261013/raw")/f"video{video}"/"sequence"
masks=(a.project/"data/phase_f_external_20261015/positive_case/fish_masks" if video==6 else a.project/"data/phase_c_20261013/fish_masks"/f"video{video}")
ff=[a.ffmpeg,"-hide_banner","-loglevel","error","-nostdin"]
profiles=[(720,64),(540,64),(360,64)] if a.clip=="short" else ([(540,64),(540,128)] if a.clip=="long108" else [(540,64)])
source=base/"source"

if a.phase=="prepare":
    if (base/"inputs.json").exists() and read(base/"inputs.json")["mask_dir"]==str(masks):print("Input already prepared");sys.exit(0)
    source.mkdir(exist_ok=True); records=[]
    for i in range(count):
        n=start+i; src=raw/f"{n:05d}.jpg"; image=rgb(src)
        if image.shape!=(720,1280,3):raise ValueError(f"Unexpected shape {src}: {image.shape}")
        dst=source/f"im{i+1:05d}.png"
        if not dst.exists():Image.fromarray(image).save(dst)
        if not np.array_equal(image,rgb(dst)):raise ValueError("Input cache mismatch")
        mp=masks/f"{n:05d}.png"
        rec={"source_frame":n,"source_sha256":sha(src),"rgb_sha256":hashlib.sha256(image.tobytes()).hexdigest(),"png_sha256":sha(dst),"gt_available":mp.exists()}
        if mp.exists():rec.update({"mask_sha256":sha(mp),"fish_pixels":int((np.asarray(Image.open(mp))>0).sum())})
        records.append(rec)
    for h in sorted({h for h,g in profiles}|{720},reverse=True):
        work=base/f"input_{h}";work.mkdir(exist_ok=True);w=h*16//9
        for i in range(count):
            src=source/f"im{i+1:05d}.png";dst=work/src.name
            if not dst.exists():
                if h==720:os.link(src,dst)
                else:Image.fromarray(cv2.resize(rgb(src),(w,h),interpolation=cv2.INTER_AREA)).save(dst)
        movie=base/f"input_{h}.mkv"
        if not movie.exists():run(ff+["-framerate",fps,"-start_number",1,"-i",work/"im%05d.png","-frames:v",count,"-an","-c:v","ffv1","-level",3,"-pix_fmt","bgr0",movie],base/f"prepare_{h}.log")
    save(base/"inputs.json",{"video":video,"source_start":start,"frames":count,"fps":fps,"fps_basis":"source timing metadata" if video==91 else "publisher nominal fps; not independently verified","source_role":"previously used development material","source_dir":str(raw),"mask_dir":str(masks),"width":1280,"height":720,"records":records,"script_sha256":sha(__file__)})
    print(f"Prepared {a.clip}: {count} frames, {fps} fps",flush=True)

elif a.phase=="glc":
    info=read(base/"inputs.json")
    for h,gop in profiles:
        dest=base/f"glc_{h}_g{gop}";dest.mkdir(exist_ok=True)
        if (dest/"verified.json").exists():continue
        common=["--repo",a.legacy/"tools/GLC","--weights",a.legacy/"checkpoints/glc","--ffmpeg",a.ffmpeg]
        exe=a.legacy/".venv-glc/Scripts/python.exe";script=a.legacy/"scripts/glc_video_codec.py"
        if not (dest/"sender/encode.json").exists():
            run([exe,script,"encode","--input",base/f"input_{h}.mkv","--output",dest/"stream.glcv","--workdir",dest/"sender","--frames",count,"--width",h*16//9,"--q",0,"--intra-period",gop,"--reset-interval",32]+common,dest/"encode.log",a.legacy)
        enc=read(dest/"sender/encode.json")
        assert enc["header"]["frames"]==count and enc["header"]["intra_period"]==gop and enc["header"]["height"]==h
        for i in range(count):assert np.array_equal(rgb(base/f"input_{h}"/f"im{i+1:05d}.png"),rgb(dest/"sender/source"/f"im{i+1:05d}.png"))
        if not (dest/"receiver/decode.json").exists():run([exe,script,"decode","--input",dest/"stream.glcv","--output",dest/"receiver"]+common,dest/"decode.log",dest)
        dec=read(dest/"receiver/decode.json")
        for i in range(count):assert np.array_equal(rgb(dest/"sender/frames"/f"im{i+1:05d}.png"),rgb(dest/"receiver/frames"/f"im{i+1:05d}.png"))
        assert dec["all_frame_checksums_match"] and not dec["source_media_reads"]
        save(dest/"verified.json",{"clip":a.clip,"codec":"glc","height":h,"gop":gop,"reset":32,"frames":count,"file_bytes":(dest/"stream.glcv").stat().st_size,"sha256":sha(dest/"stream.glcv"),"same_input_frames":count,"same_decoded_frames":count,"source_media_reads":dec["source_media_reads"],"header":enc["header"],"container_overhead_bytes":enc["container_overhead_bytes"],"encode_seconds":enc["pipeline_seconds_before_preview"],"decode_seconds":dec["pipeline_seconds_before_preview"],"peak_encode_mib":enc["peak_allocated_mib"],"peak_decode_mib":dec["peak_allocated_mib"]})
        print(f"VERIFIED {a.clip}/{dest.name}: {enc['file_bytes']} bytes",flush=True)

elif a.phase=="standards":
    for h,gop in profiles:
        target=(base/f"glc_{h}_g{gop}/stream.glcv").stat().st_size
        heights=sorted({h,720},reverse=True) if a.clip=="short" else [h]
        for ch in heights:
            for codec in ["h265","av1"]:
                dest=base/f"{codec}_{ch}_for_{h}_g{gop}";dest.mkdir(exist_ok=True)
                if (dest/"selection.json").exists():continue
                low,high=(0.,51.) if codec=="h265" else (0,63);tried=set();candidates=[]
                for attempt in range(10 if codec=="h265" else 8):
                    crf=round((low+high)/2,5) if codec=="h265" else (low+high)//2
                    if crf in tried:break
                    tried.add(crf); file=dest/f"candidate_{attempt:02d}.mp4"
                    cmd=ff+["-framerate",fps,"-start_number",1,"-i",base/f"input_{ch}/im%05d.png","-frames:v",count,"-an","-vf","scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p","-color_primaries","bt709","-color_trc","bt709","-colorspace","bt709","-color_range","tv"]
                    if codec=="h265":cmd += ["-c:v","libx265","-preset","slow","-crf",crf,"-x265-params",f"keyint={gop}:min-keyint={gop}:scenecut=0:bframes=4:log-level=error:pools=4:frame-threads=2","-tag:v","hvc1","-force_key_frames",f"expr:gte(n,n_forced*{gop})"]
                    else:cmd += ["-c:v","libaom-av1","-cpu-used",6,"-row-mt",1,"-threads",4,"-lag-in-frames",25,"-g",gop,"-keyint_min",gop,"-crf",crf,"-b:v",0]
                    cmd += ["-movflags","+faststart",file]
                    record=run(cmd,dest/f"candidate_{attempt:02d}.log",timeout=600)
                    size=file.stat().st_size;delta=(size-target)/target
                    candidates.append({"crf":crf,"bytes":size,"relative_error":delta,"file":file.name,"seconds":record["seconds"]})
                    print(f"{a.clip}/{dest.name}: CRF {crf}, {size} B, {delta:+.2%}",flush=True)
                    if abs(delta)<=.02:break
                    if size>target:low=crf if codec=="h265" else crf+1
                    else:high=crf if codec=="h265" else crf-1
                    if high<low:break
                best=min(candidates,key=lambda x:abs(x["relative_error"]))
                shutil.copy2(dest/best["file"],dest/"selected.mp4")
                frames=dest/"receiver/frames";frames.mkdir(parents=True)
                run(ff+["-i",dest/"selected.mp4","-vf","scale=in_range=tv:out_range=pc:in_color_matrix=bt709,format=rgb24","-frames:v",count,frames/"im%05d.png"],dest/"decode.log")
                assert len(list(frames.glob('*.png')))==count
                probe=json.loads(subprocess.check_output([str(a.ffmpeg.with_name('ffprobe.exe')),'-v','error','-select_streams','v:0','-show_entries','stream=width,height,pix_fmt,avg_frame_rate,nb_frames:frame=key_frame,pict_type','-of','json',str(dest/'selected.mp4')]))
                keys=[i for i,x in enumerate(probe['frames']) if x['key_frame']]
                assert len(probe['frames'])==count and keys==list(range(0,count,gop)),{'frames':len(probe['frames']),'actual_keys':keys,'expected_keys':list(range(0,count,gop))}
                save(dest/"selection.json",{"clip":a.clip,"codec":codec,"height":ch,"target_height":h,"gop":gop,"target_bytes":target,"within_2_percent":abs(best['relative_error'])<=.02,"selected_sha256":sha(dest/'selected.mp4'),"frames":count,"streams":probe['streams'],"key_frame_indices":[i for i,x in enumerate(probe['frames']) if x['key_frame']],**best,"candidates":candidates})

else:
    expected=[]
    for h,gop in profiles:
        expected.append(f'glc_{h}_g{gop}')
        for ch in (sorted({h,720},reverse=True) if a.clip=='short' else [h]):
            expected.extend(f'{codec}_{ch}_for_{h}_g{gop}' for codec in ['h265','av1'])
    missing=[name for name in expected if not (base/name/('verified.json' if name.startswith('glc_') else 'selection.json')).exists()]
    if missing:raise RuntimeError(f'Complete encodes and independent decodes before evaluation: {missing}')
    os.environ['TORCH_HOME']=str(root/'assets/torch')
    import torch
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    metricpath=a.project/'upstream/mlvc/video/src/metrics/msssim.py'
    ms=module('round2_ms',metricpath).MS_SSIM(channels=3,data_range=1).cuda().eval()
    dpath=a.legacy/'tools/GLC/src/utils/DISTS_pytorch/DISTS_pt.py'
    dm=module('round2_dists',dpath).DISTS().cuda().eval()
    info=read(base/'inputs.json');all_results=[]
    with torch.inference_mode():
        identity=torch.from_numpy(rgb(source/'im00001.png').copy()).permute(2,0,1)[None].float().cuda()/255
        identity_check={'dists':float(dm(identity,identity).item()),'ms_ssim':float(ms(identity,identity).item())}
        assert abs(identity_check['dists'])<1e-4 and abs(identity_check['ms_ssim']-1)<1e-4,identity_check
        del identity
    save(base/'metric_validation.json',{'identity':identity_check,'dists_vgg_sha256':sha(root/'assets/torch/hub/checkpoints/vgg16-397923af.pth'),'dists_calibration_sha256':sha(dpath.with_name('weights.pt')),'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'expected_runs':expected})
    def psnr(total,n):return 10*math.log10(255**2/(total/n)) if n and total else None
    for dest in sorted(base.iterdir()):
        if not (dest/'verified.json').exists() and not (dest/'selection.json').exists():continue
        if (dest/'metrics.json').exists():all_results.append(read(dest/'metrics.json'));continue
        config=read(dest/('verified.json' if (dest/'verified.json').exists() else 'selection.json'))
        gop=config['gop'];dframes=set(range(count)) if count==50 else set(range(0,count,10))
        dframes|={i for i,r in enumerate(info['records']) if r.get('fish_pixels',0)>0}
        for i in range(count):
            # Same sampled frames for GOP64 and GOP128, including every potential I boundary.
            if i%64==0 or i%32==1:dframes|={j for j in [i-1,i,i+1] if 0<=j<count}
        rows=[];sse=0.;samples=0;fish_sse=0.;fish_samples=0;previous_error=None
        for i in range(count):
            ref=rgb(source/f'im{i+1:05d}.png')
            with Image.open(dest/'receiver/frames'/f'im{i+1:05d}.png') as im:
                if im.size!=(1280,720):im=im.resize((1280,720),Image.Resampling.LANCZOS)
                rec=np.asarray(im.convert('RGB'))
            err=rec.astype(np.float64)-ref.astype(np.float64);sq=err**2
            sse+=float(sq.sum());samples+=sq.size
            row={'index':i,'source_frame':start+i,'psnr_db':psnr(float(sq.sum()),sq.size),'dists':None,'temporal_error_delta_mae_8bit':None,'glc_i_position':i%gop==0,'glc_reset_position':i%32==1}
            if previous_error is not None:row['temporal_error_delta_mae_8bit']=float(np.abs(err-previous_error).mean())
            previous_error=err
            mp=masks/f'{start+i:05d}.png'
            if mp.exists():
                mask=np.asarray(Image.open(mp))>0;sub=sq[mask];row['fish_pixels']=int(mask.sum());row['fish_psnr_db']=psnr(float(sub.sum()),sub.size)
                fish_sse+=float(sub.sum());fish_samples+=sub.size
            with torch.inference_mode():
                x=torch.from_numpy(ref.copy()).permute(2,0,1)[None].float().cuda()/255
                y=torch.from_numpy(rec.copy()).permute(2,0,1)[None].float().cuda()/255
                row['ms_ssim']=float(ms(x,y).item())
                if i in dframes:row['dists']=float(dm(x,y).item())
            rows.append(row)
            if (i+1)%50==0:print(f'METRICS {a.clip}/{dest.name}: {i+1}/{count}',flush=True)
        size=(dest/('stream.glcv' if config['codec']=='glc' else 'selected.mp4')).stat().st_size
        result={'run':dest.name,'config':config,'file_bytes':size,'duration_seconds':count/fps,'payload_wait_seconds':{str(k):8*size/(1000*k) for k in [20,30,40]},'global_psnr_db':psnr(sse,samples),'fish_psnr_db':psnr(fish_sse,fish_samples),'fish_pixels':fish_samples//3,'ms_ssim_mean':float(np.mean([r['ms_ssim'] for r in rows])),'dists_mean':float(np.mean([r['dists'] for r in rows if r['dists'] is not None])),'dists_frames':sorted(dframes),'temporal_error_delta_mae_mean':float(np.mean([r['temporal_error_delta_mae_8bit'] for r in rows[1:]])),'per_frame':rows,'evaluation':{'display_resolution':[1280,720],'dists_resize':False,'dists_source_sha256':sha(dpath),'msssim_source_sha256':sha(metricpath),'script_sha256':sha(__file__)}}
        save(dest/'metrics.json',result);all_results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k not in ['config','per_frame','evaluation','dists_frames']},ensure_ascii=False),flush=True)
    assert {r['run'] for r in all_results}==set(expected)
    save(base/'results.json',{'clip':a.clip,'input':info,'metric_validation':read(base/'metric_validation.json'),'results':all_results})
