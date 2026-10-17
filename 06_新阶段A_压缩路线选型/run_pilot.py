"""Bounded codec-selection pilot. Reuses installed codecs; never trains a model."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def command(cmd, log, cwd=None, timeout=900):
    started = time.perf_counter()
    last = ""
    with Path(log).open("w", encoding="utf-8") as stream:
        proc = subprocess.Popen([str(x) for x in cmd], stdout=stream, stderr=subprocess.STDOUT, cwd=cwd)
        while True:
            try:
                code = proc.wait(timeout=15)
                break
            except subprocess.TimeoutExpired:
                if time.perf_counter() - started > timeout:
                    proc.kill()
                    proc.wait()
                    raise TimeoutError(f"Command exceeded {timeout}s: {log}")
                lines = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()
                progress = next((s for s in reversed(lines) if s.startswith(("encode ", "decode "))), "")
                if progress and progress != last:
                    print(Path(log).stem + ": " + progress, flush=True)
                    last = progress
    record = {"command": [str(x) for x in cmd], "cwd": str(cwd) if cwd else None, "seconds": time.perf_counter()-started, "returncode": code}
    save(str(log)+".json", record)
    if code:
        raise RuntimeError(Path(log).read_text(encoding="utf-8", errors="replace")[-3000:])
    return record["seconds"]

parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=["prepare", "glc", "standards", "evaluate"])
parser.add_argument("--project", type=Path, required=True)
parser.add_argument("--legacy", type=Path, required=True)
parser.add_argument("--ffmpeg", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
out = args.output.resolve()
source = out / "source"
data = args.project / "data/phase_f_external_20261015/positive_case"
ff = [str(args.ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin"]

if args.phase == "prepare":
    out.mkdir(parents=True, exist_ok=False)
    source.mkdir()
    rows = []
    for n in range(50, 100):
        src = data / "sequence" / f"{n:05d}.jpg"
        with Image.open(src) as im:
            rgb = np.asarray(im.convert("RGB"))
        assert rgb.shape == (720, 1280, 3)
        dest = source / f"im{n-49:05d}.png"
        Image.fromarray(rgb).save(dest)
        rows.append({"source_frame": n, "source_sha256": digest(src), "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(), "png_sha256": digest(dest)})
    save(out / "input_manifest.json", {"clip":"video6_50_99", "role":"previously used development clip", "frames":50, "fps":25, "width":1280, "height":720, "gt_frames":[60,75,90], "source_frames":rows})
    cmd = ff + ["-framerate","25","-start_number","1","-i",str(source/"im%05d.png"),"-frames:v","50","-an","-c:v","ffv1","-level","3","-pix_fmt","bgr0",str(out/"input_lossless.mkv")]
    command(cmd, out/"prepare.log")
    save(out/"protocol_files.json", {"script_sha256":digest(__file__), "project":str(args.project), "legacy":str(args.legacy), "ffmpeg":str(args.ffmpeg), "ffmpeg_sha256":digest(args.ffmpeg), "training":False, "network_test":False})
    print("Prepared 50 RGB source frames and lossless GLC input", flush=True)

elif args.phase == "glc":
    py = args.legacy / ".venv-glc/Scripts/python.exe"
    script = args.legacy / "scripts/glc_video_codec.py"
    for q in [0, 63]:
        run = out / f"glc_q{q:02d}"
        if (run/"verification.json").exists():
            print(f"GLC q{q}: already complete", flush=True)
            continue
        run.mkdir(exist_ok=False)
        common = ["--repo", args.legacy/"tools/GLC", "--weights",args.legacy/"checkpoints/glc","--ffmpeg",args.ffmpeg]
        enc = command([py,script,"encode","--input",out/"input_lossless.mkv","--output",run/"stream.glcv","--workdir",run/"sender","--frames","50","--width","1280","--q",str(q),"--intra-period","32","--reset-interval","32"]+common, run/"encode.log", args.legacy)
        for n in range(1,51):
            a=np.asarray(Image.open(source/f"im{n:05d}.png"));b=np.asarray(Image.open(run/"sender/source"/f"im{n:05d}.png"))
            assert np.array_equal(a,b), f"GLC input mismatch {n}"
        dec = command([py,script,"decode","--input",run/"stream.glcv","--output",run/"receiver"]+common,run/"decode.log",run)
        for n in range(1,51):
            a=np.asarray(Image.open(run/"sender/frames"/f"im{n:05d}.png"));b=np.asarray(Image.open(run/"receiver/frames"/f"im{n:05d}.png"))
            assert np.array_equal(a,b), f"GLC reconstruction mismatch {n}"
        save(run/"verification.json", {"input_rgb_equal_frames":50,"independent_decoded_rgb_equal_frames":50,"file_bytes":(run/"stream.glcv").stat().st_size,"file_sha256":digest(run/"stream.glcv"),"encode_process_seconds":enc,"decode_process_seconds":dec})
        print(f"GLC q{q} complete: {(run/'stream.glcv').stat().st_size} bytes", flush=True)

elif args.phase == "standards":
    all_choices = []
    for q in [0,63]:
        target=(out/f"glc_q{q:02d}"/"stream.glcv").stat().st_size
        for codec in ["h265","av1"]:
            run=out/f"{codec}_match_q{q:02d}"
            if (run/"selection.json").exists():
                all_choices.append(json.loads((run/"selection.json").read_text(encoding="utf-8")))
                continue
            run.mkdir(exist_ok=False)
            low,high=(0.,51.) if codec=="h265" else (0,63)
            candidates=[]; tried=set()
            for attempt in range(10 if codec=="h265" else 8):
                crf=round((low+high)/2,5) if codec=="h265" else (low+high)//2
                if crf in tried:break
                tried.add(crf)
                path=run/f"candidate_{attempt:02d}.mp4"
                cmd=ff+["-framerate","25","-start_number","1","-i",str(source/"im%05d.png"),"-frames:v","50","-an","-vf","scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p","-color_primaries","bt709","-color_trc","bt709","-colorspace","bt709","-color_range","tv"]
                if codec=="h265":
                    cmd += ["-c:v","libx265","-preset","slow","-crf",str(crf),"-x265-params","keyint=32:min-keyint=32:scenecut=0:bframes=4:log-level=error:pools=4:frame-threads=2","-tag:v","hvc1"]
                else:
                    cmd += ["-c:v","libaom-av1","-cpu-used","6","-row-mt","1","-threads","4","-lag-in-frames","25","-g","32","-crf",str(crf),"-b:v","0"]
                cmd += ["-force_key_frames","expr:gte(n,n_forced*32)","-movflags","+faststart",str(path)]
                elapsed=command(cmd,run/f"candidate_{attempt:02d}.log",timeout=300)
                size=path.stat().st_size;error=(size-target)/target
                candidates.append({"crf":crf,"file":path.name,"bytes":size,"relative_error":error,"encode_process_seconds":elapsed})
                print(f"{codec} target q{q}: candidate {attempt+1}, CRF {crf}, {size} bytes ({error:+.2%})",flush=True)
                if abs(error)<=.02:break
                if size>target:
                    low=crf if codec=="h265" else crf+1
                else:
                    high=crf if codec=="h265" else crf-1
                if high<low:break
            best=min(candidates,key=lambda x:abs(x["relative_error"]))
            shutil.copy2(run/best["file"],run/"selected.mp4")
            frames=run/"receiver/frames";frames.mkdir(parents=True)
            elapsed=command(ff+["-i",str(run/"selected.mp4"),"-vf","scale=in_range=tv:out_range=pc:in_color_matrix=bt709,format=rgb24","-frames:v","50",str(frames/"im%05d.png")],run/"decode.log")
            assert len(list(frames.glob("*.png")))==50
            choice={"codec":codec,"target_glc_q":q,"target_bytes":target,**best,"decode_process_seconds":elapsed,"within_2_percent":abs(best["relative_error"])<=.02,"selected_sha256":digest(run/"selected.mp4"),"candidates":candidates}
            save(run/"selection.json",choice);all_choices.append(choice)
    save(out/"selections.json",all_choices)

else:
    import importlib.util
    import torch
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metric_path = args.project / "upstream/mlvc/video/src/metrics/msssim.py"
    spec = importlib.util.spec_from_file_location("pilot_msssim", metric_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ms = module.MS_SSIM(channels=3, data_range=1).to(device)
    def psnr(sse,count):return 10*math.log10(255**2/(sse/count)) if sse else None
    results=[]
    for q in [0,63]:
        for codec in ["glc","h265","av1"]:
            name=f"glc_q{q:02d}" if codec=="glc" else f"{codec}_match_q{q:02d}"
            run=out/name;recon=run/"receiver/frames"
            file=run/("stream.glcv" if codec=="glc" else "selected.mp4")
            sse=0.; count=0; fish_sse=0.;fish_count=0;bg_sse=0.;bg_count=0;per_frame=[]
            for i in range(1,51):
                a=np.asarray(Image.open(source/f"im{i:05d}.png").convert("RGB"),dtype=np.float64)
                b=np.asarray(Image.open(recon/f"im{i:05d}.png").convert("RGB"),dtype=np.float64)
                assert a.shape==b.shape
                error=(a-b)**2;sse+=float(error.sum());count+=error.size
                n=i+49;row={"source_frame":n,"rgb_psnr_db":psnr(float(error.sum()),error.size)}
                with torch.inference_mode():
                    ta=torch.from_numpy(a.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(device)/255.0
                    tb=torch.from_numpy(b.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(device)/255.0
                    row["rgb_ms_ssim"] = float(ms(ta,tb).mean().item())
                maskpath=data/"fish_masks"/f"{n:05d}.png"
                if maskpath.exists():
                    mask=np.asarray(Image.open(maskpath))>0
                    assert mask.ndim==2
                    e=error[mask];fish_sse+=float(e.sum());fish_count+=e.size
                    e_bg=error[~mask];bg_sse+=float(e_bg.sum());bg_count+=e_bg.size
                    row.update({"fish_pixels":int(mask.sum()),"gt_fish_psnr_db":psnr(float(e.sum()),e.size)})
                per_frame.append(row)
            target=(out/f"glc_q{q:02d}"/"stream.glcv").stat().st_size
            meta=json.loads((run/("verification.json" if codec=="glc" else "selection.json")).read_text(encoding="utf-8"))
            result={"codec":codec,"glc_budget_q":q,"run":name,"file_bytes":file.stat().st_size,"relative_budget_error":(file.stat().st_size-target)/target,"global_rgb_psnr_db":psnr(sse,count),"gt_fish_psnr_db":psnr(fish_sse,fish_count),"gt_background_psnr_db":psnr(bg_sse,bg_count),"gt_fish_pixels":fish_count//3,"file_sha256":digest(file),"payload_only_seconds":{str(k):file.stat().st_size*8/(k*1000) for k in [20,30,40]},"encode_process_seconds":meta['encode_process_seconds'],"decode_process_seconds":meta['decode_process_seconds'],"per_frame":per_frame}
            results.append(result)
            result["rgb_ms_ssim_mean"] = float(np.mean([x["rgb_ms_ssim"] for x in per_frame]))
            print(json.dumps({k:v for k,v in result.items() if k!='per_frame'},ensure_ascii=False),flush=True)
    save(out/"metrics.json",{"scope":"single previously-used development clip; no fresh holdout, no DISTS, no new human or task accuracy score", "evaluation_script_sha256":digest(__file__), "ms_ssim_source_sha256":digest(metric_path), "results":results})
