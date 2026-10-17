"""Sequential fixed-baseline codecs and frozen teacher scoring on reserved data."""
from __future__ import annotations
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT, FFMPEG, FFPROBE, sha256, save_json

LEGACY = Path(r"C:\Users\lenovo\Desktop\works\work\semantic-video-training")
OUT = ROOT / "runs/phase_b_codecs"
OUT.mkdir(exist_ok=False)
LOG = ROOT / "audit/phase_b_commands"
LOG.mkdir(exist_ok=False)
commands = []


def run(name, command):
    t0 = time.perf_counter()
    with (LOG / f"{name}.log").open("x",encoding="utf-8") as log:
        result = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    commands.append({"name":name,"command":command,"exit_code":result.returncode,"seconds":time.perf_counter()-t0})
    save_json(LOG / "commands.json",commands)
    print(f"{name}: exit {result.returncode}, {commands[-1]['seconds']:.2f}s",flush=True)
    if result.returncode: raise RuntimeError(f"{name} failed; see {LOG / (name+'.log')}")


corpus = json.loads((ROOT / "data/phase_b_annotation_corpus/manifest.json").read_text("utf-8"))
raw_sequences = [{**r,"streams":[]} for r in corpus["sequences"]]
save_json(ROOT / "configs/phase_b_gt_raw.json",{"sequences":raw_sequences})
run("raw_gt_teacher",[sys.executable,"adapters/evaluate_semantics.py","--manifest","configs/phase_b_gt_raw.json","--output","runs/phase_b_gt_raw"])
sequences = []
for sequence in corpus["sequences"]:
    name = sequence["name"]
    folder = OUT / name
    folder.mkdir()
    streams = []
    for q in [0,21]:
        directory = folder / f"mlvc_q{q}"
        run(f"{name}_mlvc_q{q}_encode",[sys.executable,"adapters/mlvc_file.py","encode","--source",sequence["source"],"--reference-dir",sequence["reference_dir"],
            "--fps","5/1","--run",str(directory),"--q-index",str(q),"--i-weight",str(ROOT / "checkpoints/DMCI-6.0-mirror.pth.tar")])
        run(f"{name}_mlvc_q{q}_decode",[sys.executable,"adapters/mlvc_file.py","decode","--file",str(directory / "video.mlvcv"),"--output",str(directory / "receiver"),
            "--i-weight",str(ROOT / "checkpoints/DMCI-6.0-mirror.pth.tar")])
        run(f"{name}_mlvc_q{q}_verify",[sys.executable,"adapters/mlvc_file.py","verify","--run",str(directory),"--decoded",str(directory / "receiver")])
        streams.append({"name":f"mlvc_q{q}","frames":str(directory / "receiver/rgb"),"file":str(directory / "video.mlvcv")})
    for crf in [44,50]:
        directory = folder / f"hevc_crf{crf}"
        directory.mkdir()
        (directory / "rgb").mkdir()
        file = directory / "video.mp4"
        run(f"{name}_hevc_crf{crf}_encode",[str(FFMPEG),"-v","error","-framerate","5/1","-start_number","0","-i",str(Path(sequence["reference_dir"]) / "%05d.png"),
             "-an","-vf","scale=in_range=full:out_range=tv:out_color_matrix=bt709","-c:v","libx265","-preset","medium","-tune","zerolatency",
             "-crf",str(crf),"-x265-params","keyint=32:min-keyint=32:scenecut=0:bframes=0:log-level=error","-pix_fmt","yuv420p",
             "-color_range","tv","-colorspace","bt709","-color_trc","bt709","-color_primaries","bt709",str(file)])
        run(f"{name}_hevc_crf{crf}_decode",[str(FFMPEG),"-v","error","-i",str(file),"-an","-fps_mode","passthrough","-start_number","0",str(directory / "rgb/%05d.png")])
        frame_types = json.loads(subprocess.check_output([str(FFPROBE),"-v","error","-select_streams","v:0","-show_entries","frame=pict_type","-of","json",str(file)]))["frames"]
        if len(frame_types)!=sequence["frame_count"] or sum(f["pict_type"]=="I" for f in frame_types)!=math.ceil(sequence["frame_count"]/32):
            raise ValueError("HEVC frame/GOP count differs")
        save_json(directory / "verification.json",{"complete_file_bytes":file.stat().st_size,"file_sha256":sha256(file),"frames":len(frame_types),
                    "i_frames":sum(f["pict_type"]=="I" for f in frame_types),"separate_encode_decode_processes":True,"profile":"medium zerolatency BT.709 8-bit YUV420"})
        streams.append({"name":f"hevc_crf{crf}","frames":str(directory / "rgb"),"file":str(file)})
    directory = folder / "glc_q0"
    directory.mkdir()
    interpreter = LEGACY / ".venv-glc/Scripts/python.exe"
    script = LEGACY / "scripts/glc_video_codec.py"
    shared = ["--repo",str(LEGACY / "tools/GLC"),"--weights",str(LEGACY / "checkpoints/glc"),"--ffmpeg",str(FFMPEG)]
    file = directory / "video.glcv"
    run(f"{name}_glc_q0_encode",[str(interpreter),"-B",str(script),"encode","--input",sequence["source"],"--output",str(file),"--workdir",str(directory / "sender"),
            "--frames",str(sequence["frame_count"]),"--width","1280","--q","0","--intra-period","32","--reset-interval","32",*shared])
    run(f"{name}_glc_q0_decode",[str(interpreter),"-B",str(script),"decode","--input",str(file),"--output",str(directory / "receiver"),*shared])
    decoder = json.loads((directory / "receiver/decode.json").read_text("utf-8"))
    if not decoder["all_frame_checksums_match"] or decoder["source_media_reads"]: raise ValueError("GLC independent checks failed")
    streams.append({"name":"glc_q0","frames":str(directory / "receiver/frames"),"file":str(file)})
    sequences.append({**sequence,"streams":streams})
    save_json(ROOT / "configs/phase_b_gt_codecs.json",{"sequences":sequences,"policy":"Fixed q0,q21 MLVC, CRF44,50 HEVC, q0 GLC; no tuning using these reserved GT sequences; playback 5fps"})
run("gt_codec_semantics",[sys.executable,"adapters/evaluate_semantics.py","--manifest","configs/phase_b_gt_codecs.json","--output","runs/phase_b_gt_codec_semantics"])
