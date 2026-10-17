"""Match HEVC to fixed MLVC file bytes, choose by size only, then score once."""
import json
from pathlib import Path
import subprocess
import sys
from common import ROOT, FFMPEG, save_json, sha256

corpus=json.loads((ROOT / "data/phase_b_annotation_corpus/manifest.json").read_text("utf-8"))
sequence=next(s for s in corpus["sequences"] if s["name"]=="sfishtrack_027")
mlvc=ROOT / "runs/phase_b_codecs/sfishtrack_027/mlvc_q21/video.mlvcv"
target=mlvc.stat().st_size
directory=ROOT / "runs/phase_b_rate_match"
directory.mkdir(exist_ok=False)
candidates=[]
low,high=44.,50.
best=None
for attempt in range(8):
    crf=(low+high)/2
    file=directory / f"candidate_{attempt}_crf{crf:.5f}.mp4"
    command=[str(FFMPEG),"-v","error","-framerate","5/1","-start_number","0","-i",str(Path(sequence["reference_dir"]) / "%05d.png"),
             "-an","-vf","scale=in_range=full:out_range=tv:out_color_matrix=bt709","-c:v","libx265","-preset","medium","-tune","zerolatency",
             "-crf",str(crf),"-x265-params","keyint=32:min-keyint=32:scenecut=0:bframes=0:log-level=error","-pix_fmt","yuv420p",
             "-color_range","tv","-colorspace","bt709","-color_trc","bt709","-color_primaries","bt709",str(file)]
    subprocess.run(command,check=True)
    size=file.stat().st_size
    record={"crf":crf,"file":str(file),"bytes":size,"absolute_difference":abs(size-target),"command":command}
    candidates.append(record)
    if best is None or record["absolute_difference"]<best["absolute_difference"]: best=record
    save_json(directory / "search.json",{"target_mlvc_bytes":target,"candidates":candidates,"selection_metric":"absolute full-file byte difference only; no GT/PSNR/task scores used"})
    if best["absolute_difference"]/target<=0.003: break
    if size>target: low=crf
    else: high=crf
(directory / "hevc_rgb").mkdir()
subprocess.run([str(FFMPEG),"-v","error","-i",best["file"],"-an","-fps_mode","passthrough","-start_number","0",str(directory / "hevc_rgb/%05d.png")],check=True)
save_json(directory / "selected.json",{"target_mlvc_file":str(mlvc),"target_bytes":target,"hevc_file":best["file"],"hevc_bytes":best["bytes"],
          "hevc_crf":best["crf"],"relative_file_size_difference":best["absolute_difference"]/target,"selection_using_gt":False,
          "hevc_sha256":sha256(best["file"]),"limitation":"Single matched diagnostic example, not a full rate-distortion conclusion"})
manifest={"sequences":[{**sequence,"streams":[{"name":"mlvc_q21","file":str(mlvc),"frames":str(mlvc.parent / "receiver/rgb")},
          {"name":"hevc_matched","file":best["file"],"frames":str(directory / "hevc_rgb")}]}]}
save_json(directory / "manifest.json",manifest)
subprocess.run([sys.executable,str(ROOT / "adapters/evaluate_semantics.py"),"--manifest",str(directory / "manifest.json"),"--output",str(directory / "semantic_results")],cwd=ROOT,check=True)
print(f"Matched target {target}B, HEVC {best['bytes']}B, difference {best['absolute_difference']/target:.4%}",flush=True)
