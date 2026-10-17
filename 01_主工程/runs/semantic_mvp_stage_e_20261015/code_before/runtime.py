"""CPU-only shared utilities. The receiver has no ML/source dependency."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = Path(os.environ.get("FFMPEG_PATH", r"C:\Users\lenovo\Desktop\works\Python\video-baseline\tools\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"))
FFPROBE = FFMPEG.with_name("ffprobe.exe")
PROFILE = "semantic-video-layered-v1"

def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")

def run(command):
    result = subprocess.run([str(x) for x in command], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {command[0]}\n{result.stderr[-4000:]}")
    return result.stdout

def probe(path):
    value = json.loads(run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", path]))
    if len(value["streams"]) != 1:
        raise ValueError("Expected one video stream")
    return value["streams"][0]

def fps_string(value):
    fps = Fraction(str(value)).limit_denominator(10000)
    if not 0 < fps <= 120:
        raise ValueError("FPS must be in (0,120]")
    return f"{fps.numerator}/{fps.denominator}"

def encode_h265(frame_dir, output, fps, crf=36, scale=None, bbox=None, start=0, count=None, offline=False):
    filters=[]
    if bbox:
        x,y,w,h=bbox
        filters.append(f"crop={w}:{h}:{x}:{y}")
    if scale:
        filters.append(f"scale={scale[0]}:{scale[1]}:flags=lanczos")
    filters.append("scale=in_range=full:out_range=tv:out_color_matrix=bt709")
    args=[FFMPEG,"-v","error","-framerate",fps,"-start_number",str(start),"-i",Path(frame_dir)/"%05d.png","-an","-vf",",".join(filters),"-c:v","libx265","-preset","medium","-crf",str(crf)]
    if not offline:
        args += ["-tune","zerolatency"]
    args += ["-x265-params",f"keyint=32:min-keyint=32:scenecut=0:bframes={4 if offline else 0}:pools=4:frame-threads=2:log-level=error", "-pix_fmt","yuv420p","-color_range","tv","-colorspace","bt709","-color_trc","bt709","-color_primaries","bt709","-map_metadata","-1"]
    if count is not None:
        args += ["-frames:v",str(count)]
    args += [output]
    run(args)
    return [str(x) for x in args]

def decode_png(video, directory, size=None):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    args=[FFMPEG,"-v","error","-i",video,"-an","-fps_mode","passthrough"]
    if size:
        args += ["-vf",f"scale={size[0]}:{size[1]}:flags=lanczos"]
    args += ["-start_number","0","-compression_level","1",directory/"%05d.png"]
    run(args)
    return [str(x) for x in args]

def preview(frame_dir, output, fps):
    run([FFMPEG,"-v","error","-framerate",fps,"-start_number","0","-i",Path(frame_dir)/"%05d.png","-an","-c:v","libx264","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",output])

def fresh(path):
    path=Path(path).resolve()
    path.mkdir(parents=True,exist_ok=False)
    return path

def require_runtime():
    if not FFMPEG.is_file() or not FFPROBE.is_file():
        raise FileNotFoundError("FFmpeg/FFprobe missing. Set FFMPEG_PATH to ffmpeg.exe")
