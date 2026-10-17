"""Small, explicit adapter around pinned Microsoft MLVC; no legacy imports."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream/mlvc"
COMMIT = "64b10d4c0108106e66b7c54d6e5e470200a57abf"
P_WEIGHT_SHA = "1b86b757ddb115342293efb57719d6216c6ee2e459ae796ec41723b5c05ca896"
FFMPEG = Path(r"C:\Users\lenovo\Desktop\works\Python\video-baseline\tools\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe")
FFPROBE = FFMPEG.with_name("ffprobe.exe")
sys.path.insert(0, str(UPSTREAM / "video"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml
from src.utils.model_factory import create_video_model, create_image_model
from src.utils.stream_helper import get_state_dict, get_padding_size
from src.transforms.functional import rgb2ycbcr, ycbcr2rgb, down_and_upsample
from src.metrics.msssim import MS_SSIM


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def configure(device="cuda"):
    torch.set_num_threads(4)
    torch.manual_seed(20261013)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")


def sync(device="cuda"):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def load_models(weight, i_weight=None, device="cuda", entropy=True):
    config = yaml.safe_load((UPSTREAM / "video/configs/train_video-mlvc_s.yaml").read_text("utf-8"))["model"]
    p_model = create_video_model(config["p_frame"])
    p_model.load_state_dict(get_state_dict(weight), strict=True)
    p_model.eval()
    # Build probability tables on CPU, identically in encoder and cold decoder.
    if entropy:
        p_model.build_encoder_pmf()
    p_model.to(device)
    i_model = None
    if i_weight:
        i_model = create_image_model(config["i_frame"])
        i_model.load_state_dict(get_state_dict(i_weight), strict=True)
        i_model.eval()
        if entropy:
            i_model.build_encoder_pmf()
        i_model.to(device)
    return p_model, i_model, config


def probe(video):
    data = json.loads(subprocess.check_output([str(FFPROBE), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration,pix_fmt", "-of", "json", str(video)]))
    return data["streams"][0]


def extract(video, directory, width=None, height=None, frames=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    args = [str(FFMPEG), "-v", "error", "-i", str(video), "-map", "0:v:0", "-an", "-fps_mode", "passthrough"]
    if width or height:
        if not width or not height:
            raise ValueError("Both width and height are required")
        args += ["-vf", f"scale={width}:{height}:flags=lanczos"]
    if frames:
        args += ["-frames:v", str(frames)]
    args += ["-start_number", "0", str(directory / "%05d.png")]
    subprocess.run(args, check=True)
    paths = sorted(directory.glob("*.png"))
    if not paths:
        raise RuntimeError("No video frames extracted")
    return paths


def rgb_tensor(path, device="cuda"):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0


def prepare(rgb, padding_size):
    height, width = rgb.shape[-2:]
    if width % 2 or height % 2:
        raise ValueError("Initial YUV420 profile requires even image dimensions")
    yuv = down_and_upsample(rgb2ycbcr(rgb))
    padding = get_padding_size(height, width, padding_size)
    return F.pad(yuv, padding, mode="replicate")


def gray_dpb(shape, device="cuda"):
    return {"ref_frame": torch.full(shape, 0.5, device=device), "ref_feature": None}


def rgb8(xhat, height, width):
    rgb = ycbcr2rgb(xhat[..., :height, :width])
    return (rgb.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()


def save_rgb(path, pixels):
    Image.fromarray(pixels).save(path)


def dpb_hash(dpb):
    # Verifies floating reconstruction and temporal feature state, not just PNGs.
    hashes = {}
    for key in ("ref_frame", "ref_feature"):
        value = dpb.get(key)
        hashes[key] = None if value is None else hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return hashes


def metrics(reference_dir, decoded_dir, device="cuda"):
    refs = sorted(Path(reference_dir).glob("*.png"))
    decs = sorted(Path(decoded_dir).glob("*.png"))
    if len(refs) != len(decs) or not refs:
        raise ValueError(f"Frame count mismatch: {len(refs)} vs {len(decs)}")
    ms = MS_SSIM(channels=3, data_range=1).to(device)
    records = []
    total_squared_error = 0.0
    total_samples = 0
    with torch.inference_mode():
        for ref, dec in zip(refs, decs, strict=True):
            a, b = rgb_tensor(ref, device), rgb_tensor(dec, device)
            if a.shape != b.shape:
                raise ValueError("Frame resolution mismatch")
            mse = (a - b).double().square().mean().item()
            total_squared_error += mse * a.numel()
            total_samples += a.numel()
            psnr = None if mse == 0 else float(-10 * np.log10(mse))
            records.append({"frame": ref.name, "rgb_psnr_db": psnr, "rgb_ms_ssim": ms(a, b).mean().item(), "rgb_mse": mse})
    global_mse = total_squared_error / total_samples
    return {"rgb_psnr_global_db": None if global_mse == 0 else float(-10 * np.log10(global_mse)),
            "rgb_psnr_mean_per_frame_db": float(np.mean([r["rgb_psnr_db"] for r in records])) if all(r["rgb_psnr_db"] is not None for r in records) else None,
            "rgb_ms_ssim_mean": float(np.mean([r["rgb_ms_ssim"] for r in records])),
            "metric_profile": "decoded uint8 RGB PNG; original RGB reference; BT.709 MLVC conversion; pinned upstream MS_SSIM channels=3", "frames": records}


def preview(frame_dir, video, fps):
    subprocess.run([str(FFMPEG), "-v", "error", "-framerate", str(fps), "-start_number", "0", "-i", str(Path(frame_dir) / "%05d.png"), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(video)], check=True)


def environment():
    import platform
    return {"python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "upstream_commit": COMMIT, "precision": "fp32", "tf32": False, "deterministic": True}
