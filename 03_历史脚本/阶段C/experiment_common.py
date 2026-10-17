from __future__ import annotations
import json
from pathlib import Path
import sys

PROJECT = Path(r"C:\Users\lenovo\Desktop\works\work\semantic-video-mlvc")
DATA = PROJECT / "data/phase_c_20261013"
RUN = PROJECT / "runs/phase_c_20261013"
OUT = Path(r"C:\Users\lenovo\Documents\Codex\2026-10-12\wo-d\outputs\semantic_roi_phase_c")
sys.path.insert(0,str(PROJECT/"adapters"))
from common import torch, F, configure, load_models, prepare, save_json, sha256, environment
from precomputed_roi_loss import PrecomputedROILoss
from src.losses.codec import CodecLoss, loss_total_rdc_psnr


def manifest():
    value=json.loads((DATA/"manifest.json").read_text("utf-8"))
    if value["protocol_sha256"]!=sha256(DATA/"protocol.json"):raise ValueError("Data preparation has not finalized the current protocol")
    if not any(v["video"]=="video108" and v["official_split"]=="val" for v in value["videos"]):raise ValueError("Official video108 validation data missing")
    return value


def ordinary_loss():
    return CodecLoss(frame_loss=loss_total_rdc_psnr(is_yuv420=True,mse_y_weight=8,mse_yuv_420=True,mse_rgb_weight=0),mse_lambda_exponent=1.,bpp_weight=1.)


def rng_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def binary_metrics(tp,fp,fn):
    return {"tp":int(tp),"fp":int(fp),"fn":int(fn),"fish_iou":tp/(tp+fp+fn) if tp+fp+fn else None,
            "fish_recall":tp/(tp+fn) if tp+fn else None,"fish_precision":tp/(tp+fp) if tp+fp else None,
            "fish_f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
