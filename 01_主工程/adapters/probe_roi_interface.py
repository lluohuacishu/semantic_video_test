"""Mask interface/gradient check only: no codec or classifier weight updates."""
from pathlib import Path
import numpy as np
from PIL import Image
from common import ROOT, torch, configure, save_json, sha256, UPSTREAM
from precomputed_roi_loss import PrecomputedROILoss

configure("cpu")
path=ROOT / "data/phase_b_annotation_corpus/sfishtrack_001/ground_truth_fish/00000.png"
mask=torch.from_numpy(np.asarray(Image.open(path)).copy()).float()[None]
target=torch.full((1,3,*mask.shape[-2:]),0.5)
criterion=PrecomputedROILoss(4)
reconstruction=target.clone()
reconstruction[:,0]+=0.01
reconstruction.requires_grad_(True)
result=criterion({"x_hat":reconstruction,"bpp":torch.tensor([0.1])},target,mask,torch.tensor([1.]))
result["loss"].mean().backward()
gradient=reconstruction.grad[0,0].abs()
roi_grad=gradient[mask[0].bool()].mean().item()
bg_grad=gradient[~mask[0].bool()].mean().item()
ratio=roi_grad/bg_grad
if not 3.99<ratio<4.01: raise RuntimeError(f"Unexpected ROI gradient weight ratio {ratio}")
checks=[]
for name,binary in [("empty",torch.zeros_like(mask)),("full",torch.ones_like(mask))]:
    rd={"x_hat":target+0.01,"bpp":torch.tensor([0.1])}
    stats=criterion(rd,target,binary,torch.tensor([1.]))
    if not torch.isfinite(stats["loss"]).all(): raise ValueError(f"Nonfinite {name} mask loss")
    checks.append({"mask":name,"loss":stats["loss"].item(),"finite":True})
try:
    criterion({"x_hat":target,"bpp":torch.tensor([0.1])},target,None,torch.tensor([1.]))
except ValueError: missing_rejected=True
else: raise AssertionError("Missing mask silently accepted")
output=ROOT / "runs/phase_b_roi_interface"
output.mkdir(exist_ok=False)
save_json(output / "result.json",{"native_class":"loss_total_rdc_psnr_roi_per_pixel","native_source_sha256":sha256(UPSTREAM / "video/src/losses/codec.py"),
            "native_source_modified":False,"auxiliary_model_loaded":False,"mask_required":missing_rejected,"weight_updates":False,
            "actual_mask_file":str(path),"roi_fraction":mask.mean().item(),"configured_per_pixel_ratio":4.,"observed_y_gradient_ratio":ratio,
            "normalization":"per-frame, p_roi=None; ROI pixel weight is 4x BG, average pixel weight normalized to1",
            "supported_loss_profile":"YUV420, geometric YUV MSE/PSNR loss, mse_rgb_weight=0, rate weight1",
            "empty_full_mask_checks":checks,"limitation":"Gradient/interface test only; not a trained ROI codec or evidence of compression benefit"})
print(f"Native precomputed ROI interface passed: gradient ratio {ratio:.6f}; masks empty/full finite; no weights updated")
