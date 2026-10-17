"""Actual MLVC forward/backward with official ROI loss; no optimizer or updates."""
from pathlib import Path
import numpy as np
from PIL import Image
from common import ROOT, torch, configure, load_models, rgb_tensor, prepare, save_json, sha256
from precomputed_roi_loss import PrecomputedROILoss

configure()
corpus=ROOT / "data/phase_b_annotation_corpus/sfishtrack_001"
mask_full=np.asarray(Image.open(corpus / "ground_truth_fish/00001.png"))
ys,xs=np.nonzero(mask_full)
if len(xs)==0: raise ValueError("Diagnostic pair has no fish GT")
cx,cy=int(np.mean(xs)),int(np.mean(ys))
left=max(0,min(mask_full.shape[1]-256,cx-128))
top=max(0,min(mask_full.shape[0]-256,cy-128))
output=ROOT / "runs/phase_b_roi_backprop"
output.mkdir(exist_ok=False)
for idx in [0,1]:
    Image.open(corpus / "reference" / f"{idx:05d}.png").convert("RGB").crop((left,top,left+256,top+256)).save(output / f"crop_{idx}.png")
mask=torch.from_numpy(mask_full[top:top+256,left:left+256].copy()).float()[None].cuda()
model,i_model,_=load_models(ROOT / "checkpoints/mlvc-s-psnr-v1.ckpt",ROOT / "checkpoints/DMCI-6.0-mirror.pth.tar",entropy=False)
frames=[prepare(rgb_tensor(output / f"crop_{idx}.png"),model.padding_size) for idx in [0,1]]
with torch.no_grad(): reference=i_model(frames[0],q_index=32)["x_hat"].detach()
del i_model
initial={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
model.train()
criterion=PrecomputedROILoss(4)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
rd=model(frames[1],{"ref_frame":reference,"ref_feature":None},q_index=32,fa_idx=1)
stats=criterion(rd,frames[1],mask,torch.tensor([768**(32/63)],device="cuda"))
loss=stats["loss"].mean()
loss.backward()
grads={n:float(p.grad.norm().item()) for n,p in model.named_parameters() if p.grad is not None}
if not torch.isfinite(loss) or not all(np.isfinite(v) for v in grads.values()) or not any(n.startswith("encoder.") and v>0 for n,v in grads.items()):
    raise RuntimeError("ROI loss did not reach encoder with finite gradients")
if any(not torch.equal(initial[n],p.detach().cpu()) for n,p in model.named_parameters()): raise RuntimeError("Parameters unexpectedly changed during diagnostic")
save_json(output / "result.json",{"purpose":"Integration/backprop only, no optimizer; reserved GT used to check shape/gradient path, not training or hyperparameter selection",
           "success":True,"codec_parameter_updates":False,"segmentation_model_loaded":False,"batch_size":1,"frames":2,"gradient_frames":1,"crop_size":256,
           "crop_origin":{"left":left,"top":top},"gt_mask_fraction":mask.mean().item(),"q_index":32,"rate_weight":1.,"roi_per_pixel_ratio":4.,
           "loss":loss.item(),"bpp_estimate":rd["bpp"].mean().item(),"parameter_tensors_with_nonzero_gradient":sum(v>0 for v in grads.values()),
           "encoder_gradient_nonzero":True,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
           "grad_norm_by_parameter":grads,"p_weight_sha256":sha256(ROOT / "checkpoints/mlvc-s-psnr-v1.ckpt"),
           "limitation":"Not an ROI-trained checkpoint or evidence of rate/semantic gain"})
print(f"ROI forward/backward reached MLVC encoder; {sum(v>0 for v in grads.values())} tensors have gradients; no weights changed; peak {torch.cuda.max_memory_allocated()/2**20:.2f}MiB")
