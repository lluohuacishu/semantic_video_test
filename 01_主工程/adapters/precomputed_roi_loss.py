"""Reuse pinned native ROI loss with explicitly supplied masks, no auxiliary model."""
from common import torch
from torch import nn
from src.losses.codec import CodecLoss, loss_total_rdc_psnr_roi_per_pixel


class RequirePrecomputedMask(nn.Module):
    def forward(self,*args,**kwargs):
        raise RuntimeError("Precomputed water-target mask required; auxiliary segmentation inference is disabled")


class PrecomputedROILoss(nn.Module):
    def __init__(self,roi_pixel_weight_ratio=4.0):
        super().__init__()
        if roi_pixel_weight_ratio<1: raise ValueError("ROI pixel weight ratio must be >=1")
        frame_loss=loss_total_rdc_psnr_roi_per_pixel(is_yuv420=True,mse_y_weight=8,mse_yuv_420=True,mse_rgb_weight=0,
                   roi_bg_ratio=float(roi_pixel_weight_ratio),p_roi=None,segmentation_model=RequirePrecomputedMask())
        self.loss=CodecLoss(frame_loss=frame_loss,mse_lambda_exponent=1.,bpp_weight=1.)

    def forward(self,rd,target,mask,lambdas):
        if mask is None: raise ValueError("An explicit precomputed ROI mask is required")
        if mask.ndim==4 and mask.shape[1]==1: mask=mask[:,0]
        if mask.shape!=(target.shape[0],*target.shape[-2:]): raise ValueError("Mask must align with BxHxW target")
        mask=mask.to(device=target.device,dtype=target.dtype)
        if not torch.isfinite(mask).all() or mask.min()<0 or mask.max()>1: raise ValueError("ROI mask values must be finite in [0,1]")
        return self.loss(rd,target,mask=mask,lambdas=lambdas)
