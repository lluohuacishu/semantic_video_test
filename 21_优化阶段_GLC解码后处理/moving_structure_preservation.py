"""Source-free local motion mask after robust dominant affine motion removal."""
import cv2
import numpy as np
from mild_reliability_gate import smooth_risk
from current_structure_preservation import preserve_current

SETTINGS=dict(base='CS50',grid_stride=8,minimum_fit_points=64,trim_passes=3,
              keep_percentile=70,minimum_inlier_radius=1.5,
              local_motion_thresholds=[.75,3.0],weight_blur_sigma=2.0)

def local_motion_weight(flow,alpha):
    h,w=alpha.shape
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    grid=(slice(4,h,8),slice(4,w,8))
    x=xx[grid].ravel();y=yy[grid].ravel();f=flow[grid].reshape(-1,2)
    good=(alpha[grid].ravel()>.5)&np.isfinite(f).all(axis=1)
    X=np.column_stack((x/w,y/h,np.ones_like(x)))[good];Y=f[good]
    if len(X)<SETTINGS['minimum_fit_points']:
        return np.zeros_like(alpha),dict(fit_points=len(X),inliers=0,fallback='insufficient_support',coefficients=None)
    keep=np.ones(len(X),bool);coef=None
    for _ in range(SETTINGS['trim_passes']):
        if keep.sum()<SETTINGS['minimum_fit_points'] or np.linalg.matrix_rank(X[keep])<3:break
        coef=np.linalg.lstsq(X[keep],Y[keep],rcond=None)[0]
        residual=np.linalg.norm(Y-X@coef,axis=1)
        threshold=max(SETTINGS['minimum_inlier_radius'],float(np.percentile(residual,SETTINGS['keep_percentile'])))
        keep=residual<=threshold
    if coef is None:
        return np.zeros_like(alpha),dict(fit_points=len(X),inliers=int(keep.sum()),fallback='rank_deficient',coefficients=None)
    expected=(xx[...,None]/w*coef[0]+yy[...,None]/h*coef[1]+coef[2])
    residual=np.linalg.norm(flow-expected,axis=2)
    weight=smooth_risk(residual,*SETTINGS['local_motion_thresholds']).astype(np.float32)
    weight=cv2.GaussianBlur(weight,(0,0),SETTINGS['weight_blur_sigma'])
    weight=np.where(alpha>0,np.clip(weight,0,1),0).astype(np.float32)
    return weight,dict(fit_points=len(X),inliers=int(keep.sum()),fallback=None,coefficients=coef.tolist())

def preserve_moving_structure(raw,q9,flow,alpha):
    weight,fit=local_motion_weight(flow,alpha)
    # Use unrounded half-strength structure correction, preserving the base operation.
    r=raw.astype(np.float32);q=q9.astype(np.float32);d=r-q
    low=cv2.GaussianBlur(d,(0,0),1.)
    bounded=np.minimum(np.maximum(low,np.minimum(d,0)),np.maximum(d,0))
    rgb=np.rint(q+.5*weight[...,None]*bounded).astype(np.uint8)
    return rgb,weight,fit

