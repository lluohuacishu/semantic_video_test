"""Receiver-only exit from Q9 corrections contradicted by current decoded RGB."""
import cv2
import numpy as np
from mild_reliability_gate import smooth_risk

PROFILES={
    'E32':dict(sigma3=[12.,32.],sigma1=None,dilation=0),
    'EL12':dict(sigma3=[6.,12.],sigma1=[12.,24.],dilation=1),
}


def exit_frame(raw,q9,profile):
    """No semantics/source/ROI: blend the final Q9 correction back to current raw."""
    p=PROFILES[profile];r=raw.astype(np.float32);q=q9.astype(np.float32)
    def mismatch(sigma):
        return np.max(np.abs(cv2.GaussianBlur(r,(0,0),sigma)-cv2.GaussianBlur(q,(0,0),sigma)),axis=2)
    weight=smooth_risk(mismatch(3),*p['sigma3'])
    if p['sigma1'] is not None:weight*=smooth_risk(mismatch(1),*p['sigma1'])
    if p['dilation']:weight=cv2.dilate(weight,np.ones((3,3),np.uint8))
    # This added final blend is explicit; it does not change old two-stage rendering.
    rgb=np.rint(np.clip(q*(1-weight[...,None])+r*weight[...,None],0,255)).astype(np.uint8)
    return rgb,weight.astype(np.float32)
