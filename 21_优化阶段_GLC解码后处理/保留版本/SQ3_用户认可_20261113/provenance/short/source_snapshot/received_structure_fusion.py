"""Experimental use of already transmitted guide RGB as current-frame evidence."""
import cv2
import numpy as np

PROFILES={
    'GB25':dict(kind='rgb_blend',strength=.25,sigma=None),
    'GS25':dict(kind='low_frequency_residual',strength=.25,sigma=1.0),
    'GS50':dict(kind='low_frequency_residual',strength=.50,sigma=1.0),
}

def fuse_received_structure(q9,guide,profile):
    p=PROFILES[profile];q=q9.astype(np.float32);g=guide.astype(np.float32)
    difference=g-q
    if p['sigma'] is not None:
        difference=cv2.GaussianBlur(difference,(0,0),p['sigma'])
    unbounded=q+p['strength']*difference
    rgb=np.rint(np.clip(unbounded,0,255)).astype(np.uint8)
    return rgb,dict(clipped_channel_fraction=float(((unbounded<0)|(unbounded>255)).mean()),
                    mean_abs_change=float(np.abs(rgb.astype(np.float32)-q).mean()))

