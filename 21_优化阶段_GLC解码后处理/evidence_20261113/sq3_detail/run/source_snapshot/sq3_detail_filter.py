"""Experimental, bounded luminance sharpening of the accepted SQ3 output.

No source, raw-frame residual, future frame, or recursive temporal state.
This enhances existing contrast; it cannot recover missing true detail.
"""
import cv2
import numpy as np

SETTINGS=dict(sigma=1.0,maximum_correction_levels=3.0,
    luminance_weights=[.299,.587,.114],border='reflect101',
    profiles=dict(SQ3_U15=.15,SQ3_U30=.30),added_history=0,
    source_used=False,added_transmission_bytes=0)

def enhance(rgb,gain):
    x=rgb.astype(np.float32)
    y=x[...,0]*.299+x[...,1]*.587+x[...,2]*.114
    low=cv2.GaussianBlur(y,(0,0),SETTINGS['sigma'],borderType=cv2.BORDER_REFLECT101)
    correction=np.clip(gain*(y-low),-SETTINGS['maximum_correction_levels'],SETTINGS['maximum_correction_levels'])
    # A common offset preserves RGB differences before rounding. Limit it by
    # all three channels together to avoid independent-channel saturation.
    correction=np.maximum(-x.min(axis=2),np.minimum(255-x.max(axis=2),correction))
    # Quantize the shared offset once: rounding each RGB sum separately can
    # change chroma at half-integer ties because channel parities differ.
    return (x+np.rint(correction)[...,None]).astype(np.uint8)
