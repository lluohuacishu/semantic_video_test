"""Current-frame structure correction with unchanged Q9 fine-detail reference."""
import cv2
import numpy as np

PROFILES = {
    'CS50': {'sigma': 1.0, 'structure_strength': .5},
    'CS100': {'sigma': 1.0, 'structure_strength': 1.0},
}

def preserve_current(raw, q9, profile):
    p = PROFILES[profile]
    r, q = raw.astype(np.float32), q9.astype(np.float32)
    residual = r-q
    smooth = cv2.GaussianBlur(residual, (0,0), p['sigma'])
    # Preserve zero corrections/rejections and avoid reversing pixel corrections.
    bounded = np.minimum(np.maximum(smooth, np.minimum(residual,0)), np.maximum(residual,0))
    correction = p['structure_strength']*bounded
    rgb = np.rint(np.clip(q+correction, 0, 255)).astype(np.uint8)
    return rgb

