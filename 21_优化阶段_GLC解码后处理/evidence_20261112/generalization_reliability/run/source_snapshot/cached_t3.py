"""Exact T3 computation cache; no change to reliability or RGB blend rules."""

from functools import lru_cache

import cv2
import numpy as np

from receive_guide_texture import texture_gate


def cached_texture_gate(images):
    """Build once per sequence; charge construction to render time."""
    # Current/new-anchor/old-anchor need at most three live frame identities.
    # A four-entry LRU avoids scaling this extra RGB cache with video length.
    @lru_cache(maxsize=4)
    def blurred(frame):
        return (cv2.GaussianBlur(images[frame].astype(np.float32), (0, 0), 3),
                cv2.GaussianBlur(images[frame].astype(np.float32), (0, 0), 1))
    height, width = images.shape[1:3]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)

    def reliable_cached(values, current, reference, back, forward):
        mx, my = xx + back[..., 0], yy + back[..., 1]
        inside = (mx >= 2) & (mx < width - 2) & (my >= 2) & (my < height - 2)
        inverse = cv2.remap(forward, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        cycle = np.linalg.norm(back + inverse, axis=2)
        low, mid = blurred(current)
        old_low, old_mid = blurred(reference)
        dlow = np.max(np.abs(low - cv2.remap(old_low, mx, my, cv2.INTER_LINEAR)), axis=2)
        dmid = np.max(np.abs(mid - cv2.remap(old_mid, mx, my, cv2.INTER_LINEAR)), axis=2)
        grad = np.maximum(np.linalg.norm(np.gradient(back, axis=0), axis=2),
                          np.linalg.norm(np.gradient(back, axis=1), axis=2))
        safe = inside & (cycle < 1.5) & (dlow < 16) & (dmid < 24) & (grad < .75)
        safe = cv2.erode(safe.astype(np.uint8), kernel)
        alpha = np.minimum(cv2.distanceTransform(safe, cv2.DIST_L2, 5) / 2., 1.)
        return alpha, mx, my

    return texture_gate(images, base_gate=reliable_cached)
