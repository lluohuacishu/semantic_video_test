"""Fixed receiver-only, bounded attenuation inside the existing T3 support."""
from functools import lru_cache

import cv2
import numpy as np

from cached_t3 import cached_texture_gate


def smooth_risk(value, start, end):
    z = np.clip((value - start) / (end - start), 0, 1)
    return z * z * (3 - 2 * z)


def mild_gate(images, raw, mode):
    if mode not in ('R25', 'D25'):
        raise ValueError(mode)
    base = cached_texture_gate(images)

    @lru_cache(maxsize=4)
    def guide_blurs(frame):
        rgb = images[frame].astype(np.float32)
        return cv2.GaussianBlur(rgb, (0, 0), 3), cv2.GaussianBlur(rgb, (0, 0), 1)

    @lru_cache(maxsize=4)
    def raw_blur(frame):
        return cv2.GaussianBlur(raw[frame].astype(np.float32), (0, 0), 3)

    stats = dict(calls=0, active_pixels=0, attenuated_pixels=0, penalty_sum=0.,
                 raw_dominant_pixels=0, maximum_attenuation=0.25)

    def gate(values, current, reference, back, forward):
        alpha, mx, my = base(values, current, reference, back, forward)
        inverse = cv2.remap(forward, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        cycle = np.linalg.norm(back + inverse, axis=2)
        low, mid = guide_blurs(current)
        old_low, old_mid = guide_blurs(reference)
        dlow = np.max(np.abs(low - cv2.remap(old_low, mx, my, cv2.INTER_LINEAR)), axis=2)
        dmid = np.max(np.abs(mid - cv2.remap(old_mid, mx, my, cv2.INTER_LINEAR)), axis=2)
        gradient = np.maximum(np.linalg.norm(np.gradient(back, axis=0), axis=2),
                              np.linalg.norm(np.gradient(back, axis=1), axis=2))
        risk = np.maximum.reduce((smooth_risk(cycle, .75, 1.5), smooth_risk(dlow, 8, 16),
                                  smooth_risk(dmid, 12, 24), smooth_risk(gradient, .375, .75)))
        active = alpha > 0
        if mode == 'D25':
            error = np.max(np.abs(raw_blur(current) -
                                 cv2.remap(raw_blur(reference), mx, my, cv2.INTER_LINEAR)), axis=2)
            raw_risk = smooth_risk(error, 12, 32)
            stats['raw_dominant_pixels'] += int(((raw_risk > risk) & active).sum())
            risk = np.maximum(risk, raw_risk)
        # Keep the existing hard rejection and 2px feather. Confident pixels
        # retain full weight; uncertain but accepted pixels lose at most 25%.
        weight = alpha * (1 - .25 * risk)
        assert np.all(weight <= alpha + 1e-7) and np.all(weight >= .75 * alpha - 1e-7)
        assert np.array_equal(weight > 0, active)
        stats['calls'] += 1
        stats['active_pixels'] += int(active.sum())
        stats['attenuated_pixels'] += int(((risk > 0) & active).sum())
        stats['penalty_sum'] += float((.25 * risk)[active].sum())
        return weight, mx, my

    gate.diagnostics = stats
    return gate
