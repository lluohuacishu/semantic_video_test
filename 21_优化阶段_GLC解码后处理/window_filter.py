"""Motion-aligned GLC RGB filtering with a centered temporal window."""

from pathlib import Path
import sys

import cv2
import numpy as np

STAGE11 = Path(__file__).resolve().parents[1] / '11_优化阶段_纹理稳定性'
sys.path.insert(0, str(STAGE11))
from temporal_filter import Settings, correspondence, lowpass, warp

KERNEL = np.ones((3, 3), np.uint8)
FLOW_EDGE_LIMIT = 1.5
FAR_PAIR_WEIGHT = 0.5


def aligned_detail(neighbor, current_low, flow, settings):
    neighbor_low = lowpass(neighbor, settings.sigma)
    if float(np.mean(np.abs(current_low - neighbor_low))) > settings.cut_difference:
        return None
    backward, forward = flow
    x, y, inside, error = correspondence(backward, forward)
    photo = np.mean(np.abs(current_low - warp(neighbor_low, x, y)), axis=2)
    gradient = cv2.morphologyEx(backward, cv2.MORPH_GRADIENT, KERNEL)
    edge = np.linalg.norm(gradient, axis=2) > FLOW_EDGE_LIMIT
    safe = inside & (error < settings.fb_limit) & (photo < settings.photo_limit) & ~edge
    safe = cv2.erode(safe.astype(np.uint8), KERNEL)
    confidence = safe * np.clip(1 - error / settings.fb_limit, 0, 1)
    confidence *= np.clip(1 - photo / settings.photo_limit, 0, 1)
    return warp(neighbor.astype(np.float32) - neighbor_low, x, y), confidence


def blend_center(previous, current, following, before, after, settings=Settings(), *,
                 fusion='consensus', distant=None):
    """Use only raw decoded frames. Flows are bidirectional (later to earlier first)."""
    if previous is None or following is None:
        return current.copy(), 0.0
    current_low = lowpass(current, settings.sigma)
    past = aligned_detail(previous, current_low, before, settings)
    future = aligned_detail(following, current_low, after[::-1], settings)
    if past is None or future is None:
        return current.copy(), 0.0

    current_high = current.astype(np.float32) - current_low
    past_change = past[0] - current_high
    future_change = future[0] - current_high
    if fusion == 'mean':
        detail = (past_change + future_change) * 0.5
    else:
        # Require agreement and use the smaller proposed correction.
        agreed = (past_change * future_change) > 0
        magnitude = np.minimum(np.abs(past_change), np.abs(future_change))
        detail = np.where(agreed, np.sign(past_change) * magnitude, 0)
    weight = settings.strength * np.minimum(past[1], future[1])
    if distant is not None:
        past2, future2, before2, after2 = distant
        far_past = aligned_detail(past2, current_low, before2, settings)
        far_future = aligned_detail(future2, current_low, after2[::-1], settings)
        if far_past is not None and far_future is not None:
            far_weight = FAR_PAIR_WEIGHT * np.minimum(far_past[1], far_future[1])[..., None]
            far_detail = (far_past[0] + far_future[0]) * 0.5 - current_high
            # Keep the same current-frame anchor as the three-frame filter.
            detail = (detail + far_weight * far_detail) / (1 + far_weight)
    change = weight[..., None] * detail
    change = np.clip(change, -settings.max_change, settings.max_change)
    output = np.clip(np.rint(current + change), 0, 255).astype(np.uint8)
    return output, float(np.mean(weight > 0))
