"""Repair an isolated GLC texture jump when both aligned neighbors support it."""

import cv2
import numpy as np

from window_filter import FLOW_EDGE_LIMIT, KERNEL
from temporal_filter import correspondence, lowpass, motion, warp

SETTINGS = dict(agreement_sigma=2.0, outlier_margin=2.0, outlier_ramp=10.0,
                coarse_sigma=3.0, coarse_reject=24.0, coarse_ramp=16.0,
                coarse_keep=0.65, correction_limit=36.0, feather_pixels=8,
                cut_threshold=35.0, fb_limit=1.5, strength=0.9)


def feather(box, shape):
    x0, y0, x1, y1 = box
    height, width = shape[:2]
    y, x = np.ogrid[:height, :width]
    dx = np.minimum(x - x0, x1 - 1 - x)
    dy = np.minimum(y - y0, y1 - 1 - y)
    return np.clip(np.minimum(dx, dy) / SETTINGS['feather_pixels'], 0, 1).astype(np.float32)


def aligned_rgb(frame, pair):
    backward, forward = pair
    x, y, inside, error = correspondence(backward, forward)
    gradient = cv2.morphologyEx(backward, cv2.MORPH_GRADIENT, KERNEL)
    safe = inside & (error < SETTINGS['fb_limit']) & (np.linalg.norm(gradient, axis=2) <= FLOW_EDGE_LIMIT)
    safe = cv2.erode(safe.astype(np.uint8), KERNEL)
    confidence = safe * np.clip(1 - error / SETTINGS['fb_limit'], 0, 1)
    return warp(frame.astype(np.float32), x, y), confidence


def gap(left, right):
    difference = np.mean(np.abs(left - right), axis=2)
    return cv2.GaussianBlur(difference, (0, 0), SETTINGS['agreement_sigma'])


def repair_frame(previous, current, following, before, after, region):
    current_float = current.astype(np.float32)
    current_low = lowpass(current)
    if any(np.abs(lowpass(frame) - current_low).mean() > SETTINGS['cut_threshold']
           for frame in (previous, following)):
        return current.copy()
    past, past_confidence = aligned_rgb(previous, before)
    future, future_confidence = aligned_rgb(following, after[::-1])
    # The middle frame must disagree with both neighbors more than they
    # disagree with each other. This avoids bridging a persistent change.
    neighbor_gap = gap(past, future)
    center_gap = np.minimum(gap(current_float, past), gap(current_float, future))
    outlier = np.clip((center_gap - neighbor_gap - SETTINGS['outlier_margin']) /
                      SETTINGS['outlier_ramp'], 0, 1)
    change = (past + future) * 0.5 - current_float
    coarse = lowpass(change, SETTINGS['coarse_sigma'])
    coarse_gap = np.mean(np.abs(coarse), axis=2)
    geometry_guard = np.clip((SETTINGS['coarse_reject'] - coarse_gap) / SETTINGS['coarse_ramp'], 0, 1)
    confidence = np.minimum(past_confidence, future_confidence)
    weight = SETTINGS['strength'] * region * confidence * outlier * geometry_guard
    correction = np.clip(change - SETTINGS['coarse_keep'] * coarse,
                         -SETTINGS['correction_limit'], SETTINGS['correction_limit'])
    return np.clip(np.rint(current_float + weight[..., None] * correction), 0, 255).astype(np.uint8)


def repair_clip(raw_with_context, box):
    region = feather(box, raw_with_context.shape[1:])
    output = np.empty_like(raw_with_context[1:-1])
    before = motion(raw_with_context[0], raw_with_context[1])
    for i in range(1, len(raw_with_context) - 1):
        after = motion(raw_with_context[i], raw_with_context[i + 1])
        output[i - 1] = repair_frame(raw_with_context[i - 1], raw_with_context[i],
                                     raw_with_context[i + 1], before, after, region)
        before = after
    return output
