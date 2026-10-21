"""Small sender-only protection rule; takes predictions, never ground truth."""
from __future__ import annotations
import numpy as np
from scipy import ndimage


def seed_masks(packed, plan, config):
    fish_bit = plan['packed_bits'][f"fish_{config['fish_threshold']}"]
    diver_bit = plan['packed_bits'][f"diver_{config['diver_threshold']}"]
    return (packed & (1 << fish_bit)) != 0, (packed & (1 << diver_bit)) != 0


def hard_protection(current, previous, config):
    seed = current.copy()
    if config['previous_seed_frames'] and previous is not None:
        seed |= previous
    radius = config['margin_at_1920']
    radius = max(1, round(radius * max(seed.shape) / 1920)) if radius else 0
    if radius:
        seed = ndimage.maximum_filter(seed, size=2*radius+1, mode='constant', cval=0)
    return seed


def alpha_protection(current, hard):
    """Return uint8 alpha (255=preserve input); fallback is reported separately."""
    if not np.any(current):
        return np.full(current.shape, 255, dtype=np.uint8), 'empty_current_prediction'
    if int(hard.sum()) * 2 > hard.size:
        return np.full(current.shape, 255, dtype=np.uint8), 'protection_exceeds_half_frame'
    width = max(1, round(4 * max(hard.shape) / 1920))
    distance = ndimage.distance_transform_edt(~hard)
    alpha = np.clip(1.0 - distance / width, 0, 1)
    alpha[hard] = 1.0
    return np.rint(255*alpha).astype(np.uint8), None
