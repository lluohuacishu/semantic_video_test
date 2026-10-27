"""Causal, two-frame high-frequency stabilization after GLC decoding."""
from dataclasses import asdict, dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class Settings:
    strength: float = 0.45
    sigma: float = 1.2
    flow_scale: float = 0.5
    fb_limit: float = 1.5
    photo_limit: float = 12.0
    max_change: float = 10.0
    cut_difference: float = 35.0


def lowpass(rgb, sigma=1.2):
    return cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), sigma,
                            borderType=cv2.BORDER_REFLECT101)


def gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def motion(previous, current, scale=0.5):
    h, w = current.shape[:2]
    size = (max(8, round(w * scale)), max(8, round(h * scale)))
    a = cv2.resize(gray(previous), size, interpolation=cv2.INTER_AREA)
    b = cv2.resize(gray(current), size, interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(a, (5, 5), 1)
    b = cv2.GaussianBlur(b, (5, 5), 1)
    def flow(x, y):
        small = cv2.calcOpticalFlowFarneback(x, y, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        full = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        full *= np.array([w / size[0], h / size[1]], np.float32)
        return full
    return flow(b, a), flow(a, b)


def correspondence(backward, forward):
    h, w = backward.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    mx, my = xx + backward[..., 0], yy + backward[..., 1]
    valid = (mx >= 2) & (mx < w - 2) & (my >= 2) & (my < h - 2)
    inverse = warp(forward, mx, my)
    error = np.linalg.norm(backward + inverse, axis=2)
    return mx, my, valid, error


def warp(value, mx, my):
    return cv2.remap(value, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


class TemporalFilter:
    def __init__(self, settings=Settings()):
        self.settings = settings
        self.previous = None

    def reset(self):
        self.previous = None

    def apply(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Expected uint8 RGB')
        previous = self.previous
        self.previous = rgb.copy()
        if previous is None or previous.shape != rgb.shape:
            return rgb.copy(), dict(reset=True, mean_weight=0.0, active_fraction=0.0)
        s = self.settings
        cur_low, prev_low = lowpass(rgb, s.sigma), lowpass(previous, s.sigma)
        if np.mean(np.abs(cur_low - prev_low)) > s.cut_difference:
            return rgb.copy(), dict(reset=True, mean_weight=0.0, active_fraction=0.0)
        backward, forward = motion(previous, rgb, s.flow_scale)
        mx, my, valid, fb = correspondence(backward, forward)
        photo = np.mean(np.abs(cur_low - warp(prev_low, mx, my)), axis=2)
        safe = (valid & (fb < s.fb_limit) & (photo < s.photo_limit)).astype(np.uint8)
        safe = cv2.erode(safe, np.ones((3, 3), np.uint8))
        weight = s.strength * safe * np.clip(1 - fb / s.fb_limit, 0, 1)
        weight *= np.clip(1 - photo / s.photo_limit, 0, 1)
        cur_high = rgb.astype(np.float32) - cur_low
        prev_high = previous.astype(np.float32) - prev_low
        change = weight[..., None] * (warp(prev_high, mx, my) - cur_high)
        change = np.clip(change, -s.max_change, s.max_change)
        output = np.clip(np.rint(rgb.astype(np.float32) + change), 0, 255).astype(np.uint8)
        return output, dict(reset=False, mean_weight=float(weight.mean()),
            active_fraction=float(np.mean(weight > 0.05)),
            mean_abs_change=float(np.abs(output.astype(np.float32) - rgb).mean()))


def description():
    return dict(settings=asdict(Settings()), history='one unfiltered decoded previous frame',
        future_frames=0, source_required=False, codec_reference_modified=False,
        precision='OpenCV CPU float32, uint8 RGB output',
        scope='postprocessing only; no additional transmitted bytes; no FEC')
