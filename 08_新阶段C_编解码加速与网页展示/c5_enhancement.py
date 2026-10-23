"""Frozen D2 sender preprocessing, shared by the web adapter and its worker.

Importing this module starts no model. Masks never enter the transmitted stream.
"""
import hashlib
import json
import os
from pathlib import Path

from c4_config import REPO, discover_workspace

MODES = {
    "ordinary": "普通 GLC",
    "lowpass": "全图低通 · σ 1.2",
    "light_lowpass": "全图轻度低通 · σ 0.6",
    "semantic_original": "语义保护 · 目标原像素",
    "semantic_light": "语义保护 · 目标轻度低通",
}
SEMANTIC = {"semantic_original", "semantic_light"}


def mode_option(value):
    if not isinstance(value, str) or value not in MODES:
        raise ValueError("请选择有效的低通 / 语义保护模式")
    return value


def runtime_paths():
    workspace = discover_workspace(REPO)
    links = workspace / "01_项目/运行环境"
    python = links / ("语义模型_Python环境/Scripts/python.exe" if os.name == "nt" else "语义模型_Python环境/bin/python")
    checkpoint = links / "旧训练工程/runs/suim_segformer_b0_20260915/segmentation_best.pt"
    return {"python": Path(os.environ.get("GLC_SEMANTIC_PYTHON", str(python))).resolve(),
            "checkpoint": Path(os.environ.get("GLC_SEMANTIC_CHECKPOINT", str(checkpoint))).resolve()}


def availability():
    paths = runtime_paths()
    return {"modes": MODES, "preprocess_available": paths["python"].is_file(),
            "semantic_available": all(p.is_file() for p in paths.values()),
            "policy": "D2 frozen prediction; sender-only; no masks transmitted; q=0"}


def transform(rgb, mode, alpha=None, fallback=None):
    """Same float32 Gaussian, reflect boundary and rounding as D2/D2.2."""
    import numpy as np
    from scipy.ndimage import gaussian_filter
    mode_option(mode)
    if mode == "ordinary" or (mode in SEMANTIC and fallback):
        return rgb.copy()
    def blur(sigma):
        return np.rint(gaussian_filter(rgb.astype(np.float32), sigma=(sigma, sigma, 0),
                       mode="reflect", truncate=4)).clip(0, 255).astype(np.uint8)
    if mode in ("lowpass", "light_lowpass"):
        return blur(1.2 if mode == "lowpass" else .6)
    if alpha is None or alpha.shape != rgb.shape[:2] or alpha.dtype != np.uint8:
        raise ValueError("Semantic mode requires an aligned uint8 prediction alpha")
    target = rgb if mode == "semantic_original" else blur(.6)
    weight = alpha.astype(np.float32)[..., None] / 255
    return np.rint(weight * target + (1 - weight) * blur(1.2)).clip(0, 255).astype(np.uint8)


class PreparedFrames:
    """Verify each processed frame against the original reference before coding."""
    def __init__(self, manifest, frames, width, height):
        self.root = Path(manifest).parent
        data = json.loads(Path(manifest).read_text("utf-8"))
        self.summary = data["summary"]
        self.rows = data["records"]
        if (self.summary["frames"] != frames or len(self.rows) != frames
                or self.summary["size"] != [width, height] or self.summary["state"] != "completed"):
            raise ValueError("Prepared frame count / size differs from encoder input")

    def frame(self, index, original):
        import numpy as np
        from PIL import Image
        row = self.rows[index]
        if row["index"] != index or hashlib.sha256(original.tobytes()).hexdigest() != row["source_rgb_sha256"]:
            raise ValueError("Preprocessing source frame alignment mismatch")
        # Paths are generated here, never taken from an HTTP request or manifest.
        with Image.open(self.root / "frames" / f"{index:05d}.png") as image:
            rgb = np.asarray(image.convert("RGB"))
        if rgb.shape != original.shape or hashlib.sha256(rgb.tobytes()).hexdigest() != row["coding_rgb_sha256"]:
            raise ValueError("Prepared coding frame checksum / shape mismatch")
        return rgb
