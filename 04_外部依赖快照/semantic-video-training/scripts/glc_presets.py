"""Selectable B4 presets; legacy encoding defaults remain unchanged without one."""
from __future__ import annotations

import json
from pathlib import Path

CONFIG = json.loads(Path(__file__).with_name("glc_presets.json").read_text(encoding="utf-8"))
PRESETS = CONFIG["presets"]
LEGACY_DEFAULTS = {
    "frames": 50, "width": 1280, "frame_stride": 1, "container_version": 1,
    "q": 63, "intra_period": 32, "reset_interval": 32,
}


def apply_encode_preset(args):
    """Resolve settings before encoding; explicit frame counts control clip length."""
    name = getattr(args, "preset", None)
    if name is not None and name not in PRESETS:
        raise ValueError(f"Unknown GLC preset: {name}")
    selected = ({**CONFIG["common"], **PRESETS[name]} if name else LEGACY_DEFAULTS)
    for field in LEGACY_DEFAULTS:
        supplied = getattr(args, field, None)
        expected = selected[field]
        if name and field != "frames" and supplied is not None and supplied != expected:
            raise ValueError(f"--{field.replace('_', '-')} conflicts with preset {name}: expected {expected}.")
        setattr(args, field, expected if supplied is None else supplied)
    return args
