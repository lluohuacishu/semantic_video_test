"""Record the existing target-specific teachers; no training or threshold search."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "01_主工程/semantic_mvp"))
    from teacher import Teacher, FISH_CONFIG
    from semantic_teacher import CHECKPOINT, LABELS
    import torch
    fixed = json.loads(FISH_CONFIG.read_text("utf-8"))
    targets = {}
    for target in ("fish", "diver"):
        teacher = Teacher(target)
        checkpoint = Path(fixed["checkpoint"]) if target == "fish" else CHECKPOINT
        targets[target] = {
            **teacher.profile, "checkpoint": str(checkpoint.resolve()),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256_verified": sha(checkpoint) == teacher.profile["model_sha256"],
            "loaded_strictly": True, "parameters_require_grad": any(p.requires_grad for p in teacher.model.parameters()),
            "mask_rule": "class6 probability >= 0.9 (no argmax requirement)" if target == "fish" else "argmax == class1 and winning probability >= 0.7",
        }
        if not targets[target]["checkpoint_sha256_verified"] or targets[target]["parameters_require_grad"]:
            raise ValueError("Teacher freeze check failed")
        del teacher
    result = {
        "status": "frozen_before_B2", "selection": "Reuse existing per-target Teacher profiles unchanged; no outcome-based selection",
        "targets": targets, "labels": list(LABELS),
        "normalization": {"rgb_range": [0, 1], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "roi": {"layout": "halves", "segment_frames": 32, "padding": 16, "min_pixels": 64, "connectivity": 8},
        "python": sys.version, "torch": torch.__version__, "device": torch.cuda.get_device_name(),
        "training_performed": False, "inference_or_quality_evaluation_performed": False,
        "limitations": ["Fish and diver use different checkpoints and preprocessing; report separately.",
                        "Known small-fish misses remain; thresholds were not lowered.", "No B2 semantic benefit has been measured."],
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print("Existing fish and diver profiles loaded and frozen", flush=True)


if __name__ == "__main__":
    main()
