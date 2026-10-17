"""Validate ported frozen teacher on the unchanged 110-image legacy GT split."""
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from common import ROOT, configure, sha256, save_json, torch, environment
from semantic_teacher import LEGACY, CHECKPOINT, load_teacher, fit_image, confusion, class_metrics

configure()
model, state = load_teacher()
manifest_file = LEGACY / "data/SUIM/manifest.json"
if sha256(manifest_file) != state["manifest_sha256"]: raise ValueError("SUIM manifest changed")
metadata = json.loads(manifest_file.read_text("utf-8"))
data_root = manifest_file.parent / metadata.get("root", ".")
matrix = np.zeros((8, 8), dtype=np.int64)
start = time.perf_counter()
with torch.inference_mode():
    for row in metadata["splits"]["test"]:
        tensor, target, _ = fit_image(Image.open(data_root / row["image"]).convert("RGB"), state["size"], Image.open(data_root / row["mask_index"]))
        prediction = model(tensor[None].cuda()).argmax(1)[0].cpu().numpy()
        matrix += confusion(prediction, target.numpy())
result = class_metrics(matrix)
expected = json.loads((LEGACY / "runs/suim_segformer_b0_20260915/evaluation_test/summary.json").read_text("utf-8"))
delta = result["mean_iou_present_union"] - expected["miou"]
if abs(delta) > 0.0005: raise RuntimeError(f"Teacher port changed mIoU by {delta}")
result.update({"images": len(metadata["splits"]["test"]), "ground_truth": True, "size": state["size"], "checkpoint": str(CHECKPOINT),
               "checkpoint_sha256": sha256(CHECKPOINT), "historical_miou": expected["miou"], "delta_miou": delta,
               "environment": environment(), "seconds": time.perf_counter()-start,
               "purpose": "Compatibility validation only; no training or selecting settings using this test split"})
directory = ROOT / "runs/phase_b_teacher_validation"
directory.mkdir(exist_ok=False)
save_json(directory / "result.json", result)
print(json.dumps({k:result[k] for k in ["images", "mean_iou_present_union", "historical_miou", "delta_miou", "seconds"]}, indent=2))
