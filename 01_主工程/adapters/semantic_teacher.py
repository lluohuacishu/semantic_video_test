"""Frozen legacy SUIM SegFormer-B0 with its exact letterbox preprocessing."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from common import ROOT, sha256

LEGACY = Path(r"C:\Users\lenovo\Desktop\works\work\semantic-video-training")
CHECKPOINT = LEGACY / "runs/suim_segformer_b0_20260915/segmentation_best.pt"
EXPECTED_SHA = "ff00cb439a4af290dc3efde8ae76d4acfad797784483b669ab6db79aaf5b1476"
LABELS = ("BW", "HD", "PF", "WR", "RO", "RI", "FV", "SR")
# Display colors differ from SUIM's RGB bit encoding; numeric class IDs do not.
DISPLAY_PALETTE = np.asarray([[25, 45, 65], [255, 110, 50], [75, 205, 100], [255, 205, 65],
                              [55, 210, 225], [180, 95, 230], [40, 240, 155], [195, 160, 120]], dtype=np.uint8)


class FrozenSegFormer(nn.Module):
    def __init__(self):
        super().__init__()
        config = SegformerConfig(num_labels=8, id2label=dict(enumerate(LABELS)), label2id={n:i for i,n in enumerate(LABELS)},
                                 hidden_sizes=[32, 64, 160, 256], depths=[2, 2, 2, 2], num_attention_heads=[1, 2, 5, 8],
                                 decoder_hidden_size=256, semantic_loss_ignore_index=255)
        self.network = SegformerForSemanticSegmentation(config)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):
        logits = self.network(pixel_values=(image-self.mean)/self.std).logits
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


def load_teacher(device="cuda"):
    if sha256(CHECKPOINT) != EXPECTED_SHA:
        raise ValueError("Legacy SegFormer checkpoint hash differs")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if state["model_name"] != "segformer_b0_suim_v1" or state["preprocessing"] != "rgb_0_1_long_edge_resize_center_pad_ignore_255":
        raise ValueError("Unsupported legacy teacher/preprocessing")
    if tuple(c["symbol"] for c in state["classes"]) != LABELS:
        raise ValueError("Class mapping differs")
    model = FrozenSegFormer()
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, {k:v for k,v in state.items() if k != "model"}


def fit_image(image, size=256, mask=None):
    width, height = image.size
    scale = size/max(width, height)
    new_w, new_h = max(1, round(width*scale)), max(1, round(height*scale))
    left, top = (size-new_w)//2, (size-new_h)//2
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(image.convert("RGB").resize((new_w, new_h), Image.Resampling.BILINEAR), (left, top))
    image_tensor = torch.from_numpy(np.asarray(canvas, dtype=np.uint8).copy()).permute(2, 0, 1).float()/255.0
    target = None
    if mask is not None:
        if mask.size != image.size or mask.mode != "L":
            raise ValueError("Label size/mode mismatch")
        target_canvas = Image.new("L", (size, size), 255)
        target_canvas.paste(mask.resize((new_w, new_h), Image.Resampling.NEAREST), (left, top))
        target = torch.from_numpy(np.asarray(target_canvas).copy())
    geometry = {"width": width, "height": height, "scaled_width": new_w, "scaled_height": new_h, "left": left, "top": top}
    return image_tensor, target, geometry


def predict(model, path, size=256, device="cuda"):
    image = Image.open(path).convert("RGB")
    tensor, _, g = fit_image(image, size)
    logits = model(tensor.unsqueeze(0).to(device))
    logits = logits[:, :, g["top"]:g["top"]+g["scaled_height"], g["left"]:g["left"]+g["scaled_width"]]
    logits = F.interpolate(logits, size=(g["height"], g["width"]), mode="bilinear", align_corners=False)
    probabilities = logits.softmax(1)
    confidence, labels = probabilities.max(1)
    return labels[0].cpu().numpy().astype(np.uint8), confidence[0].cpu().numpy(), g


def confusion(prediction, target, classes=8, valid=None):
    pred, truth = np.asarray(prediction), np.asarray(target)
    keep = (truth < classes) if valid is None else (truth < classes) & valid
    return np.bincount(classes*truth[keep].astype(np.int64)+pred[keep].astype(np.int64), minlength=classes**2).reshape(classes, classes)


def class_metrics(matrix, labels=LABELS):
    records = []
    for idx, name in enumerate(labels):
        tp = int(matrix[idx, idx])
        support, predicted = int(matrix[idx].sum()), int(matrix[:, idx].sum())
        union = support+predicted-tp
        records.append({"class_id": idx, "symbol": name, "iou": tp/union if union else None,
                        "f1": 2*tp/(support+predicted) if support+predicted else None,
                        "recall": tp/support if support else None, "precision": tp/predicted if predicted else None,
                        "target_pixels": support, "predicted_pixels": predicted})
    present = [r["iou"] for r in records if r["iou"] is not None]
    return {"mean_iou_present_union": float(np.mean(present)) if present else None,
            "pixel_agreement": int(np.trace(matrix))/int(matrix.sum()) if matrix.sum() else None,
            "per_class": records, "confusion_matrix": matrix.tolist()}


def overlay(path, labels):
    image = np.asarray(Image.open(path).convert("RGB"))
    colors = DISPLAY_PALETTE[labels]
    alpha = np.where(labels == 0, 0.0, 0.45)[..., None]
    return Image.fromarray((image*(1-alpha)+colors*alpha).round().astype(np.uint8))
