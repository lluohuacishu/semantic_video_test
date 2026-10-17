"""在固定 SUIM 划分上训练联合 RGB / 语义码流；真值只用于损失。"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from prepare_image_codec import SOURCE_REVISION, verify_weight, MODEL_DIR
from segmentation import (IGNORE_INDEX, NUM_CLASSES, SegmentationDataset, confusion_matrix,
                          load_segmentation_model, manifest_info, metrics_from_confusion,
                          segmentation_loss, write_json)
from semantic_image_codec import condition_image, pretrained_joint_model
from image_codec import SEMANTIC_ARCHITECTURE
from segment_video import choose_device

ROOT = Path(__file__).resolve().parents[1]
TEACHER = ROOT / "runs/suim_segformer_b0_20260915/segmentation_best.pt"


def subset(records, limit):
    if not limit or limit >= len(records):
        return records
    return [records[int(i)] for i in np.linspace(0, len(records) - 1, limit).round()]


class ConditionedDataset(Dataset):
    def __init__(self, dataset, guidance, augment=False):
        self.dataset, self.guidance, self.augment = dataset, guidance, augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        guidance = self.guidance[index].long()
        if self.augment and torch.rand(()).item() < 0.5:
            image, label, guidance = image.flip(-1), label.flip(-1), guidance.flip(-1)
        return image, label, guidance


def predict_guidance(teacher, data, device, batch_size):
    cached, matrix = [], torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    with torch.inference_mode():
        for images, labels in DataLoader(data, batch_size=batch_size, shuffle=False):
            prediction = teacher(images.to(device)).argmax(1).cpu()
            matrix += confusion_matrix(prediction, labels)
            prediction[labels == IGNORE_INDEX] = 0
            cached.extend(prediction.byte().unbind(0))
    return cached, matrix


def objective(result, image, target, guidance, weights, args):
    valid = target != IGNORE_INDEX
    mse = ((result["x_hat"] - image).square() * valid.unsqueeze(1)).sum() / (3 * valid.sum())
    rate = sum(-torch.log2(p.clamp_min(1e-9)).sum() for p in result["likelihoods"].values()) / valid.sum()
    semantic = segmentation_loss(result["semantic_logits"], target, weights)
    # 发送端预测是编码条件；这一项是辅助蒸馏，人工真值损失仍是主监督。
    prediction_target = guidance.masked_fill(~valid, IGNORE_INDEX)
    guidance_loss = F.cross_entropy(result["semantic_logits"], prediction_target, ignore_index=IGNORE_INDEX)
    total = rate + args.image_lambda * 255**2 * mse + args.semantic_weight * semantic + args.guidance_weight * guidance_loss
    return {"total": total, "rate_bpp_estimate": rate, "mse": mse, "semantic": semantic,
            "guidance": guidance_loss}


def gradient_audit(model, batch, weights, device):
    """只反传语义损失，排除像素损失引起编码器梯度的可能。"""
    image, target, guidance = (t.to(device) for t in batch)
    model.zero_grad(set_to_none=True)
    result = model(condition_image(image, guidance))
    segmentation_loss(result["semantic_logits"], target, weights).backward()
    gradient = model.g_a[0].weight.grad
    audit = {"encoder_rgb_gradient_l1": float(gradient[:, :3].abs().sum()),
             "encoder_guidance_gradient_l1": float(gradient[:, 3:].abs().sum()),
             "semantic_head_gradient_l1": float(model.semantic_head[-1].weight.grad.abs().sum()),
             "semantic_loss_reaches_sender_encoder": bool(torch.isfinite(gradient).all() and gradient.abs().sum() > 0),
             "loss_used": "ground_truth_semantic_loss_only"}
    model.zero_grad(set_to_none=True)
    if not audit["semantic_loss_reaches_sender_encoder"]:
        raise RuntimeError("语义损失没有反传至编码器")
    return audit


def evaluate(model, loader, weights, classes, device, args):
    totals = {key: 0.0 for key in ("total", "rate_bpp_estimate", "mse", "semantic", "guidance")}
    pixels = 0
    matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            image, target, guidance = (t.to(device) for t in batch)
            result = model(condition_image(image, guidance))
            losses = objective(result, image, target, guidance, weights, args)
            count = int((target != IGNORE_INDEX).sum())
            pixels += count
            for key in totals:
                totals[key] += float(losses[key]) * count
            matrix += confusion_matrix(result["semantic_logits"].argmax(1), target).cpu()
    average = {key: value / pixels for key, value in totals.items()}
    average["psnr_db_valid_pixels"] = -10 * math.log10(max(average["mse"], 1e-12))
    return {**average, "direct_semantics": metrics_from_confusion(matrix.numpy(), classes),
            "confusion_matrix": matrix.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/SUIM/manifest.json")
    parser.add_argument("--teacher", type=Path, default=TEACHER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-validation", type=int, default=0)
    parser.add_argument("--image-lambda", type=float, default=0.0067)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--guidance-weight", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if (args.epochs <= args.warmup_epochs or args.warmup_epochs < 0 or args.size < 64 or args.size % 64
            or args.batch_size < 1 or args.limit_train < 0 or args.limit_validation < 0
            or any(not math.isfinite(v) or v <= 0 for v in (args.image_lambda, args.semantic_weight,
                                                          args.learning_rate, args.head_learning_rate))
            or not math.isfinite(args.guidance_weight) or args.guidance_weight < 0):
        parser.error("需要至少一轮联合训练、64 的整数倍尺寸、正的批量与损失/学习率参数")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    root, metadata, manifest_sha = manifest_info(args.manifest)
    train_records = subset(metadata["splits"]["train"], args.limit_train)
    valid_records = subset(metadata["splits"]["validation"], args.limit_validation)
    print(f"{device}: train={len(train_records)}, validation={len(valid_records)}, size={args.size}", flush=True)
    train_data = SegmentationDataset(root, train_records, args.size, cache=True)
    valid_data = SegmentationDataset(root, valid_records, args.size, cache=True)
    histogram = train_data.class_histogram().float()
    weights = (histogram.sum() / histogram.clamp_min(1)).sqrt()
    weights = (weights / weights.mean()).clamp(0.25, 4).to(device)
    teacher, teacher_state = load_segmentation_model(args.teacher, device)
    if teacher_state["manifest_sha256"] != manifest_sha or teacher_state["size"] != args.size:
        raise ValueError("教师模型的固定划分或输入尺寸与这次实验不一致")
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    print("缓存发送端预测（不将真值送入编码器）…", flush=True)
    train_guidance, _ = predict_guidance(teacher, train_data, device, max(2, args.batch_size))
    valid_guidance, teacher_matrix = predict_guidance(teacher, valid_data, device, max(2, args.batch_size))
    teacher_metrics = metrics_from_confusion(teacher_matrix.numpy(), metadata["classes"])
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    train_loader = DataLoader(ConditionedDataset(train_data, train_guidance, augment=True),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(ConditionedDataset(valid_data, valid_guidance),
                              batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = pretrained_joint_model().to(device)
    main_parameters, head_parameters, aux_parameters = [], [], []
    for name, parameter in model.named_parameters():
        (aux_parameters if name.endswith(".quantiles") else
         head_parameters if name.startswith("semantic_head.") else main_parameters).append(parameter)
    optimizer = torch.optim.Adam([{"params": main_parameters, "lr": args.learning_rate},
                                  {"params": head_parameters, "lr": args.head_learning_rate}])
    aux_optimizer = torch.optim.Adam(aux_parameters, lr=1e-3)
    audit = gradient_audit(model, next(iter(train_loader)), weights, device)
    _, base_sha = verify_weight(MODEL_DIR, 3)
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "model_name": SEMANTIC_ARCHITECTURE, "source_revision": SOURCE_REVISION,
              "manifest_sha256": manifest_sha, "sender_teacher_sha256": hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
              "base_codec_sha256": base_sha, "num_classes": NUM_CLASSES, "quality": 3,
              "classes": metadata["classes"], "class_weights": weights.cpu().tolist(),
              "train_files": [r["image"] for r in train_records], "validation_files": [r["image"] for r in valid_records],
              "encoder_guidance": "frozen SegFormer predictions; never ground truth",
              "loss": "estimated_bpp + image_lambda*255^2*MSE + semantic_weight*(0.7 weighted CE+0.3 Dice) + guidance_weight*teacher CE",
              "validation_protocol": "256 long-edge center pad, ignore 255; likelihood rate estimate excludes file header",
              "model_selection": "lowest joint validation objective; test split never used",
              "teacher_validation": teacher_metrics}
    write_json(args.output / "config.json", config)
    write_json(args.output / "semantic_gradient_audit.json", audit)
    history, best = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        warmup = epoch <= args.warmup_epochs
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(not warmup or name.startswith("semantic_head."))
        model.train()
        average, batches = 0.0, 0
        for batch_index, batch in enumerate(train_loader, 1):
            image, target, guidance = (t.to(device) for t in batch)
            optimizer.zero_grad(set_to_none=True)
            result = model(condition_image(image, guidance))
            losses = objective(result, image, target, guidance, weights, args)
            loss = (args.semantic_weight * losses["semantic"] + args.guidance_weight * losses["guidance"]
                    if warmup else losses["total"])
            if not torch.isfinite(loss):
                raise RuntimeError("训练损失非有限，停止并保留已有轮次")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(main_parameters + head_parameters, 1.0)
            optimizer.step()
            if not warmup:
                aux_optimizer.zero_grad(set_to_none=True)
                aux_loss = model.aux_loss()
                aux_loss.backward()
                aux_optimizer.step()
            average += float(loss.detach())
            batches += 1
            if batch_index == 1 or batch_index % 100 == 0:
                print(f"epoch {epoch}/{args.epochs} {'head' if warmup else 'joint'} {batch_index}/{len(train_loader)} loss={float(loss.detach()):.4f}", flush=True)
        validation = evaluate(model, valid_loader, weights, metadata["classes"], device, args)
        row = {"epoch": epoch, "warmup_only": warmup, "train_loss": average / batches,
               "epoch_seconds": time.perf_counter() - epoch_start, "validation": validation}
        history.append(row)
        write_json(args.output / "history.json", {"epochs": history})
        state = {"model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                 "model_name": SEMANTIC_ARCHITECTURE, "quality": 3, "num_classes": NUM_CLASSES,
                 "source_revision": SOURCE_REVISION, "classes": metadata["classes"],
                 "sender_teacher_sha256": config["sender_teacher_sha256"], "manifest_sha256": manifest_sha,
                 "size": args.size, "epoch": epoch, "joint_training_completed": not warmup,
                 "validation": validation, "training_config": config}
        pending = args.output / "last.tmp"
        torch.save(state, pending)
        pending.replace(args.output / "semantic_codec_last.pt")
        if not warmup and validation["total"] < best:
            best = validation["total"]
            pending = args.output / "best.tmp"
            torch.save(state, pending)
            pending.replace(args.output / "semantic_codec_best.pt")
        print(f"epoch {epoch}: direct val mIoU={validation['direct_semantics']['miou']*100:.2f}%, "
              f"PSNR={validation['psnr_db_valid_pixels']:.2f}, est bpp={validation['rate_bpp_estimate']:.3f}, "
              f"objective={validation['total']:.3f}, {row['epoch_seconds']:.1f}s", flush=True)
    checkpoint = args.output / "semantic_codec_best.pt"
    best_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    summary = {"best_checkpoint": str(checkpoint.resolve()), "best_epoch": best_state["epoch"],
               "validation": best_state["validation"], "teacher_validation": teacher_metrics,
               "train_images": len(train_records), "validation_images": len(valid_records),
               "test_images_used": 0, "epochs_completed": args.epochs,
               "elapsed_seconds": time.perf_counter() - started, "semantic_gradient_audit": audit,
               "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None,
               "note": "首轮联合语义源编码实验；前向量化指标需再用真实 CPU 熵码流验证。不是论文复现或完整水声信道系统。"}
    write_json(args.output / "summary.json", summary)
    print(f"completed: {checkpoint.resolve()}", flush=True)


if __name__ == "__main__":
    main()
