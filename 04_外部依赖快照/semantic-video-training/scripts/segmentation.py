"""SUIM 分割的数据读取、轻量 U-Net 和指标计算。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from dataset_manifest import load_manifest

NUM_CLASSES = 8
IGNORE_INDEX = 255
MODEL_NAME = "tiny_unet_v1"
TRANSFER_MODEL_NAME = "lraspp_mobilenet_v3_large_suim_v1"
HR_TRANSFER_MODEL_NAME = "lraspp_mobilenet_v3_large_hr4_suim_v1"
FPN_TRANSFER_MODEL_NAME = "lraspp_mobilenet_v3_large_fpn_suim_v1"
SEGFORMER_MODEL_NAME = "segformer_b0_suim_v1"
TRANSFER_MODEL_NAMES = (TRANSFER_MODEL_NAME, HR_TRANSFER_MODEL_NAME, FPN_TRANSFER_MODEL_NAME)
PREPROCESSING = "rgb_0_1_long_edge_resize_center_pad_ignore_255"


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def manifest_info(path: Path):
    root, metadata, _ = load_manifest(path)
    if [row["index"] for row in metadata["classes"]] != list(range(NUM_CLASSES)):
        raise ValueError("当前模型要求 SUIM 的 8 类标签表，编号为 0～7")
    return root, metadata, hashlib.sha256(path.read_bytes()).hexdigest()


def fit_image(image: Image.Image, size: int, mask: Image.Image | None = None):
    width, height = image.size
    scale = size / max(width, height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (size - new_w) // 2, (size - new_h) // 2
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(image.convert("RGB").resize((new_w, new_h), Image.Resampling.BILINEAR), (left, top))
    label = None
    if mask is not None:
        if mask.mode != "L" or mask.size != image.size:
            raise ValueError("索引标签必须为单通道 L 模式，且尺寸与图片一致")
        if np.asarray(mask).max() >= NUM_CLASSES:
            raise ValueError("源索引标签只允许 0～7；255 仅用于本脚本新增的补边")
        label_canvas = Image.new("L", (size, size), IGNORE_INDEX)
        label_canvas.paste(mask.resize((new_w, new_h), Image.Resampling.NEAREST), (left, top))
        label = torch.from_numpy(np.array(label_canvas))
    geometry = {"width": width, "height": height, "scaled_width": new_w, "scaled_height": new_h,
                "left": left, "top": top}
    return TF.pil_to_tensor(canvas), label, geometry


class SegmentationDataset(Dataset):
    def __init__(self, root: Path, records: list[dict], size: int, *, augment=False, cache=False, augmentation="basic"):
        if not records:
            raise ValueError("数据集不能为空")
        self.root, self.records, self.size = root, records, size
        self.augment = augment
        if augmentation not in ("basic", "strong"):
            raise ValueError("augmentation 只支持 basic 或 strong")
        self.augmentation = augmentation
        for row in records:
            path = (root / row["mask_index"]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file() or not row.get("label_aligned", False):
                raise ValueError(f"缺少对齐的索引标签：{row['id']}")
        self.cached = [self.read_pair(i) for i in range(len(records))] if cache else None

    def __len__(self):
        return len(self.records)

    def read_pair(self, index):
        row = self.records[index]
        with Image.open(self.root / row["image"]) as image, Image.open(self.root / row["mask_index"]) as mask:
            tensor, label, _ = fit_image(image, self.size, mask)
        return tensor, label

    def pair(self, index):
        return self.cached[index] if self.cached is not None else self.read_pair(index)

    def __getitem__(self, index):
        image, mask = self.pair(index)
        if self.augment and self.augmentation == "strong":
            angle = (torch.rand(()).item() - 0.5) * 16
            scale = 1.0 + 0.2 * torch.rand(()).item()
            translate = [round((torch.rand(()).item() - 0.5) * 0.04 * self.size) for _ in range(2)]
            image = TF.affine(image, angle, translate, scale, [0., 0.], interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = TF.affine(mask.unsqueeze(0), angle, translate, scale, [0., 0.], interpolation=InterpolationMode.NEAREST, fill=IGNORE_INDEX)[0]
        image = image.float().div(255)
        if self.augment:
            if torch.rand(()).item() < 0.5:
                image, mask = image.flip(-1), mask.flip(-1)
            # 几何操作同步；亮度/饱和度只改变图片，不修改类别编号。
            strength = 0.2 if self.augmentation == "strong" else 0.1
            image = TF.adjust_brightness(image, 1 - strength + 2 * strength * torch.rand(()).item())
            image = TF.adjust_saturation(image, 1 - strength + 2 * strength * torch.rand(()).item())
        return image, mask.long()

    def class_histogram(self):
        counts = torch.zeros(NUM_CLASSES, dtype=torch.int64)
        for i in range(len(self)):
            mask = self.pair(i)[1].long()
            counts += torch.bincount(mask[mask != IGNORE_INDEX], minlength=NUM_CLASSES)
        return counts


def balanced_sampling_probabilities(records):
    """一半均匀采样，一半按类别出现次数平衡；只从训练记录计算。"""
    pixels = np.asarray([row["class_pixels"] for row in records], dtype=np.float64)
    present = pixels >= np.maximum(1, 0.005 * pixels.sum(axis=1, keepdims=True))
    frequency = present.sum(axis=0)
    available = frequency > 0
    if not available.any():
        raise ValueError("没有可用于平衡采样的类别")
    balanced = (present[:, available] / frequency[available]).mean(axis=1)
    probability = 0.5 / len(records) + 0.5 * balanced
    return torch.from_numpy(probability / probability.sum()).double()


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels), nn.ReLU(inplace=True),
        )


class TinyUNet(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels 必须为至少 8 的 8 倍数")
        b = base_channels
        self.enc1, self.enc2 = ConvBlock(3, b), ConvBlock(b, 2*b)
        self.enc3, self.bridge = ConvBlock(2*b, 4*b), ConvBlock(4*b, 8*b)
        self.dec3, self.dec2, self.dec1 = ConvBlock(12*b, 4*b), ConvBlock(6*b, 2*b), ConvBlock(3*b, b)
        self.head = nn.Conv2d(b, NUM_CLASSES, 1)

    def forward(self, image):
        x1 = self.enc1((image - 0.5) / 0.5)
        x2 = self.enc2(F.max_pool2d(x1, 2))
        x3 = self.enc3(F.max_pool2d(x2, 2))
        x4 = self.bridge(F.max_pool2d(x3, 2))
        def up(x, skip):
            return torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], dim=1)
        return self.head(self.dec1(up(self.dec2(up(self.dec3(up(x4, x3)), x2)), x1)))


class FusedLRASPPHead(nn.Module):
    """把 1/16、1/8、1/4 三个尺度逐级融合的轻量解码头。"""
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.high_cbr = nn.Sequential(
            nn.Conv2d(960, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.scale = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(960, 128, 1, bias=False), nn.Sigmoid())
        self.low_proj = nn.Sequential(
            nn.Conv2d(40, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.low_refine = nn.Sequential(
            nn.Conv2d(192, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.fine_proj = nn.Sequential(
            nn.Conv2d(24, 48, 3, padding=1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True)
        )
        self.fine_refine = nn.Sequential(
            nn.Conv2d(176, 96, 3, padding=1, bias=False), nn.BatchNorm2d(96), nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(96, num_classes, 1)

    def forward(self, input):
        fine, low, high = input["fine"], input["low"], input["high"]
        context = self.high_cbr(high) * self.scale(high)
        context = F.interpolate(context, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low_context = self.low_refine(torch.cat((context, self.low_proj(low)), dim=1))
        low_context = F.interpolate(low_context, size=fine.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fine_refine(torch.cat((low_context, self.fine_proj(fine)), dim=1))
        return self.classifier(fused)


class TransferSegmentationModel(nn.Module):
    """MobileNetV3 + LR-ASPP；使用官方预训练特征，八类输出层重新学习。

    ``low_layer=4`` 是 torchvision 默认的 1/8 低层特征；``low_layer=3``
    输出 1/4 特征图，给边界和小目标保留更多空间采样点。
    """
    def __init__(self, pretrained_path: Path | None = None, low_layer: str = "4", fused: bool = False):
        super().__init__()
        from torchvision.models.segmentation import lraspp_mobilenet_v3_large
        # 显式禁止隐式下载，推理时只需要本项目训练得到的权重。
        self.network = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=NUM_CLASSES)
        if low_layer not in {"3", "4"}:
            raise ValueError("low_layer 只支持 3（1/4）或 4（1/8）")
        self.low_layer = low_layer
        self.fused = fused
        if fused:
            self.network.backbone.return_layers = {"3": "fine", "4": "low", "16": "high"}
            self.network.classifier = FusedLRASPPHead(NUM_CLASSES)
        else:
            self.network.backbone.return_layers = {low_layer: "low", "16": "high"}
        low_channels = {"3": 24, "4": 40}[low_layer]
        if low_layer == "3" and not fused:
            # LR-ASPP 默认低层分类器接收 block 4 的 40 通道；改接 block 3
            # 时必须同步修改输入通道，其他高层结构和预训练特征保持不变。
            self.network.classifier.low_classifier = nn.Conv2d(low_channels, NUM_CLASSES, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            retained = {key: value for key, value in state.items() if not key.startswith("classifier.")}
            result = self.network.load_state_dict(retained, strict=False)
            # BatchNorm 的 num_batches_tracked 在 strict=False 加载时不会列入
            # missing_keys，因此从预期缺失集合中排除这些缓冲区。
            expected_missing = {key for key in self.network.state_dict()
                                if key.startswith("classifier.") and not key.endswith("num_batches_tracked")}
            if set(result.missing_keys) != expected_missing or result.unexpected_keys:
                raise ValueError("预训练文件与当前 LR-ASPP 结构不兼容")

    def forward(self, image):
        return self.network((image - self.mean) / self.std)["out"]


def create_segmentation_model(model_name, base_channels=16, pretrained_path=None):
    if model_name == SEGFORMER_MODEL_NAME:
        from segformer_suim import SegFormerSegmentationModel
        return SegFormerSegmentationModel(pretrained_path)
    if model_name == MODEL_NAME:
        if pretrained_path is not None:
            raise ValueError("TinyUNet 不接受 LR-ASPP 的预训练文件")
        return TinyUNet(base_channels)
    if model_name == TRANSFER_MODEL_NAME:
        return TransferSegmentationModel(pretrained_path, low_layer="4")
    if model_name == HR_TRANSFER_MODEL_NAME:
        return TransferSegmentationModel(pretrained_path, low_layer="3")
    if model_name == FPN_TRANSFER_MODEL_NAME:
        return TransferSegmentationModel(pretrained_path, low_layer="4", fused=True)
    raise ValueError(f"不支持的分割模型：{model_name}")


def segmentation_loss(logits, target, weights):
    ce = F.cross_entropy(logits, target, weight=weights, ignore_index=IGNORE_INDEX)
    valid = target != IGNORE_INDEX
    if not valid.any():
        raise ValueError("此批次没有可用于监督的像素")
    probabilities = logits.float().softmax(dim=1) * valid.unsqueeze(1)
    truth = F.one_hot(target.masked_fill(~valid, 0), NUM_CLASSES).permute(0, 3, 1, 2).float()
    truth = truth * valid.unsqueeze(1)
    intersection = (probabilities * truth).sum((0, 2, 3))
    target_area = truth.sum((0, 2, 3))
    total_area = probabilities.sum((0, 2, 3)) + target_area
    present = target_area > 0
    dice = 1 - ((2 * intersection[present] + 1e-6) / (total_area[present] + 1e-6)).mean()
    return 0.7 * ce + 0.3 * dice


def confusion_matrix(prediction, target):
    valid = target != IGNORE_INDEX
    return torch.bincount(NUM_CLASSES * target[valid].long() + prediction[valid].long(),
                          minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_confusion(confusion, classes):
    c = np.asarray(confusion, dtype=np.float64)
    tp, support, predicted = np.diag(c), c.sum(axis=1), c.sum(axis=0)
    union = support + predicted - tp
    present = union > 0
    iou = np.divide(tp, union, out=np.zeros(NUM_CLASSES), where=present)
    dice = np.divide(2*tp, support+predicted, out=np.zeros(NUM_CLASSES), where=(support+predicted)>0)
    recall = np.divide(tp, support, out=np.zeros(NUM_CLASSES), where=support>0)
    return {
        "miou": float(iou[present].mean()) if present.any() else None,
        "pixel_accuracy": float(tp.sum() / c.sum()) if c.sum() else None,
        "mean_class_accuracy": float(recall[support > 0].mean()) if (support > 0).any() else None,
        "valid_pixels": int(c.sum()),
        "per_class": [{**row, "iou": float(iou[i]) if present[i] else None,
                       "dice": float(dice[i]) if present[i] else None,
                       "recall": float(recall[i]) if support[i] else None,
                       "true_pixels": int(support[i]), "predicted_pixels": int(predicted[i])}
                      for i, row in enumerate(classes)],
        "metric_definition": "混淆矩阵先跨全部图片累计；行=真值、列=预测。mIoU 对并集非空的类别等权平均，含背景。标签 255 不计分。",
    }


def palette_from_classes(classes):
    return np.array([row["rgb"] for row in classes], dtype=np.uint8)


def color_mask(mask, palette):
    result = np.full((*mask.shape, 3), 80, dtype=np.uint8)
    valid = mask < NUM_CLASSES
    result[valid] = palette[mask[valid]]
    return result


def save_comparison(samples, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    palette = palette_from_classes(classes)
    fig, axes = plt.subplots(len(samples), 4, figsize=(12, 2.7*len(samples) + 0.5),
                             squeeze=False, layout="constrained")
    for row, (name, image, target, prediction) in zip(axes, samples):
        ref = (image.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        pred_colors = color_mask(prediction, palette)
        overlay = (0.55 * ref + 0.45 * pred_colors).round().astype(np.uint8)
        overlay[target == IGNORE_INDEX] = 80
        for ax, pixels in zip(row, (ref, color_mask(target, palette), pred_colors, overlay)):
            ax.imshow(pixels)
            ax.axis("off")
        row[0].set_title(name, fontsize=8)
    for ax, title in zip(axes[0], ("Input (padded)", "Ground truth", "Prediction", "Overlay")):
        ax.set_title(title + ("\n" + samples[0][0] if title.startswith("Input") else ""), fontsize=10)
    fig.legend(handles=[Patch(color=np.array(row["rgb"])/255, label=row["symbol"]) for row in classes],
               loc="outside lower center", ncol=8, fontsize=9)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def load_segmentation_model(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("model_name") not in (MODEL_NAME, *TRANSFER_MODEL_NAMES, SEGFORMER_MODEL_NAME) or state.get("preprocessing") != PREPROCESSING:
        raise ValueError("不是当前版本的分割模型，不能使用重建模型或其他预处理的权重")
    model = create_segmentation_model(state["model_name"], state.get("base_channels", 16)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def predict_original(model, image, size, device):
    tensor, _, geometry = fit_image(image, size)
    with torch.inference_mode():
        logits = model(tensor.unsqueeze(0).float().div(255).to(device))
        g = geometry
        logits = logits[:, :, g["top"]:g["top"]+g["scaled_height"], g["left"]:g["left"]+g["scaled_width"]]
        logits = F.interpolate(logits, size=(g["height"], g["width"]), mode="bilinear", align_corners=False)
        prediction = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    return prediction
