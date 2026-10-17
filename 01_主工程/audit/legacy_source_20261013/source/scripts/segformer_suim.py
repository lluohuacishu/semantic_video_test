"""SegFormer-B0 的 SUIM 八类适配，复用现有数据和评价流程。"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from prepare_segformer import MODEL_ID, REVISION, sha256

LABELS = ("BW", "HD", "PF", "WR", "RO", "RI", "FV", "SR")


class SegFormerSegmentationModel(nn.Module):
    def __init__(self, pretrained_path: Path | None = None):
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as error:
            raise RuntimeError("SegFormer 需要 .venv-segformer 环境，请使用对应的启动脚本") from error
        id2label = dict(enumerate(LABELS))
        label2id = {name: index for index, name in id2label.items()}
        if pretrained_path is None:
            config = SegformerConfig(
                num_labels=8, id2label=id2label, label2id=label2id,
                hidden_sizes=[32, 64, 160, 256], depths=[2, 2, 2, 2],
                num_attention_heads=[1, 2, 5, 8], decoder_hidden_size=256,
                semantic_loss_ignore_index=255,
            )
            self.network = SegformerForSemanticSegmentation(config)
        else:
            pretrained_path = Path(pretrained_path)
            provenance = json.loads((pretrained_path / "provenance.json").read_text(encoding="utf-8"))
            if provenance.get("model_id") != MODEL_ID or provenance.get("revision") != REVISION:
                raise ValueError("预训练目录不是已核对来源的 SegFormer-B0 权重")
            for name in ("config.json", "model.safetensors"):
                if sha256(pretrained_path / name) != provenance.get("files", {}).get(name, {}).get("sha256"):
                    raise ValueError(f"预训练文件与准备时的记录不一致：{name}")
            self.network, info = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_path, local_files_only=True, trust_remote_code=False,
                use_safetensors=True, output_loading_info=True,
            )
            if any(info.get(name) for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise ValueError(f"预训练权重未完整加载：{info}")
            if list(self.network.config.hidden_sizes) != [32, 64, 160, 256] or list(self.network.config.depths) != [2]*4:
                raise ValueError("预训练结构不是 B0")
            # 只重建最终分类层；主干和解码器的多尺度融合权重继续参与微调。
            self.network.decode_head.classifier = nn.Conv2d(256, 8, kernel_size=1)
            nn.init.normal_(self.network.decode_head.classifier.weight, std=self.network.config.initializer_range)
            nn.init.zeros_(self.network.decode_head.classifier.bias)
            self.network.config.num_labels = 8
            self.network.config.id2label = id2label
            self.network.config.label2id = label2id
            self.network.config.semantic_loss_ignore_index = 255
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):
        logits = self.network(pixel_values=(image - self.mean) / self.std).logits
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
