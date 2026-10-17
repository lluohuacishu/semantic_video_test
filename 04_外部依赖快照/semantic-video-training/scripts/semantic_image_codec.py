"""联合语义源编码：同一潜变量码流解码 RGB 和 SUIM 类别，不依赖原图。"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import compressai
import numpy as np
import torch
from compressai.models import ScaleHyperprior
from torch import nn
from torch.nn import functional as F
from PIL import Image

from image_codec import (ImageCodec, SEMANTIC_ARCHITECTURE, read_package, read_rgb,
                         validate_dimensions, write_package)
from prepare_image_codec import SOURCE_REVISION
from segmentation import NUM_CLASSES, color_mask, palette_from_classes


class JointSemanticHyperprior(ScaleHyperprior):
    """RGB 与发送端预测类别共同进入编码器；量化潜变量共用两个解码头。"""

    def __init__(self):
        super().__init__(128, 192)
        first = self.g_a[0]
        self.g_a[0] = nn.Conv2d(3 + NUM_CLASSES, first.out_channels,
                                first.kernel_size, first.stride, first.padding)
        self.semantic_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, NUM_CLASSES, 1),
        )

    def synthesize(self, y_hat):
        shared = self.g_s[:4](y_hat)
        image = self.g_s[4:](shared)
        semantics = F.interpolate(self.semantic_head(shared), size=image.shape[-2:],
                                  mode="bilinear", align_corners=False)
        return {"x_hat": image, "semantic_logits": semantics}

    def forward(self, conditioned):
        y = self.g_a(conditioned)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        result = self.synthesize(y_hat)
        result["likelihoods"] = {"y": y_likelihoods, "z": z_likelihoods}
        return result

    def decompress(self, strings, shape):
        if not isinstance(strings, list) or len(strings) != 2:
            raise ValueError("联合语义模型需要两组熵编码流")
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        result = self.synthesize(y_hat)
        result["x_hat"] = result["x_hat"].clamp(0, 1)
        return result


def condition_image(image, semantic_index):
    """训练与实际编码共用条件表示；输入语义只能来自发送端预测。"""
    if semantic_index.ndim != 3 or semantic_index.shape != (image.shape[0], *image.shape[-2:]):
        raise ValueError("语义索引与图片尺寸不一致")
    if semantic_index.dtype != torch.long or torch.any((semantic_index < 0) | (semantic_index >= NUM_CLASSES)):
        raise ValueError("语义条件必须是 0～7 的 long 类别索引")
    guidance = F.one_hot(semantic_index, NUM_CLASSES).permute(0, 3, 1, 2).to(image.dtype)
    return torch.cat((image, guidance), dim=1)


def pretrained_joint_model():
    baseline, _ = ImageCodec().model(3)
    model = JointSemanticHyperprior()
    state = dict(baseline.state_dict())
    rgb_weight = state["g_a.0.weight"]
    expanded = torch.zeros_like(model.g_a[0].weight)
    expanded[:, :3] = rgb_weight
    state["g_a.0.weight"] = expanded
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not name.startswith("semantic_head.") for name in missing):
        raise ValueError("通用压缩模型与联合模型初始化结构不一致")
    return model


def load_joint_checkpoint(path):
    from compressai.version import git_version
    if git_version != SOURCE_REVISION:
        raise ValueError("CompressAI 实现与联合码流的锁定版本不一致")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state.get("model_name") != SEMANTIC_ARCHITECTURE or state.get("source_revision") != SOURCE_REVISION
            or state.get("num_classes") != NUM_CLASSES or state.get("quality") != 3
            or state.get("joint_training_completed") is not True
            or [row["index"] for row in state.get("classes", [])] != list(range(NUM_CLASSES))):
        raise ValueError("这不是受支持的 SUIM 联合语义编码权重")
    model = JointSemanticHyperprior()
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, state


class SemanticImageCodec:
    def __init__(self, checkpoint: Path):
        torch.set_num_threads(4)
        compressai.set_entropy_coder("ans")
        self.checkpoint = Path(checkpoint)
        self.digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.model, self.state = load_joint_checkpoint(self.checkpoint)
        # 实际码流两端固定 CPU；训练前向使用 GPU 不代表跨设备 rANS 可互换。
        self.model.update(force=True)
        self.palette = palette_from_classes(self.state["classes"])

    def encode(self, rgb, semantic_index, output):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("图片必须是 uint8 RGB")
        height, width = rgb.shape[:2]
        validate_dimensions(width, height)
        if semantic_index.dtype != np.uint8 or semantic_index.shape != (height, width):
            raise ValueError("发送端语义条件必须是同尺寸的 uint8 类别图")
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float().div(255)
        mask = torch.from_numpy(np.ascontiguousarray(semantic_index)).unsqueeze(0).long()
        if torch.any(mask >= NUM_CLASSES):
            raise ValueError("发送端语义条件只能包含 0～7")
        padding = (0, (-width) % 64, 0, (-height) % 64)
        x, mask = F.pad(x, padding), F.pad(mask, padding, value=0)
        started = time.perf_counter()
        with torch.inference_mode():
            result = self.model.compress(condition_image(x, mask))
        elapsed = time.perf_counter() - started
        streams = [group[0] for group in result["strings"]]
        header = write_package(Path(output), {
            "architecture": SEMANTIC_ARCHITECTURE, "quality": 3, "semantic_classes": NUM_CLASSES,
            "width": width, "height": height, "shape": list(result["shape"]), "entropy_coder": "ans",
            "weights_sha256": self.digest, "source_revision": SOURCE_REVISION, "inference_backend": "cpu",
        }, streams)
        total = Path(output).stat().st_size
        return {"width": width, "height": height, "quality": 3, "file_bytes": total,
                "payload_bytes": sum(map(len, streams)), "header_bytes": total - sum(map(len, streams)),
                "bpp": total * 8 / (width * height), "raw_rgb_ratio": width * height * 3 / total,
                "encode_compute_seconds": elapsed, "header": header}

    def decode(self, source, image_output, semantic_output):
        for output in (Path(image_output), Path(semantic_output)):
            if output.exists():
                raise FileExistsError(f"解码输出已经存在：{output}")
        if Path(image_output).resolve() == Path(semantic_output).resolve():
            raise ValueError("图片输出和类别输出不能使用同一个路径")
        header, streams = read_package(Path(source))
        if header["architecture"] != SEMANTIC_ARCHITECTURE or header["weights_sha256"] != self.digest:
            raise ValueError("码流与接收端联合语义模型不匹配")
        started = time.perf_counter()
        with torch.inference_mode():
            result = self.model.decompress([[stream] for stream in streams], tuple(header["shape"]))
        compute_seconds = time.perf_counter() - started
        height, width = header["height"], header["width"]
        rgb = result["x_hat"][0, :, :height, :width].mul(255).round().byte().permute(1, 2, 0).numpy()
        index = result["semantic_logits"][0, :, :height, :width].argmax(0).byte().numpy()
        Image.fromarray(rgb).save(image_output)
        Image.fromarray(index).save(semantic_output)
        return rgb, index, {"width": width, "height": height, "quality": 3,
                            "decode_compute_seconds": compute_seconds,
                            "semantic_source": "direct_latent_decoder"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("encode", "decode"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--semantic-output", type=Path)
    parser.add_argument("--segmentation-checkpoint", type=Path,
                        help="仅发送端需要；接收端不会加载分割模型")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                        help="发送端分割设备；码流编解码统一使用 CPU")
    args = parser.parse_args()
    codec = SemanticImageCodec(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "encode":
        if not args.segmentation_checkpoint:
            parser.error("发送端需要 --segmentation-checkpoint")
        from segment_video import choose_device, predict_frame
        from segmentation import load_segmentation_model
        device = choose_device(args.device)
        teacher, teacher_state = load_segmentation_model(args.segmentation_checkpoint, device)
        digest = hashlib.sha256(args.segmentation_checkpoint.read_bytes()).hexdigest()
        if digest != codec.state["sender_teacher_sha256"]:
            raise ValueError("发送端分割权重与联合编码模型的训练条件不一致")
        rgb = read_rgb(args.input)
        mask, _ = predict_frame(teacher, rgb, int(teacher_state["size"]), device)
        report = codec.encode(rgb, mask, args.output)
    else:
        semantic_output = args.semantic_output or args.output.with_name(args.output.stem + "_classes.png")
        semantic_output.parent.mkdir(parents=True, exist_ok=True)
        color_output = semantic_output.with_name(semantic_output.stem + "_color.png")
        if color_output.exists() or color_output.resolve() == args.output.resolve():
            raise FileExistsError(color_output)
        _, index, report = codec.decode(args.input, args.output, semantic_output)
        report["semantic_index_output"] = str(semantic_output.resolve())
        Image.fromarray(color_mask(index, codec.palette)).save(color_output)
        report["semantic_color_output"] = str(color_output.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
