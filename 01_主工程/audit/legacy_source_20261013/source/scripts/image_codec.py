"""图片神经编码器：实际 rANS 码流、带尺寸信息的 SVIC 文件和独立解码。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

from prepare_image_codec import ARCHITECTURE, MODEL_DIR, SOURCE_REVISION, URLS, verify_weight

MAGIC = b"SVIC\x01"
MAX_HEADER = 4096
MAX_PACKAGE = 256 * 1024 * 1024
MAX_PIXELS = 4_194_304
SEMANTIC_ARCHITECTURE = "suim-joint-hyperprior-v1"


def validate_dimensions(width, height):
    if (type(width) is not int or type(height) is not int or not 0 < width <= 4096
            or not 0 < height <= 4096 or width * height > MAX_PIXELS):
        raise ValueError("图片尺寸无效；最长边不超过 4096，最多 419 万像素")


def validate_header(header):
    if not isinstance(header, dict):
        raise ValueError("压缩文件头应为对象")
    if (header.get("architecture") not in (ARCHITECTURE, SEMANTIC_ARCHITECTURE) or type(header.get("quality")) is not int
            or header.get("quality") not in URLS or header.get("entropy_coder") != "ans"
            or header.get("source_revision") != SOURCE_REVISION
            or header.get("inference_backend") != "cpu"):
        raise ValueError("不支持这个压缩模型、质量档位或码流版本")
    if header["architecture"] == SEMANTIC_ARCHITECTURE:
        if header["quality"] != 3 or type(header.get("semantic_classes")) is not int or header["semantic_classes"] != 8:
            raise ValueError("语义码流需要八类联合模型，当前只支持 q3 初始化的训练版本")
    validate_dimensions(header.get("width"), header.get("height"))
    shape = header.get("shape")
    expected = [(header["height"] + 63) // 64, (header["width"] + 63) // 64]
    if shape != expected or not isinstance(shape, list) or any(type(v) is not int for v in shape):
        raise ValueError("潜变量尺寸与原图不一致")
    lengths = header.get("stream_lengths")
    if (not isinstance(lengths, list) or len(lengths) != 2
            or any(type(v) is not int or not 0 < v <= MAX_PACKAGE for v in lengths)
            or sum(lengths) > MAX_PACKAGE):
        raise ValueError("压缩码流长度无效")
    for key in ("weights_sha256", "payload_sha256"):
        value = header.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("压缩文件缺少有效校验记录")


def write_package(output: Path, header: dict, streams: list[bytes]):
    header = dict(header, stream_lengths=[len(stream) for stream in streams],
                  payload_sha256=hashlib.sha256(b"".join(streams)).hexdigest())
    validate_header(header)
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(encoded) > MAX_HEADER:
        raise ValueError("压缩文件头过长")
    with output.open("xb") as handle:
        handle.write(MAGIC + struct.pack("!I", len(encoded)) + encoded)
        for stream in streams:
            handle.write(stream)
    return header


def read_package(path: Path):
    size = path.stat().st_size
    if not len(MAGIC) + 4 < size <= MAX_PACKAGE:
        raise ValueError("压缩文件长度无效")
    with path.open("rb") as handle:
        if handle.read(len(MAGIC)) != MAGIC:
            raise ValueError("文件不是支持的 SVIC 图片压缩包")
        length_bytes = handle.read(4)
        if len(length_bytes) != 4:
            raise ValueError("压缩文件头不完整")
        length = struct.unpack("!I", length_bytes)[0]
        if not 0 < length <= MAX_HEADER:
            raise ValueError("压缩文件头长度无效")
        try:
            header = json.loads(handle.read(length))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("压缩文件头损坏") from exc
        validate_header(header)
        if len(MAGIC) + 4 + length + sum(header["stream_lengths"]) != size:
            raise ValueError("压缩文件未收完整或含多余数据")
        streams = [handle.read(count) for count in header["stream_lengths"]]
    if hashlib.sha256(b"".join(streams)).hexdigest() != header["payload_sha256"]:
        raise ValueError("压缩码流校验失败")
    return header, streams


def read_rgb(path: Path):
    with Image.open(path) as original:
        validate_dimensions(*original.size)
        image = ImageOps.exif_transpose(original).convert("RGB")
        validate_dimensions(*image.size)
        return np.array(image)


class ImageCodec:
    def __init__(self, device="cpu", model_dir=MODEL_DIR):
        self.device = torch.device(device)
        if self.device.type != "cpu":
            raise ValueError("当前图片码流统一使用 CPU 编解码；GPU 概率计算可能造成跨设备解码不一致")
        self.model_dir = Path(model_dir)
        self.models = {}
        torch.set_num_threads(4)

    def model(self, quality):
        if quality in self.models:
            return self.models[quality]
        try:
            import compressai
            from compressai.zoo import bmshj2018_hyperprior
            from compressai.zoo.pretrained import load_pretrained
        except ImportError as exc:
            raise RuntimeError("图片编码环境未准备好，请运行 setup_image_codec.cmd") from exc
        from compressai.version import git_version
        if git_version != SOURCE_REVISION:
            raise ValueError("CompressAI 版本与 SVIC 码流的锁定实现不一致")
        path, digest = verify_weight(self.model_dir, quality)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]
        model = bmshj2018_hyperprior(quality=quality, metric="mse", pretrained=False)
        model.load_state_dict(load_pretrained(state), strict=True)
        model.eval()
        # 不在 CPU/CUDA 之间切换：预测的尺度落在不同区间会让 rANS 解码失步。
        model.update(force=True)
        model = model.to(self.device)
        compressai.set_entropy_coder("ans")
        self.models[quality] = model, digest
        return model, digest

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def encode(self, rgb: np.ndarray, output: Path, quality=3):
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("编码输入必须为 uint8 RGB 图片")
        height, width = rgb.shape[:2]
        validate_dimensions(width, height)
        model, digest = self.model(quality)
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float().div(255).to(self.device)
        x = F.pad(x, (0, (-width) % 64, 0, (-height) % 64))
        self.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            result = model.compress(x)
        self.synchronize()
        compute_seconds = time.perf_counter() - started
        header = {"architecture": ARCHITECTURE, "quality": quality, "width": width, "height": height,
                  "shape": list(result["shape"]), "entropy_coder": "ans", "weights_sha256": digest,
                  "source_revision": SOURCE_REVISION, "inference_backend": "cpu"}
        streams = [group[0] for group in result["strings"]]
        header = write_package(output, header, streams)
        total = output.stat().st_size
        return {"width": width, "height": height, "quality": quality, "file_bytes": total,
                "payload_bytes": sum(len(stream) for stream in streams),
                "header_bytes": total - sum(len(stream) for stream in streams),
                "bpp": total * 8 / (width * height), "raw_rgb_ratio": width * height * 3 / total,
                "encode_compute_seconds": compute_seconds, "header": header}

    def decode(self, source: Path, output: Path):
        if output.exists():
            raise FileExistsError(f"还原图片已经存在：{output}")
        header, streams = read_package(source)
        if header["architecture"] != ARCHITECTURE:
            raise ValueError("这是联合语义码流，请使用 semantic_image_codec.py 和对应训练权重解码")
        model, digest = self.model(header["quality"])
        if digest != header["weights_sha256"]:
            raise ValueError("接收端的本地模型权重与编码端不同")
        self.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            result = model.decompress([[stream] for stream in streams], tuple(header["shape"]))
        self.synchronize()
        compute_seconds = time.perf_counter() - started
        rgb = (result["x_hat"][0, :, :header["height"], :header["width"]].clamp(0, 1)
               .mul(255).round().byte().permute(1, 2, 0).cpu().numpy())
        if rgb.shape != (header["height"], header["width"], 3):
            raise ValueError("解码图片尺寸异常")
        Image.fromarray(rgb).save(output)
        return rgb, {"decode_compute_seconds": compute_seconds, "width": header["width"],
                     "height": header["height"], "quality": header["quality"]}


def quality_metrics(reference: np.ndarray, reconstructed: np.ndarray):
    if reference.shape != reconstructed.shape:
        raise ValueError("原图和还原图片尺寸不同，无法计算质量")
    mse = float(np.square(reference.astype(np.float64) - reconstructed.astype(np.float64)).mean())
    psnr = 10 * math.log10(255 ** 2 / mse) if mse else None
    ssim = None
    if min(reference.shape[:2]) >= 7:
        from pytorch_msssim import ssim as calculate_ssim
        tensors = [torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).float().div(255)
                   for array in (reference, reconstructed)]
        with torch.inference_mode():
            ssim = float(calculate_ssim(*tensors, data_range=1, size_average=True, win_size=7))
    return {"psnr_db": psnr, "ssim": ssim, "mse_rgb_8bit": mse, "identical_pixels": mse == 0,
            "definition": "原尺寸、8-bit RGB PNG；PSNR 使用全图 MSE，SSIM 为 pytorch-msssim 的 7x7 Gaussian 单尺度 SSIM"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("encode", "decode", "inspect"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quality", type=int, choices=tuple(URLS), default=3)
    parser.add_argument("--device", choices=("cpu",), default="cpu", help="码流编解码固定使用 CPU")
    args = parser.parse_args()
    if args.command == "inspect":
        result = read_package(args.input)[0]
    else:
        if args.output is None:
            parser.error("编码和解码需要 --output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        codec = ImageCodec(args.device)
        result = codec.encode(read_rgb(args.input), args.output, args.quality) if args.command == "encode" else codec.decode(args.input, args.output)[1]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
