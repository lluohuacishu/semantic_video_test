"""用 NVIDIA 的 SegFormer-B0 / ADE20K 权重做离线图片分割。"""
from __future__ import annotations

import argparse
import html
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

# 准备模型时才联网；推理只加载已下载的配置和 safetensors。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
import torch
import transformers
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from prepare_segformer import MODEL_DIR, MODEL_ID, REVISION, ROOT, SAMPLE_PATH, sha256

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def inputs_for_run(inputs: list[Path], manifest: Path, samples: int) -> list[Path]:
    paths = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(p for p in item.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()))
        else:
            paths.append(item)
    if not inputs:
        paths.append(SAMPLE_PATH)
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        root = (manifest.parent / metadata.get("root", ".")).resolve()
        records = metadata["splits"]["validation"]
        if not records:
            raise ValueError("验证集为空")
        # 均匀取样，索引由数据清单决定，不按预测效果挑图。
        indices = np.linspace(0, len(records) - 1, min(samples, len(records)), dtype=int)
        for index in indices:
            path = (root / records[int(index)]["image"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("数据清单包含目录外的图片")
            paths.append(path)
    unique = list(dict.fromkeys(p.resolve() for p in paths))
    if not unique:
        raise ValueError("没有可处理的图片")
    for path in unique:
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"图片不存在或格式不受支持：{path}")
    return unique


def visual_palette(classes: int) -> np.ndarray:
    # 固定的显示配色，类别含义始终以模型 config.json 为准。
    colors = np.zeros((classes, 3), dtype=np.uint8)
    for index in range(classes):
        value = index
        for bit in range(8):
            for channel in range(3):
                colors[index, channel] |= ((value >> channel) & 1) << (7 - bit)
            value >>= 3
    return colors


def font(size: int):
    for name in ("C:/Windows/Fonts/msyh.ttc", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def save_panel(image: Image.Image, colors: Image.Image, overlay: Image.Image, row: dict, path: Path) -> None:
    canvas = Image.new("RGB", (1200, 358), "#f4f6f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 8), f"{row['number']:02d}  {Path(row['input']).name[:65]}", font=font(19), fill="#172b3a")
    for col, (label, picture) in enumerate(zip(("原图", "类别图 · ADE20K", "叠加图"), (image, colors, overlay))):
        draw.text((col * 400 + 14, 36), label, font=font(18), fill="#172b3a")
        tile = ImageOps.contain(picture, (384, 248), Image.Resampling.LANCZOS)
        canvas.paste(tile, (col * 400 + (400 - tile.width) // 2, 66 + (248 - tile.height) // 2))
    legend = " / ".join(f"{r['name']} {r['fraction']:.1%}" for r in row["predicted_classes"][:5])
    draw.text((14, 326), legend[:116], font=font(16), fill="#334155")
    canvas.save(path)


def save_report(output: Path, summary: dict) -> None:
    parts = []
    for row in summary["images"]:
        legend = " ".join(
            f'<span><i style="background:rgb({",".join(str(c) for c in item["rgb"])})"></i>'
            f'{html.escape(item["name"])} {item["fraction"]:.1%}</span>'
            for item in row["predicted_classes"] if item["fraction"] >= 0.001
        )
        folder = row["folder"]
        parts.append(f'<article><h2>{row["number"]}. {html.escape(Path(row["input"]).name)}</h2>'
                     f'<img src="{folder}/comparison.png" alt="原图、类别图和叠加图" loading="lazy">'
                     f'<div class="legend">{legend}</div><p>'
                     f'<a href="{folder}/input.png">原图</a> · <a href="{folder}/prediction_index.png">类别索引</a> · '
                     f'<a href="{folder}/overlay.png">叠加图</a> · <a href="{folder}/summary.json">数据</a></p></article>')
    benchmark = summary["benchmark"]
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page += '<title>SegFormer-B0 试跑结果</title><style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f4f6f8;color:#172b3a;max-width:1200px;margin:32px auto;padding:0 18px;line-height:1.7}h1{font-size:28px}h2{font-size:20px}article{background:white;padding:20px;margin:20px 0;border:1px solid #dbe2e8;border-radius:10px}img{width:100%;height:auto}.legend{display:flex;flex-wrap:wrap;gap:8px 20px}.legend span{white-space:nowrap}i{display:inline-block;width:14px;height:14px;margin-right:5px;border:1px solid #bbb}a{color:#005fa8}.note{border-left:4px solid #4680a0;padding:12px 18px;background:#e9f0f5}</style>'
    page += f'<h1>SegFormer-B0 预训练模型试跑</h1><p>{html.escape(summary["device_name"])} · 输入 {summary["input_size"]} × {summary["input_size"]} · {summary["parameters"]:,} 个参数</p>'
    page += '<p class="note">这是 NVIDIA 发布的 ADE20K 150 类权重，尚未在 SUIM 上微调。水下图片只用于检查输出，不报告 SUIM 准确率，也不与现有八类模型直接比较 mIoU。颜色表示预测类别，不表示预测正确。</p>'
    page += f'<p>预热后的模型前向中位耗时：{benchmark["forward_ms_median"]:.1f} ms；共测量 {benchmark["repeats"]} 次。该时间不含图片读取、预处理、上采样和文件保存，不能当作视频端到端帧率。</p>'
    page += '<p><a href="summary.json">完整运行数据</a> · <a href="classes.json">150 类名称和配色</a> · <a href="https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512">权重来源</a></p>'
    page += "".join(parts) + '</html>'
    (output / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="图片或图片目录；不传时使用作者示例及固定抽取的 SUIM 验证图片")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/SUIM/manifest.json")
    parser.add_argument("--samples", type=int, default=4, help="默认试跑时从 SUIM 验证集均匀取多少张图片")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--benchmark-repeats", type=int, default=20)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.samples < 1 or args.size < 64 or args.size % 32 or args.benchmark_repeats < 1:
        parser.error("samples、benchmark-repeats 必须为正；size 必须为至少 64 的 32 倍数")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用")
    output = (args.output_dir or ROOT / "runs" / datetime.now().strftime("segformer_b0_pretrained_%Y%m%d-%H%M%S-%f")).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"输出目录非空，不覆盖：{output}")
    try:
        paths = inputs_for_run(args.inputs, args.manifest, args.samples)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    for name in ("config.json", "preprocessor_config.json", "model.safetensors", "provenance.json"):
        if not (args.model_dir / name).is_file():
            parser.error(f"缺少 {name}，请先运行 setup_segformer.cmd")
    provenance = json.loads((args.model_dir / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("model_id") != MODEL_ID or provenance.get("revision") != REVISION:
        parser.error("模型目录的来源记录不属于本次 SegFormer-B0 / ADE20K 权重")
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        if sha256(args.model_dir / name) != provenance.get("files", {}).get(name, {}).get("sha256"):
            parser.error(f"{name} 与准备模型时的记录不同，请检查模型目录")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    torch.set_num_threads(4)
    processor = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    model, loading_info = SegformerForSemanticSegmentation.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False, use_safetensors=True, output_loading_info=True,
    )
    if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"权重未完整匹配模型：{loading_info}")
    loading_info = {key: sorted(value) if isinstance(value, set) else value for key, value in loading_info.items()}
    if model.config.num_labels != 150:
        raise ValueError("本脚本只用于 ADE20K 150 类的预训练试跑")
    model.to(device).eval()
    palette = visual_palette(model.config.num_labels)
    classes = [{"index": index, "name": model.config.id2label[index], "rgb": palette[index].tolist()}
               for index in range(model.config.num_labels)]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "classes.json", classes)
    print(f"设备：{device}；模型输入：{args.size}×{args.size}；待处理：{len(paths)} 张", flush=True)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def prepare(image):
        return processor(images=image, size={"height": args.size, "width": args.size}, return_tensors="pt").to(device)

    with Image.open(paths[0]) as opened:
        first = ImageOps.exif_transpose(opened).convert("RGB")
    benchmark_inputs = prepare(first)
    durations = []
    with torch.inference_mode():
        for _ in range(3):
            model(**benchmark_inputs)
        sync()
        for _ in range(args.benchmark_repeats):
            start = time.perf_counter()
            model(**benchmark_inputs)
            sync()
            durations.append((time.perf_counter() - start) * 1000)
        feature_output = model(**benchmark_inputs, output_hidden_states=True)
        features = [{"shape": list(tensor.shape), "downsample_factor": args.size // tensor.shape[-1]}
                    for tensor in feature_output.hidden_states]
        raw_logits_shape = list(feature_output.logits.shape)
        del feature_output, benchmark_inputs
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        rows = []
        for number, path in enumerate(paths, 1):
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            sync()
            start = time.perf_counter()
            inputs = prepare(image)
            outputs = model(**inputs)
            if not torch.isfinite(outputs.logits).all():
                raise RuntimeError(f"模型输出包含 NaN/Inf：{path}")
            prediction = processor.post_process_semantic_segmentation(outputs, target_sizes=[(image.height, image.width)])[0]
            prediction = prediction.cpu().numpy().astype(np.uint8)
            sync()
            processing_ms = (time.perf_counter() - start) * 1000
            if prediction.shape != (image.height, image.width) or prediction.max() >= model.config.num_labels:
                raise RuntimeError("输出尺寸或类别编号异常")
            del outputs, inputs
            counts = np.bincount(prediction.ravel(), minlength=model.config.num_labels)
            predicted_classes = [{**classes[int(index)], "pixels": int(counts[index]), "fraction": float(counts[index] / counts.sum())}
                                 for index in np.argsort(-counts) if counts[index] > 0]
            folder = f"{number:02d}"
            item_dir = output / folder
            item_dir.mkdir()
            colors = Image.fromarray(palette[prediction])
            overlay = Image.blend(image, colors, 0.45)
            image.save(item_dir / "input.png")
            Image.fromarray(prediction).save(item_dir / "prediction_index.png")
            colors.save(item_dir / "prediction_color.png")
            overlay.save(item_dir / "overlay.png")
            row = {"number": number, "folder": folder, "input": str(path), "input_sha256": sha256(path),
                   "width": image.width, "height": image.height, "processing_ms": processing_ms,
                   "processing_time_definition": "预处理、复制到 GPU、模型前向、logits 上采样、argmax、复制到 CPU；不含读盘和保存",
                   "predicted_classes": predicted_classes}
            save_panel(image, colors, overlay, row, item_dir / "comparison.png")
            write_json(item_dir / "summary.json", row)
            rows.append(row)
            print(f"[{number}/{len(paths)}] {path.name}：{processing_ms:.1f} ms；主要预测 {predicted_classes[0]['name']}", flush=True)

    summary = {
        "experiment": "segformer_b0_ade20k_pretrained_inference", "created_at": datetime.now().astimezone().isoformat(),
        "model_id": MODEL_ID, "revision": REVISION, "model_dir": str(args.model_dir.resolve()),
        "weights_sha256": sha256(args.model_dir / "model.safetensors"), "loading_info": loading_info,
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "python": platform.python_version(), "torch": str(torch.__version__), "transformers": transformers.__version__,
        "dtype": str(next(model.parameters()).dtype), "parameters": sum(p.numel() for p in model.parameters()),
        "input_size": args.size, "preprocessing": "RGB；按权重配置归一化；直接缩放到指定正方形尺寸，未使用等比补边",
        "processor_class": type(processor).__name__, "num_classes": model.config.num_labels,
        "feature_maps": features, "raw_logits_shape": raw_logits_shape,
        "benchmark": {"warmup": 3, "repeats": args.benchmark_repeats,
                      "forward_ms_median": float(np.median(durations)), "forward_ms_mean": float(np.mean(durations)),
                      "forward_ms_p95": float(np.percentile(durations, 95)), "forward_ms_samples": durations,
                      "definition": "单张已预处理输入；FP32；包含模型前向和 CUDA 同步，不含预处理、后处理和文件读写"},
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == "cuda" else None,
        "memory_definition": "图片处理阶段 PyTorch 显存峰值，包含模型；不包含其他应用、驱动和部分 CUDA 上下文开销",
        "selection": "用户指定图片" if args.inputs else f"作者仓库示例 + SUIM 验证集按清单顺序均匀抽取 {len(paths)-1} 张；未按效果挑选",
        "images": rows, "accuracy_metrics": None,
        "note": "这是 Transformers 实现及 NVIDIA 发布权重的推理验证；未微调 SUIM，未复现论文训练或全量 ADE20K 评测。150 类与 SUIM 8 类不对应，不报告 SUIM mIoU。",
    }
    write_json(output / "summary.json", summary)
    panels = []
    for row in rows[:5]:
        with Image.open(output / row["folder"] / "comparison.png") as panel:
            panels.append(panel.copy())
    overview = Image.new("RGB", (1200, 60 + 358 * len(panels)), "#f4f6f8")
    ImageDraw.Draw(overview).text((14, 14), "SegFormer-B0 · ADE20K 预训练试跑（非 SUIM 八类结果）", font=font(23), fill="#172b3a")
    for index, panel in enumerate(panels):
        overview.paste(panel, (0, 60 + index * 358))
    overview.save(output / "comparison.png")
    save_report(output, summary)
    print(f"完成。报告：{output / 'index.html'}\n汇总图：{output / 'comparison.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
