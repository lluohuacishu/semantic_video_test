"""在相同缓存输入上比较两个分割模型的 GPU 前向耗时。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from segmentation import SegmentationDataset, load_segmentation_model, manifest_info, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/SUIM/manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("此基准使用 CUDA Event，需要可用的 CUDA")
    if args.repeats < 10:
        parser.error("至少重复 10 次")
    if args.output.exists():
        parser.error("基准结果文件已存在，不覆盖")
    torch.set_num_threads(4)
    root, metadata, digest = manifest_info(args.manifest)
    device = torch.device("cuda")
    results = []
    expected_size = None
    for label, checkpoint in (("before", args.baseline), ("after", args.candidate)):
        model, state = load_segmentation_model(checkpoint, device)
        if state["manifest_sha256"] != digest:
            raise ValueError("模型的数据清单与当前清单不一致")
        if expected_size is not None and state["size"] != expected_size:
            raise ValueError("模型输入尺寸不同，不能作为同尺寸速度比较")
        expected_size = state["size"]
        dataset = SegmentationDataset(root, metadata["splits"]["validation"][:1], state["size"])
        image = dataset[0][0].unsqueeze(0).to(device)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for _ in range(10):
                model(image)
            torch.cuda.synchronize()
            timings = []
            for _ in range(args.repeats):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                model(image)
                end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end))
        results.append({
            "label": label, "model_name": state["model_name"], "checkpoint": str(checkpoint.resolve()),
            "batch_size": 1, "size": state["size"], "precision": "float32", "memory_format": "contiguous",
            "mean_ms": float(np.mean(timings)), "median_ms": float(np.median(timings)),
            "p95_ms": float(np.percentile(timings, 95)), "repeats": args.repeats,
            "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated()/1024**2, 1),
        })
        del model, image
        torch.cuda.empty_cache()
    report = {
        "gpu": torch.cuda.get_device_name(), "results": results,
        "source_image": metadata["splits"]["validation"][0]["image"], "manifest_sha256": digest,
        "note": "CUDA Event、预热后、缓存单图输入，仅模型前向。包括模型内部归一化和输出上采样；不含磁盘读取、预处理、显存传入传出、原图尺寸恢复、视频编解码或网络。不是端到端实时延迟。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    for result in results:
        print(f"{result['label']}: median={result['median_ms']:.2f}ms, p95={result['p95_ms']:.2f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
