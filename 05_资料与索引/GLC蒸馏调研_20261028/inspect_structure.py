"""Inspect existing GLC weights on CPU and candidate structures on meta.

No CUDA operation, model forward, backward, optimizer or trained model output.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import torch


def inspect_checkpoints(weights):
    rows = []
    for filename in ("GLC_image.pth.tar", "GLC_Video.pth.tar"):
        raw = torch.load(weights / filename, map_location="cpu", weights_only=True, mmap=True)
        wrappers = []
        for wrapper in ("state_dict", "net", "net_g"):
            if wrapper in raw:
                raw = raw[wrapper]
                wrappers.append(wrapper)
        state = {key.removeprefix("module."): value for key, value in raw.items()}
        groups = {}
        for prefix in ("vqgan.generator.", "vqgan.encoder.", "code_pred_loss.", "code_pred_pix_loss."):
            tensors = {key: value for key, value in state.items()
                       if key.startswith(prefix) and isinstance(value, torch.Tensor)}
            groups[prefix] = {
                "tensors": len(tensors),
                "elements": sum(value.numel() for value in tensors.values()),
                "tensor_bytes": sum(value.numel() * value.element_size() for value in tensors.values()),
            }
        generator = {key: value for key, value in state.items()
                     if key.startswith("vqgan.generator.")}
        if not generator or not groups["vqgan.encoder."]["tensors"]:
            raise ValueError(f"Missing generator/encoder tensors in {filename}")
        digest = hashlib.sha256()
        for key, value in sorted(generator.items()):
            digest.update(key.encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.contiguous().numpy().tobytes())
        rows.append({
            "checkpoint": filename, "unwrapped_keys": wrappers, "groups": groups,
            "generator_state_sha256": digest.hexdigest(),
            "generator_dtypes": sorted({str(value.dtype) for value in generator.values()}),
        })
        del raw, state, generator
        gc.collect()
    return {
        "inspection": "CPU checkpoint tensor inspection only; no inference, backward or optimizer",
        "torch": torch.__version__, "checkpoints": rows,
        "image_and_video_generator_identical":
            rows[0]["generator_state_sha256"] == rows[1]["generator_state_sha256"],
    }


def inspect_candidates(upstream):
    source = upstream / "src/models/vqgan_arch.py"
    spec = importlib.util.spec_from_file_location("glc_vqgan_structure_inspection", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for name, width, blocks in (
        ("teacher_structure", 128, 3),
        ("candidate_width64_blocks2", 64, 2),
        ("candidate_width64_blocks1", 64, 1),
    ):
        with torch.device("meta"):
            model = module.Generator(width, 256, [1, 1, 2, 2, 4], blocks, 256, [16],
                                     swish_last=True, quant_conv=True, patch=32)
        count = sum(parameter.numel() for parameter in model.parameters())
        rows.append({
            "name": name, "base_width": width, "resblocks_per_scale": blocks,
            "parameters": count, "fp32_parameter_bytes": count * 4,
            "input_channels": 256, "output_channels": 3, "spatial_upsample": 16,
        })
    assert rows[0]["parameters"] == 42515587
    for row in rows:
        row["fraction_of_teacher_parameters"] = row["parameters"] / rows[0]["parameters"]
    return {
        "method": "meta-device structure construction and parameter counting only; no forward, backward or trained student weights",
        "candidates": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Existing official GLC checkpoint directory")
    parser.add_argument("--upstream", type=Path, required=True, help="Existing GLC source checkout")
    args = parser.parse_args()
    print(json.dumps({
        "weights": inspect_checkpoints(args.weights),
        "candidate_structures": inspect_candidates(args.upstream),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
