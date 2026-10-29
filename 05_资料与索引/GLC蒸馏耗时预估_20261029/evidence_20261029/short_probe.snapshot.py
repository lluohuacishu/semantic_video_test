"""Bounded student-step timing probe. Uses synthetic data and zero learning rate.

No official checkpoints are loaded or modified. No trained student is saved.
The VGG case is an untrained structural compute surrogate, not LPIPS quality.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
import statistics
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", "utf-8")
    temp.replace(path)


def gpu_snapshot():
    try:
        value = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader"], text=True, timeout=5)
        return value.strip()
    except Exception as error:
        return type(error).__name__


def checksum(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class VGGFeatureSurrogate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models.vgg import cfgs, make_layers
        self.features = make_layers(cfgs["D"])
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def forward(self, value):
        outputs = []
        for index, layer in enumerate(self.features):
            value = layer(value)
            if index in (3, 8, 15, 22, 29):
                outputs.append(value)
        return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=240)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    origin = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(29)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    free, total = torch.cuda.mem_get_info()
    fraction = min(0.45, max(0.1, (free - 1024**3) / total))
    torch.cuda.set_per_process_memory_fraction(fraction)
    module_path = args.upstream / "src/models/vqgan_arch.py"
    spec = importlib.util.spec_from_file_location("glc_probe_vqgan", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = dict(
        date="2026-10-29", device=torch.cuda.get_device_name(), torch=torch.__version__,
        precision_policy="FP32 parameters; TF32 disabled; optional BF16 autocast",
        cudnn_benchmark=False, cpu_threads=1,
        synthetic_data=True, official_checkpoints_loaded=False, learned_model_saved=False,
        optimizer="AdamW lr=0 weight_decay=0; optimizer moments updated, model parameters unchanged",
        teacher="assumed cached offline; teacher forward not in measured step",
        io="input/targets preloaded on GPU; disk loader and H2D excluded",
        perceptual="untrained frozen VGG16 feature stack; includes two forwards and student-output gradient; NOT LPIPS quality validation",
        memory_fraction_limit=fraction, start_gpu=gpu_snapshot(), cases=[])
    write_json(args.output / "timings.json", result)
    bf16 = torch.cuda.is_bf16_supported()
    cases = [
        ("pixel_fp32_256_b1", 256, 256, 1, False, False),
        ("pixel_bf16_256_b1", 256, 256, 1, True, False),
        ("vgg_bf16_256_b1", 256, 256, 1, True, True),
        ("vgg_bf16_256_b2", 256, 256, 2, True, True),
        ("vgg_bf16_384_b1", 384, 384, 1, True, True),
        ("vgg_bf16_512x320_b1", 512, 320, 1, True, True),
    ]
    for name, width, height, batch, mixed, perceptual in cases:
        if time.perf_counter() - origin > args.max_seconds:
            result["budget_stopped"] = True
            break
        row = dict(name=name, width=width, height=height, batch=batch,
                   bf16=mixed, perceptual=perceptual, before_gpu=gpu_snapshot())
        model = optimizer = feature = inputs = targets = None
        try:
            if mixed and not bf16:
                row["status"] = "bf16_unavailable"
                continue
            free_now, _ = torch.cuda.mem_get_info()
            if free_now < 1400 * 1024**2:
                row["status"] = "insufficient_free_device_memory"
                continue
            model = module.Generator(64, 256, [1, 1, 2, 2, 4], 2, 256, [16],
                                     swish_last=True, quant_conv=True, patch=32).cuda().train()
            row["parameters"] = sum(parameter.numel() for parameter in model.parameters())
            assert row["parameters"] == 8800707
            before_hash = checksum(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.0)
            feature = VGGFeatureSurrogate().cuda() if perceptual else None
            inputs = torch.randn(batch, 256, height // 16, width // 16, device="cuda")
            targets = torch.randn(batch, 3, height, width, device="cuda").tanh()

            def step():
                optimizer.zero_grad(set_to_none=True)
                context = torch.autocast("cuda", dtype=torch.bfloat16) if mixed else nullcontext()
                with context:
                    output = model(inputs)
                    loss = F.l1_loss(output.float(), targets)
                    if feature is not None:
                        with torch.no_grad():
                            target_features = feature(targets)
                        output_features = feature(output)
                        loss = loss + 0.1 * sum(
                            F.l1_loss(left.float(), right.float())
                            for left, right in zip(output_features, target_features))
                loss.backward()
                optimizer.step()
                return loss

            for _ in range(5):
                loss = step()
            torch.cuda.synchronize()
            if not torch.isfinite(loss).item():
                raise FloatingPointError("non-finite loss during warmup")
            row["all_gradients_finite"] = all(
                parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                for parameter in model.parameters())
            if not row["all_gradients_finite"]:
                raise FloatingPointError("non-finite gradients")
            torch.cuda.reset_peak_memory_stats()
            durations = []
            tick_case = time.perf_counter()
            for _ in range(24):
                torch.cuda.synchronize()
                tick = time.perf_counter()
                loss = step()
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - tick)
                if len(durations) >= 10 and time.perf_counter() - tick_case > 25:
                    break
            row.update(
                status="ok", timed_steps=len(durations),
                mean_seconds=statistics.mean(durations),
                median_seconds=statistics.median(durations),
                min_seconds=min(durations), max_seconds=max(durations),
                mean_samples_per_second=batch / statistics.mean(durations),
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                loss_finite=bool(torch.isfinite(loss).item()),
                parameters_unchanged=before_hash == checksum(model),
                durations_seconds=durations,
            )
            assert row["parameters_unchanged"], "Zero learning rate unexpectedly changed parameters"
            print(json.dumps({key: value for key, value in row.items() if key != "durations_seconds"}), flush=True)
        except torch.cuda.OutOfMemoryError as error:
            row.update(status="bounded_cuda_oom", error=str(error).splitlines()[0])
            print(json.dumps({"name":name,"status":row["status"]}), flush=True)
        except Exception as error:
            row.update(status="error", error=type(error).__name__ + ": " + str(error))
            print(json.dumps({"name":name,"status":row["status"],"error":row["error"]}), flush=True)
        finally:
            model = optimizer = feature = inputs = targets = None
            if "loss" in locals():
                del loss
            if "step" in locals():
                del step
            gc.collect()
            torch.cuda.empty_cache()
            row["after_gpu"] = gpu_snapshot()
            result["cases"].append(row)
            write_json(args.output / "timings.json", result)
    result.update(end_gpu=gpu_snapshot(), wall_seconds=time.perf_counter()-origin, complete=True)
    write_json(args.output / "timings.json", result)


if __name__ == "__main__":
    main()

