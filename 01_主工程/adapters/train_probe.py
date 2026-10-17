"""Small pre-trained fine-tuning feasibility probe, not a research training run."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

from common import ROOT, torch, F, Image, configure, load_models, rgb_tensor, prepare, gray_dpb, save_json, sha256, environment, sync


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="Two continuous reference RGB PNGs, directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--weight", default=str(ROOT / "checkpoints/mlvc-s-psnr-v1.ckpt"))
    parser.add_argument("--i-weight")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--q-index", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--checkpointing", action="store_true")
    args = parser.parse_args()
    configure("cuda")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    paths = sorted(Path(args.frames).glob("*.png"))[:2]
    if len(paths) != 2:
        raise ValueError("Need two consecutive frames")
    (output / "crop_rgb").mkdir()
    for idx, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        width, height = image.size
        left, top = (width - args.crop)//2, (height - args.crop)//2
        if left < 0 or top < 0:
            raise ValueError("Crop exceeds source image")
        image.crop((left, top, left+args.crop, top+args.crop)).save(output / "crop_rgb" / f"{idx:05d}.png")
    model, i_model, _ = load_models(args.weight, args.i_weight, entropy=False)
    xs = [prepare(rgb_tensor(output / "crop_rgb" / f"{idx:05d}.png"), model.padding_size) for idx in range(2)]
    with torch.no_grad():
        if i_model:
            reference = i_model(xs[0], q_index=args.q_index)["x_hat"].detach()
            dpb = {"ref_frame": reference, "ref_feature": None}
        else:
            dpb = model(xs[0], gray_dpb(xs[0].shape), q_index=args.q_index, fa_idx=1)["dpb"]
            dpb = {k: v.detach() if v is not None else None for k, v in dpb.items()}
    del i_model
    model.train()
    if args.checkpointing:
        model.set_use_ckpt(True, recon_module=True, hyper_module=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lam = 768 ** (args.q_index / 63)
    config = {"purpose": "feasibility only; fixed two-frame sample, not held-out evaluation", "batch_size": 1,
              "frames": 2, "gradient_frames": 1, "reference": "frozen DMCI reconstruction" if args.i_weight else "detached MLVC gray-initialized reconstruction",
              "crop": args.crop, "q_index": args.q_index, "lambda": lam, "rate_weight": 1.0,
              "loss": "bpp + lambda*(6*MSE_Y + MSE_U420 + MSE_V420)/8", "lr": args.lr,
              "precision": "fp32", "checkpointing": args.checkpointing, "steps": args.steps,
              "p_weight_sha256": sha256(args.weight), "environment": environment()}
    save_json(output / "config.json", config)
    initial = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    logs = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            rd = model(xs[1], dpb, q_index=args.q_index, fa_idx=1 if args.i_weight else 0)
            error_y = F.mse_loss(rd["x_hat"][:, :1], xs[1][:, :1])
            error_uv = (F.avg_pool2d(rd["x_hat"][:, 1:], 2) - F.avg_pool2d(xs[1][:, 1:], 2)).square().mean(dim=(0, 2, 3))
            distortion = (6*error_y + error_uv.sum())/8
            rate = rd["bpp"].mean()
            loss = rate + lam*distortion
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            loss.backward()
            nonzero_grads = sum(p.grad is not None and torch.count_nonzero(p.grad).item() > 0 for p in model.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm) or nonzero_grads == 0:
                raise RuntimeError("Invalid or zero gradients")
            optimizer.step()
            sync()
            record = {"step": step+1, "loss": loss.item(), "bpp_estimate": rate.item(), "distortion_yuv420": distortion.item(),
                      "grad_norm_before_clip": grad_norm.item(), "parameters_with_nonzero_gradient": nonzero_grads,
                      "allocated_bytes": torch.cuda.memory_allocated(), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            logs.append(record)
            save_json(output / "steps.json", logs)
            print(record, flush=True)
        changed = [name for name, value in model.named_parameters() if not torch.equal(initial[name], value.detach().cpu())]
        if not changed:
            raise RuntimeError("Optimizer did not change parameters")
        ckpt = output / "probe_finetuned.ckpt"
        torch.save({"state_dict": model.state_dict(), "probe": config}, ckpt)
        save_json(output / "result.json", {"success": True, "effective_parameter_update": True, "changed_parameters": len(changed),
                    "first_loss": logs[0]["loss"], "last_loss": logs[-1]["loss"],
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "seconds": time.perf_counter()-start, "checkpoint_sha256": sha256(ckpt),
                    "changed_parameter_names": changed, "limitation": "Only one fixed pair and one QP; does not establish convergence, generalization or full-sequence training feasibility."})
    except Exception as exc:
        save_json(output / "failure.json", {"error": repr(exc), "completed_steps": len(logs),
                     "allocated_bytes": torch.cuda.memory_allocated(), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                     "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        raise


if __name__ == "__main__":
    main()
