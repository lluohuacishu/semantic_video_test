"""Bounded, supervised web adapter for existing D2 lowpass / prediction rules."""
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

from c2_watch import Heartbeat, write
from c5_enhancement import MODES, SEMANTIC, mode_option, transform
from c5_timebase import TimedFrames, prepare


def run(config, heartbeat):
    import numpy as np
    from PIL import Image
    mode = mode_option(config["mode"])
    output, stop = Path(config["output"]), Path(config["stop_file"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    def cancelled():
        if stop.exists():
            raise InterruptedError("预处理已停止")
    cancelled()
    tick = time.perf_counter()
    opts = config["options"]
    plan = prepare(Path(config["input"]), Path(config["ffmpeg"]), opts["preset"], opts.get("clip"),
                   cancelled, video=opts.get("video"))
    model, context = None, nullcontext()
    if mode in SEMANTIC:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "09_新阶段D_语义目标增强"))
        from infer_d1 import load_model, tiled_fish, letterbox_diver, torch
        from d1_protection import hard_protection, alpha_protection
        torch.set_num_threads(4)
        torch.manual_seed(20261013)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        cancelled()
        heartbeat.update(state="loading_semantic_model")
        model = load_model(Path(config["checkpoint"]))
        context = torch.inference_mode()
    args = argparse.Namespace(input=Path(config["input"]), ffmpeg=Path(config["ffmpeg"]),
                              time_plan=plan, workdir=output)
    rows, previous = [], None
    reader = TimedFrames(args, cancel=cancelled)
    try:
        with context:
            for index, rgb in enumerate(reader):
                cancelled()
                alpha, fallback, seed_fraction, hard_fraction = None, None, None, None
                if model is not None:
                    # Inference sees the native source, exactly as D2; filtering sees delivery resolution.
                    original = reader.last.copy()
                    fish = tiled_fish(model, original) >= .5
                    labels, confidence = letterbox_diver(model, Image.fromarray(original))
                    seed = fish | ((labels == 1) & (confidence >= .5))
                    hard = hard_protection(seed, previous, {"margin_at_1920": 16, "previous_seed_frames": 1})
                    alpha_native, fallback = alpha_protection(seed, hard)
                    previous = seed
                    alpha = np.asarray(Image.fromarray(alpha_native).resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST))
                    seed_fraction, hard_fraction = float(seed.mean()), float(hard.mean())
                coding = transform(rgb, mode, alpha, fallback)
                Image.fromarray(coding).save(output / "frames" / f"{index:05d}.png", compress_level=1)
                rows.append({"index": index, "source_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                             "coding_rgb_sha256": hashlib.sha256(coding.tobytes()).hexdigest(),
                             "fallback": fallback, "seed_fraction": seed_fraction, "hard_fraction": hard_fraction,
                             "alpha_mean": float(alpha.mean() / 255) if alpha is not None else None})
                heartbeat.update(state="preprocessing", last_progress={"frame_completed": index + 1, "frames": plan["frames"]})
                print(f"preprocess {index + 1}/{plan['frames']} {mode} fallback={fallback}", flush=True)
    finally:
        reader.close()
    cancelled()
    if len(rows) != plan["frames"]:
        raise ValueError("Preprocessing ended before all planned frames were read")
    fallbacks = Counter(r["fallback"] for r in rows if r["fallback"])
    summary = {"state": "completed", "mode": mode, "label": MODES[mode], "frames": len(rows),
               "size": [plan["width"], plan["height"]], "seconds": time.perf_counter() - tick,
               "semantic": mode in SEMANTIC, "active_frames": len(rows) - sum(fallbacks.values()) if mode in SEMANTIC else 0,
               "fallback_frames": sum(fallbacks.values()), "fallback_reasons": dict(fallbacks),
               "mask_transmitted": False, "ground_truth_used": False, "training": False,
               "psnr_reference": "original sampled/resized RGB before lowpass or semantic protection",
               "rules": {"background_sigma": 1.2, "target_sigma": .6 if mode == "semantic_light" else 0,
                         "fish_threshold": .5, "diver_threshold": .5, "margin_at_1920": 16,
                         "previous_seed_frames": 1, "empty_or_over_half": "original RGB"} if mode in SEMANTIC else None}
    write(output / "prepared.json", {"summary": summary, "records": rows})
    heartbeat.update(state="completed", summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    config = json.loads(parser.parse_args().config.read_text("utf-8"))
    with Heartbeat(config["process_folder"]) as beat:
        try:
            run(config, beat)
        except BaseException as exc:
            beat.update(state="cancelled" if isinstance(exc, InterruptedError) else "failed", error=str(exc))
            traceback.print_exc()
            return 2 if isinstance(exc, InterruptedError) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
