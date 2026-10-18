"""Read-only source/received-RGB diagnostics and synchronized B2 viewing copies.

Runs outside the receiver. Source predictions are not ground truth.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from b2_common import HERE, Journal, git, read, require, sha, write


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def mask(path):
    with Image.open(path) as image:
        return np.asarray(image) > 0


def score(squared, samples):
    return 10 * math.log10(255 ** 2 * samples / squared) if squared > 0 and samples else None


def measurements(sender, receiver, config, output):
    manifest = read(receiver / "received_manifest.json")
    source_audit = read(sender / "source_audit.json")
    n, w, h = manifest["frames"], manifest["width"], manifest["height"]
    targets = list(manifest["targets"])
    names = ["full", "outside_transmitted_ROI", "inside_transmitted_ROI", "fixed_stripe_crop"]
    names += [f"{target}_{kind}_source_prediction" for target in targets for kind in ("raw", "kept")]
    aggregates = {method: {name: [0.0, 0] for name in names} for method in ("GLC_base", "GLC_auto_ROI")}
    previous = {}
    temporal = {method: {"all": [], "window_boundaries": [], "non_boundaries": [], "fixed_stripe_crop": []} for method in aggregates}
    boundary = {method: [] for method in aggregates}
    frame_rows, coverage_rows = [], []
    x1, y1, x2, y2 = config["fixed_diagnostic_crop_xyxy"]
    focus_start, focus_end = config["slow_view_source_frame_range"]
    for index in range(n):
        source = rgb(sender / "private_frames" / f"{index:05d}.png")
        require(hashlib.sha256(source.tobytes()).hexdigest() == source_audit["mapping"][index]["rgb_sha256"], "Evaluation source changed")
        images = {"GLC_base": rgb(receiver / "base_decoded/frames" / f"im{index + 1:05d}.png"),
                  "GLC_auto_ROI": rgb(receiver / "rgb" / f"{index:05d}.png")}
        rectangle = np.zeros((h, w), dtype=bool)
        active = [patch for patch in manifest["patches"] if patch["start"] <= index < patch["start"] + patch["frames"]]
        for patch in active:
            x, y, pw, ph = patch["box"]
            rectangle[y:y+ph, x:x+pw] = True
        focus = np.zeros((h, w), dtype=bool)
        if focus_start <= index <= focus_end:
            focus[y1:y2, x1:x2] = True
        masks = {"full": np.ones((h, w), dtype=bool), "outside_transmitted_ROI": ~rectangle,
                 "inside_transmitted_ROI": rectangle, "fixed_stripe_crop": focus}
        coverage = {"frame": index, "transmitted_rectangle_pixels": int(rectangle.sum()), "rectangle_fraction": float(rectangle.mean())}
        for target in targets:
            for kind in ("raw", "kept"):
                target_mask = mask(sender / "private_masks" / target / kind / f"{index:05d}.png")
                masks[f"{target}_{kind}_source_prediction"] = target_mask
                count, covered = int(target_mask.sum()), int(np.count_nonzero(target_mask & rectangle))
                coverage[f"{target}_{kind}_pixels"] = count
                coverage[f"{target}_{kind}_covered_pixels"] = covered
                coverage[f"{target}_{kind}_coverage"] = covered / count if count else None
        require(all(coverage[f"{target}_kept_pixels"] == coverage[f"{target}_kept_covered_pixels"] for target in targets), "ROI plan omitted a kept predicted pixel")
        coverage_rows.append(coverage)
        row = {"frame": index, "source_frame": source_audit["mapping"][index]["source_frame"], "is_window_boundary": index > 0 and index % config["roi"]["segment_frames"] == 0}
        for method, image in images.items():
            error = image.astype(np.float32) - source.astype(np.float32)
            squared = np.square(error)
            for name, selected in masks.items():
                total, samples = float(squared[selected].sum(dtype=np.float64)), int(selected.sum()) * 3
                aggregates[method][name][0] += total
                aggregates[method][name][1] += samples
                row[f"{method}_{name}_psnr_db"] = score(total, samples)
            if method in previous:
                change = np.abs(error - previous[method])
                value = float(change.mean(dtype=np.float64))
                temporal[method]["all"].append(value)
                temporal[method]["window_boundaries" if row["is_window_boundary"] else "non_boundaries"].append(value)
                row[f"{method}_error_change_mae"] = value
                if focus_start < index <= focus_end:
                    temporal[method]["fixed_stripe_crop"].append(float(change[y1:y2, x1:x2].mean(dtype=np.float64)))
            previous[method] = error
            edge_values = []
            for patch in active:
                x, y, pw, ph = patch["box"]
                if x > 0:
                    edge_values.append(np.abs(error[y:y+ph, x] - error[y:y+ph, x-1]).ravel())
                if x + pw < w:
                    edge_values.append(np.abs(error[y:y+ph, x+pw-1] - error[y:y+ph, x+pw]).ravel())
                if y > 0:
                    edge_values.append(np.abs(error[y, x:x+pw] - error[y-1, x:x+pw]).ravel())
                if y + ph < h:
                    edge_values.append(np.abs(error[y+ph-1, x:x+pw] - error[y+ph, x:x+pw]).ravel())
            if edge_values:
                value = float(np.concatenate(edge_values).mean(dtype=np.float64))
                boundary[method].append(value)
                row[f"{method}_ROI_edge_error_difference_mae"] = value
        frame_rows.append(row)
    metric_rows = []
    for method in aggregates:
        metric_rows.append({"method": method,
                           "rgb_psnr_db": {name: score(*totals) for name, totals in aggregates[method].items()},
                           "sample_counts_RGB_values": {name: totals[1] for name, totals in aggregates[method].items()},
                           "error_change_mae_0_255": {name: float(np.mean(values)) if values else None for name, values in temporal[method].items()},
                           "ROI_edge_error_difference_mae_0_255": float(np.mean(boundary[method])) if boundary[method] else None})
    coverage_summary = {}
    for target in targets:
        item = {}
        for kind in ("raw", "kept"):
            count = sum(r[f"{target}_{kind}_pixels"] for r in coverage_rows)
            covered = sum(r[f"{target}_{kind}_covered_pixels"] for r in coverage_rows)
            item[kind] = {"pixels": count, "covered_pixels": covered, "coverage": covered / count if count else None,
                          "present_frames": sum(r[f"{target}_{kind}_pixels"] > 0 for r in coverage_rows)}
        coverage_summary[target] = item
    write(output / "per_frame_metrics.json", frame_rows)
    write(output / "prediction_coverage.json", {"summary": coverage_summary, "frames": coverage_rows,
          "meaning": "Coverage of source model predictions only; no ground-truth target recall or physical object count."})
    with (output / "per_frame_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = sorted({k for row in frame_rows for k in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(frame_rows)
    result = {"frames": n, "data_role": "previously_used_development_clip", "methods": metric_rows, "source_prediction_coverage": coverage_summary,
              "fixed_crop_xyxy": config["fixed_diagnostic_crop_xyxy"], "fixed_crop_source_frames": config["slow_view_source_frame_range"],
              "ground_truth_available": False, "human_viewing_scores": None, "matched_budget_control": "B3 pending",
              "limitations": ["ROI masks are model predictions used for transmission selection; their PSNR is not independent task accuracy.",
                             "Temporal error changes and ROI-edge error differences are diagnostics, not validated perceptual flicker scores.",
                             "No ground-truth recall, object-count accuracy, diver quality score, or semantic-budget advantage is inferred."]}
    write(output / "metrics.json", result)
    return result


def font(size=19):
    return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)


def panel(image, title, text_font):
    output = Image.new("RGB", (image.width, image.height + 36), "#101e2a")
    output.paste(image, (0, 36))
    ImageDraw.Draw(output).text((10, 5), title, font=text_font, fill="white")
    return output


def comparison_frame(sender, receiver, index, crop=None):
    paths = [sender / "private_frames" / f"{index:05d}.png", receiver / "base_decoded/frames" / f"im{index + 1:05d}.png", receiver / "rgb" / f"{index:05d}.png"]
    titles = ["源图", "GLC 基础层", "GLC + 自动 ROI（增加字节）"]
    canvas = Image.new("RGB", (1920, 396))
    for number, (path, title) in enumerate(zip(paths, titles)):
        with Image.open(path) as image:
            image = image.convert("RGB")
            if crop:
                image = image.crop(tuple(crop))
            image = image.resize((640, 360), Image.Resampling.NEAREST if crop else Image.Resampling.LANCZOS)
        canvas.paste(panel(image, f"{title} · 源帧 {index}", font()), (number * 640, 0))
    return canvas


def roi_frame(sender, manifest, semantics, index):
    with Image.open(sender / "private_frames" / f"{index:05d}.png") as image:
        image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    for patch in manifest["patches"]:
        if patch["start"] <= index < patch["start"] + patch["frames"]:
            x, y, w, h = patch["box"]
            draw.rectangle((x, y, x+w-1, y+h-1), outline="white", width=3)
    for target, predicted in semantics["frames"][index]["targets"].items():
        color = (60, 240, 130) if target == "fish" else (255, 150, 50)
        for region in predicted["regions"]:
            x, y, w, h = region["box"]
            draw.rectangle((x, y, x+w-1, y+h-1), outline=color, width=2)
    return panel(image, f"源端模型预测（非GT）· 白框=传输ROI · 绿框=鱼 · 橙框=潜水员 · 源帧 {index}", font(23))


def movie(ffmpeg, output, frames, fps, make_frame, journal, purpose):
    sample = make_frame(frames[0])
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{sample.width}x{sample.height}",
               "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "11",
               "-vf", "scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p", "-pix_fmt", "yuv420p",
               "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv", "-movflags", "+faststart", output]
    argv = list(map(str, command))
    entry = journal.begin(argv)
    tick = time.perf_counter()
    log = output.with_suffix(".ffmpeg.log")
    with log.open("x", encoding="utf-8") as handle:
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=handle, stderr=subprocess.STDOUT)
        try:
            for index in frames:
                process.stdin.write(make_frame(index).tobytes())
            process.stdin.close()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    entry["log"] = str(log)
    journal.finish(entry, code, time.perf_counter() - tick)
    require(code == 0, f"Viewing encode failed: {log}")
    probe = json.loads(journal.run([Path(ffmpeg).with_name("ffprobe.exe"), "-v", "error", "-select_streams", "v:0", "-count_frames",
                     "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames,duration", "-of", "json", output]))["streams"][0]
    from fractions import Fraction
    require(int(probe["nb_read_frames"]) == len(frames) and Fraction(probe["avg_frame_rate"]) == Fraction(str(fps)), "Viewing frame count/rate mismatch")
    journal.run([ffmpeg, "-v", "error", "-i", output, "-f", "null", "-"])
    return {"path": str(output), "bytes": output.stat().st_size, "sha256": sha(output), "source_frame_range": [frames[0], frames[-1]],
            "frames": len(frames), "fps": fps, "playback_speed": fps / 15, "probe": probe, "purpose": purpose,
            "encoding": "H.264 CRF11 YUV420 BT709 limited; viewing only, not transmitted or used for metrics"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rate", type=int, default=40)
    parser.add_argument("--network", default="network")
    args = parser.parse_args()
    root = args.run.resolve()
    sender = root / "sender"
    result = read(root / args.network / f"rate_{args.rate}kbps/result.json")
    require(result["status"] == "passed", "Evaluate only accepted received RGB")
    receiver = root / args.network / f"rate_{args.rate}kbps/receiver"
    saved = read(sender / "config.json")
    config, baseline = saved["b2"], saved["baseline"]
    output = root / "evaluation"
    output.mkdir()
    journal = Journal(output)
    invocation = journal.begin([sys.executable, *sys.argv], cwd=str(Path.cwd()), code_commit=git("rev-parse", "HEAD"), evaluator_script_sha256=sha(__file__))
    tick = time.perf_counter()
    try:
        print("B2 measuring received RGB and predicted-region coverage", flush=True)
        measurements(sender, receiver, config, output)
        manifest, semantics = read(receiver / "received_manifest.json"), read(receiver / "target_information.json")
        views = []
        ffmpeg = baseline["runtime_dependencies"]["ffmpeg"]
        views.append(movie(ffmpeg, output / "video91_原图_GLC_自动ROI_同步对比.mp4", list(range(250)), 15,
                           lambda i: comparison_frame(sender, receiver, i), journal, "Full scene, synchronized, normal speed; unequal byte budgets"))
        first, last = config["slow_view_source_frame_range"]
        views.append(movie(ffmpeg, output / "video91_固定鱼纹区域_三路半速.mp4", list(range(first, last + 1)), 7.5,
                           lambda i: comparison_frame(sender, receiver, i, config["fixed_diagnostic_crop_xyxy"]), journal, "Existing A fixed diagnostic crop, half speed; includes window boundary at source frame224"))
        views.append(movie(ffmpeg, output / "video91_源端预测与传输ROI.mp4", list(range(250)), 15,
                           lambda i: roi_frame(sender, manifest, semantics, i), journal, "Sender predictions and actual transmitted rectangles; not GT or receiver accuracy"))
        sample_indices = [10, 96, 223, 224]
        canvas = Image.new("RGB", (1920, 396 * len(sample_indices)))
        for row, index in enumerate(sample_indices):
            canvas.paste(comparison_frame(sender, receiver, index), (0, row * 396))
        canvas.save(output / "固定关键帧对比.jpg", quality=95)
        crop_canvas = Image.new("RGB", (1920, 792))
        for row, index in enumerate((223, 239)):
            crop_canvas.paste(comparison_frame(sender, receiver, index, config["fixed_diagnostic_crop_xyxy"]), (0, row * 396))
        crop_canvas.save(output / "固定鱼纹局部对比.jpg", quality=95)
        write(output / "viewing.json", {"videos": views, "static_source_frame_indices": sample_indices, "local_crop_source_indices": [223, 239],
              "human_review": "pending; no user score generated", "evaluated_received_rate_kbps": args.rate})
        write(output / "completion.json", {"status": "B2_diagnostics_complete", "evaluated_rate_kbps": args.rate,
              "enhanced_rgb_sequence_sha256": result["enhanced_rgb_sequence_sha256"], "script_sha256": sha(__file__), "human_review": None})
        journal.finish(invocation, 0, time.perf_counter() - tick)
        print("B2 diagnostics and three verified viewing videos complete", flush=True)
    except BaseException as exc:
        write(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        journal.finish(invocation, 1, time.perf_counter() - tick)
        raise


if __name__ == "__main__":
    main()
