"""Evaluate received public-candidate RGB and create attributed viewing copies.

The source-prediction masks are reused selector outputs, not independent labels.
No budget-matched control or semantic-net-benefit claim is made by this script.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from b2_common import HERE, Journal, git, read, require, sha, write
from evaluate_b2 import font, mask, movie, panel, rgb, score, roi_frame
from evaluate_b3_controls import edges, rect_mask


def mean(values):
    return float(np.mean(values)) if values else None


def evaluate(summary_path, registry, output, baseline):
    network = read(summary_path)
    require(network["status"] == "B3_public_candidate_three_rate_engineering_passed", "Network not accepted")
    require({row["rate_kbps"] for row in network["results"]} == {20, 30, 40}
            and len(network["results"]) == 3 and network["all_rates_identical_enhanced_RGB"], "Three-rate evidence incomplete")
    source_id = network["source_id"]
    source_record = next(row for row in registry["candidates"] if row["id"] == source_id)
    window = next(row for row in registry["selected_windows"] if row["id"] == source_id)
    sender = Path(network["sender"])
    directories = network.get("result_directories", {str(r): str(summary_path.parent / f"rate_{r}kbps") for r in (20, 30, 40)})
    directory = Path(directories["40"])
    receiver = directory / "receiver"
    manifest, decoded = read(receiver / "received_manifest.json"), read(receiver / "decode.json")
    encode, audit = read(sender / "encode.json"), read(sender / "source_audit.json")
    require(manifest == read(sender / "manifest.json"), "Received manifest changed")
    require(sha(sender / "message.scv") == sha(receiver / "message.scv") == network["packet_sha256"], "Packet identity changed")
    require(sha(audit["input"]) == audit["input_sha256"], "Frozen source clip changed")
    require(sum(encode["byte_ledger"].values()) == encode["packet_bytes"], "Complete byte ledger mismatch")
    for target in ("fish", "diver"):
        frozen = baseline["semantic_plan"]["target_profiles"][target]
        require(all(manifest["targets"][target][key] == frozen[key] for key in ("model_sha256", "class_id", "threshold")),
                "Frozen Teacher profile differs")
    n, w, h = manifest["frames"], manifest["width"], manifest["height"]
    require(n == round(window["source_duration_seconds"] * 15) and (w, h, manifest["fps"]) == (1280, 720, "15/1"),
            "Candidate geometry or duration differs from frozen window")
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output)
    invocation = journal.begin([sys.executable, *sys.argv], code_commit=git("rev-parse", "HEAD"),
                               evaluator_sha256=sha(__file__), source_id=source_id)
    tick = time.monotonic()
    predictions = {target: read(sender / f"{target}_prediction_summary.json") for target in ("fish", "diver")}
    semantics = json.loads(gzip.decompress((sender / "sender_streams/semantics.json.gz").read_bytes()))
    paths = {"GLC_base": (receiver / "base_decoded/frames", "im{:05d}.png", 1),
             "auto_ROI": (receiver / "rgb", "{:05d}.png", 0)}
    hashes = {"GLC_base": decoded["base_rgb_sha256"], "auto_ROI": decoded["enhanced_rgb_sha256"]}
    totals = {method: {} for method in paths}
    temporal = {method: {key: [] for key in ("all", "window_boundaries", "non_boundaries")} for method in paths}
    edge = {method: [] for method in paths}
    previous, rows, coverage_rows = {}, [], []
    mask_sequence_hashes = {f"{target}_{kind}": hashlib.sha256() for target in ("fish", "diver") for kind in ("raw", "kept")}
    for index in range(n):
        original = rgb(sender / "private_frames" / f"{index:05d}.png")
        require(hashlib.sha256(original.tobytes()).hexdigest() == audit["mapping"][index]["rgb_sha256"], "Source RGB changed")
        rectangle = rect_mask(manifest["patches"], index, h, w)
        masks = {"full": np.ones((h, w), dtype=bool), "inside_ROI": rectangle, "outside_ROI": ~rectangle}
        coverage = {"frame": index, "rectangle_pixels": int(rectangle.sum())}
        for target in ("fish", "diver"):
            for kind in ("raw", "kept"):
                name = f"{target}_{kind}"
                selected = mask(sender / "private_masks" / target / kind / f"{index:05d}.png")
                require(selected.shape == (h, w), "Prediction mask shape differs")
                count = int(selected.sum())
                require(count == predictions[target][f"{kind}_mask_pixels"][index], "Prediction summary changed")
                if kind == "kept":
                    require(count == semantics["frames"][index]["targets"][target]["pixels"], "Transmitted prediction count differs")
                    require(np.array_equal(selected & rectangle, selected), "ROI omitted a kept predicted pixel")
                masks[name + "_prediction"] = selected
                coverage[name + "_pixels"] = count
                coverage[name + "_covered"] = int(np.count_nonzero(selected & rectangle))
                mask_sequence_hashes[name].update(selected.tobytes())
        coverage_rows.append(coverage)
        boundary = index > 0 and index % 32 == 0
        row = {"frame": index, "source_time_seconds": window["source_start_seconds"] + index / 15,
               "window_boundary": boundary, "methods": {}}
        images = {}
        for method, (folder, pattern, offset) in paths.items():
            image = rgb(folder / pattern.format(index + offset))
            images[method] = image
            require(hashlib.sha256(image.tobytes()).hexdigest() == hashes[method][index], "Accepted receiver RGB changed")
            error = image.astype(np.float32) - original.astype(np.float32)
            squared = np.square(error)
            values = {}
            for name, selected in masks.items():
                sse, samples = float(squared[selected].sum(dtype=np.float64)), int(selected.sum()) * 3
                aggregate = totals[method].setdefault(name, [0.0, 0])
                aggregate[0] += sse
                aggregate[1] += samples
                values[name] = {"sse": sse, "RGB_samples": samples, "psnr_db": score(sse, samples)}
            if method in previous:
                value = float(np.abs(error - previous[method]).mean(dtype=np.float64))
                temporal[method]["all"].append(value)
                temporal[method]["window_boundaries" if boundary else "non_boundaries"].append(value)
                values["error_change_mae"] = value
            previous[method] = error
            value = edges(error, manifest["patches"], index)
            if value is not None:
                edge[method].append(value)
            values["on_ROI_edges_error_difference_mae"] = value
            row["methods"][method] = values
        require(np.array_equal(images["GLC_base"][~rectangle], images["auto_ROI"][~rectangle]), "Outside-ROI RGB changed")
        rows.append(row)
        if (index + 1) % 50 == 0:
            print(f"{source_id}: diagnostic RGB {index+1}/{n}", flush=True)
    methods = {method: {"rgb_psnr_db": {name: score(*values) for name, values in regions.items()},
                        "sse_and_RGB_samples": regions,
                        "error_change_mae": {name: mean(values) for name, values in temporal[method].items()},
                        "on_ROI_edges_error_difference_mae": mean(edge[method])}
               for method, regions in totals.items()}
    region_names = list(totals["auto_ROI"])
    regressions = {name: [row["frame"] for row in rows if row["methods"]["auto_ROI"][name]["RGB_samples"] > 0
                         and row["methods"]["auto_ROI"][name]["sse"] > row["methods"]["GLC_base"][name]["sse"]]
                   for name in region_names}
    coverage_summary = {name: sum(row[name] for row in coverage_rows) for name in coverage_rows[0] if name != "frame"}
    coverage_summary["mean_ROI_fraction"] = coverage_summary["rectangle_pixels"] / (n * w * h)
    for target in ("fish", "diver"):
        for kind in ("raw", "kept"):
            name = f"{target}_{kind}"
            pixels = coverage_summary[name + "_pixels"]
            coverage_summary[name + "_predicted_fraction"] = pixels / (n * w * h)
            coverage_summary[name + "_prediction_coverage"] = coverage_summary[name + "_covered"] / pixels if pixels else None
    result = {"role": "public_candidate_diagnostics_only", "source_id": source_id, "window": window,
              "frames": n, "source_record": source_record, "methods": methods,
              "file_bytes": {"GLC_base": encode["byte_ledger"]["glc_file_bytes"], "auto_ROI": encode["packet_bytes"]},
              "byte_ledger": encode["byte_ledger"], "prediction_summary": encode["source_predictions"],
              "coverage": coverage_summary, "regression_frame_indices_auto_vs_base": regressions,
              "budget_matched_controls_complete": False, "human_review": None, "manual_ground_truth": False,
              "semantic_net_benefit_confirmed": False,
              "mask_sequence_sha256": {name: digest.hexdigest() for name, digest in mask_sequence_hashes.items()},
              "network_summary": str(summary_path), "network_summary_sha256": sha(summary_path),
              "result_directories": directories, "diagnostics_from_rate_kbps": 40,
              "limitations": ["Source prediction regions are also used by the ROI selector; not independent labels",
                              "Predicted components do not count real fish or divers",
                              "Temporal and edge proxies are not validated perceptual flicker scores",
                              "Pretraining exposure and independent capture certification remain unverified",
                              "Unequal total file budgets; no semantic-net-benefit conclusion",
                              "PSNR null means zero samples or zero error; see explicit SSE/sample counts"]}
    write(output / "metrics.json", result)
    write(output / "per_frame_metrics.json", rows)
    write(output / "prediction_coverage.json", {"frames": coverage_rows, "summary": coverage_summary})

    def view(index):
        canvas = Image.new("RGB", (1920, 450), "#101e2a")
        views = [(sender / "private_frames" / f"{index:05d}.png", "源图"),
                 (receiver / "base_decoded/frames" / f"im{index+1:05d}.png", f"GLC q0 · {result['file_bytes']['GLC_base']:,} B"),
                 (receiver / "rgb" / f"{index:05d}.png", f"自动ROI · {result['file_bytes']['auto_ROI']:,} B")]
        for col, (path, title) in enumerate(views):
            with Image.open(path) as image:
                resized = image.convert("RGB").resize((640, 360), Image.Resampling.LANCZOS)
            canvas.paste(panel(resized, f"{title} · 帧{index}", font(19)), (640 * col, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 398), "公开候选诊断；模型预测不是GT；增强方案增加字节，等预算控制和用户观看仍待完成。", font=font(17), fill="white")
        credit = source_record["creator"].split(" (")[0]
        draw.text((10, 423), f"Credit: {credit} | {source_record['license']} | {source_id} | ATTRIBUTION.md", font=font(14), fill="white")
        return canvas

    ffmpeg = baseline["runtime_dependencies"]["ffmpeg"]
    views = [movie(ffmpeg, output / "comparison_normal.mp4", list(range(n)), 15, view, journal,
                   "Source / received GLC base / received automatic ROI; all frozen frames; normal speed"),
             movie(ffmpeg, output / "comparison_half_speed.mp4", list(range(n)), 7.5, view, journal,
                   "Same complete frame sequence at half speed; no frame selection from quality results")]
    fixed_indices = [0, n // 2, n - 1]
    sheet = Image.new("RGB", (1920, 450 * len(fixed_indices)))
    roi_sheet = Image.new("RGB", (1280, 756 * len(fixed_indices)))
    for row, index in enumerate(fixed_indices):
        sheet.paste(view(index), (0, row * 450))
        roi_sheet.paste(roi_frame(sender, manifest, semantics, index), (0, row * 756))
    sheet.save(output / "fixed_frame_comparison.jpg", quality=95)
    roi_sheet.save(output / "source_prediction_regions.jpg", quality=95)
    attribution = (f"# Source and derived-view attribution\n\n{source_record['title']}\n\n"
                   f"Creator: {source_record['creator']}\n\nSource: {source_record['page']}\n\n"
                   f"License: [{source_record['license']}]({source_record['license_url']})\n\n"
                   f"Changes: frozen {window['source_start_seconds']}–{window['source_start_seconds']+window['source_duration_seconds']} s window; "
                   "15 fps / 1280×720 conversion; GLC and ROI reconstruction; labels, scaling and comparison layout. "
                   "Half-speed view changes playback speed only. Derived views retain the source license. "
                   "Model outputs may alter target details; no endorsement by the source creator is implied.\n")
    (output / "ATTRIBUTION.md").write_text(attribution, encoding="utf-8")
    write(output / "viewing.json", {"source_id": source_id, "videos": views, "fixed_frame_indices": fixed_indices,
          "attribution": str(output / "ATTRIBUTION.md"), "source_record": source_record, "human_review": "pending"})
    write(output / "completion.json", {"status": "B3_public_candidate_diagnostics_complete", "source_id": source_id,
          "code_commit": git("rev-parse", "HEAD"), "script_sha256": sha(__file__),
          "network_summary_sha256": sha(summary_path), "frames": n, "human_review": "pending"})
    journal.finish(invocation, 0, time.monotonic() - tick)
    print(f"{source_id}: received-RGB diagnostics and two attributed viewing videos complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-for-network", action="store_true")
    parser.add_argument("--baseline-config", type=Path, default=HERE / "configs/b0_baseline.json")
    args = parser.parse_args()
    require(not git("status", "--porcelain"), "Commit evaluator before running it")
    args.output.mkdir(parents=True, exist_ok=False)
    registry, baseline = read(args.registry), read(args.baseline_config)
    for path in args.network_summaries:
        if args.wait_for_network:
            deadline = time.monotonic() + 14400
            while not path.exists():
                require(not (path.parent / "failure.json").exists(), "Network failed; retain completed diagnostics")
                require(time.monotonic() < deadline, "Network wait timed out")
                time.sleep(1)
        require(path.is_file(), "Network summary missing")
        evaluate(path.resolve(), registry, args.output / read(path)["source_id"], baseline)
    write(args.output / "completion.json", {"status": "B3_public_batch_diagnostics_complete", "clips": len(args.network_summaries)})


if __name__ == "__main__":
    main()
