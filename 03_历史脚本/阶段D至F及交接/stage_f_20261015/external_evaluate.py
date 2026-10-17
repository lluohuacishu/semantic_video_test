"""Evaluate received RGB only after byte-only selection; empty GT stays explicit."""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
from PIL import Image
from scipy import ndimage
from external_preflight import PROJECT, RUN, OUT, DATA, read, write
sys.path.insert(0, str(PROJECT/"semantic_mvp"))
from runtime import sha256
from teacher import Teacher, regions


def psnr(sse, samples):
    return float(10*np.log10(255**2*samples/sse)) if sse > 0 and samples > 0 else None


def tasks(c):
    tp, fp, fn = map(int, c)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "iou": tp/(tp+fp+fn) if tp+fp+fn else None,
        "recall": tp/(tp+fn) if tp+fn else None,
        "precision": tp/(tp+fp) if tp+fp else None,
        "positive_gt_pixels": tp+fn,
        "interpretation": ("with zero positive GT, no target-protection claim; IoU=0 if FP>0, otherwise undefined"
                           if tp+fn == 0 else "raw0.9 mask against sparse publisher fish GT; before64px filtering"),
    }


def main():
    data = read(DATA/"manifest.json")
    frozen = read(RUN/"experiment_protocol.json")
    assert all(sha256(path) == expected for path, expected in frozen["input_sha256"].items())
    initial = RUN/"initial_crf32"
    source = read(initial/"source_audit.json")
    assert source["source_frame_indices"] == data["source_frame_indices"]
    assert source["rgb_sha256"] == data["source_rgb_sha256"]
    selections = {k: read(RUN/k/"selection.json") for k in ("E", "H265")}
    for name in selections:
        assert (RUN/name/"network/end_to_end.json").exists()
        decoded = read(RUN/name/"network/receiver/decode.json")
        assert decoded["frames"] == 50 and decoded["file_sha256"] == selections[name]["sha256"]
        guard = read(RUN/name/"network/receiver/receiver_read_audit.json")
        assert all(r["allowed"] for r in guard["python_reads"])
    with zipfile.ZipFile(selections["E"]["file"]) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        semantics = json.loads(gzip.decompress(archive.read("semantics.json.gz")))
    assert semantics == read(RUN/"E/network/receiver/target_information.json")
    source_plan = read(initial/"manifest.json")["patches"]
    assert source_plan == manifest["patches"]
    model = Teacher("fish")
    assert model.threshold == .9
    directories = {
        "original": initial/"private_frames",
        "E": RUN/"E/network/receiver/rgb", "H265": RUN/"H265/network/receiver/rgb",
    }
    labels = {r["local_frame"]: r for r in data["sparse_gt"]}
    decoded_hashes = {name: read(RUN/name/"network/receiver/decode.json")["decoded_rgb_sha256"]
                      for name in ("E", "H265")}
    summaries = []
    frame_rows = {name: [] for name in directories}
    inference_rows = []
    raw_cache = {}
    for i in range(50):
        image = np.asarray(Image.open(directories["original"]/f"{i:05d}.png").convert("RGB"))
        assert hashlib.sha256(image.tobytes()).hexdigest() == data["source_rgb_sha256"][i]
        raw_cache[i] = image
    for name, directory in directories.items():
        sse_total = 0.
        gt_bg_sse = 0.
        gt_bg_samples = 0
        gt_fish_sse = 0.
        gt_fish_samples = 0
        counts_total = np.zeros(3, dtype=np.int64)
        negative_frames, negative_with_pred = 0, 0
        for index in range(50):
            image = np.asarray(Image.open(directory/f"{index:05d}.png").convert("RGB"))
            assert image.shape == (720, 1280, 3)
            expected = data["source_rgb_sha256"][index] if name == "original" else decoded_hashes[name][index]
            assert hashlib.sha256(image.tobytes()).hexdigest() == expected
            error = (raw_cache[index].astype(np.float64)-image.astype(np.float64))**2
            sse = float(error.sum()); sse_total += sse
            row = {"local_frame": index, "source_frame": data["source_frame_indices"][index],
                   "global_sse": sse, "samples": int(error.size),
                   "global_rgb_psnr_db": psnr(sse, error.size)}
            if index in labels:
                label = labels[index]
                assert sha256(label["fish_mask"]) == label["fish_mask_sha256"]
                gt = np.asarray(Image.open(label["fish_mask"])) > 0
                started = time.perf_counter()
                predicted, probability = model.mask(directory/f"{index:05d}.png")
                kept, objects = regions(predicted, probability, 64)
                counts = [
                    int((predicted & gt).sum()), int((predicted & ~gt).sum()),
                    int((~predicted & gt).sum())]
                counts_total += counts
                if not gt.any():
                    negative_frames += 1
                    negative_with_pred += int(predicted.any())
                fish_error = float(error[gt].sum())
                bg_error = float(error[~gt].sum())
                gt_fish_sse += fish_error; gt_fish_samples += int(gt.sum())*3
                gt_bg_sse += bg_error; gt_bg_samples += int((~gt).sum())*3
                sparse = {
                    "condition": name, "source_frame": data["source_frame_indices"][index],
                    "gt_fish_pixels": int(gt.sum()), "metrics_raw_09": tasks(counts),
                    "predicted_pixels_before_64_filter": int(predicted.sum()),
                    "predicted_pixels_after_64_filter": int(kept.sum()),
                    "retained_regions": objects,
                    "fish_probability_max": float(probability.max()),
                    "fish_probability_mean": float(probability.mean()),
                    "fish_gt_psnr_db": psnr(fish_error, int(gt.sum())*3),
                    "background_gt_psnr_db": psnr(bg_error, int((~gt).sum())*3),
                    "inference_and_statistics_seconds": time.perf_counter()-started,
                }
                mask_dir = OUT/"evaluation_masks"/name
                mask_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(predicted.astype(np.uint8)*255).save(mask_dir/f"{index:05d}.png")
                inference_rows.append(sparse)
                row["sparse_gt"] = sparse
                print("GT_EVAL", name, index, "TP_FP_FN", counts, flush=True)
            frame_rows[name].append(row)
        summaries.append({
            "condition": name,
            "complete_file_bytes": None if name == "original" else selections[name]["file_bytes"],
            "global_rgb_psnr_all50_db": psnr(sse_total, 50*720*1280*3),
            "global_reference_is_exact": name == "original",
            "gt_background_rgb_psnr_fourframes_db": psnr(gt_bg_sse, gt_bg_samples),
            "gt_fish_rgb_psnr_db": psnr(gt_fish_sse, gt_fish_samples),
            "gt_fish_psnr_defined": gt_fish_samples > 0,
            "sparse_gt_metrics": tasks(counts_total),
            "negative_gt_frames": negative_frames,
            "negative_gt_frames_with_false_positive_mask": negative_with_pred,
            "false_positive_pixel_fraction_on_negative_gt": int(counts_total[1])/(negative_frames*720*1280) if negative_frames else None,
        })
    geometry_rows = []
    for index, record in enumerate(semantics["frames"]):
        roi = np.zeros((720, 1280), dtype=bool)
        active = [p for p in manifest["patches"] if p["start"] <= index < p["start"]+p["frames"]]
        for patch in active:
            x, y, w, h = patch["box"]
            assert not roi[y:y+h, x:x+w].any()
            roi[y:y+h, x:x+w] = True
        for region in record["regions"]:
            x, y, w, h = region["box"]
            assert roi[y:y+h, x:x+w].all()
        geometry_rows.append({
            "source_frame": data["source_frame_indices"][index], "local_frame": index,
            "gt_available": index in labels,
            "gt_fish_pixels": labels[index]["fish_pixels"] if index in labels else None,
            "sender_kept_regions": len(record["regions"]),
            "sender_kept_prediction_pixels": record["pixels"],
            "roi_pixels": int(roi.sum()), "roi_fraction": float(roi.mean()),
            "active_roi_boxes": [p["box"] for p in active],
        })
    roi_fraction = sum(r["roi_pixels"] for r in geometry_rows)/(50*720*1280)
    stream_rows = []
    for patch in manifest["patches"]:
        stream_rows.append({
            **patch, "bytes": manifest["files"][patch["file"]]["bytes"],
            "covered_publisher_gt_frames": [
                r["source_frame"] for r in data["sparse_gt"]
                if patch["start"] <= r["local_frame"] < patch["start"]+patch["frames"]],
            "important_limit": "unlabeled frames cannot be called fish-free; source IDs are separate from local patch starts",
        })
    result = {
        "date": "2026-10-15", "stage": "F", "step": frozen["step"],
        "video": "video6",
        "scope": f"50 consecutive frames {data['source_frame_indices'][0]}..{data['source_frame_indices'][-1]}, {len(labels)} sparse publisher masks",
        "rows": summaries, "roi_mean_frame_fraction": roi_fraction,
        "sender_target_present_frames": sum(bool(r["regions"]) for r in semantics["frames"]),
        "sender_total_retained_regions": sum(len(r["regions"]) for r in semantics["frames"]),
        "gt_frames": len(labels), "gt_positive_frames": sum(r["fish_pixels"] > 0 for r in labels.values()),
        "positive_target_benefit_test_completed": any(r["fish_pixels"] > 0 for r in labels.values()),
        "positive_benefit_not_implied_by_test_completion": True,
        "source_independence": "not used in local adaptation/method selection; official train split; physical capture independence unverified",
        "model_and_production_code_unchanged": True,
        "evaluation_model_matches_sender_model": True,
        "no_training_or_threshold_search": True,
        "all50_actual_received_RGB_hashes_verified_each_method": True,
    }
    write(OUT/"新视频实测评价.json", result)
    write(OUT/"逐标注帧分割评价.json", inference_rows)
    write(OUT/"逐帧ROI与稀疏标注状态.json", geometry_rows)
    write(OUT/"ROI子码流开销.json", stream_rows)
    for name, values in frame_rows.items():
        write(OUT/f"{name}_逐帧画质.json", values)
    assert all(sha256(path) == expected for path, expected in frozen["input_sha256"].items())
    print("EXTERNAL_EVALUATION_COMPLETE", json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
