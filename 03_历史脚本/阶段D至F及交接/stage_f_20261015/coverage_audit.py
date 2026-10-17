"""Stage F step 2: audit frozen human rectangles and existing received ROI plans.

No model inference, encoding, threshold search, or annotation edits.
"""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

WORKSPACE = Path(r"C:\Users\lenovo\Documents\ChatGPT\语义通信")
PROJECT = Path(r"C:\Users\lenovo\Desktop\works\work\semantic-video-mlvc")
OUT = WORKSPACE / "outputs/semantic_roi_phase_f/step2_box_coverage"
RUN = PROJECT / "runs/semantic_mvp_stage_f_20261015/step2"
CONTROL = PROJECT / "runs/semantic_mvp_roi_control_20261015"
STAGE_E = PROJECT / "runs/semantic_mvp_stage_e_20261015"
OLD = PROJECT / "runs/semantic_mvp_20261014_fish"
MANUAL = CONTROL / "manual_roi_snapshot.json"
MANUAL_SHA = "9e4da19c6b20fd1e4326638917a1caaec1eb38a7abc2254adc6477e89334338e"
WIDTH, HEIGHT, N = 1280, 720, 50
FULL_NUMERATOR, FULL_DENOMINATOR = 95, 100
EDGE_MARGIN = 16
SIZE_CUTS = [512, 2048]  # Bounding rectangle area, not fish mask pixel area.
sys.path.insert(0, str(PROJECT / "semantic_mvp"))
from roi_layout import plan_halves
from system import roi_box


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rectangles(plans, index):
    return [p["box"] for p in plans if p["start"] <= index < p["start"] + p["frames"]]


def mask(boxes):
    result = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for x, y, w, h in boxes:
        assert 0 <= x < x + w <= WIDTH and 0 <= y < y + h <= HEIGHT
        result[y:y+h, x:x+w] = True
    return result


def intersection(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return max(0, right-x) * max(0, bottom-y)


def full(inside, area):
    # Exact integer comparison; avoid floating-point ambiguity at 95%.
    return inside * FULL_DENOMINATOR >= area * FULL_NUMERATOR


def classification(inside, area):
    return "missed" if inside == 0 else "full95" if full(inside, area) else "partial"


def size_group(area):
    return "le512" if area <= 512 else "513to2048" if area <= 2048 else "gt2048"


def box_summary(rows, method):
    selected = [r[method] for r in rows]
    count = len(selected)
    return {
        "boxes": count,
        "full95": sum(r["status"] == "full95" for r in selected),
        "partial": sum(r["status"] == "partial" for r in selected),
        "missed": sum(r["status"] == "missed" for r in selected),
        "mean_box_coverage": float(np.mean([r["coverage"] for r in selected])) if count else None,
        "full95_fraction": sum(r["status"] == "full95" for r in selected)/count if count else None,
    }


def load_packet(selection):
    selected = read(selection)
    packet = Path(selected["file"])
    assert sha(packet) == selected["sha256"]
    assert packet.stat().st_size == selected["file_bytes"]
    with zipfile.ZipFile(packet) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("manifest.json"))
        semantic_bytes = archive.read("semantics.json.gz")
        semantics = json.loads(gzip.decompress(semantic_bytes))
        streams = [
            {**p, "bytes": len(archive.read(p["file"]))}
            for p in manifest["patches"]
        ]
    return selected, manifest, semantics, semantic_bytes, streams


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    assert sha(MANUAL) == MANUAL_SHA
    human = read(MANUAL)
    assert len(human["frames"]) == N
    assert sum(len(r["boxes"]) for r in human["frames"]) == 61
    assert sum(bool(r["boxes"]) for r in human["frames"]) == 38
    frozen_paths = [
        Path(__file__), MANUAL, CONTROL/"A/selection.json", STAGE_E/"selection.json",
        PROJECT/"semantic_mvp/roi_layout.py", PROJECT/"semantic_mvp/system.py",
        PROJECT/"semantic_mvp/teacher.py", PROJECT/"runs/phase_c_20261013/fixed_teacher.json",
        CONTROL/"A/network/receiver/decode.json", STAGE_E/"network/receiver/decode.json",
        OLD/"sender/source_audit.json",
        WORKSPACE/"outputs/semantic_roi_phase_e/ROI与逐帧覆盖.json",
    ]
    checkpoint = Path(read(PROJECT/"runs/phase_c_20261013/fixed_teacher.json")["checkpoint"])
    frozen_paths.append(checkpoint)
    protocol = {
        "stage": "F", "step": 2, "date": "2026-10-15",
        "scope": "61 frozen frame-box observations, 50 frames; geometric diagnosis only",
        "comparisons": {
            "A": "unchanged predictions, 32-frame single rectangle, already received packet",
            "E": "same predictions, 32-frame left/right rectangles, already received packet",
            "same_frame_E_geometry": "same kept predictions and padding, one frame only; diagnostic, not encoded",
        },
        "full_coverage_rule": "covered rectangle pixels / all rectangle pixels >= 95/100",
        "coverage_classes": ["missed: 0", "partial: >0 and <0.95", "full95: >=0.95"],
        "size_groups_bbox_pixels": ["<=512", "513..2048", ">2048"],
        "size_thresholds_not_fish_mask_pixel_thresholds": True,
        "edge_rule": "minimum bbox gap to image boundary <=16 pixels; touching means gap=0",
        "frame_success_rule": "all annotated rectangles individually >=95%; positive frames only",
        "negative_frame_rule": "frozen human/publisher no_fish=True, no boxes",
        "weighting": "each frame-box observation equal weight; no independent-fish inference",
        "no_per_frame_bitrate_estimate_from_area": True,
        "no_training_inference_encoding_or_parameter_search": True,
        "annotations_unchanged": True,
        "input_sha256": {str(p): sha(p) for p in frozen_paths},
        "source_clip": "video131 real consecutive source frames 420..469; seen diagnostic clip",
    }
    # Freeze all metric choices and inputs before computing the new coverage results.
    if (RUN/"protocol.json").exists():
        assert read(RUN/"protocol.json") == protocol, "Existing frozen protocol differs"
    write(RUN/"protocol.json", protocol)
    write(OUT/"步骤2冻结协议.json", protocol)

    loaded = {
        "A": load_packet(CONTROL/"A/selection.json"),
        "E": load_packet(STAGE_E/"selection.json"),
    }
    assert loaded["A"][3] == loaded["E"][3], "Sender semantic payload changed"
    records = loaded["E"][2]["frames"]
    assert len(records) == N
    original_semantics = None
    with zipfile.ZipFile(OLD/"receiver/message.scv") as archive:
        original_semantics = archive.read("semantics.json.gz")
    assert original_semantics == loaded["E"][3]
    plans = {k: v[1]["patches"] for k, v in loaded.items()}
    expected_a = []
    for start in range(0, N, 32):
        b = roi_box(records[start:start+32], WIDTH, HEIGHT, 16)
        if b:
            expected_a.append({"start": start, "frames": min(32, N-start), "box": b})
    expected_e = plan_halves(records, WIDTH, HEIGHT, 32, 16)
    for name, expected in (("A", expected_a), ("E", expected_e)):
        actual = [{k: p[k] for k in ("start", "frames", "box")} for p in plans[name]]
        assert actual == [{k: p[k] for k in ("start", "frames", "box")} for p in expected]
        decoded = read((CONTROL/"A" if name == "A" else STAGE_E)/"network/receiver/decode.json")
        assert decoded["file_sha256"] == loaded[name][0]["sha256"] and decoded["frames"] == N

    previous = read(WORKSPACE/"outputs/semantic_roi_phase_e/ROI与逐帧覆盖.json")
    boxes, frames = [], []
    for index, annotation in enumerate(human["frames"]):
        assert annotation["source_frame"] == 420+index and annotation["local_frame"] == index
        assert annotation["width"] == WIDTH and annotation["height"] == HEIGHT
        assert annotation["no_fish"] == (not annotation["boxes"])
        roi_masks = {name: mask(rectangles(p, index)) for name, p in plans.items()}
        assert not (roi_masks["E"] & ~roi_masks["A"]).any()
        same_frame = [p["box"] for p in plan_halves([records[index]], WIDTH, HEIGHT, 32, 16)]
        same_mask = mask(same_frame)
        human_mask = mask(annotation["boxes"])
        current = []
        for ordinal, b in enumerate(annotation["boxes"], 1):
            x, y, w, h = b
            area = w*h
            gap = min(x, y, WIDTH-x-w, HEIGHT-y-h)
            row = {
                "id": f"{420+index:05d}-{ordinal}", "source_frame": 420+index,
                "local_frame": index, "box_ordinal": ordinal, "box": b, "bbox_area": area,
                "size_group": size_group(area), "minimum_edge_gap": gap,
                "near_edge16": gap <= EDGE_MARGIN, "touches_edge": gap == 0,
                "annotation_status": annotation["status"], "annotation_source": annotation["source"],
            }
            for name in ("A", "E"):
                active = rectangles(plans[name], index)
                inside = sum(intersection(b, p) for p in active)
                assert inside == int(roi_masks[name][y:y+h, x:x+w].sum())
                row[name] = {"covered_bbox_pixels": inside, "coverage": inside/area,
                             "status": classification(inside, area)}
            same_inside = sum(intersection(b, p) for p in same_frame)
            assert same_inside == int(same_mask[y:y+h, x:x+w].sum())
            row["same_frame_E_geometry"] = {
                "covered_bbox_pixels": same_inside, "coverage": same_inside/area,
                "status": classification(same_inside, area),
            }
            row["E_window_full_but_same_frame_below95"] = (
                row["E"]["status"] == "full95" and
                row["same_frame_E_geometry"]["status"] != "full95"
            )
            row["lost_covered_bbox_pixels_A_to_E"] = (
                row["A"]["covered_bbox_pixels"] - row["E"]["covered_bbox_pixels"]
            )
            boxes.append(row)
            current.append(row)
        frame = {
            "source_frame": 420+index, "local_frame": index, "no_fish": annotation["no_fish"],
            "annotation_status": annotation["status"], "box_count": len(current),
            "box_union_pixels": int(human_mask.sum()),
            "predicted_region_count": len(records[index]["regions"]),
            "predicted_mask_pixels": sum(r["pixels"] for r in records[index]["regions"]),
            "predicted_regions": records[index]["regions"],
        }
        for name in ("A", "E"):
            area = int(roi_masks[name].sum())
            covered = int((human_mask & roi_masks[name]).sum())
            frame[name] = {
                "boxes": rectangles(plans[name], index), "roi_pixels": area,
                "roi_fraction": area/(WIDTH*HEIGHT), "covered_box_union_pixels": covered,
                "box_union_coverage": covered/int(human_mask.sum()) if human_mask.any() else None,
                "all_boxes_full95": all(r[name]["status"] == "full95" for r in current) if current else None,
                "full95_boxes": sum(r[name]["status"] == "full95" for r in current),
                "partial_boxes": sum(r[name]["status"] == "partial" for r in current),
                "missed_boxes": sum(r[name]["status"] == "missed" for r in current),
            }
            prior = previous[name]["manual_frames"][index]
            assert prior["source_frame"] == frame["source_frame"]
            assert prior["reviewed_box_union_pixels"] == frame["box_union_pixels"]
            assert prior["covered_box_union_pixels"] == covered
        frames.append(frame)
    assert len(boxes) == 61 and len(frames) == N
    assert sum(len(r["regions"]) for r in records) == 35
    positive = [f for f in frames if not f["no_fish"]]
    negative = [f for f in frames if f["no_fish"]]
    assert len(positive) == 38 and len(negative) == 12
    summaries = {}
    for name in ("A", "E"):
        avg = sum(f[name]["roi_pixels"] for f in frames)/(N*WIDTH*HEIGHT)
        old_mean = float(np.mean([f[name]["box_union_coverage"] for f in positive]))
        assert abs(avg - previous[name]["average_roi_area"]) < 1e-12
        assert abs(old_mean - previous[name]["manual_frame_mean_box_area_coverage"]) < 1e-12
        total_roi = sum(f[name]["roi_pixels"] for f in frames)
        negative_roi = sum(f[name]["roi_pixels"] for f in negative)
        summaries[name] = {
            **box_summary(boxes, name),
            "positive_frames": len(positive),
            "all_boxes_full95_frames": sum(f[name]["all_boxes_full95"] for f in positive),
            "all_boxes_full95_frame_fraction": sum(f[name]["all_boxes_full95"] for f in positive)/len(positive),
            "any_overlap_frames": sum(f[name]["covered_box_union_pixels"] > 0 for f in positive),
            "no_box_overlap_frames": sum(f[name]["covered_box_union_pixels"] == 0 for f in positive),
            "previous_frame_mean_union_coverage": old_mean,
            "previous_frame_union95_frames": sum(f[name]["box_union_coverage"] >= .95 for f in positive),
            "average_roi_area": avg,
            "negative_frames": len(negative),
            "negative_frames_with_final_roi": sum(f[name]["roi_pixels"] > 0 for f in negative),
            "negative_mean_roi_fraction": negative_roi/(len(negative)*WIDTH*HEIGHT),
            "negative_roi_pixel_frames": negative_roi,
            "negative_share_of_all_roi_pixel_frames": negative_roi/total_roi,
            "negative_predicted_positive_frames": sum(f["predicted_region_count"] > 0 for f in negative),
            "negative_predicted_regions": sum(f["predicted_region_count"] for f in negative),
            "negative_predicted_mask_pixels": sum(f["predicted_mask_pixels"] for f in negative),
            "size_groups": {g: box_summary([r for r in boxes if r["size_group"] == g], name)
                            for g in ("le512", "513to2048", "gt2048")},
            "edge_groups": {
                "near_edge16": box_summary([r for r in boxes if r["near_edge16"]], name),
                "interior_gt16": box_summary([r for r in boxes if not r["near_edge16"]], name),
                "touching_edge": box_summary([r for r in boxes if r["touches_edge"]], name),
            },
            "complete_file_bytes_existing": loaded[name][0]["file_bytes"],
            "actual_roi_stream_bytes_existing": loaded[name][0]["parts"]["roi"],
            "streams": loaded[name][4],
            "partial_box_ids": [r["id"] for r in boxes if r[name]["status"] == "partial"],
            "missed_box_ids": [r["id"] for r in boxes if r[name]["status"] == "missed"],
        }
    same_summary = box_summary(boxes, "same_frame_E_geometry")
    window_extra = [r["id"] for r in boxes if r["E_window_full_but_same_frame_below95"]]
    group_cross = {}
    for g in ("le512", "513to2048", "gt2048"):
        for edge in (False, True):
            group_cross[f"{g}_near_edge_{edge}"] = box_summary(
                [r for r in boxes if r["size_group"] == g and r["near_edge16"] == edge], "E")
    result = {
        "scope": protocol["scope"], "annotation_sha256": MANUAL_SHA,
        "methods": summaries, "same_frame_E_geometry_not_encoded": same_summary,
        "window_full_same_frame_below95_box_ids": window_extra,
        "E_size_edge_cross_groups": group_cross,
        "bbox_area_range": [min(r["bbox_area"] for r in boxes), max(r["bbox_area"] for r in boxes)],
        "annotation_status_frame_counts": {s: sum(f["status"] == s for f in human["frames"])
                                           for s in sorted({f["status"] for f in human["frames"]})},
        "independence": "frame-box observations from a single already-seen continuous video; no track identities",
    }
    assert all(sha(p) == digest for p, digest in protocol["input_sha256"].items())
    verification = {
        "annotation_rows": 50, "box_rows": 61, "positive_frames": 38, "negative_frames": 12,
        "annotation_sha256_matches_frozen_snapshot": True,
        "all_frame_indices_and_dimensions_verified": True,
        "same_semantic_payload_in_original_A_E_packets": True,
        "A_E_plans_reproduce_production_code": True,
        "A_E_packet_hashes_match_existing_independent_receiver_decode": True,
        "all_box_intersection_counts_crosschecked_by_raster_masks": True,
        "all_100_prior_frame_union_counts_reproduced": True,
        "frozen_production_code_model_and_annotations_unchanged": True,
        "E_roi_is_subset_of_A_in_every_frame": True,
        "new_model_inferences": 0, "new_video_encodes": 0, "new_tcp_transfers": 0,
        "stage_f_status": "in_progress", "step2_status": "data_complete_report_pending",
    }
    write(OUT/"逐框覆盖明细.json", boxes)
    write(OUT/"逐帧覆盖与无鱼开销.json", frames)
    write(OUT/"覆盖统计汇总.json", result)
    write(OUT/"步骤2数据核验.json", verification)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
