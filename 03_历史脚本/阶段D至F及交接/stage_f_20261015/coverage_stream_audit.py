"""Follow up the box audit with all existing A/E substream costs, without reencoding."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile
from coverage_audit import OUT, RUN, PROJECT, MANUAL, WIDTH, HEIGHT, read, write, sha, intersection


def main():
    summary = read(OUT/"覆盖统计汇总.json")
    frames = read(OUT/"逐帧覆盖与无鱼开销.json")
    human = read(MANUAL)
    paths = [Path(__file__), OUT/"步骤2冻结协议.json", OUT/"覆盖统计汇总.json",
             OUT/"逐帧覆盖与无鱼开销.json", MANUAL]
    protocol = {
        "date": "2026-10-15", "stage": "F", "step": 2,
        "scope": "report every existing A/E ROI stream, not only a selected suspicious crop",
        "geometry": "sum intersection with frozen reviewed box unions in each active frame",
        "cost": "actual full MP4 bytes from existing packet, not a frame byte allocation",
        "limit": "zero reviewed-box overlap does not mean the crop has no contextual value",
        "no_reencoding_or_production_change": True,
        "input_sha256": {str(p): sha(p) for p in paths},
    }
    if (RUN/"stream_cost_protocol.json").exists():
        assert read(RUN/"stream_cost_protocol.json") == protocol
    write(RUN/"stream_cost_protocol.json", protocol)
    write(OUT/"码流空间对应补充约定.json", protocol)
    output = []
    for method, data in summary["methods"].items():
        for stream in data["streams"]:
            local_indices = range(stream["start"], stream["start"]+stream["frames"])
            overlap_sum, region_count, region_pixels = 0, 0, 0
            intersected_ids = []
            for index in local_indices:
                import numpy as np
                x, y, w, h = stream["box"]
                human_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
                for ordinal, b in enumerate(human["frames"][index]["boxes"], 1):
                    bx, by, bw, bh = b
                    human_mask[by:by+bh, bx:bx+bw] = True
                    if intersection(stream["box"], b) > 0:
                        intersected_ids.append(f"{index+420:05d}-{ordinal}")
                overlap_sum += int(human_mask[y:y+h, x:x+w].sum())
                for region in frames[index]["predicted_regions"]:
                    if intersection(stream["box"], region["box"]) > 0:
                        region_count += 1
                        region_pixels += region["pixels"]
            roi_pixel_frames = stream["frames"]*stream["box"][2]*stream["box"][3]
            output.append({
                "method": method, **stream, "source_frames": [
                    420+stream["start"], 420+stream["start"]+stream["frames"]-1],
                "share_of_roi_stream_bytes": stream["bytes"]/data["actual_roi_stream_bytes_existing"],
                "share_of_complete_packet_bytes": stream["bytes"]/data["complete_file_bytes_existing"],
                "roi_pixel_frames": roi_pixel_frames,
                "intersected_reviewed_box_ids": intersected_ids,
                "reviewed_box_union_intersection_pixel_frames": overlap_sum,
                "reviewed_box_union_fraction_of_crop": overlap_sum/roi_pixel_frames,
                "sender_region_boxes_intersecting_crop": region_count,
                "sender_region_mask_pixels_for_intersecting_boxes": region_pixels,
                "zero_overlap_with_reviewed_boxes_all_active_frames": overlap_sum == 0,
                "cost_interpretation": "existing stream bytes; no claim that all bytes could be saved at equal quality",
            })
    assert all(sha(p) == value for p, value in protocol["input_sha256"].items())
    write(OUT/"码流与人工框空间对应.json", output)
    for row in output:
        print(json.dumps({k: row[k] for k in (
            "method", "file", "box", "bytes", "share_of_roi_stream_bytes",
            "share_of_complete_packet_bytes", "reviewed_box_union_intersection_pixel_frames",
            "sender_region_boxes_intersecting_crop")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
