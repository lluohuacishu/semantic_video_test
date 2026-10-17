"""Add common RGB measurements without re-encoding or changing semantic scores."""
import json
from pathlib import Path
from common import ROOT, configure, metrics, save_json

configure()
manifest=json.loads((ROOT / "configs/phase_b_gt_codecs.json").read_text("utf-8"))
semantic=json.loads((ROOT / "runs/phase_b_gt_codec_semantics/summary.json").read_text("utf-8"))
rows=[]
for sequence,result in zip(manifest["sequences"],semantic["sequences"],strict=True):
    for stream,score in zip(sequence["streams"],result["streams"][1:],strict=True):
        if stream["name"]!=score["name"]: raise ValueError("Stream alignment changed")
        quality=metrics(sequence["reference_dir"],stream["frames"])
        directory=ROOT / "runs/phase_b_quality" / sequence["name"]
        directory.mkdir(parents=True,exist_ok=True)
        save_json(directory / f"{stream['name']}.json",quality)
        row={"sequence":sequence["name"],"method":stream["name"],"frames":sequence["frame_count"],"duration_seconds":sequence["duration_seconds"],
             "file_bytes":score["complete_file_bytes"],"video_kbps":score["video_kbps"],"transfer_seconds_at_20kbps":score["theoretical_transfer_seconds_at_20kbps"],
             "rgb_psnr_global_db":quality["rgb_psnr_global_db"],"rgb_ms_ssim_mean":quality["rgb_ms_ssim_mean"],
             "gt_fish_rgb_psnr_db":score["ground_truth_fish_rgb_psnr_db"],"fish_iou":score["fish_gt_iou"],"fish_f1":score["fish_gt_f1"]}
        rows.append(row)
        save_json(ROOT / "runs/phase_b_quality/comparison.json",{"scoring_profile":"Same decoded RGB pixels and original-resolution fish GT grid; codec native color domains differ; nominal 5fps playback", "rows":rows})
        print(f"{sequence['name']} {stream['name']}: {row['file_bytes']}B RGB {row['rgb_psnr_global_db']:.3f}dB fish {row['gt_fish_rgb_psnr_db']:.3f}dB IoU {row['fish_iou']:.6f}",flush=True)
