"""GT-pixel ROI coverage and component diagnostics for the positive supplement."""
from __future__ import annotations
import json
import zipfile
import numpy as np
from PIL import Image
from scipy import ndimage
from external_preflight import RUN, OUT, DATA, read, write, digest as sha256


def psnr(sse, samples):
    return float(10*np.log10(255**2*samples/sse)) if sse > 0 and samples > 0 else None


def main():
    root = RUN/"positive_case"; output = OUT/"positive_case"; data = read(DATA/"positive_case/manifest.json")
    manifest = read(root/"E/network/receiver/received_manifest.json")
    components = []; frames = []
    for label in data["sparse_gt"]:
        local = label["local_frame"]; real = label["source_frame"]
        assert sha256(label["fish_mask"]) == label["fish_mask_sha256"]
        gt = np.asarray(Image.open(label["fish_mask"])) > 0
        roi = np.zeros_like(gt)
        for patch in manifest["patches"]:
            if patch["start"] <= local < patch["start"]+patch["frames"]:
                x,y,w,h = patch["box"]; roi[y:y+h,x:x+w] = True
        original = np.asarray(Image.open(root/f"initial_crf32/private_frames/{local:05d}.png").convert("RGB"), dtype=np.float64)
        reconstructions = {method: np.asarray(Image.open(root/f"{method}/network/receiver/rgb/{local:05d}.png").convert("RGB"), dtype=np.float64)
                           for method in ("E","H265")}
        masks = {method: np.asarray(Image.open(output/f"evaluation_masks/{method}/{local:05d}.png")) > 0
                 for method in ("original","E","H265")}
        labels, n = ndimage.label(gt, np.ones((3,3), dtype=np.uint8))
        for ident in range(1,n+1):
            target = labels == ident
            yy, xx = np.where(target)
            pixels = int(target.sum()); covered = int((target & roi).sum())
            components.append({
                "source_frame": real, "local_frame": local, "component": ident,
                "component_is_not_a_verified_instance": True, "fish_pixels": pixels,
                "box": [int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)],
                "E_roi_covered_fish_pixels": covered, "E_roi_coverage": covered/pixels,
                "model_gt_recall": {method: int((mask & target).sum())/pixels for method,mask in masks.items()},
                "fish_rgb_psnr_db": {method: psnr(float(((rgb-original)**2)[target].sum()), pixels*3)
                                     for method,rgb in reconstructions.items()},
            })
        count = int(gt.sum()); covered = int((gt & roi).sum())
        frames.append({
            "source_frame": real, "local_frame": local, "gt_fish_pixels": count,
            "E_roi_covered_fish_pixels": covered, "E_roi_fish_coverage": covered/count if count else None,
            "gt_components": n,
        })
    total = sum(row["gt_fish_pixels"] for row in frames)
    inside = sum(row["E_roi_covered_fish_pixels"] for row in frames)
    result = {
        "gt_frames": frames, "gt_components": components,
        "pooled_GT_fish_pixels": total, "E_covered_GT_fish_pixels": inside,
        "E_pooled_GT_fish_coverage": inside/total if total else None,
        "positive_GT_frame_any_coverage": sum(row["E_roi_covered_fish_pixels"] > 0 for row in frames),
        "positive_GT_frames": sum(row["gt_fish_pixels"] > 0 for row in frames),
        "components_full95": sum(row["E_roi_coverage"] >= .95 for row in components),
        "components_missed": sum(row["E_roi_coverage"] == 0 for row in components),
        "all_components": len(components),
        "scope": "three sparse GT frames from one additional clip of the same video6; no instance tracking",
    }
    write(output/"含鱼GT保护与逐连通域核对.json", result)
    print("POSITIVE_TARGET_AUDIT", json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
