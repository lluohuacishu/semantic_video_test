"""Check the additional GLC 720p video and calculate target pixel diagnostics."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
from PIL import Image


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def psnr(squared_error, pixels):
    return 10 * math.log10(255**2 / (squared_error / pixels)) if pixels and squared_error else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    info = json.loads((a.run / "inputs.json").read_text(encoding="utf-8"))
    run = a.run / "glc_720_g64"
    enc = json.loads((run / "sender/encode.json").read_text(encoding="utf-8"))
    dec = json.loads((run / "receiver/decode.json").read_text(encoding="utf-8"))
    assert dec["independent_decoder"] and dec["all_frame_checksums_match"]
    assert not dec["source_media_reads"]
    assert enc["header"]["frames"] == dec["header"]["frames"] == info["frames"]
    errors = []
    source_hashes = []
    squared = fish_squared = 0.0
    values = fish_values = 0
    previous = None
    rows = []
    for i in range(info["frames"]):
        filename = f"im{i+1:05d}.png"
        original = rgb(a.run / "source" / filename)
        actual_input = rgb(run / "sender/source" / filename)
        restored = rgb(run / "receiver/frames" / filename)
        sender = rgb(run / "sender/frames" / filename)
        assert np.array_equal(original, actual_input), f"Input pixels differ at {i}"
        assert np.array_equal(sender, restored), f"Receiver pixels differ at {i}"
        source_hashes.append(hashlib.sha256(original.tobytes()).hexdigest())
        error = restored.astype(np.float64) - original.astype(np.float64)
        sq = error**2
        squared += float(sq.sum())
        values += sq.size
        row = {"index": i, "full_psnr_db": psnr(float(sq.sum()), sq.size)}
        if previous is not None:
            delta = float(np.abs(error - previous).mean())
            errors.append(delta)
            row["error_delta_mae_8bit"] = delta
        previous = error
        mask_file = Path(info["mask_dir"]) / f"{info['source_start']+i:05d}.png"
        if mask_file.exists():
            mask = np.asarray(Image.open(mask_file)) > 0
            sample = sq[mask]
            fish_squared += float(sample.sum())
            fish_values += sample.size
            row["fish_pixels"] = int(mask.sum())
            row["fish_psnr_db"] = psnr(float(sample.sum()), sample.size)
        rows.append(row)
    size = (run / "stream.glcv").stat().st_size
    result = {"run": "glc_720_g64", "file_bytes": size, "frames": info["frames"],
              "fps": info["fps"], "duration_seconds": info["frames"] / info["fps"],
              "source_same_rgb_frames": len(source_hashes), "receiver_same_rgb_frames": len(rows),
              "source_media_reads": dec["source_media_reads"], "all_frame_checksums_match": True,
              "header": enc["header"], "full_psnr_db": psnr(squared, values),
              "fish_psnr_db": psnr(fish_squared, fish_values), "fish_pixels": fish_values // 3,
              "temporal_error_delta_mae_mean": float(np.mean(errors)),
              "payload_wait_seconds": {str(k): size * 8 / (1000 * k) for k in (20, 30, 40)},
              "source_rgb_hashes": source_hashes, "per_frame": rows,
              "limitations": ["Target PSNR is a pixel diagnostic, not a recognition score.",
                              "Error-delta is not an established perceptual flicker score.",
                              "720p uses more bytes than the previous 540p point; not an equal-budget codec comparison."]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("header", "source_rgb_hashes", "per_frame")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
