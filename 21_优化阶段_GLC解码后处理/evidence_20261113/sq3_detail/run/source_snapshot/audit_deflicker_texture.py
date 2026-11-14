"""Separate changes in texture amplitude from changes in aligned texture patterns."""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from compare_deflicker import verified_pilot
from probe_reef import bands
from run_deflicker import REEF_NAME, FPS, START, STOP, SHAPE, png_frames
from run_p0 import REPO, file_sha, load_json, roi, save_json
from window_filter import KERNEL
from temporal_filter import correspondence, motion, warp

NAMES = ('source', 'raw', 'center3', 'atlas', 'filter', 'deflicker', 'spatial_control')


def atlas_rgb(folder):
    expected = [f'{i:05d}.png' for i in range(STOP - START)]
    files = sorted(folder.glob('*.png'))
    if [path.name for path in files] != expected:
        raise ValueError('Atlas RGB has missing or out-of-order frames')
    frames = []
    for path in files:
        with Image.open(path) as image:
            if image.mode != 'RGB' or image.size != (SHAPE[1] // 4, SHAPE[0] // 4):
                raise ValueError(f'Unexpected atlas geometry: {path}')
            frame = np.array(image)
        frames.append(cv2.resize(frame, (SHAPE[1], SHAPE[0]), interpolation=cv2.INTER_LINEAR))
    return np.stack(frames)


def fine_energy(frames, crop):
    return float(np.mean([np.abs(bands(frame)[crop][..., 2]).mean() for frame in frames]))


def matched_blur(raw, restored, crop):
    target = fine_energy(restored, crop)
    low, high = 0., 1.5
    best = (abs(fine_energy(raw, crop) - target), low, raw)
    for _ in range(9):
        sigma = (low + high) / 2
        blurred = np.stack([cv2.GaussianBlur(frame, (0, 0), sigma) for frame in raw])
        energy = fine_energy(blurred, crop)
        if abs(energy - target) < best[0]:
            best = (abs(energy - target), sigma, blurred)
        if energy > target:
            low = sigma
        else:
            high = sigma
    return best[1], best[2]


def normalized(x):
    centered = x - np.mean(x, axis=0)
    std = np.sqrt(np.mean(centered ** 2, axis=0))
    return centered / np.maximum(std, 1e-6)


def source_match(previous, current, previous_bands, current_bands, crop):
    backward, forward = motion(previous, current)
    x, y, inside, error = correspondence(backward, forward)
    old_low = previous_bands[..., 3] - previous_bands[..., 2]
    new_low = current_bands[..., 3] - current_bands[..., 2]
    photo = np.abs(new_low - warp(old_low, x, y))
    reliable = inside & (error < 1.5) & (photo < 12)
    reliable = cv2.erode(reliable.astype(np.uint8), KERNEL).astype(bool)
    textured = cv2.GaussianBlur(np.abs(current_bands[..., 2]), (0, 0), 2) > 2.
    return x, y, (reliable & textured)[crop]


def measure(sequences, crop, start=START, *, include_coarse=False):
    rows = []
    source = sequences['source']
    prev = {name: bands(video[0]) for name, video in sequences.items()}
    for i in range(1, len(source)):
        cur = {name: bands(video[i]) for name, video in sequences.items()}
        x, y, mask = source_match(source[i - 1], source[i], prev['source'], cur['source'], crop)
        count = int(mask.sum())
        if count < 64:
            raise ValueError(f'Insufficient matched source texture at frame {start + i}')
        source_a = warp(prev['source'], x, y)[crop][mask]
        source_b = cur['source'][crop][mask]
        source_delta = source_b - source_a
        unit_delta = normalized(source_b[:, 1:3]) - normalized(source_a[:, 1:3])
        row = dict(frame=start + i, valid_pixels=count)
        for name in sequences:
            a = warp(prev[name], x, y)[crop][mask]
            b = cur[name][crop][mask]
            error = np.abs(b - a - source_delta)
            za = normalized(a[:, 1:3])
            zb = normalized(b[:, 1:3])
            ncc = (za * zb).mean(axis=0)
            norm_error = np.abs(zb - za - unit_delta).mean(axis=0)
            row[f'{name}_full_sum'] = float(error[:, 3].sum())
            row[f'{name}_fine_sum'] = float(error[:, 2].sum())
            if include_coarse:
                row[f'{name}_coarse_sum'] = float(error[:, 0].sum())
            for j, band in enumerate(('middle', 'fine')):
                row[f'{name}_{band}_ncc'] = float(ncc[j])
                row[f'{name}_{band}_normalized_error'] = float(norm_error[j])
        rows.append(row)
        prev = cur
    return rows


def summarize(rows, sequences, crop):
    count = sum(row['valid_pixels'] for row in rows)
    result = {}
    for name, video in sequences.items():
        metrics = dict(
            full_temporal_mae=sum(row[f'{name}_full_sum'] for row in rows) / count,
            fine_temporal_mae=sum(row[f'{name}_fine_sum'] for row in rows) / count,
            fine_energy=fine_energy(video, crop))
        for band in ('middle', 'fine'):
            for metric in ('ncc', 'normalized_error'):
                key = f'{band}_{metric}'
                metrics[key] = sum(row[f'{name}_{key}'] * row['valid_pixels'] for row in rows) / count
        result[name] = metrics
    return result


def strip(path, sequences, crop, frame):
    index = frame - START
    source = sequences['source']
    back, _ = motion(source[index - 1], source[index])
    _, ahead = motion(source[index], source[index + 1])
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(np.float32)
    height, width = source[index][crop].shape[:2]
    height, width = 2 * height, 2 * width
    title = 25
    labels = [('source', 'Source'), ('raw', 'GLC raw'), ('spatial_control', 'Spatial blur'),
              ('deflicker', 'Deflicker final')]
    canvas = np.full((len(labels) * (height + title), 3 * width, 3), (24, 35, 48), np.uint8)
    for row, (name, label) in enumerate(labels):
        video = sequences[name]
        aligned = (warp(video[index - 1], xx + back[..., 0], yy + back[..., 1]), video[index],
                   warp(video[index + 1], xx + ahead[..., 0], yy + ahead[..., 1]))
        y = row * (height + title)
        for col, rgb in enumerate(aligned):
            x = col * width
            suffix = 'center' if col == 1 else 'aligned'
            cv2.putText(canvas, f'{label} | {frame + col - 1} | {suffix}', (x + 6, y + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, .47, (240, 244, 250), 1, cv2.LINE_AA)
            canvas[y + title:y + title + height, x:x + width] = cv2.resize(
                rgb[crop], (width, height), interpolation=cv2.INTER_NEAREST)
    Image.fromarray(canvas).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    clip, _, reference, decoded, mean3, restored, receipt = verified_pilot(
        args.workspace, args.previous, args.pilot / 'trial')
    model = args.pilot / 'upstream/results' / REEF_NAME
    if not np.array_equal(png_frames(model / 'final/output'), restored):
        raise ValueError('Model output PNGs differ from the tested lossless result')
    crop = roi(clip['coral_roi'], SHAPE)
    sigma, blurred = matched_blur(decoded[1:-1], restored, crop)
    frames = dict(zip(NAMES, (reference[1:-1], decoded[1:-1], mean3,
                              atlas_rgb(model / 'stage_1/output'), png_frames(model / 'neural_filter/output'),
                              restored, blurred), strict=True))
    print(f'Matched spatial-only blur sigma={sigma:.5f} px', flush=True)
    rows = measure(frames, crop)
    methods = summarize(rows, frames, crop)
    earlier = load_json(args.pilot / 'comparison/summary.json')
    for name in ('raw', 'center3', 'deflicker'):
        for old_key, new_key in [('all', 'full'), ('fine', 'fine')]:
            expected = earlier['methods'][name]['temporal_mae'][old_key]
            if not math.isclose(expected, methods[name][f'{new_key}_temporal_mae'], rel_tol=1e-10):
                raise ValueError(f'Temporal baseline differs: {name}/{new_key}')
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'per_frame.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for index in (192, 215):
        strip(output / f'frame{index}_aligned.png', frames, crop, index)
    summary = dict(
        model_output_sha256=receipt['output_rgb_sha256'], audit_code_sha256=file_sha(__file__),
        frames=STOP - START, fps=FPS, source_start=START, source_stop=STOP,
        gaussian_sigma=sigma, atlas_original_size=[128, 72], methods=methods,
        compared_pairs=len(rows), valid_pixels=sum(row['valid_pixels'] for row in rows),
        normalized_error='Source-corrected temporal residual after per-pair patch-band mean/variance normalization',
        selected_frames=[row for row in rows if row['frame'] in (192, 193, 215)])
    save_json(output / 'summary.json', summary)
    for name, value in methods.items():
        print(name, value, flush=True)


if __name__ == '__main__':
    main()
