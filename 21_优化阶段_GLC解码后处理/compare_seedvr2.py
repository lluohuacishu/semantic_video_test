"""Evaluate SeedVR2 against raw GLC, Deflicker and framewise spatial controls."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from audit_deflicker_texture import fine_energy, matched_blur, measure, normalized, summarize
from compare_deflicker import perceptual, save_csv, verified_pilot
from experiment_common import array_sha
from probe_reef import bands
from run_deflicker import FPS, SHAPE, START, STOP
from run_p0 import REPO, file_sha, load_json, preview, psnr, roi, save_json


def source_detail(source, candidates, crop, start=START):
    rows = []
    for i, frame in enumerate(source):
        original = bands(frame)
        mask = (cv2.GaussianBlur(np.abs(original[..., 2]), (0, 0), 2) > 2.0)[crop]
        ref = normalized(original[crop][mask, 1:3])
        row = dict(frame=start + i, valid_pixels=int(mask.sum()))
        for method, video in candidates.items():
            actual = normalized(bands(video[i])[crop][mask, 1:3])
            middle, fine = (ref * actual).mean(axis=0)
            row[f'{method}_middle_ncc_to_source'] = float(middle)
            row[f'{method}_fine_ncc_to_source'] = float(fine)
        rows.append(row)
    return rows


def plot(path, temporal, detail, candidates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, layout='constrained')
    styles = {
        'raw': ('GLC raw', '#4d5259'), 'deflicker': ('Deflicker', '#087f88'),
        'spatial_control': ('Deflicker-matched blur', '#a67525'),
        'seedvr2': ('SeedVR2 3B Q8', '#7a3b9c'),
        'seed_matched_blur': ('SeedVR2-matched blur', '#668e35'),
    }
    for method in candidates:
        label, color = styles[method]
        axes[0].plot([r['frame'] / FPS for r in temporal], [r[f'{method}_fine_ncc'] for r in temporal],
                     color=color, label=label, linewidth=1.3)
        axes[1].plot([r['frame'] / FPS for r in detail], [r[f'{method}_fine_ncc_to_source'] for r in detail],
                     color=color, label=label, linewidth=1.3)
    for axis in axes:
        axis.axvline(192 / FPS, color='#aaaaaa', linewidth=1, linestyle=':')
        axis.grid(alpha=.18)
        axis.set_ylabel('Fine-texture NCC')
    axes[0].set_title('Across adjacent frames after source-motion alignment', loc='left')
    axes[1].set_title('Same-frame correspondence to source texture', loc='left')
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set_xlabel('Time in source video (seconds)')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--deflicker-run', type=Path, required=True)
    parser.add_argument('--seedvr2-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    parser.add_argument('--metrics-root', type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    clip, _, reference, decoded, _, deflicker, _ = verified_pilot(
        args.workspace, args.previous, args.deflicker_run)
    source, raw = reference[1:-1], decoded[1:-1]
    inputs_path = args.seedvr2_run / 'inputs.json'
    seed_summary = load_json(args.seedvr2_run / 'run.json')
    inputs = load_json(inputs_path)
    seed_path = args.seedvr2_run / 'seedvr2_rgb.npy'
    if (not seed_summary['complete'] or seed_summary['inputs_sha256'] != file_sha(inputs_path)
            or seed_summary['output_sha256'] != file_sha(seed_path) or seed_summary['fps'] != FPS
            or (inputs['start'], inputs['stop']) != (START, STOP)
            or inputs['input_rgb_sha256'] != array_sha(raw)):
        raise ValueError('SeedVR2 result or fixed RGB comparison input differs')
    seed = np.load(seed_path, mmap_mode='r', allow_pickle=False)
    if seed.shape != source.shape or seed.dtype != np.uint8:
        raise ValueError('SeedVR2 changed frame count, geometry or RGB precision')
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    crop = roi(clip['coral_roi'], SHAPE)
    panels = (source, raw, deflicker, seed)
    labels = ('Source RGB', 'GLC raw', 'All-In-One-Deflicker', 'SeedVR2 3B Q8')
    videos = {
        name: preview(args.ffmpeg, output / name, panels, FPS, START, labels, region)
        for name, region in [('compare.mp4', None), ('zoom.mp4', crop)]
    }
    print('Verified both comparison previews at the original frame rate', flush=True)

    sigma, spatial = matched_blur(raw, deflicker, crop)
    methods = dict(raw=raw, deflicker=deflicker, spatial_control=spatial, seedvr2=seed)
    sigmas = {'spatial_control': sigma}
    if fine_energy(seed, crop) < fine_energy(raw, crop):
        seed_sigma, seed_blur = matched_blur(raw, seed, crop)
        if seed_sigma != sigma:
            methods['seed_matched_blur'] = seed_blur
            sigmas['seed_matched_blur'] = seed_sigma
    all_sequences = dict(source=source, **methods)
    temporal = measure(all_sequences, crop)
    measures = summarize(temporal, all_sequences, crop)
    detail = source_detail(source, methods, crop)
    save_csv(output / 'per_frame_temporal.csv', temporal)
    save_csv(output / 'per_frame_source_detail.csv', detail)
    plot(output / 'texture_curve.png', temporal, detail, methods)
    total = sum(row['valid_pixels'] for row in detail)
    for method, frames in methods.items():
        error = frames.astype(np.float32) - source
        for label, area in [('rgb_psnr_db', error), ('coral_rgb_psnr_db', error[(slice(None), *crop)])]:
            measures[method][label] = psnr(float(np.mean(np.square(area), dtype=np.float64)))
        for band in ('middle', 'fine'):
            key = f'{band}_ncc_to_source'
            measures[method][key] = sum(row['valid_pixels'] * row[f'{method}_{key}'] for row in detail) / total
        measures[method]['fine_energy_vs_source'] = measures[method]['fine_energy'] / measures['source']['fine_energy']
    provenance = None
    if args.metrics_root is not None:
        scores, provenance = perceptual(source, methods, args.metrics_root.resolve())
        save_csv(output / 'per_frame_perceptual.csv', scores)
        for method in methods:
            for metric in ('lpips', 'dists'):
                measures[method][metric] = float(np.mean([row[f'{method}_{metric}'] for row in scores]))
    save_json(output / 'summary.json', dict(
        methods=measures, frames=STOP - START, fps=FPS, shape=SHAPE,
        gaussian_sigmas=sigmas, paired_temporal_frames=len(temporal), metric_provenance=provenance,
        model_run_sha256=file_sha(args.seedvr2_run / 'run.json'), previews=videos,
        boundary_rows=[row for row in temporal if row['frame'] in (192, 193, 215)]))
    print('PASS: aligned texture, source correspondence and perceptual evaluation complete', flush=True)


if __name__ == '__main__':
    main()
