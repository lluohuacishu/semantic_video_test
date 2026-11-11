"""Render matched underwater previews and a restrained UVE-Net grade."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image

from audit_deflicker_texture import measure, normalized, summarize
from compare_deflicker import perceptual, save_csv
from probe_reef import bands
from run_underwater_enhancement import REPO, array_sha, file_sha, fixed_clips, load_json, load_range, save_json
from run_p0 import preview, psnr, roi
from c1_io import RawPreview
from temporal_filter import gray

METHODS = ('uve', 'undive')
SAMPLE_FRAMES = {'01_coral_wall': 169, '02_fish_turn': 492, '03_moving_reef': 634}


def checked_model(folder, run, clip, baseline, receipt, raw):
    record = next(item for item in run['clips'] if item['clip']['id'] == clip['id'])
    path = folder / f'{clip["id"]}.npy'
    if (record['clip'] != clip or record['glc_baseline'] != receipt
            or (record['frames'], record['fps']) != (len(raw), baseline['fps'])
            or record['input_rgb_sha256'] != array_sha(raw)
            or record['output_npy_sha256'] != file_sha(path)):
        raise ValueError(f'Model result does not match the frozen clip: {path}')
    frames = np.load(path, mmap_mode='r', allow_pickle=False)
    if frames.shape != raw.shape or frames.dtype != raw.dtype:
        raise ValueError(f'Invalid lossless RGB: {path}')
    return frames


def encode(ffmpeg, path, frames, size, fps, count):
    width, height = size
    writer = RawPreview(ffmpeg, path, width, height, fps)
    try:
        for frame in frames:
            writer.add(frame)
        writer.finish()
    except BaseException:
        writer.abort()
        raise
    ffprobe = ffmpeg.with_name('ffprobe' + ffmpeg.suffix)
    result = json.loads(subprocess.check_output([str(ffprobe), '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames,duration',
        '-of', 'json', str(path)]))['streams'][0]
    if ((result['width'], result['height']) != size or Fraction(result['avg_frame_rate']) != Fraction(fps)
            or int(result['nb_read_frames']) != count or abs(float(result['duration']) - count / fps) > .001):
        raise ValueError(f'Output timebase or geometry differs: {path}')
    return dict(result, sha256=file_sha(path))


def paired_frames(raw, enhanced, start, fps):
    h, w = raw.shape[1:3]
    title = 26
    canvas = np.full((h + title, 2 * w, 3), (24, 35, 48), np.uint8)
    for index, (before, after) in enumerate(zip(raw, enhanced, strict=True)):
        canvas[:title] = (24, 35, 48)
        canvas[title:, :w] = before
        canvas[title:, w:] = after
        for col, label in enumerate(('GLC raw', 'UVE-Net 70%')):
            cv2.putText(canvas, f'{label} | {(start + index) / fps:.2f}s', (col * w + 8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, .47, (240, 244, 250), 1, cv2.LINE_AA)
        yield canvas


def color_stats(frames):
    luma = np.stack([gray(frame) for frame in frames])
    p05, p50, p95 = np.percentile(luma, (5, 50, 95))
    return dict(luma_p05=float(p05), luma_p50=float(p50), luma_p95=float(p95),
                luma_at_least_245_fraction=float(np.mean(luma >= 245)))


def fine_structure(raw, candidate, crop):
    total, pixels = 0., 0
    for original, edited in zip(raw, candidate, strict=True):
        reference = bands(original)[..., 2]
        actual = bands(edited)[..., 2]
        mask = cv2.GaussianBlur(np.abs(reference), (0, 0), 2)[crop] > 2.
        x, y = reference[crop][mask], actual[crop][mask]
        total += float((normalized(x) * normalized(y)).sum())
        pixels += len(x)
    return total / pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--uve-run', type=Path, required=True)
    parser.add_argument('--undive-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    parser.add_argument('--metrics-root', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    folders = dict(uve=args.uve_run, undive=args.undive_run)
    runs = {key: load_json(folder / 'run.json') for key, folder in folders.items()}
    if any(not run['complete'] or run['method'] != key for key, run in runs.items()):
        raise ValueError('Expected both completed underwater model runs')
    args.output.mkdir(parents=True, exist_ok=False)
    reports, stills = [], []
    for clip, baseline, decoded, reference, receipt in fixed_clips(args.workspace):
        start, stop = clip['start'], clip['stop']
        fps = baseline['fps']
        source = load_range(reference, start, stop, baseline['shape'][1:])
        raw = load_range(decoded, start, stop, baseline['shape'][1:])
        candidates = {key: checked_model(folders[key], runs[key], clip, baseline, receipt, raw) for key in METHODS}
        balanced = ((7 * candidates['uve'].astype(np.uint16) + 3 * raw.astype(np.uint16) + 5) // 10).astype(np.uint8)
        methods = dict(raw=raw, **candidates, balanced=balanced)
        seq = dict(source=source, **methods)
        out = args.output / clip['id']
        out.mkdir()
        rgb_file = out / 'enhanced_rgb.npy'
        np.save(rgb_file, balanced, allow_pickle=False)
        h, w = raw.shape[1:3]
        outputs = {
            'enhanced.mp4': encode(args.ffmpeg, out / 'enhanced.mp4', balanced, (w, h), fps, len(raw)),
            'before_after.mp4': encode(args.ffmpeg, out / 'before_after.mp4',
                                       paired_frames(raw, balanced, start, fps), (2 * w, h + 26), fps, len(raw)),
        }
        panels = (source, raw, candidates['uve'], candidates['undive'])
        labels = ('Source RGB', 'GLC raw', 'UVE-Net', 'UnDIVE RGB')
        outputs['models.mp4'] = preview(args.ffmpeg, out / 'models.mp4', panels, fps, start, labels)
        if clip['id'] == '02_fish_turn':
            outputs['fish_zoom.mp4'] = preview(args.ffmpeg, out / 'fish_zoom.mp4', panels, fps, start,
                                              labels, roi(clip['motion_roi'], raw.shape[1:]))
        coral = roi(clip['coral_roi'], raw.shape[1:])
        details = roi(clip['motion_roi'], raw.shape[1:])
        temporal = measure(seq, coral, start)
        save_csv(out / 'per_frame_temporal.csv', temporal)
        results = summarize(temporal, seq, coral)
        for name, rgb in seq.items():
            results[name].update(color_stats(rgb))
        for name, rgb in methods.items():
            results[name]['rgb_psnr_db'] = psnr(float(np.mean((rgb.astype(np.float32) - source) ** 2, dtype=np.float64)))
            results[name]['motion_roi_fine_ncc_to_glc'] = fine_structure(raw, rgb, details)
        scores, metric_provenance = perceptual(source, methods, args.metrics_root.resolve(), start=start)
        save_csv(out / 'per_frame_perceptual.csv', scores)
        for name in methods:
            for metric in ('lpips', 'dists'):
                results[name][metric] = float(np.mean([row[f'{name}_{metric}'] for row in scores]))
        reports.append(dict(clip=clip, fps=fps, frames=len(raw), shape=list(raw.shape),
                            source_rgb_sha256=array_sha(source), glc_rgb_sha256=array_sha(raw),
                            enhanced_rgb_file_sha256=file_sha(rgb_file), previews=outputs,
                            paired_temporal_frames=len(temporal), results=results, metric_provenance=metric_provenance))
        sample = SAMPLE_FRAMES[clip['id']] - start
        stills.append(next(paired_frames(raw[sample:sample+1], balanced[sample:sample+1], start + sample, fps)).copy())
        print(f'{clip["id"]}: model and balanced previews plus full-reference scores verified', flush=True)
    Image.fromarray(np.vstack(stills)).save(args.output / 'balanced_samples.png')
    save_json(args.output / 'summary.json', dict(complete=True, clips=reports, uve_mix=.70,
        model_run_sha256={key: file_sha(folder / 'run.json') for key, folder in folders.items()},
        code_sha256=file_sha(__file__)))


if __name__ == '__main__':
    main()
