"""Evaluate the delivered motion by reliability ablation on fixed reef regions."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from audit_deflicker_texture import measure
from cache_motion_inputs import FPS, FRAME_SHAPE, HEIGHT, WIDTH, base_frames
from compare_cache_motion import read_rgb, score_scope
from compare_deflicker import save_csv
from compare_seedvr2 import source_detail
from compare_source_dense_flow import load_flow
from coral_reference_sidecar import file_sha, gray, load_json, save_json
from experiment_common import array_sha
from gate_aux_stream import MOTION, byte_limit, expand, unpack_grid
from run_cache_motion import CACHE_FRAMES
from run_dense_batch import CASES
from run_dense_transfer import source_input
from run_gate_ablation import PILOTS, WIRES, editable, receiver_flow
from run_p0 import preview, roi


DETAILS = {'upper_plate': (342, 10, 502, 105), 'lower_reef': (348, 186, 494, 260)}
LABELS = dict(raw='GLC raw', A='A / Receiver only', B='B / Sent reliability',
    C='C / Sent motion', D='D / Both sent', upper='O / Offline upper')


def checked_variants(run, raw, clip, base_sha, allowed):
    sender = load_json(run/'sender/result.json')
    sequences, rates, coverage = {}, {}, {}
    outputs = {key: run/'receive'/key for key in WIRES} | {'upper': run/'upper'}
    for key, path in outputs.items():
        data = load_json(path/'result.json')
        if data['base_sha256'] != base_sha or data['offline_upper'] != (key == 'upper'):
            raise ValueError(f'Incorrect {key} receipt')
        if key in WIRES:
            packets = {name: run/'delivery'/key/name for name in WIRES[key]}
            files = {name: dict(sha256=file_sha(path), bytes=path.stat().st_size)
                     for name, path in packets.items()}
            size = sum(row['bytes'] for row in files.values())
            if (files != {name: sender['files'][name] for name in packets}
                    or data['files'] != {name: row['sha256'] for name, row in files.items()}
                    or size != data['extra_bytes'] or size > byte_limit(clip)
                    or not data['source_media_reads_forbidden']):
                raise ValueError(f'{key} auxiliary file provenance differs from its ablation')
            rates[key] = dict(bytes=size, kbps=size*8*FPS/clip.count/1000, files=files)
        rgb = read_rgb(path, 'rgb', data['rgb_sha256'], clip)
        alpha = np.load(path/'alpha.npy', mmap_mode='r', allow_pickle=False)
        if (alpha.shape != raw.shape[:3] or alpha.dtype != np.uint8
                or array_sha(alpha) != data['alpha_sha256'] or alpha[:, ~allowed].any()
                or not np.array_equal(rgb[:, ~allowed], raw[:, ~allowed])
                or not np.array_equal(rgb[::CACHE_FRAMES], raw[::CACHE_FRAMES])):
            raise ValueError(f'{key} changed protected pixels or anchor frames')
        sequences[key] = rgb
        coverage[key] = alpha
    return sequences, rates, coverage


def alignment(source, fields, crop, allowed):
    yy, xx = np.mgrid[:HEIGHT, :WIDTH].astype(np.float32)
    luma = np.stack([gray(frame).astype(np.float32) for frame in source])
    count = 0
    sums = dict.fromkeys(fields, 0.)
    for i, target in enumerate(luma):
        if i % CACHE_FRAMES == 0:
            anchor = target
            continue
        shared = allowed[crop].copy()
        predictions = {}
        for name, flow in fields.items():
            x, y = xx+flow[i, ..., 0], yy+flow[i, ..., 1]
            shared &= ((x >= 2) & (y >= 2) & (x < WIDTH-2) & (y < HEIGHT-2))[crop]
            predictions[name] = cv2.remap(anchor, x, y, cv2.INTER_LINEAR)[crop]
        count += int(shared.sum())
        for name, values in predictions.items():
            sums[name] += float(np.abs(values-target[crop])[shared].sum())
    return {name: total/count for name, total in sums.items()}


def coarse_scores(rows, methods):
    pixels = sum(row['valid_pixels'] for row in rows)
    return {name: sum(row[f'{name}_coarse_sum'] for row in rows)/pixels for name in methods}


def region_scores(output, name, sequences, coverage, crop, clip):
    temporal = measure(sequences, crop, start=clip.start, include_coarse=True)
    spatial = source_detail(sequences['source'], sequences, crop, start=clip.start)
    save_csv(output/f'{name}_temporal.csv', temporal)
    scopes = {}
    for key, rows, same, frames in (
            ('full', temporal, spatial, np.s_[:]),
            ('last2', temporal[-2*FPS:], spatial[-2*FPS:], np.s_[-2*FPS:])):
        metrics = score_scope(rows, same, {name: video[frames] for name, video in sequences.items()}, crop)
        coarse = coarse_scores(rows, sequences)
        for method in sequences:
            metrics['scores'][method]['coarse_temporal_mae'] = coarse[method]
        metrics['coverage'] = {method: float((alpha[frames][:, crop[0], crop[1]] > 0).mean())
                               for method, alpha in coverage.items()}
        scopes[key] = metrics
    boundaries = [row for row in temporal if (row['frame']-clip.start) % CACHE_FRAMES == 0]
    scopes['coarse_at_cache_refresh'] = coarse_scores(boundaries, sequences)
    return scopes


def evaluate(base, run, clip, case, ffmpeg):
    raw, base_sha, size = base_frames(base, clip)
    source = source_input(base, clip)
    allowed = editable(case)
    restored, rates, coverage = checked_variants(run, raw, clip, base_sha, allowed)
    sequences = dict(source=source, raw=raw, **restored)
    source_motion, _ = load_flow(base/'flow', source, clip)
    sent = expand(unpack_grid((run/'delivery/C'/MOTION).read_bytes(), MOTION, clip, base_sha))
    fields = dict(receiver=receiver_flow(run, raw, clip), sent=sent, upper=source_motion)
    boxes = dict(reef=clip.coral)
    if case == 'video1':
        boxes.update(DETAILS)
    crops = {'image': np.s_[:, :]} | {name: roi(box, FRAME_SHAPE) for name, box in boxes.items()}
    geometry = {name: alignment(source, fields, crops[name], allowed) for name in boxes}
    del fields, sent

    output = run/'comparison'
    output.mkdir(exist_ok=False)
    scopes = {name: region_scores(output, name, sequences, coverage, crop, clip)
              for name, crop in crops.items()}
    focus = crops['upper_plate'] if case == 'video1' else crops['reef']
    videos = (
        ('ablation_full', ('A','B','C','D'), None),
        ('ablation_detail', ('A','B','C','D'), focus),
        ('offline_upper_detail', ('raw','A','D','upper'), focus),
    )
    previews = {name: preview(ffmpeg, output/f'{name}.mp4', [sequences[key] for key in keys],
        FPS, clip.start, [LABELS[key] for key in keys], crop) for name, keys, crop in videos}
    save_json(run/'summary.json', dict(case=case, base_sha256=base_sha, base_bytes=size,
        base_kbps=size*8*FPS/clip.count/1000, fps=FPS, frames=clip.count,
        protected_pixel_fraction=float((~allowed).mean()), protected_pixels_bit_exact=True,
        fixed_regions=boxes, actual_auxiliary_files=rates, upper_offline_only=True,
        source_anchor_warp_mae=geometry, fixed_scores=scopes, previews=previews,
        code_sha256=file_sha(__file__)))
    print(f'{case}: fixed-region ablation and {len(previews)} previews verified', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=PILOTS, required=True)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    clip, _ = CASES[args.case]
    evaluate(args.batch.resolve()/args.case, args.run.resolve()/args.case,
             clip, args.case, args.ffmpeg.resolve())


if __name__ == '__main__':
    main()
