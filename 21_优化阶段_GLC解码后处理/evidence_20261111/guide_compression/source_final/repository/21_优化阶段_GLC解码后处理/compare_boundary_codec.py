"""Compare GLC boundary placements at the same source timestamps."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from audit_deflicker_texture import measure, summarize
from compare_deflicker import save_csv
from run_boundary_codec import CASES, REPO, array_sha, file_sha, load_json, load_range, open_baseline, reef_inputs, save_json
from run_p0 import preview, psnr, roi

KINDS = ('i', 'first_p', 'reset_p', 'ordinary_p')
LABELS = {'g64_r32': 'GOP 64 / reset 32', 'g72_r32': 'GOP 72 / reset 32', 'g64_r24': 'GOP 64 / reset 24'}
BASELINE = 'g64_r32'


def frame_kind(index, gop, reset):
    if index % gop == 0:
        return 'i'
    if (index - 1) % gop == 0:
        return 'first_p'
    if reset and index % reset == 1:
        return 'reset_p'
    return 'ordinary_p'


def aligned_metrics(rows, name):
    total = sum(row['valid_pixels'] for row in rows)
    return dict(frames=len(rows), valid_pixels=total,
                full_mae=sum(row[f'{name}_full_sum'] for row in rows) / total,
                fine_mae=sum(row[f'{name}_fine_sum'] for row in rows) / total,
                fine_ncc=sum(row[f'{name}_fine_ncc'] * row['valid_pixels'] for row in rows) / total)


def crossovers(rows):
    result = {}
    for name in CASES:
        if name == BASELINE:
            continue
        groups = {}
        for row in rows:
            original, candidate = row[f'{BASELINE}_kind'], row[f'{name}_kind']
            if original == candidate:
                continue
            old_mae, new_mae = (row[f'{m}_full_sum'] / row['valid_pixels'] for m in (BASELINE, name))
            groups.setdefault(f'{original}_to_{candidate}', []).append(dict(
                frame=row['frame'], baseline_mae=old_mae, candidate_mae=new_mae, delta=new_mae - old_mae))
        result[name] = {kind: dict(pairs=items, mean_delta=float(np.mean([item['delta'] for item in items])))
                        for kind, items in groups.items()}
    return result


def plot(path, rows, fps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, sharey=True, layout='constrained')
    colors = ('#395a84', '#a2426e', '#277e69')
    markers = {'i': ('^', 'I frame'), 'first_p': ('o', 'First P'), 'reset_p': ('D', 'Effective reset')}
    time = np.array([row['frame'] / fps for row in rows])
    for axis, name, color in zip(axes, CASES, colors, strict=True):
        mae = np.array([row[f'{name}_full_sum'] / row['valid_pixels'] for row in rows])
        axis.plot(time, mae, color=color, lw=1.3)
        for kind, (marker, label) in markers.items():
            matched = [row[f'{name}_kind'] == kind for row in rows]
            axis.scatter(time[matched], mae[matched], marker=marker, s=28, label=label, color=color, zorder=3)
        axis.set_title(LABELS[name], loc='left')
        axis.set_ylabel('Temporal MAE')
        axis.grid(alpha=.20)
    axes[0].legend(ncol=3, fontsize=8, loc='upper left')
    axes[-1].set_xlabel('Time in source video (seconds)')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    plan, clip, baseline = reef_inputs(args.workspace)
    _, reference, original = open_baseline(args.workspace, baseline, plan)
    start, stop = clip['start'], clip['stop']
    source = load_range(reference, 0, stop, baseline['shape'][1:])
    source_digest = array_sha(source)
    receipt_path = args.run / 'codec_summary.json'
    receipt = load_json(receipt_path)
    if not receipt['complete'] or [case['id'] for case in receipt['cases']] != list(CASES):
        raise ValueError('Expected all three boundary cases')
    sequences = dict(source=source[start:stop])
    for case in receipt['cases']:
        name = case['id']
        path = args.run / name / 'receiver/decoded_rgb.npy'
        if (file_sha(path) != case['decoded_rgb_sha256'] or case['source_rgb_sha256'] != source_digest
                or (case['intra_period'], case['reset_interval']) != CASES[name]
                or (case['q'], case['fps'], case['frames']) != (plan['q'], baseline['fps'], stop)):
            raise ValueError(f'Boundary result differs from the matched input and settings: {name}')
        frames = np.load(path, mmap_mode='r', allow_pickle=False)
        if frames.shape != source.shape or frames.dtype != source.dtype:
            raise ValueError(f'Unexpected receiver shape or precision: {name}')
        sequences[name] = frames[start:stop]
    if not receipt['cases'][0]['original_packet_and_rgb_match']:
        raise ValueError('Expected a bit-exact baseline reproduction')
    crop = roi(clip['coral_roi'], source.shape[1:])
    rows = measure(sequences, crop, start)
    metrics = summarize(rows, sequences, crop)
    for row in rows:
        for name, (gop, reset) in CASES.items():
            row[f'{name}_kind'] = frame_kind(row['frame'], gop, reset)
    for name in CASES:
        metrics[name]['by_kind'] = {
            kind: aligned_metrics([row for row in rows if row[f'{name}_kind'] == kind], name)
            for kind in KINDS}
        error = sequences[name].astype(np.float32) - sequences['source']
        metrics[name]['rgb_psnr_db'] = psnr(float(np.mean(error ** 2, dtype=np.float64)))
        metrics[name]['coral_rgb_psnr_db'] = psnr(float(np.mean(error[(slice(None), *crop)] ** 2, dtype=np.float64)))
    shared_ordinary = [row for row in rows if all(
        frame_kind(index, *settings) == 'ordinary_p'
        for settings in CASES.values() for index in (row['frame'] - 1, row['frame']))]
    same_gop_boundaries = [row['frame'] for row in rows if row[f'{BASELINE}_kind'] in ('i', 'first_p')]
    if not all(np.array_equal(sequences[BASELINE][i - start], sequences['g64_r24'][i - start])
               for i in same_gop_boundaries):
        raise ValueError('Reset-only case changed the I frame or its immediate first P')
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    save_csv(output / 'per_frame.csv', rows)
    fps = baseline['fps']
    plot(output / 'boundaries.png', rows, fps)
    labels = ('Source RGB', *(LABELS[name] for name in CASES))
    previews = {name: preview(args.ffmpeg, output / name, tuple(sequences.values()), fps, start, labels, region)
                for name, region in (('compare.mp4', None), ('zoom.mp4', crop))}
    save_json(output / 'summary.json', dict(
        start=start, stop=stop, fps=fps, coral_roi=clip['coral_roi'], paired_frames=len(rows),
        cases=receipt['cases'], metrics=metrics, same_frame_crossovers=crossovers(rows), previews=previews,
        shared_ordinary_pairs={name: aligned_metrics(shared_ordinary, name) for name in CASES},
        reset_only_i_and_first_p_bit_exact=same_gop_boundaries,
        codec_summary_sha256=file_sha(receipt_path), original=original, code_sha256=file_sha(__file__)))
    print('PASS: matched RGB boundary analysis and both original-speed previews', flush=True)


if __name__ == '__main__':
    main()
