"""Compare ordinary P-frame changes before and after RGB generation."""

import argparse
from pathlib import Path
from statistics import correlation

import cv2
import numpy as np

from audit_deflicker_texture import source_match
from capture_p_trace import STAGES
from compare_boundary_codec import frame_kind
from compare_deflicker import save_csv
from probe_reef import bands
from run_boundary_codec import REPO, array_sha, file_sha, load_json, load_range, open_baseline, reef_inputs, save_json
from run_deflicker import START, STOP
from run_p0 import roi
from temporal_filter import warp

FEATURES = ('context', 'y_hat', 'latent')
QUIET, PEAK = 175, 215


def feature_grid(x, y, crop, mask, padded, shape):
    height, width = x.shape
    sy, sx = padded[0] / shape[0], padded[1] / shape[1]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    weights = np.zeros(padded, np.float32)
    weights[:height, :width][crop] = mask
    area = cv2.resize(weights, shape[::-1], interpolation=cv2.INTER_AREA)
    offsets = []
    for flow, scale in ((x - xx, sx), (y - yy, sy)):
        field = weights.copy()
        field[:height, :width] *= flow
        area_flow = cv2.resize(field, shape[::-1], interpolation=cv2.INTER_AREA)
        offsets.append(area_flow / np.maximum(area, 1e-6) / scale)
    my, mx = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
    mx += offsets[0]
    my += offsets[1]
    keep = ((area >= .5) & (mx >= 0) & (mx <= shape[1] - 1) &
            (my >= 0) & (my <= shape[0] - 1))
    return mx, my, keep


def area_luma(luma, padded, shape):
    field = cv2.copyMakeBorder(luma, 0, padded[0] - luma.shape[0], 0,
                               padded[1] - luma.shape[1], cv2.BORDER_REPLICATE)
    return cv2.resize(field, shape[::-1], interpolation=cv2.INTER_AREA)


def compare(source, decoded, features, metadata, crop):
    height, width = decoded.shape[1:3]
    padded = features['generator_rgb'].shape[-2:]
    shape = features['latent'].shape[-2:]
    gop, reset = (metadata['header'][key] for key in ('intra_period', 'reset_interval'))
    schedule = metadata['reference_adaptor_schedule']
    slots = {i: slot for slot, i in enumerate(metadata['p_frames'])}
    ordinary = [slot for i, slot in slots.items() if frame_kind(i, gop, reset) == 'ordinary_p']
    fixed_roi = np.zeros(padded, np.float32)
    fixed_roi[:height, :width][crop] = 1.
    fixed_cells = cv2.resize(fixed_roi, shape[::-1], interpolation=cv2.INTER_AREA) >= .5
    channels_last = {key: features[key].transpose(0, 2, 3, 1) for key in FEATURES}
    scales = {}
    for key, video in channels_last.items():
        values = video[ordinary][:, fixed_cells]
        std = np.std(values, axis=(0, 1), dtype=np.float64).astype(np.float32)
        scales[key] = np.maximum(std, max(1e-6, .01 * float(np.median(std))))

    generator = features['generator_rgb'][:, :, :height, :width].transpose(0, 2, 3, 1)
    for i, slot in slots.items():
        display = np.rint((np.clip(generator[slot], -1., 1.) + 1.) * 127.5).astype(np.uint8)
        if not np.array_equal(display, decoded[i - START]):
            raise ValueError(f'Captured generator output differs from display RGB at frame {i}')
    rows = []
    for i in range(START + 1, STOP):
        if i - 1 not in slots or i not in slots:
            continue
        j = i - START
        left, right = slots[i - 1], slots[i]
        before, after = bands(source[j - 1]), bands(source[j])
        x, y, mask = source_match(source[j - 1], source[j], before, after, crop)
        mx, my, cells = feature_grid(x, y, crop, mask, padded, shape)
        if int(cells.sum()) < 16:
            raise ValueError(f'Insufficient aligned feature cells at frame {i}')
        previous_kind, kind = (frame_kind(index, gop, reset) for index in (i - 1, i))
        previous_fa, fa = (schedule[index % len(schedule)] for index in (i - 1, i))
        row = dict(frame=i, previous_kind=previous_kind, kind=kind, phase=f'{previous_fa}->{fa}',
                   valid_pixels=int(mask.sum()), valid_feature_cells=int(cells.sum()))
        for key, video in channels_last.items():
            current, previous = video[right], video[left]
            aligned = warp(np.ascontiguousarray(previous), mx, my)
            row[f'{key}_aligned_mae'] = float((np.abs(current[cells] - aligned[cells]) / scales[key]).mean())
            row[f'{key}_unwarped_mae'] = float((np.abs(current[cells] - previous[cells]) / scales[key]).mean())
        true_delta = after[..., 3] - warp(before[..., 3], x, y)
        displayed = [bands(decoded[index])[..., 3] for index in (j - 1, j)]
        floating = [bands((generator[index] + 1.) * 127.5)[..., 3] for index in (left, right)]
        for key, (old, new) in (('rgb', displayed), ('generator_rgb', floating)):
            delta = new - warp(old, x, y)
            row[f'{key}_aligned_mae'] = float(np.abs(delta - true_delta)[crop][mask].mean())
        old_small, new_small = [area_luma(frame, padded, shape) for frame in displayed]
        source_old, source_new = [area_luma(frame[..., 3], padded, shape) for frame in (before, after)]
        coarse_error = (new_small - warp(old_small, mx, my)) - (source_new - warp(source_old, mx, my))
        row['rgb_at_feature_scale_mae'] = float(np.abs(coarse_error)[cells].mean())
        rows.append(row)
    return rows


def same_phase_summary(rows):
    chosen = {row['frame']: row for row in rows if row['frame'] in (QUIET, PEAK)}
    quiet, peak = chosen[QUIET], chosen[PEAK]
    if quiet['phase'] != peak['phase']:
        raise ValueError('The selected P-frame pairs have different feature-adaptor phases')
    peers = [row for row in rows if row['phase'] == peak['phase']
             and row['kind'] == row['previous_kind'] == 'ordinary_p']
    if quiet not in peers or peak not in peers:
        raise ValueError('Expected two ordinary P-to-P pairs')
    keys = [key for key in peak if key.endswith('_mae')]
    metrics = {}
    for key in keys:
        values = np.array([row[key] for row in peers])
        metrics[key] = dict(quiet=quiet[key], peak=peak[key], peer_median=float(np.median(values)),
                            peak_rank_from_largest=int(1 + np.count_nonzero(values > peak[key])))
    return dict(quiet_frame=QUIET, peak_frame=PEAK, phase=peak['phase'], peer_count=len(peers),
                peer_frames=[row['frame'] for row in peers], metrics=metrics)


def phase_correlations(rows):
    groups = {}
    for row in rows:
        if row['kind'] == row['previous_kind'] == 'ordinary_p':
            groups.setdefault(row['phase'], []).append(row)
    keys = [f'{key}_aligned_mae' for key in FEATURES] + ['rgb_at_feature_scale_mae']
    return {phase: dict(pairs=len(group), spearman_to_display={
        key: correlation([row[key] for row in group], [row['rgb_aligned_mae'] for row in group], method='ranked')
        for key in keys}) for phase, group in sorted(groups.items())}


def plot(path, rows, sample, fps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, layout='constrained')
    time = [row['frame'] / fps for row in rows]
    for name, color in zip(FEATURES, ('#395a84', '#c17730', '#277e69'), strict=True):
        axes[0].plot(time, [row[f'{name}_aligned_mae'] for row in rows], label=name, color=color, lw=1.35)
    for key, label, style in [('rgb', 'Displayed RGB', '-'), ('generator_rgb', 'Generator before clamp', '--')]:
        axes[1].plot(time, [row[f'{key}_aligned_mae'] for row in rows], style, label=label, lw=1.35)
    for axis in axes:
        for key, label in [('quiet_frame', '175'), ('peak_frame', '215')]:
            x = sample[key] / fps
            axis.axvline(x, color='#666666', alpha=.5, lw=1, linestyle=':')
            axis.text(x, .96, f' {label}', transform=axis.get_xaxis_transform(), va='top', fontsize=8)
        axis.legend(ncol=3, fontsize=8)
        axis.grid(alpha=.20)
    axes[0].set_ylabel('Feature MAE / channel SD')
    axes[1].set_ylabel('Luma temporal MAE')
    axes[1].set_xlabel('Time in source video (seconds)')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    plan, clip, baseline = reef_inputs(args.workspace)
    reference, source, original = open_baseline(args.workspace, baseline, plan)
    source = load_range(source, START, STOP, baseline['shape'][1:])
    expected = load_range(reference, START, STOP, baseline['shape'][1:])
    metadata_path = args.capture / 'capture.json'
    metadata = load_json(metadata_path)
    if (not metadata['complete'] or not metadata['source_media_reads_forbidden']
            or metadata['stream_sha256'] != original['stream_sha256']
            or (metadata['start'], metadata['stop']) != (START, STOP)
            or any(metadata['header'][key] != value for key, value in plan.items())):
        raise ValueError('Trace does not correspond to the original FP32 pilot')
    arrays = {}
    for name in (*STAGES, 'decoded_rgb'):
        path = args.capture / f'{name}.npy'
        info = metadata['files'][name]
        if file_sha(path) != info['sha256']:
            raise ValueError(f'Trace tensor changed: {name}')
        arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
        if list(arrays[name].shape) != info['shape'] or str(arrays[name].dtype) != info['dtype']:
            raise ValueError(f'Invalid trace shape or dtype: {name}')
    decoded = arrays.pop('decoded_rgb')
    if not np.array_equal(decoded, expected):
        raise ValueError('Captured RGB differs from the archived original decoder output')
    rows = compare(source, decoded, arrays, metadata, roi(clip['coral_roi'], source.shape[1:]))
    sample = same_phase_summary(rows)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    save_csv(output / 'per_frame.csv', rows)
    plot(output / 'p_trace.png', rows, sample, baseline['fps'])
    save_json(output / 'summary.json', dict(
        start=START, stop=STOP, fps=baseline['fps'], coral_roi=clip['coral_roi'], compared_p_pairs=len(rows),
        ordinary_p_pairs=sum(row['kind'] == row['previous_kind'] == 'ordinary_p' for row in rows),
        source_rgb_sha256=array_sha(source), capture_sha256=file_sha(metadata_path),
        decoded_rgb_bit_exact=True, generator_capture_round_trip_bit_exact=True, same_phase=sample,
        phase_correlations=phase_correlations(rows),
        code_sha256=file_sha(__file__)))
    print('PASS: source-aligned P-frame features and display RGB analyzed', flush=True)


if __name__ == '__main__':
    main()
