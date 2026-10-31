"""CPU-only evaluation. Ground truth is opened only after independent decoding."""
from __future__ import annotations

import argparse
import math
import statistics
from decoder_common import *
import numpy as np
from evaluate_experiment import quality
from run_decoder_experiment import verify_results


def summary_quality(expected, actual):
    mse, maximum, changed, count = 0., 0, 0, 0
    for left, right in zip(expected, actual):
        error = right.astype(np.float64) - left
        mse += float(np.sum(error * error))
        maximum = max(maximum, int(np.abs(error).max()))
        changed += int(np.count_nonzero(error))
        count += error.size
    return dict(mse=mse/count, psnr_vs_original_rgb_db=10*math.log10(255**2/max(mse/count, 1e-12)),
                maximum_rgb_error=maximum, changed_rgb_values=changed,
                changed_rgb_fraction=changed/count, bit_exact=changed == 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=('short', 'multi', 'full'), required=True)
    args = p.parse_args()
    runs = load_json(TRIAL / (args.suite + '_summary.json'))['runs']
    plan = load_json(ARCHIVE / 'plan.json')
    reports = []
    for row in runs:
        folder = Path(row['folder'])
        if args.suite == 'short' and row['repeat'] != 1:
            continue
        if args.suite == 'short':
            clip = 'long108_q0_640'
            baseline_folder = TRIAL / 'short/eager1'
        elif args.suite == 'multi':
            clip = row['clip']
            baseline_folder = TRIAL / 'multi' / clip / 'eager'
        else:
            clip = 'uploaded_half_1239_q0_640'
            baseline_folder = TRIAL / 'full1239/eager'
        result, baseline = load_json(folder / 'result.json'), load_json(baseline_folder / 'result.json')
        verify_results(baseline, result)
        for path, expected_sha in ((folder / 'decoded_rgb.npy', result['decoded_rgb_sha256']),
                                   (baseline_folder / 'decoded_rgb.npy', baseline['decoded_rgb_sha256'])):
            if file_sha(path) != expected_sha:
                raise ValueError('Decoded RGB changed before CPU evaluation')
        actual = np.load(folder / 'decoded_rgb.npy', mmap_mode='r')
        old = np.load(baseline_folder / 'decoded_rgb.npy', mmap_mode='r')
        if actual.shape != old.shape or actual.dtype != np.uint8:
            raise ValueError('RGB array shape/type differs')
        mask = None
        if args.suite != 'full':
            item = next(c for c in plan['clips'] if c['id'] == clip)
            if file_sha(item['source']) != item['source_sha256']:
                raise ValueError('GT source identity differs')
            gt = np.load(item['source'], mmap_mode='r')
            if clip.startswith(('sf007', 'sf025')):
                name = 'v007' if clip.startswith('sf007') else 'v025'
                mask_path = WORKSPACE / '02_实验数据/优化_GLC蒸馏/pilot_20261029_01' / name / 'fish_mask.npy'
                mask = np.load(mask_path, mmap_mode='r')
            q = quality(gt, old, actual, mask)
        else:
            # Read original upload on CPU at the exact existing half-rate preprocessing,
            # rather than treating the lossy H.264 preview as ground truth.
            from types import SimpleNamespace
            from c1_io import MemoryFrames
            source = WORKSPACE / '02_实验数据/C_编解码加速与网页展示/C5_loopback/sender/inference_lab/cfb1a1b4cba37b3d569a04b4ab73bcf2/upload.mp4'
            reader = MemoryFrames(SimpleNamespace(input=source, width=640, frames=2**31-1,
                                                   start_frame=0, frame_stride=2), collect_hashes=False)
            try:
                q = quality(reader, old, actual)
            finally:
                reader.close()
            if len(q['frames']) != result['frames']:
                raise ValueError('Uploaded source frame selection count differs')
        if len(q['frames']) != result['frames']:
            raise ValueError('GT/mask/decoded length differs')
        timing = [load_json(Path(r['folder']) / 'result.json')['core_seconds'] for r in runs
                  if r['backend'] == row['backend'] and (args.suite != 'multi' or r['clip'] == clip)]
        base_timing = [load_json(Path(r['folder']) / 'result.json')['core_seconds'] for r in runs
                       if r['backend'] == 'eager' and (args.suite != 'multi' or r['clip'] == clip)]
        report = dict(clip=clip, backend=row['backend'], frames=result['frames'],
            stream_sha256=result['stream_sha256'], stream_bytes=result['stream_bytes'],
            core_seconds=timing, median_core_seconds=statistics.median(timing),
            median_fps=result['frames']/statistics.median(timing),
            speedup_vs_eager=statistics.median(base_timing)/statistics.median(timing),
            all_reference_states_and_entropy_identical=True,
            rgb_difference=summary_quality(old, actual), quality=q,
            fish_mask_policy='Existing annotation-union mask, not a precise fish silhouette' if mask is not None else None)
        reports.append(report)
        print(f'{clip} {row["backend"]}: {report["median_fps"]:.3f}fps, {report["speedup_vs_eager"]:.3f}x; P PSNR delta={q["groups"]["P"]["psnr_delta_db"]:.6f}dB', flush=True)
    save_json(TRIAL / (args.suite + '_quality.json'), dict(complete=True, reports=reports,
        gt_read_by_receiver=False, rgb_evaluated_before_preview_compression=True,
        temporal_metric_is_error_difference_proxy=True, independent_source_split_established=False,
        Orin_tested=False, training_steps=0))


if __name__ == '__main__':
    main()
