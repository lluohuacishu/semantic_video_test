"""Locate visible texture jumps on two fixed GLC reef clips."""

import argparse
import csv
from pathlib import Path
import time

import cv2
import numpy as np

from run_p0 import HERE, REPO, file_sha, load_json, load_range, open_baseline, preview, psnr, roi, save_json
from reef_repair import SETTINGS, repair_clip
from window_filter import KERNEL, blend_center
from temporal_filter import correspondence, gray, motion, warp

METHODS = ('raw', 'center3', 'source_flow_control')
BANDS = ('coarse', 'middle', 'fine', 'all')
REEF_IDS = ('01_coral_wall', '03_moving_reef')


def bands(rgb):
    image = gray(rgb).astype(np.float32)
    low = cv2.GaussianBlur(image, (0, 0), 1.2)
    coarse = cv2.GaussianBlur(image, (0, 0), 3.0)
    return np.stack((coarse, low - coarse, image - low, image), axis=2)


def alignment_strip(path, source, raw, three, local_index, crop, start, repair=None):
    center = local_index + 1
    back, _ = motion(source[center - 1], source[center])
    _, ahead = motion(source[center], source[center + 1])
    yy, xx = np.mgrid[:back.shape[0], :back.shape[1]].astype(np.float32)
    rows = [('Source', source[center - 1:center + 2]),
            ('GLC raw', raw[center - 1:center + 2]),
            ('Mean 3', three[local_index - 1:local_index + 2])]
    if repair is not None:
        rows.append(('Jump repair', repair[local_index - 1:local_index + 2]))
    height, width = source[center][crop].shape[:2]
    width, height = 2 * width, 2 * height
    title = 25
    sheet = np.full((len(rows) * (height + title), 3 * width, 3), (24, 35, 48), np.uint8)
    for row, (label, (previous, current, following)) in enumerate(rows):
        images = (warp(previous, xx + back[..., 0], yy + back[..., 1]), current,
                  warp(following, xx + ahead[..., 0], yy + ahead[..., 1]))
        y = row * (height + title)
        for col, image in enumerate(images):
            x = col * width
            i = start + local_index + col - 1
            direction = 'center' if col == 1 else 'aligned to center'
            cv2.putText(sheet, f'{label} | {i} | {direction}', (x + 6, y + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (245, 245, 245), 1, cv2.LINE_AA)
            sheet[y + title:y + title + height, x:x + width] = cv2.resize(
                image[crop], (width, height), interpolation=cv2.INTER_NEAREST)
    success, encoded = cv2.imencode('.png', cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError(f'Cannot render {path}')
    path.write_bytes(encoded.tobytes())


def probe(source, raw, three, clip, codec, repair=None):
    crop = roi(clip['coral_roi'], source.shape[1:])
    methods = (*METHODS, 'repair') if repair is not None else METHODS
    coral_sse = dict.fromkeys(methods, 0.0)
    hf_energy = dict.fromkeys(methods, 0.0)
    source_energy = 0.0
    before = motion(source[0], source[1])
    previous = None
    rows = []
    disagreement = []
    for i, old_output in enumerate(three, start=1):
        after = motion(source[i], source[i + 1])
        oracle, _ = blend_center(raw[i - 1], raw[i], raw[i + 1], before, after, fusion='mean')
        current_source = bands(source[i])
        frames = (raw[i], old_output, oracle)
        if repair is not None:
            frames = (*frames, repair[i - 1])
        current = {}
        source_energy += float(np.abs(current_source[crop][..., 2]).sum())
        for name, frame in zip(methods, frames, strict=True):
            current[name] = bands(frame)
            coral_sse[name] += float(np.square(frame[crop].astype(np.float32) - source[i][crop]).sum())
            hf_energy[name] += float(np.abs(current[name][crop][..., 2]).sum())
        if previous is not None:
            index = clip['start'] + i - 1
            is_i = index % codec['intra_period'] == 0
            is_reset = not is_i and index % codec['reset_interval'] == 1
            kind = 'i_frame' if is_i else 'ref_y_reset' if is_reset else 'other'
            back, forward = before
            x, y, inside, fb = correspondence(back, forward)
            previous_source = previous['source']
            photo = np.abs((current_source[..., 3] - current_source[..., 2])
                           - warp(previous_source[..., 3] - previous_source[..., 2], x, y))
            safe = inside & (fb < 1.5) & (photo < 12)
            safe = cv2.erode(safe.astype(np.uint8), KERNEL).astype(bool)
            mask = (safe & (cv2.GaussianBlur(np.abs(current_source[..., 2]), (0, 0), 2) > 2.0))[crop]
            row = dict(frame=index, kind=kind, valid_pixels=int(mask.sum()))
            if row['valid_pixels']:
                true_change = current_source - warp(previous_source, x, y)
                for name in methods:
                    error = np.abs(current[name] - warp(previous[name], x, y) - true_change)[crop][mask]
                    row.update({f'{name}_{band}_sum': float(error[:, j].sum())
                                for j, band in enumerate(BANDS)})
                raw_back, _ = motion(raw[i - 1], raw[i])
                diff = np.linalg.norm(raw_back - back, axis=2)[crop][mask]
                disagreement.append(diff)
                row['flow_disagreement_mean_px'] = float(diff.mean())
            else:
                row.update({f'{m}_{band}_sum': 0.0 for m in methods for band in BANDS})
                row['flow_disagreement_mean_px'] = None
            rows.append(row)
        previous = {**current, 'source': current_source}
        before = after

    total = sum(row['valid_pixels'] for row in rows)
    means = {name: {band: sum(row[f'{name}_{band}_sum'] for row in rows) / total
                    for band in BANDS} for name in methods}
    ranked = sorted((row for row in rows if row['valid_pixels']),
                    key=lambda row: row['raw_all_sum'] / row['valid_pixels'], reverse=True)
    top = ranked[:10]
    flow = np.concatenate(disagreement)
    coral_channels = source[0][crop].size * len(three)
    summary = dict(error=means, valid_pixels=total, spatial={
                       name: dict(coral_rgb_psnr_db=psnr(coral_sse[name] / coral_channels),
                                  coral_fine_energy_vs_source=hf_energy[name] / source_energy)
                       for name in methods
                   },
                   flow_disagreement_px=dict(p50=float(np.median(flow)), p90=float(np.percentile(flow, 90)),
                                             fraction_gt_2=float((flow > 2).mean())),
                   top_raw_all_frames=[dict(frame=row['frame'], kind=row['kind'],
                                            mae={name: row[f'{name}_all_sum'] / row['valid_pixels']
                                                 for name in methods}) for row in top],
                   band_protocol=dict(coarse='Gaussian sigma=3 px', middle='Gaussian sigma=1.2 minus sigma=3',
                                      fine='RGB luma minus Gaussian sigma=1.2', all='unfiltered luma'),
                   mean3_coral_change_from_raw=float(np.mean(
                       np.abs(three[(slice(None), *crop)].astype(np.int16) - raw[1:-1][(slice(None), *crop)]))))
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--previous', type=Path, required=True, help='Completed mean three-frame run')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repair', action='store_true', help='Test agreement-gated repair in the frozen reef ROI')
    parser.add_argument('--ffmpeg', type=Path)
    args = parser.parse_args()
    if args.repair and args.ffmpeg is None:
        parser.error('--repair requires --ffmpeg')
    plan_path = HERE / 'plan_p0.json'
    plan = load_json(plan_path)
    previous = load_json(args.previous / 'summary.json')
    if (not previous['complete'] or previous['center3_fusion'] != 'mean'
            or previous['plan_sha256'] != file_sha(plan_path)):
        raise ValueError('Expected the verified mean three-frame run')
    manifest = args.workspace / '01_项目/GLC后处理/接口与基线.json'
    if previous['baseline_manifest_sha256'] != file_sha(manifest):
        raise ValueError('The RGB baseline manifest changed')
    baselines = {item['id']: item for item in load_json(manifest)['baselines']}
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(code_sha256=file_sha(__file__), input_run_sha256=file_sha(args.previous / 'summary.json'), clips=[])
    if args.repair:
        report.update(repair_code_sha256=file_sha(HERE / 'reef_repair.py'), repair_settings=SETTINGS)
    for old in previous['clips']:
        clip = old['clip']
        identity = clip['id']
        if identity not in REEF_IDS:
            continue
        print(f'{identity}: isolating source-aligned residuals', flush=True)
        baseline = baselines[clip['baseline']]
        decoded, source, receipt = open_baseline(args.workspace, baseline, plan['codec'])
        if receipt != old['input']:
            raise ValueError(f'RGB baseline changed: {identity}')
        start, stop = clip['start'], clip['stop']
        shape = baseline['shape'][1:]
        rgb = load_range(decoded, start - 1, stop + 1, shape)
        reference = load_range(source, start - 1, stop + 1, shape)
        saved = args.previous / identity / 'center3_rgb.npy'
        if file_sha(saved) != old['output_sha256']['center3']:
            raise ValueError(f'Three-frame result changed: {identity}')
        three = np.load(saved, mmap_mode='r', allow_pickle=False)
        if three.shape != rgb[1:-1].shape or three.dtype != np.uint8:
            raise ValueError(f'Incompatible three-frame RGB: {identity}')
        repair = None
        if args.repair:
            tick = time.perf_counter()
            repair = repair_clip(rgb, clip['coral_roi'])
            repair_ms = 1000 * (time.perf_counter() - tick) / len(repair)
        rows, summary = probe(reference, rgb, three, clip, plan['codec'], repair)
        with (args.output / f'{identity}.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        peaks = [item['frame'] for item in summary['top_raw_all_frames'] if start < item['frame'] < stop - 1]
        if peaks:
            alignment_strip(args.output / f'{identity}_aligned.png', reference, rgb, three,
                            peaks[0] - start, roi(clip['coral_roi'], shape), start, repair)
        if repair is not None:
            folder = args.output / identity
            folder.mkdir()
            saved = folder / 'repair_rgb.npy'
            np.save(saved, repair, allow_pickle=False)
            panels = (reference[1:-1], rgb[1:-1], three, repair)
            labels = ('Source RGB', 'GLC raw', '3 frames / mean', 'Local jump repair')
            summary.update(repair_mean_ms=repair_ms, repair_rgb_sha256=file_sha(saved), previews={})
            for name, crop in [('compare.mp4', None), ('zoom.mp4', roi(clip['coral_roi'], shape))]:
                summary['previews'][name] = preview(args.ffmpeg, folder / name, panels,
                                                     baseline['fps'], start, labels, crop)
        report['clips'].append(dict(clip=clip, fps=baseline['fps'], peak_strip_frame=peaks[0] if peaks else None,
                                    **summary))
        print(identity, summary['error'], flush=True)
    save_json(args.output / 'summary.json', report)


if __name__ == '__main__':
    main()
