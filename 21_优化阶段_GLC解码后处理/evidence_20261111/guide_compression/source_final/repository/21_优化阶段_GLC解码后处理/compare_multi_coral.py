"""Verify coverage, temporal metrics and original-speed videos for the multi-tile stream."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from compare_coral_reference import measure
from compare_deflicker import save_csv
from compare_underwater_enhancement import encode
from coral_reference_sidecar import (
    BASE_START, START, STOP, FPS, HEIGHT, WIDTH, LIMIT, REPO,
    baseline, file_sha, load_json, load_range, open_baseline, save_json,
)
from multi_coral_reference import read_package
from run_p0 import preview


def paired(raw, enhanced, labels=('GLC raw', 'Multi-tile source')):
    title = 26
    canvas = np.full((HEIGHT+title, WIDTH*2, 3), (24, 35, 48), np.uint8)
    for i, (before, after) in enumerate(zip(raw, enhanced, strict=True)):
        canvas[:title] = (24, 35, 48)
        canvas[title:, :WIDTH] = before
        canvas[title:, WIDTH:] = after
        for x, label in zip((8, WIDTH+8), labels, strict=True):
            cv2.putText(canvas, f'{label} | {(START+i)/FPS:.2f}s', (x, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, .47, (240, 244, 250), 1, cv2.LINE_AA)
        yield canvas


def summarize(rows, names):
    valid = [row for row in rows if row['raw_ncc'] is not None]
    pixels = sum(row['shared_active_pixels'] for row in valid)
    if not pixels:
        raise ValueError('No reliable frame pairs in this evaluation region')
    scores = {key: {metric: sum(row[f'{key}_{metric}']*row['shared_active_pixels'] for row in valid)/pixels
                    for metric in ('ncc','to_source_ncc','energy')} for key in names}
    return dict(pairs=len(valid), mean_shared_metric_pixels=pixels/len(valid), scores=scores)


def prior_result(folder, base_sha, method):
    receipt = load_json(folder/'received/result.json')
    path = folder/'received'/f'{method}.npy'
    if receipt['base_sha256'] != base_sha or receipt['files'][method] != file_sha(path):
        raise ValueError(f'The prior {method} result changed')
    return np.load(path, mmap_mode='r', allow_pickle=False)


def by_second(mask):
    return mask.mean(axis=(1,2)).reshape(-1, FPS).mean(axis=1).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--base-run', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prior-multi-run', type=Path, help='Verified all-source multi-tile run for the hybrid comparison')
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    base_run, run = args.base_run, args.run
    base_sha = file_sha(base_run/'base.glcsnd')
    send = load_json(run/'send.json')
    receive = load_json(run/'received/result.json')
    package = run/'coral_multi.glcsem'
    if (not receive['complete'] or not receive['source_media_reads_forbidden']
            or not receive['excluded_fish_region_bit_exact']
            or not base_sha == receive['base_sha256'] == send['base_sha256']
            or not file_sha(package) == send['semantic_sha256'] == receive['semantic_sha256']
            or not package.stat().st_size == sum(send['ledger'].values()) == send['semantic_bytes'] <= LIMIT):
        raise ValueError('The base stream or semantic byte ledger changed')
    tiles, boxes = read_package(package, base_sha)
    spec = baseline(args.workspace)
    _, reference, _ = open_baseline(args.workspace, spec, dict(q=0, intra_period=64, reset_interval=32))
    source = load_range(reference, START, STOP, (HEIGHT, WIDTH, 3))
    raw = np.load(base_run/'base_receive/decoded_rgb.npy', mmap_mode='r', allow_pickle=False)[START-BASE_START:]
    arrays = {}
    for name, digest in receive['files'].items():
        path = run/'received'/f'{name}.npy'
        if file_sha(path) != digest:
            raise ValueError(f'Changed receiver array {name}')
        arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
    alpha = arrays.pop('alpha')
    source_alpha = arrays.pop('source_alpha', None)
    hybrid = source_alpha is not None
    if hybrid != (args.prior_multi_run is not None):
        parser.error('--prior-multi-run is required for the hybrid result')
    untouched = alpha == 0
    for video in arrays.values():
        if not np.array_equal(video[untouched], raw[untouched]):
            raise ValueError('Composition affected pixels beyond the transmitted support')
        for i, (x0,y0,x1,y1) in enumerate(boxes):
            if not np.array_equal(video[i,y0:y1,x0:x1], raw[i,y0:y1,x0:x1]):
                raise ValueError('Foreground exclusion failed')
    if hybrid:
        prior = prior_result(args.prior_multi_run, base_sha, 'source_multi')
        prior_alpha = prior_result(args.prior_multi_run, base_sha, 'alpha')
        source_outside = source_alpha == 0
        if (np.any(source_alpha > alpha) or
                not np.array_equal(arrays['hybrid'][source_outside], arrays['glc_multi'][source_outside])):
            raise ValueError('Hybrid source texture changed the cache-only area')
        prior_name, method = 'all_source', 'hybrid'
        before_label, after_label = 'Prior all-source tiles', 'Hybrid references'
    else:
        prior = prior_result(base_run, base_sha, 'source_anchor')
        prior_name, method = 'single', 'source_multi'
        before_label, after_label = 'One source tile', 'Multiple source tiles'
    sequences = dict(source=source, raw=raw, **{prior_name: prior}, **arrays)
    enhanced = arrays[method]
    active = alpha >= 128
    area = float(active.sum()/len(raw))
    if area != send['mean_effective_pixels'] or area != receive['mean_effective_pixels']:
        raise ValueError('Sender and receiver disagree on applied coverage')
    output = run/'comparison'
    output.mkdir(exist_ok=False)
    labels = ('Source RGB','GLC raw',before_label,after_label)
    panels = (source, raw, prior, enhanced)
    previews = {
        'before_after.mp4': encode(args.ffmpeg, output/'before_after.mp4',
                                   paired(raw, enhanced, ('GLC raw', after_label)), (2*WIDTH,HEIGHT+26), FPS, len(raw)),
        'compare.mp4': preview(args.ffmpeg, output/'compare.mp4', panels, FPS, START, labels),
        'reef_zoom.mp4': preview(args.ffmpeg, output/'reef_zoom.mp4', panels, FPS, START, labels,
                                 np.s_[102:288,104:512]),
        f'{method}.mp4': encode(args.ffmpeg, output/f'{method}.mp4', enhanced,
                                   (WIDTH,HEIGHT), FPS, len(raw)),
    }
    if hybrid:
        previews['previous_vs_hybrid.mp4'] = encode(
            args.ffmpeg, output/'previous_vs_hybrid.mp4', paired(prior, enhanced, (before_label, after_label)),
            (2*WIDTH,HEIGHT+26), FPS, len(raw))
    samples = []
    for i in (10,44,73):
        tinted = raw[i].copy()
        tinted[active[i]] = np.rint(.55*tinted[active[i]] + .45*np.array([105,245,180])).astype(np.uint8)
        if hybrid:
            from_source = source_alpha[i] >= 128
            tinted[from_source] = np.rint(.55*raw[i][from_source] + .45*np.array([255,178,63])).astype(np.uint8)
        samples.append(np.hstack((raw[i], tinted, enhanced[i])))
    Image.fromarray(np.vstack(samples)).save(output/'coverage.png')
    rows = measure(sequences, alpha)
    save_csv(output/'per_frame.csv', rows)
    report = summarize(rows, sequences)
    extra = {}
    if hybrid:
        prior_area = float((prior_alpha >= 128).sum()/len(raw))
        source_area = float((source_alpha >= 128).sum()/len(raw))
        if (source_area != send['mean_effective_source_pixels'] or source_area != receive['mean_effective_source_pixels']
                or send['source_tiles'] != sum(tile.image is not None for tile in tiles)):
            raise ValueError('Sender and receiver disagree on transmitted source coverage')
        regions = {}
        for name, mask in [('source', source_alpha), ('cached_only', np.where(source_outside, alpha, 0)),
                           ('shared_prior', np.minimum(alpha, prior_alpha))]:
            region_rows = measure(sequences, mask)
            save_csv(output/f'per_frame_{name}.csv', region_rows)
            regions[name] = summarize(region_rows, sequences)
        late = [row for row in rows if row['frame'] >= STOP-2*FPS]
        extra.update(
            source_tiles=send['source_tiles'], source_fraction=source_area/(HEIGHT*WIDTH),
            coverage_by_second=dict(all_source=by_second(prior_alpha>=128),
                                    hybrid=by_second(active), hybrid_source=by_second(source_alpha>=128)),
            source_outside_bit_exact_to_cache=True, region_metrics=regions,
            last_two_seconds=summarize(late, sequences))
    else:
        prior_area = load_json(base_run/'comparison/summary.json')['mean_effective_area_pixels']
    duration = len(raw)/FPS
    save_json(output/'summary.json', dict(complete=True, frames=len(raw), fps=FPS, tiles=len(tiles),
        semantic_bytes=send['semantic_bytes'], semantic_kbps=send['semantic_bytes']*8/duration/1000,
        semantic_byte_limit=LIMIT, base_sha256=base_sha, excluded_fish_region_bit_exact=True,
        mean_effective_pixels=area, effective_frame_fraction=area/(HEIGHT*WIDTH), prior_effective_pixels=prior_area,
        coverage_multiple=area/prior_area, **report, **extra, previews=previews, send_sha256=file_sha(run/'send.json'),
        receive_sha256=file_sha(run/'received/result.json'), code_sha256=file_sha(__file__)))
    print(f'Verified {len(raw)} frames at {FPS} FPS, {area/(HEIGHT*WIDTH):.1%} mean coverage', flush=True)


if __name__ == '__main__':
    main()
