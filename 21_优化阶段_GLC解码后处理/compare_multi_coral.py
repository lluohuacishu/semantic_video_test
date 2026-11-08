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


def paired(raw, enhanced):
    title = 26
    canvas = np.full((HEIGHT+title, WIDTH*2, 3), (24, 35, 48), np.uint8)
    for i, (before, after) in enumerate(zip(raw, enhanced, strict=True)):
        canvas[:title] = (24, 35, 48)
        canvas[title:, :WIDTH] = before
        canvas[title:, WIDTH:] = after
        for x, label in ((8, 'GLC raw'), (WIDTH+8, 'Multi-tile source')):
            cv2.putText(canvas, f'{label} | {(START+i)/FPS:.2f}s', (x, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, .47, (240, 244, 250), 1, cv2.LINE_AA)
        yield canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--base-run', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
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
    untouched = alpha == 0
    for video in arrays.values():
        if not np.array_equal(video[untouched], raw[untouched]):
            raise ValueError('Composition affected pixels beyond the transmitted support')
        for i, (x0,y0,x1,y1) in enumerate(boxes):
            if not np.array_equal(video[i,y0:y1,x0:x1], raw[i,y0:y1,x0:x1]):
                raise ValueError('Foreground exclusion failed')
    old = load_json(base_run/'received/result.json')
    single_path = base_run/'received/source_anchor.npy'
    if old['base_sha256'] != base_sha or old['files']['source_anchor'] != file_sha(single_path):
        raise ValueError('The prior single-tile result changed')
    single = np.load(single_path, mmap_mode='r', allow_pickle=False)
    sequences = dict(source=source, raw=raw, single=single, **arrays)
    active = alpha >= 128
    area = float(active.sum()/len(raw))
    if area != send['mean_effective_pixels'] or area != receive['mean_effective_pixels']:
        raise ValueError('Sender and receiver disagree on applied coverage')
    output = run/'comparison'
    output.mkdir(exist_ok=False)
    labels = ('Source RGB','GLC raw','One source tile','Multiple source tiles')
    panels = (source, raw, single, arrays['source_multi'])
    previews = {
        'before_after.mp4': encode(args.ffmpeg, output/'before_after.mp4', paired(raw, arrays['source_multi']),
                                   (2*WIDTH,HEIGHT+26), FPS, len(raw)),
        'compare.mp4': preview(args.ffmpeg, output/'compare.mp4', panels, FPS, START, labels),
        'reef_zoom.mp4': preview(args.ffmpeg, output/'reef_zoom.mp4', panels, FPS, START, labels,
                                 np.s_[102:288,104:512]),
        'source_multi.mp4': encode(args.ffmpeg, output/'source_multi.mp4', arrays['source_multi'],
                                   (WIDTH,HEIGHT), FPS, len(raw)),
    }
    samples = []
    for i in (10,44,73):
        tinted = raw[i].copy()
        tinted[active[i]] = np.rint(.55*tinted[active[i]] + .45*np.array([105,245,180])).astype(np.uint8)
        samples.append(np.hstack((raw[i], tinted, arrays['source_multi'][i])))
    Image.fromarray(np.vstack(samples)).save(output/'coverage.png')
    rows = measure(sequences, alpha)
    save_csv(output/'per_frame.csv', rows)
    valid = [row for row in rows if row['raw_ncc'] is not None]
    pixels = sum(row['shared_active_pixels'] for row in valid)
    scores = {key: {metric: sum(row[f'{key}_{metric}']*row['shared_active_pixels'] for row in valid)/pixels
                    for metric in ('ncc','to_source_ncc','energy')} for key in sequences}
    old_area = load_json(base_run/'comparison/summary.json')['mean_effective_area_pixels']
    duration = len(raw)/FPS
    save_json(output/'summary.json', dict(complete=True, frames=len(raw), fps=FPS, tiles=len(tiles),
        semantic_bytes=send['semantic_bytes'], semantic_kbps=send['semantic_bytes']*8/duration/1000,
        semantic_byte_limit=LIMIT, base_sha256=base_sha, excluded_fish_region_bit_exact=True,
        mean_effective_pixels=area, effective_frame_fraction=area/(HEIGHT*WIDTH), prior_effective_pixels=old_area,
        coverage_multiple=area/old_area, pairs=len(valid), mean_shared_metric_pixels=pixels/len(valid),
        scores=scores, previews=previews, send_sha256=file_sha(run/'send.json'),
        receive_sha256=file_sha(run/'received/result.json'), code_sha256=file_sha(__file__)))
    print(f'Verified {len(raw)} frames at {FPS} FPS, {area/(HEIGHT*WIDTH):.1%} mean coverage', flush=True)


if __name__ == '__main__':
    main()
