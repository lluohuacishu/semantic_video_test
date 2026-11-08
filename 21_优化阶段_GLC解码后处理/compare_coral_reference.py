"""Verify and compare the fixed 15 FPS coral-reference pilot."""

import argparse
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from audit_deflicker_texture import normalized, source_match
from compare_deflicker import save_csv
from compare_underwater_enhancement import encode
from coral_reference_sidecar import (
    BASE_START, START, STOP, FPS, HEIGHT, WIDTH, LIMIT, RECORD, SCALES,
    REPO, baseline, file_sha, load_json, load_range, open_baseline, save_json,
)
from probe_reef import bands
from run_p0 import preview
from temporal_filter import warp


def measure(sequences, alpha):
    source = sequences['source']
    crop = np.s_[:, :]
    previous = {key: bands(rgb[0]) for key, rgb in sequences.items()}
    rows = []
    for i in range(1, len(source)):
        current = {key: bands(rgb[i]) for key, rgb in sequences.items()}
        mx, my, mask = source_match(source[i-1], source[i], previous['source'], current['source'], crop)
        mask &= (alpha[i] >= 128) & (warp(alpha[i-1], mx, my) >= 128)
        count = int(mask.sum())
        row = dict(frame=START+i, shared_active_pixels=count)
        for key in sequences:
            row[f'{key}_ncc'] = row[f'{key}_to_source_ncc'] = row[f'{key}_energy'] = None
        if count >= 64:
            ref = normalized(current['source'][..., 2][mask])
            for key in sequences:
                new = current[key][..., 2][mask]
                old = warp(previous[key][..., 2], mx, my)[mask]
                z = normalized(new)
                row[f'{key}_ncc'] = float((normalized(old)*z).mean())
                row[f'{key}_to_source_ncc'] = float((z*ref).mean())
                row[f'{key}_energy'] = float(np.abs(new).mean())
        rows.append(row)
        previous = current
    return rows


def tracked_zoom(videos, matrices, tile_shape):
    h, w = tile_shape
    result = []
    for video in videos:
        frames = []
        for rgb, matrix in zip(video, matrices, strict=True):
            patch = cv2.warpAffine(rgb, matrix, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            frames.append(cv2.resize(patch, (w*3, h*3), interpolation=cv2.INTER_NEAREST))
        result.append(np.stack(frames))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    run = args.run
    send = load_json(run / 'send.json')
    received = run / 'received'
    receive = load_json(received / 'result.json')
    semantic, base = run / 'coral.glcsem', run / 'base.glcsnd'
    if (not receive['complete'] or not receive['source_media_reads_forbidden']
            or not file_sha(semantic) == send['semantic_sha256'] == receive['semantic_sha256']
            or file_sha(base) != receive['base_sha256']
            or not semantic.stat().st_size == sum(send['ledger'].values()) == send['semantic_bytes'] <= LIMIT):
        raise ValueError('Semantic receipt or independent byte ledger changed')
    spec = baseline(args.workspace)
    decoded, source, _ = open_baseline(args.workspace, spec, dict(q=0, intra_period=64, reset_interval=32))
    expected = load_range(decoded, BASE_START, STOP, (HEIGHT, WIDTH, 3))
    received_base = np.load(run / 'base_receive/decoded_rgb.npy', mmap_mode='r', allow_pickle=False)
    if not np.array_equal(received_base, expected):
        raise ValueError('The independent fragment decode differs from the original 15 FPS GLC RGB')
    raw = received_base[START-BASE_START:]
    source = load_range(source, START, STOP, (HEIGHT, WIDTH, 3))
    arrays = {}
    for name, digest in receive['files'].items():
        path = received / f'{name}.npy'
        if file_sha(path) != digest:
            raise ValueError(f'Receiver output changed: {name}')
        arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
    alpha = arrays.pop('alpha')
    untouched = alpha == 0
    if any(not np.array_equal(video[untouched], raw[untouched]) for video in arrays.values()):
        raise ValueError('Composition changed pixels outside its transmitted support')
    sequences = dict(source=source, raw=raw, **arrays)
    x0, y0, x1, y1 = send['manifest']['anchor_box']
    with ZipFile(semantic) as archive:
        tile = cv2.imdecode(np.frombuffer(archive.read('anchor.png'), np.uint8), cv2.IMREAD_COLOR)
        records = np.frombuffer(archive.read('track.bin'), dtype=RECORD)
    if not np.array_equal(tile[..., ::-1], source[0, y0:y1, x0:x1]):
        raise ValueError('Received reference PNG differs from the original texture')
    matrices = records['affine'].astype(np.float32) / SCALES
    output = run / 'comparison'
    output.mkdir(exist_ok=False)
    labels = ('Source RGB', 'GLC raw', 'Reuse GLC tile', 'Reuse sent source tile')
    panels = tuple(sequences.values())
    previews = {
        'compare.mp4': preview(args.ffmpeg, output / 'compare.mp4', panels, FPS, START, labels),
        'tracking_zoom.mp4': preview(args.ffmpeg, output / 'tracking_zoom.mp4',
            tracked_zoom(panels, matrices, (y1-y0, x1-x0)), FPS, START, labels),
        'source_anchor.mp4': encode(args.ffmpeg, output / 'source_anchor.mp4', arrays['source_anchor'],
                                    (WIDTH, HEIGHT), FPS, len(raw)),
    }
    rows = measure(sequences, alpha)
    save_csv(output / 'per_frame.csv', rows)
    valid = [row for row in rows if row['raw_ncc'] is not None]
    pixels = sum(row['shared_active_pixels'] for row in valid)
    scores = {key: {metric: sum(row[f'{key}_{metric}'] * row['shared_active_pixels'] for row in valid) / pixels
                    for metric in ('ncc', 'to_source_ncc', 'energy')} for key in sequences}
    duration = len(raw) / FPS
    result = dict(
        complete=True, frames=len(raw), fps=FPS, duration_seconds=duration,
        base_decoded_rgb_bit_exact=True, source_tile_bit_exact=True, outside_support_bit_exact=True,
        base_bytes=base.stat().st_size, semantic_bytes=semantic.stat().st_size, semantic_limit_bytes=LIMIT,
        base_coded_duration_seconds=len(received_base)/FPS, base_kbps=base.stat().st_size*8/(len(received_base)/FPS)/1000,
        semantic_kbps=semantic.stat().st_size*8/duration/1000,
        combined_kbps_over_display=(semantic.stat().st_size+base.stat().st_size)*8/duration/1000,
        pairs=len(valid), mean_effective_area_pixels=float(np.mean((alpha>=128).sum(axis=(1, 2)))),
        mean_shared_metric_pixels=pixels/len(valid), active_frames=receive['active_frames'],
        scores=scores, previews=previews, send_sha256=file_sha(run/'send.json'),
        receive_sha256=file_sha(received/'result.json'), code_sha256=file_sha(__file__))
    save_json(output / 'summary.json', result)
    print(f'PASS: {len(raw)} frames at {FPS} FPS; sidecar {result["semantic_kbps"]:.3f} kbps', flush=True)


if __name__ == '__main__':
    main()
