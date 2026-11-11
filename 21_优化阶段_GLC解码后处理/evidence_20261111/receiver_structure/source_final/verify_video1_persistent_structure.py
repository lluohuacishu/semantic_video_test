"""Read-back checks for the pilot, expanded coral, and autoencoder diagnostics."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from coral_reference_sidecar import file_sha, load_json, save_json
from run_video1_persistent_structure import START, STOP, VIEW_BOX


def inspect_preview(ffmpeg, path, expected, crop=None):
    probe = json.loads(subprocess.check_output([str(ffmpeg.with_name('ffprobe.exe')),
        '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,avg_frame_rate,nb_read_frames,duration', '-of', 'json', str(path)]))['streams'][0]
    count = len(expected[0])
    if int(probe['nb_read_frames']) != count or Fraction(probe['avg_frame_rate']) != 15:
        raise ValueError('Wrong preview timeline')
    decoded = subprocess.check_output([str(ffmpeg), '-v', 'error', '-i', str(path),
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'])
    frames = np.frombuffer(decoded, np.uint8).reshape(count, probe['height'], probe['width'], 3)
    h, w = expected[0].shape[1:3] if crop is None else (2*(crop[3]-crop[1]), 2*(crop[2]-crop[0]))
    errors = []
    for panel, reference in enumerate(expected):
        x, y = (panel % 2)*w, (panel//2)*(h+26)+26
        sample_errors = []
        for i in (0, count//2, count-1):
            target = reference[i]
            if crop is not None:
                x0, y0, x1, y1 = crop
                target = cv2.resize(target[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_NEAREST)
            diff = np.abs(frames[i, y:y+h, x:x+w].astype(np.float32)-target)
            sample_errors.append(float(diff.mean()))
        if max(sample_errors) > 6:
            raise ValueError(f'Preview panel differs from its declared lossless source: {path}/{panel}')
        errors.append(sample_errors)
    return dict(probe=probe, frame_indices=[0, count//2, count-1], panel_rgb_mae=errors,
                sha256=file_sha(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    receipts = {name: load_json(root/name/'summary.json') for name in ('pilot', 'expanded', 'bypass')}
    inputs = load_json(root/'pilot/inputs.json')
    inputs_exact = all(file_sha(Path(path)) == digest for path, digest in inputs['files'].items())
    if not inputs_exact:
        raise ValueError('A frozen input changed')
    base_path = next(Path(p) for p in inputs['files'] if Path(p).name == 'decoded_rgb.npy')
    source_path = next(Path(p) for p in inputs['files'] if Path(p).name == 'source.npy')
    s_path = next(Path(p) for p in inputs['files'] if Path(p).parent.name == 'structure')
    raw = np.load(base_path, mmap_mode='r')
    source = np.load(source_path, mmap_mode='r')
    s = np.load(s_path, mmap_mode='r')
    arrays = {}
    for folder, receipt in receipts.items():
        key = 'outputs_sha256' if folder == 'pilot' else 'output_sha256'
        for name, digest in receipt[key].items():
            path = root/folder/name
            if file_sha(path) != digest:
                raise ValueError(f'Changed output: {path}')
            a = np.load(path, allow_pickle=False)
            arrays[(folder, name)] = a
            if name.endswith('_rgb.npy') and (a.dtype != np.uint8 or a.shape != ((150 if folder == 'bypass' else 23), 288, 512, 3)):
                raise ValueError('Invalid RGB output')
    for folder, names in (('pilot', ('C', 'R', 'B0')), ('expanded', ('C2', 'R2'))):
        alpha = arrays[(folder, 'alpha.npy')]
        if not np.isfinite(alpha).all() or alpha.min() < 0 or alpha.max() > 1:
            raise ValueError('Invalid blend mask')
        for name in names:
            rgb = arrays[(folder, name+'_rgb.npy')]
            if not np.array_equal(rgb[alpha == 0], raw[START:STOP][alpha == 0]):
                raise ValueError('Zero-alpha pixels changed')
    for name in ('source', 'raw', 'C'):
        if receipts['pilot']['scores']['fixed'][name] != receipts['expanded']['scores']['fixed'][name]:
            raise ValueError('Frozen pilot metrics differ under the expanded evaluation')
    for name in ('source', 'raw'):
        if receipts['pilot']['scores'][ 'tracked'][name] != receipts['bypass']['pilot_scores']['tracked'][name]:
            raise ValueError('Bypass evaluation does not use the exact pilot masks')
    checks = [
        ('pilot/pilot_coral_4way.mp4', [raw[START:STOP], s[START:STOP], arrays[('pilot','C_rgb.npy')], arrays[('pilot','R_rgb.npy')]], VIEW_BOX),
        ('expanded/expanded_coral_4way.mp4', [raw[START:STOP], arrays[('pilot','C_rgb.npy')], arrays[('expanded','C2_rgb.npy')], arrays[('expanded','R2_rgb.npy')]], VIEW_BOX),
        ('expanded/expanded_coral_pair.mp4', [raw[START:STOP], arrays[('expanded','C2_rgb.npy')]], VIEW_BOX),
        ('expanded/expanded_full_pair.mp4', [raw[START:STOP], arrays[('expanded','C2_rgb.npy')]], None),
        ('bypass/bypass_coral_10s.mp4', [source, raw, arrays[('bypass','fixed_video_rgb.npy')], arrays[('bypass','switched_IP_rgb.npy')]], VIEW_BOX),
    ]
    previews = {name: inspect_preview(args.ffmpeg, root/name, sequence, crop) for name, sequence, crop in checks}
    save_json(root/'verification.json', dict(complete=True, frozen_inputs_unchanged=inputs_exact,
        all_saved_array_hashes_checked=True, rgb_dtype_shape_verified=True,
        shared_fixed_roi_metrics_exact=True, bypass_pilot_mask_metrics_exact=True,
        previews=previews, code_sha256=file_sha(__file__)))
    print('PASS: frozen inputs, saved arrays, shared metrics, and five decoded previews', flush=True)


if __name__ == '__main__':
    main()
