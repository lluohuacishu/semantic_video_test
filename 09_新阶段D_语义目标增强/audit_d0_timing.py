"""Match selected annotated RGB to CPU-decoded MP4 frames and preserve actual PTS.

This audits timing only; publisher RGB remains the quality reference. Ambiguous
matches remain unassigned instead of inventing a frame rate or fixed offset.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageFilter

from prepare_d0 import file_sha, load_json, save_json


def classify_matches(matches, timestamps, time_base):
    accepted = all(row['accepted'] for row in matches)
    indices = [row['best_source_frame_index'] for row in matches]
    increasing = all(b > a for a, b in zip(indices, indices[1:]))
    dense = accepted and increasing and all(b == a + 1 for a, b in zip(indices, indices[1:]))
    pts = [Fraction(timestamps[i]['best_effort_timestamp']) * time_base for i in indices]
    steps = [b - a for a, b in zip(pts, pts[1:])]
    positive_time = all(step > 0 for step in steps)
    constant_step = positive_time and bool(steps) and len(set(steps)) == 1
    return {'all_selected_matches_accepted': accepted, 'strictly_increasing_source_indices': increasing,
            'dense_source_frames': dense, 'positive_pts_steps': positive_time, 'constant_pts_step': constant_step,
            'source_indices': indices, 'fps_if_constant': str(1 / steps[0]) if constant_step else None,
            'ready_for_dense_video_comparison': dense and constant_step}


def audit(sequence, root, ffmpeg, ffprobe):
    path = root / sequence['video_path']
    if file_sha(path) != sequence['video_sha256']:
        raise ValueError('Source MP4 changed since corpus preparation')
    probe_command = [str(ffprobe), '-v', 'error', '-select_streams', 'v:0', '-show_streams',
                     '-show_frames', '-show_entries',
                     'stream=width,height,time_base,avg_frame_rate,r_frame_rate,nb_frames,duration:'
                     'frame=best_effort_timestamp,best_effort_timestamp_time,duration,duration_time',
                     '-of', 'json', str(path)]
    probe = json.loads(subprocess.check_output(probe_command))
    stream = probe['streams'][0]
    timestamps = probe['frames']
    if any('best_effort_timestamp' not in row for row in timestamps):
        raise ValueError('Missing source PTS')
    width, height = 480, 270
    command = [str(ffmpeg), '-v', 'error', '-threads', '2', '-i', str(path), '-map', '0:v:0', '-an',
               '-vf', f'scale={width}:{height}:flags=bilinear', '-fps_mode', 'passthrough',
               '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1']
    raw = subprocess.check_output(command)
    frame_bytes = width * height * 3
    if len(raw) % frame_bytes:
        raise ValueError('Partial raw frame')
    video = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)
    if len(video) != len(timestamps):
        raise ValueError('Probe/decode frame count mismatch')
    matches = []
    for row in sequence['frames']:
        rgb_path = root / row['rgb_path']
        mask_path = root / row['mask_path']
        if file_sha(rgb_path) != row['source_file_sha256'] or file_sha(mask_path) != row['mask_file_sha256']:
            raise ValueError('Selected RGB or mask changed')
        reference = np.asarray(Image.open(rgb_path).convert('RGB').resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
        mask = Image.open(mask_path).resize((width, height), Image.Resampling.NEAREST).filter(ImageFilter.MaxFilter(9))
        roi = np.asarray(mask) > 0
        if not roi.any():
            raise ValueError('Timing matcher requires a positive annotated target')
        target = reference[roi]
        mse = []
        for start in range(0, len(video), 32):
            delta = video[start:start + 32, roi].astype(np.float32) - target
            mse.extend(np.square(delta).mean(axis=(1, 2)).tolist())
        order = np.argsort(mse, kind='stable')
        best, second = int(order[0]), int(order[1])
        global_mse = float(np.square(video[best].astype(np.float32) - reference).mean())
        psnr = float(10 * np.log10(255 ** 2 / max(global_mse, 1e-12)))
        ratio = mse[second] / max(mse[best], 1e-12)
        accepted = psnr >= 30 and ratio >= 1.03
        matches.append({'sample_index': row['index'], 'original_file_name': row['original_file_name'],
                        'annotation_frame_id': row['annotation_frame_id'], 'best_source_frame_index': best,
                        'runner_up_source_frame_index': second, 'best_roi_mse': mse[best],
                        'runner_up_roi_mse': mse[second], 'runner_up_to_best_ratio': ratio,
                        'global_psnr_db_at_best': psnr, 'accepted': accepted,
                        'source_pts_ticks': timestamps[best]['best_effort_timestamp'] if accepted else None,
                        'source_pts_seconds': timestamps[best]['best_effort_timestamp_time'] if accepted else None})
    time_base = Fraction(stream['time_base'])
    result = {'video_id': sequence['video_id'], 'role': sequence['role'], 'source_stream': stream,
              'decoded_source_frames': len(video), 'comparison_size': [width, height],
              'method': 'All MP4 frames searched by MSE on GT union dilated 4 pixels at 480x270; global PSNR verifies full frame',
              'acceptance_frozen': {'minimum_global_psnr_db': 30, 'minimum_runner_up_to_best_roi_mse_ratio': 1.03},
              'probe_command': probe_command, 'decode_command': command, 'matches': matches,
              **classify_matches(matches, timestamps, time_base)}
    if result['ready_for_dense_video_comparison']:
        first, last = result['source_indices'][0], result['source_indices'][-1]
        frame_step = Fraction(1, 1) / Fraction(result['fps_if_constant'])
        result['clip_duration_seconds_exact'] = str(len(matches) * frame_step)
        result['clip_duration_seconds'] = float(len(matches) * frame_step)
        result['source_pts_start_ticks'] = timestamps[first]['best_effort_timestamp']
        result['source_pts_last_ticks'] = timestamps[last]['best_effort_timestamp']
    print(json.dumps({'video_id': sequence['video_id'], 'accepted': sum(r['accepted'] for r in matches),
                      'frames': len(matches), 'dense_ready': result['ready_for_dense_video_comparison'],
                      'first_last_best_index': [matches[0]['best_source_frame_index'], matches[-1]['best_source_frame_index']],
                      'minimum_psnr': min(r['global_psnr_db_at_best'] for r in matches),
                      'minimum_uniqueness_ratio': min(r['runner_up_to_best_ratio'] for r in matches)}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    parser.add_argument('--ffprobe', type=Path, required=True)
    args = parser.parse_args()
    output = args.corpus / 'timing_audit.json'
    if output.exists():
        raise FileExistsError(output)
    manifest = load_json(args.corpus / 'manifest.json')
    results = []
    for sequence in manifest['fish_sequences']:
        result = audit(sequence, args.corpus, args.ffmpeg, args.ffprobe)
        results.append(result)
        save_json(args.corpus / 'fish' / f'video_{sequence["video_id"]:03d}' / 'timing.json', result)
    report = {'schema': 'd0-timing-audit-v1', 'model_loaded': False, 'gpu_used': False,
              'source_manifest_sha256': file_sha(args.corpus / 'manifest.json'),
              'script_sha256': file_sha(Path(__file__)), 'ffmpeg_version': subprocess.check_output([str(args.ffmpeg), '-version'], text=True).splitlines()[0],
              'source_independence_established': False, 'sequences': results}
    save_json(output, report)


if __name__ == '__main__':
    main()
