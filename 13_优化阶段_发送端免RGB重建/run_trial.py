"""Freeze three existing clips, run three alternating sender pairs, decode alone."""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

from sender_common import HERE, REPO, SCRIPTS, file_sha, load_json, load_settings, save_json


def prepare(output):
    import cv2
    from PIL import Image
    from glc_video_probe import extract_frames
    workspace = load_settings()['workspace']
    data = workspace / '02_实验数据'
    output.mkdir(parents=True, exist_ok=False)
    prepared = output / 'prepared'
    prepared.mkdir()
    source = data / 'A_压缩路线选型/A2_GLC验证/long108/input_540.mkv'
    info = extract_frames(argparse.Namespace(input=source, frames=66, start_frame=0,
                          width=640, frame_stride=2), prepared, cv2, Image)
    if (info['processed_frames'], info['width'], info['height']) != (66, 640, 360):
        raise ValueError('Unexpected diagnostic video geometry/length')
    array = np.stack([np.asarray(Image.open(prepared / 'source' / f'im{i+1:05d}.png').convert('RGB'))
                      for i in range(66)])
    np.save(prepared / 'long108.npy', array)
    definitions = [dict(id='long108_q0_640', source=prepared / 'long108.npy', width=640, height=360,
                        frames=66, fps=info['fps'], q=0,
                        original_source=str(source), original_source_sha256=file_sha(source),
                        historical_baseline_stream=str(data / 'C_编解码加速与网页展示/GLC_operator_profile_20261025_171041/encode/baseline/stream.glcv'))]
    specs = [('sf007_q12_512', 'pilot_20261029_01/v007', 96, 12),
             ('uv108_q17_512', 'round2_20261029_01/uv108', 66, 17)]
    for name, rel, frames, q in specs:
        cached = data / '优化_GLC蒸馏' / rel / 'source.npy'
        target = prepared / (name + '.npy')
        np.save(target, np.asarray(np.load(cached, mmap_mode='r')[:frames]))
        entry = dict(id=name, source=target, width=512, height=288, frames=frames, fps=30.0, q=q,
                     original_source=str(cached), original_source_sha256=file_sha(cached),
                     timebase='frame-order only; nominal FPS is not capture timing')
        if name.startswith('sf007'):
            entry['historical_baseline_stream'] = str(cached.with_name('stream.glcv'))
        definitions.append(entry)
    for clip in definitions:
        clip['source'] = str(clip['source'])
        clip['source_sha256'] = file_sha(clip['source'])
        if 'historical_baseline_stream' in clip:
            clip['historical_baseline_sha256'] = file_sha(clip['historical_baseline_stream'])
    files = [*HERE.glob('*.py'), SCRIPTS / 'glc_video_codec.py', SCRIPTS / 'glc_container.py',
             REPO / '08_新阶段C_编解码加速与网页展示/c2_native.py']
    plan = dict(schema='glc-sender-no-rgb-trial-v1', pairs=3, warmup_frames=8, clips=definitions,
                pair_orders=[['baseline', 'no_rgb'], ['no_rgb', 'baseline'], ['baseline', 'no_rgb']],
                code_sha256={str(p.relative_to(REPO)): file_sha(p) for p in files},
                source_preparation=info,
                scope='One RTX 4060 Laptop GPU; fresh sequential processes; no Orin/power/acoustic-channel test')
    save_json(output / 'plan.json', plan)
    save_json(output / 'plan_identity.json', dict(sha256=file_sha(output / 'plan.json')))
    return plan


def job(output, clip, role, folder, stream=None):
    result = folder / 'result.json'
    if result.exists():
        return load_json(result)
    command = [str(load_settings()['runtime']['python']), str(HERE / 'trial_worker.py'),
               '--plan', str(output / 'plan.json'), '--clip', clip['id'], '--role', role, '--output', str(folder)]
    if stream is not None:
        command += ['--stream', str(stream)]
    folder.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with folder.with_suffix('.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    value = load_json(result)
    value['controller_process_wall_seconds'] = time.perf_counter() - start
    save_json(result, value)
    print(f'{clip["id"]} {role} core={value["core_seconds"]:.3f}s task={value["task_seconds"]:.3f}s', flush=True)
    return value


def compare_frames(left, right, rgb=False):
    keys = ['index', 'kind', 'latent_sha256', 'payload_sha256', 'entropy', 'payload_bytes']
    if rgb:
        keys.append('rgb_sha256')
    if len(left['records']) != len(right['records']):
        raise ValueError('Frame count differs')
    for a, b in zip(left['records'], right['records']):
        for key in keys:
            if a[key] != b[key]:
                raise ValueError(f'Frame {a["index"]}: {key} differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    plan = load_json(output / 'plan.json') if (output / 'plan.json').exists() else prepare(output)
    if file_sha(output / 'plan.json') != load_json(output / 'plan_identity.json')['sha256']:
        raise ValueError('Frozen plan changed')
    for name, digest in plan['code_sha256'].items():
        if file_sha(REPO / name) != digest:
            raise ValueError(f'Frozen trial code changed: {name}; use a new output directory')
    if args.prepare_only:
        print('Prepared frozen inputs', flush=True)
        return
    reports = []
    for clip in plan['clips']:
        runs = []
        for pair, order in enumerate(plan['pair_orders'], 1):
            pair_runs = {}
            for role in order:
                folder = output / clip['id'] / f'pair{pair}_{role}'
                pair_runs[role] = job(output, clip, role, folder)
            compare_frames(pair_runs['baseline'], pair_runs['no_rgb'])
            if pair > 1:
                compare_frames(runs[0]['baseline'], pair_runs['baseline'], rgb=True)
                if runs[0]['no_rgb']['stream_sha256'] != pair_runs['no_rgb']['stream_sha256']:
                    raise ValueError('No-RGB output changed across repetitions')
            runs.append(pair_runs)
        baseline, sender = runs[0]['baseline'], runs[0]['no_rgb']
        if clip.get('historical_baseline_sha256') and baseline['stream_sha256'] != clip['historical_baseline_sha256']:
            raise ValueError('Default RGB path changed versus the historical byte-identical baseline')
        decoded = job(output, clip, 'decode', output / clip['id'] / 'receiver',
                      output / clip['id'] / 'pair1_no_rgb/stream.glcsnd')
        compare_frames(baseline, decoded, rgb=True)
        report = dict(clip=clip['id'], frames=clip['frames'], width=clip['width'], height=clip['height'], q=clip['q'],
                      payload_bytes=sender['payload_bytes'], baseline_bytes=baseline['stream_bytes'],
                      no_rgb_bytes=sender['stream_bytes'],
                      payload_symbols_cdf_reference_exact=True, receiver_rgb_exact_to_original=True,
                      sender_generator_calls=sender['generator_calls'],
                      historical_baseline_bit_exact=bool(clip.get('historical_baseline_sha256')),
                      runs=runs, receiver=decoded)
        for metric in ['core_seconds', 'task_seconds', 'controller_process_wall_seconds', 'peak_allocated_mib']:
            report['baseline_median_' + metric] = statistics.median(r['baseline'][metric] for r in runs)
            report['no_rgb_median_' + metric] = statistics.median(r['no_rgb'][metric] for r in runs)
        report['median_paired_core_speedup'] = statistics.median(r['baseline']['core_seconds'] / r['no_rgb']['core_seconds'] for r in runs)
        report['median_paired_task_speedup'] = statistics.median(r['baseline']['task_seconds'] / r['no_rgb']['task_seconds'] for r in runs)
        reports.append(report)
        save_json(output / 'summary.json', dict(complete=len(reports) == len(plan['clips']),
                  plan_sha256=file_sha(output / 'plan.json'), clips=reports,
                  Orin_tested=False, power_measured=False, FEC_implemented=False,
                  production_web_integrated=False, student_used=False, training_steps=0))
        print(f'VERIFIED {clip["id"]}: exact RGB, speedup={report["median_paired_core_speedup"]:.3f}x', flush=True)


if __name__ == '__main__':
    main()
