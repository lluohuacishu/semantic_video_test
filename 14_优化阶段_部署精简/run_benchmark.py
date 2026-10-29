"""Matched full-vs-role-specific sender/receiver trials; no source recaching."""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import subprocess
import time

from role_codec import REPO
from sender_common import file_sha, load_json, load_settings, save_json
from run_trial import compare_frames

HERE = Path(__file__).resolve().parent


def prepare(output, previous, assets):
    output.mkdir(parents=True, exist_ok=False)
    old = load_json(previous / 'plan.json')
    clips = []
    for item in old['clips']:
        clip = dict(item)
        clip.update(stream=str(previous / clip['id'] / 'pair1_no_rgb/stream.glcsnd'),
                    teacher_result=str(previous / clip['id'] / 'receiver/result.json'))
        clip['stream_sha256'] = file_sha(clip['stream'])
        clip['teacher_result_sha256'] = file_sha(clip['teacher_result'])
        clips.append(clip)
    files = [*HERE.glob('*.py'), *sorted((REPO / '13_优化阶段_发送端免RGB重建').glob('*.py')),
             REPO / '04_外部依赖快照/semantic-video-training/scripts/glc_video_codec.py',
             REPO / '08_新阶段C_编解码加速与网页展示/c1_io.py',
             REPO / '08_新阶段C_编解码加速与网页展示/c2_native.py']
    plan = dict(schema='glc-role-deployment-trial-v1', assets=str(assets),
                assets_manifest_sha256=file_sha(assets / 'manifest.json'), clips=clips,
                code_sha256={p.relative_to(REPO).as_posix(): file_sha(p) for p in files},
                pair_orders=[['original', 'slim_fast'], ['slim_fast', 'original'], ['original', 'slim_fast']],
                baseline='Sender already skips RGB; receiver renders original full RGB',
                scope='Same-GPU FP32/native rANS; three new process pairs per role and setting; no Orin or power test')
    save_json(output / 'plan.json', plan)
    save_json(output / 'plan_identity.json', dict(sha256=file_sha(output / 'plan.json')))
    return plan


def job(output, clip, role, variant, folder, assets, stream):
    done = folder / 'result.json'
    if done.exists():
        value = load_json(done)
        path = folder / 'stream.glcsnd' if role == 'sender' else stream
        if value['stream_sha256'] != file_sha(path):
            raise ValueError('Saved stream changed')
        if role == 'receiver' and value['receiver_rgb_sha256'] != file_sha(folder / 'decoded_rgb.npy'):
            raise ValueError('Saved receiver array changed')
        return value
    folder.parent.mkdir(parents=True, exist_ok=True)
    command = [str(load_settings()['runtime']['python']), '-X', 'utf8', str(HERE / 'benchmark_worker.py'),
               '--variant', variant, '--assets', str(assets), '--plan', str(output / 'plan.json'),
               '--clip', clip['id'], '--role', 'no_rgb' if role == 'sender' else 'decode', '--output', str(folder)]
    if role == 'receiver':
        command += ['--stream', str(stream)]
    started = time.perf_counter()
    with folder.with_suffix('.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    value = load_json(done)
    value['controller_process_wall_seconds'] = time.perf_counter() - started
    save_json(done, value)
    print(f'{clip["id"]} {role} {variant}: {value["core_seconds"]:.3f}s, '
          f'peak {value["peak_allocated_mib"]:.1f} MiB', flush=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args()
    output, assets = args.output.resolve(), args.assets.resolve()
    plan = load_json(output / 'plan.json') if (output / 'plan.json').exists() else prepare(output, args.previous.resolve(), assets)
    if file_sha(output / 'plan.json') != load_json(output / 'plan_identity.json')['sha256']:
        raise ValueError('Frozen plan changed')
    if str(assets) != plan['assets'] or file_sha(assets / 'manifest.json') != plan['assets_manifest_sha256']:
        raise ValueError('Deployment manifest changed')
    for path, digest in plan['code_sha256'].items():
        if file_sha(REPO / path) != digest:
            raise ValueError('Frozen execution code changed: ' + path)
    reports = []
    for clip in plan['clips']:
        for key, digest_key in [('source', 'source_sha256'), ('stream', 'stream_sha256'),
                                ('teacher_result', 'teacher_result_sha256')]:
            if file_sha(clip[key]) != clip[digest_key]:
                raise ValueError('Frozen input changed: ' + key)
        teacher = load_json(clip['teacher_result'])
        for role in ['sender', 'receiver']:
            runs = []
            for pair, order in enumerate(plan['pair_orders'], 1):
                row = {}
                for variant in order:
                    folder = output / clip['id'] / role / f'pair{pair}_{variant}'
                    row[variant] = job(output, clip, role, variant, folder, assets, Path(clip['stream']))
                    compare_frames(teacher, row[variant], rgb=role == 'receiver')
                    if role == 'sender' and row[variant]['stream_sha256'] != clip['stream_sha256']:
                        raise ValueError('Whole sender stream differs from prior verified output')
                compare_frames(row['original'], row['slim_fast'], rgb=role == 'receiver')
                runs.append(row)
            result = dict(clip=clip['id'], role=role, frames=clip['frames'], q=clip['q'],
                          width=clip['width'], height=clip['height'], runs=runs,
                          all_reference_symbols_cdf_exact=True,
                          receiver_rgb_exact=role == 'receiver', sender_entire_stream_exact=role == 'sender')
            for metric in ['core_seconds', 'task_seconds', 'init_seconds', 'controller_process_wall_seconds',
                           'resident_allocated_mib', 'peak_allocated_mib', 'live_parameters', 'tensor_bytes']:
                for variant in ['original', 'slim_fast']:
                    result[variant + '_median_' + metric] = statistics.median(row[variant][metric] for row in runs)
            for metric in ['core_seconds', 'task_seconds', 'controller_process_wall_seconds']:
                result['median_paired_speedup_' + metric] = statistics.median(row['original'][metric] / row['slim_fast'][metric] for row in runs)
            if clip == plan['clips'][0]:
                probe = job(output, clip, role, 'slim', output / clip['id'] / role / 'slim_only_probe', assets, Path(clip['stream']))
                compare_frames(teacher, probe, rgb=role == 'receiver')
                result['slim_only_single_probe'] = probe
            reports.append(result)
            save_json(output / 'summary.json', dict(complete=len(reports)==len(plan['clips'])*2,
                      plan_sha256=file_sha(output / 'plan.json'), reports=reports,
                      training_steps=0, Orin_tested=False, power_measured=False,
                      default_codec_changed=False, production_web_changed=False))
            print(f'VERIFIED {clip["id"]} {role}: {result["median_paired_speedup_core_seconds"]:.3f}x', flush=True)


if __name__ == '__main__':
    main()
