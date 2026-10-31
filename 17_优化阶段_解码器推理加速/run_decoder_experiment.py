"""Sequential, resumable decoder-only runs on existing, unchanged bitstreams."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request

from decoder_common import *


def web_idle():
    # A live user job takes priority. Never kill or cancel it to benchmark.
    with urllib.request.urlopen('http://127.0.0.1:8773/inference/api/jobs', timeout=5) as response:
        jobs = json.load(response)
    if any(j['status'] not in ('done', 'error', 'cancelled', 'interrupted', 'staged') for j in jobs):
        raise RuntimeError('User web job active/queued; decoder experiment will not start')


def gpu_snapshot():
    query='pstate,clocks.current.graphics,clocks.current.memory,temperature.gpu,power.draw,utilization.gpu'
    return subprocess.check_output(['nvidia-smi','--query-gpu='+query,'--format=csv'],text=True).strip()


def verify_results(baseline, actual):
    for key in ('stream_sha256', 'stream_bytes', 'reference_checksums', 'entropy_symbols_and_cdf_sha256'):
        if baseline[key] != actual[key]:
            raise ValueError('Decoder backend changed bitstream/reference/entropy identity: ' + key)
    left, right = baseline['records'], actual['records']
    if len(left) != len(right) or any(any(a[k] != b[k] for k in ('index', 'kind', 'latent_sha256'))
                                      for a, b in zip(left, right)):
        raise ValueError('Per-frame reference chain differs')


def run(folder, stream, backend, micro=False, video=False):
    path = folder / 'result.json'
    if not path.exists():
        web_idle()
        folder.parent.mkdir(parents=True, exist_ok=True)
        command = [str(load_settings()['runtime']['python']), '-X', 'utf8', '-u', str(HERE / 'decoder_worker.py'),
                   '--stream', str(stream), '--output', str(folder), '--backend', backend]
        if micro:
            command.append('--microbenchmark')
        if video:
            command.append('--video')
        launch=dict(argv=command,code_sha256=code_hashes(),input_sha256=file_sha(stream),started_unix=time.time(),gpu_before=gpu_snapshot())
        save_json(folder.with_suffix('.launch.json'),launch)
        with folder.with_suffix('.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
        launch.update(completed_unix=time.time(),returncode=completed.returncode,gpu_after=gpu_snapshot())
        save_json(folder.with_suffix('.launch.json'),launch)
        if completed.returncode:
            raise RuntimeError(f'Decoder failed: {folder.with_suffix(".log")} ({completed.returncode})')
    result = load_json(path)
    if result['code_sha256'] != code_hashes() or result['stream_sha256'] != file_sha(stream):
        raise ValueError('Cached result is from a different implementation/stream')
    if not result['reference_verified'] or not result['source_media_reads_forbidden']:
        raise ValueError('Independent receiver evidence incomplete')
    print(f'{folder.relative_to(TRIAL)}: {result["core_seconds"]:.3f}s, {result["fps"]:.3f} fps', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('short', 'multi', 'full'), required=True)
    parser.add_argument('--backends', nargs='+', choices=BACKENDS, default=list(BACKENDS))
    args = parser.parse_args()
    if 'eager' not in args.backends:
        raise ValueError('A same-stream eager baseline is required')
    results = []
    if args.suite == 'short':
        stream = ARCHIVE / 'compact/long108_q0_640/eager/stream.glcsnd'
        for repeat in range(1, 4):
            order = args.backends if repeat % 2 else list(reversed(args.backends))
            for backend in order:
                name = f'eager{repeat}' if backend == 'eager' else f'{backend}_{repeat}'
                result = run(TRIAL / 'short' / name, stream, backend, micro=True, video=repeat == 1)
                results.append(dict(folder=str(TRIAL / 'short' / name), backend=backend, repeat=repeat,
                                    core_seconds=result['core_seconds'], fps=result['fps']))
                save_json(TRIAL / 'short_progress.json', results)
        baseline = load_json(TRIAL / 'short/eager1/result.json')
        for row in results:
            verify_results(baseline, load_json(Path(row['folder']) / 'result.json'))
    elif args.suite == 'multi':
        for clip in ('sf007_q12_512', 'sf025_q32_512', 'uv096_q63_512_255'):
            stream = ARCHIVE / 'compact' / clip / 'eager/stream.glcsnd'
            baseline = None
            for backend in args.backends:
                folder = TRIAL / 'multi' / clip / backend
                result = run(folder, stream, backend, video=clip.startswith('sf'))
                baseline = baseline or result
                verify_results(baseline, result)
                results.append(dict(folder=str(folder), clip=clip, backend=backend,
                                    core_seconds=result['core_seconds'], fps=result['fps']))
                save_json(TRIAL / 'multi_progress.json', results)
    else:
        directory = WORKSPACE / '02_实验数据/C_编解码加速与网页展示/C5_loopback/sender/inference_lab'
        stream = directory / '9223a0d0559c1475a7087e8df16d1f5a/trt_fp16_stream_sender/stream.glcsnd'
        baseline = None
        for backend in args.backends:
            folder = TRIAL / 'full1239' / backend
            result = run(folder, stream, backend, video=True)
            baseline = baseline or result
            verify_results(baseline, result)
            results.append(dict(folder=str(folder), backend=backend, core_seconds=result['core_seconds'], fps=result['fps']))
            save_json(TRIAL / 'full_progress.json', results)
    save_json(TRIAL / (args.suite + '_summary.json'), dict(complete=True, runs=results,
        source_bitstreams_reencoded=False, per_frame_references_identical=True, training_steps=0))


if __name__ == '__main__':
    main()
