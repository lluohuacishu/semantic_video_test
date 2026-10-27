"""Finite, alternating-order serial/pipeline comparison on one local GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from pipeline_probe import HERE, WORK, dump, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = WORK / '02_实验数据/C_编解码加速与网页展示/GLC_precision_20261026_native_05/fp32/encode/run_1'
    source = WORK / '02_实验数据/A_压缩路线选型/A2_GLC验证/long108/input_540.mkv'
    schedule = [(40, 1, 'serial'), (40, 1, 'pipeline'), (40, 2, 'pipeline'), (40, 2, 'serial'),
                (20, 1, 'pipeline'), (20, 1, 'serial'), (20, 2, 'serial'), (20, 2, 'pipeline')]
    fingerprints = {name: sha(HERE / name) for name in ('pipeline_probe.py', 'stream_protocol.py', 'benchmark_pairs.py')}
    dump(args.output / 'plan.json', {'date': '2026-10-27', 'schedule': schedule,
        'frames': 66, 'warmup_frames_per_endpoint': 8, 'transport_version': 2,
        'input': str(source), 'baseline': str(baseline), 'code_sha256': fingerprints,
        'measure': 'first measured encode start to receiver final verification; initialization/warmup excluded',
        'quality': '640x360,12.5fps,q0,GOP64,reset32,FP32,native rANS,no previews',
        'scope': 'two processes share one RTX4060 Laptop GPU; no Orin or acoustic channel'})
    results = []
    for kbps, repeat, mode in schedule:
        assert all(sha(HERE / name) == value for name, value in fingerprints.items())
        name = f'{kbps}kbps_pair{repeat}_{mode}'
        print('START ' + name, flush=True)
        cmd = [sys.executable, str(HERE / 'pipeline_probe.py'), 'run', '--frames', '66',
               '--kbps', str(kbps), '--mode', mode, '--warmup-frames', '8',
               '--warmup-stream', str(baseline / 'stream.glcv'), '--baseline-root', str(baseline),
               '--input', str(source), '--output', str(args.output / name)]
        with (args.output / f'{name}.log').open('w', encoding='utf-8') as log:
            run = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 env={**os.environ, 'PYTHONUTF8': '1'})
        if run.returncode:
            dump(args.output / 'progress.json', {'complete': False, 'failed': name, 'runs': results})
            raise RuntimeError(f'{name} failed; original output preserved')
        result = json.loads((args.output / name / 'summary.json').read_text('utf-8'))
        assert result['verified'] and result['identical_to_baseline_prefix']
        assert result['mode'] == mode and result['warmup_frames'] == 8
        assert result['first_decoded_before_last_frame_received'] == (mode == 'pipeline')
        result.update(name=name, pair=repeat, rate_kbps=kbps)
        results.append(result)
        dump(args.output / 'progress.json', {'complete': False, 'runs': results})
        timing = result['seconds_from_first_encode_start']
        print(f"DONE {name} total={timing['receiver_verified']:.3f}s first={timing['first_frame_decoded']:.3f}s bytes={result['forward_application_bytes']}", flush=True)
    assert len({r['archive_sha256'] for r in results}) == 1
    assert len({r['forward_application_bytes'] for r in results}) == 1
    pairs = []
    for kbps in (40, 20):
        for repeat in (1, 2):
            a = next(r for r in results if (r['rate_kbps'], r['pair'], r['mode']) == (kbps, repeat, 'serial'))
            b = next(r for r in results if (r['rate_kbps'], r['pair'], r['mode']) == (kbps, repeat, 'pipeline'))
            serial = a['seconds_from_first_encode_start']['receiver_verified']
            pipeline = b['seconds_from_first_encode_start']['receiver_verified']
            pairs.append({'rate_kbps': kbps, 'pair': repeat, 'serial_seconds': serial,
                'pipeline_seconds': pipeline, 'seconds_saved': serial-pipeline,
                'reduction_percent': (serial-pipeline)/serial*100, 'speedup': serial/pipeline})
    summary = {'complete': True, 'runs': results, 'pairs': pairs, 'aggregates': []}
    for kbps in (40, 20):
        group = [r for r in pairs if r['rate_kbps'] == kbps]
        summary['aggregates'].append({'rate_kbps': kbps,
            'mean_serial_seconds': statistics.mean(r['serial_seconds'] for r in group),
            'mean_pipeline_seconds': statistics.mean(r['pipeline_seconds'] for r in group),
            'median_paired_reduction_percent': statistics.median(r['reduction_percent'] for r in group),
            'paired_reduction_range': [min(r['reduction_percent'] for r in group), max(r['reduction_percent'] for r in group)]})
    dump(args.output / 'summary.json', summary)
    dump(args.output / 'progress.json', {'complete': True, 'runs': results})
    print(json.dumps(summary['aggregates'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
