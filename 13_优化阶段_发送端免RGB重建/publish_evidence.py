"""Validate saved artifacts and export small public evidence, without GPU work."""
from __future__ import annotations

import argparse
from pathlib import Path

from run_trial import compare_frames
from sender_common import HERE, REPO, file_sha, load_json, save_json
from sender_container import read_sender_stream
from glc_container import read_stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    output, destination = args.output.resolve(), args.destination.resolve()
    summary, plan = load_json(output / 'summary.json'), load_json(output / 'plan.json')
    if not summary['complete'] or summary['plan_sha256'] != file_sha(output / 'plan.json'):
        raise ValueError('Incomplete or changed trial plan')
    for name, expected in plan['code_sha256'].items():
        if file_sha(REPO / name) != expected:
            raise ValueError('Execution source changed: ' + name)
    for clip in plan['clips']:
        if file_sha(clip['source']) != clip['source_sha256']:
            raise ValueError('Frozen source array changed')
    public, timings, traces, raw = [], [], [], {}
    for clip in summary['clips']:
        name = clip['clip']
        small = {k: v for k, v in clip.items() if k not in ('runs', 'receiver')}
        small['runs'] = []
        for pair, run in enumerate(clip['runs'], 1):
            compare_frames(run['baseline'], run['no_rgb'])
            base_path = output / name / f'pair{pair}_baseline/stream.glcv'
            sender_path = output / name / f'pair{pair}_no_rgb/stream.glcsnd'
            _, base_packets = read_stream(base_path)
            _, sender_packets, _ = read_sender_stream(sender_path)
            if len(base_packets) != len(sender_packets):
                raise ValueError('Saved packet count differs')
            for baseline_packet, sender_packet in zip(base_packets, sender_packets):
                if any(baseline_packet[key] != sender_packet[key] for key in ('kind', 'z', 'y')):
                    raise ValueError('Actual saved z/y bytes differ')
            if pair > 1:
                compare_frames(clip['runs'][0]['baseline'], run['baseline'], rgb=True)
            for role, result in run.items():
                folder = output / name / f'pair{pair}_{role}'
                path = folder / ('stream.glcv' if role == 'baseline' else 'stream.glcsnd')
                if file_sha(path) != result['stream_sha256']:
                    raise ValueError('Saved sender file changed')
                small['runs'].append(dict(pair=pair, **{k: v for k, v in result.items() if k not in ('records', 'header')}))
                timings.append(dict(clip=name, pair=pair, role=role,
                                    frames=[{k: r[k] for k in ('index', 'kind', 'seconds')} for r in result['records']]))
        receiver = clip['receiver']
        decoded = output / name / 'receiver/decoded_rgb.npy'
        if not receiver['reference_verified'] or file_sha(decoded) != receiver['receiver_rgb_sha256']:
            raise ValueError('Saved receiver RGB changed or was not verified')
        compare_frames(clip['runs'][0]['baseline'], receiver, rgb=True)
        small['receiver'] = {k: v for k, v in receiver.items() if k not in ('records', 'header')}
        small['all_pairs_saved_payload_byte_comparison_exact'] = True
        public.append(small)
        keys = ('index', 'kind', 'latent_sha256', 'rgb_sha256', 'payload_sha256', 'payload_bytes', 'entropy')
        trace = dict(clip=name)
        for role, result in [('baseline', clip['runs'][0]['baseline']), ('no_rgb', clip['runs'][0]['no_rgb']), ('receiver', receiver)]:
            trace[role] = [{k: r[k] for k in keys if k in r} for r in result['records']]
        traces.append(trace)
    for path in output.rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.log', '.glcv', '.glcsnd', '.npy', '.mp4'):
            raw[str(path.relative_to(output)).replace('\\', '/')] = dict(sha256=file_sha(path), bytes=path.stat().st_size)
    destination.mkdir(parents=True, exist_ok=False)
    save_json(destination / 'summary.json', dict(
        schema='glc-sender-no-rgb-public-evidence-v1', complete=True,
        **{k: v for k, v in summary.items() if k not in ('clips', 'complete')}, clips=public))
    save_json(destination / 'timings.json', timings)
    save_json(destination / 'verified_frames.json', traces)
    save_json(destination / 'plan.json', plan)
    verification = load_json(output / 'verification.json')
    if any(check['returncode'] for check in verification['checks']):
        raise ValueError('Regression tests failed')
    save_json(destination / 'verification.json', verification)
    for check in verification['checks']:
        target = destination / check['log']
        target.write_text((output / check['log']).read_text('utf-8'), encoding='utf-8', newline='\n')
    manifest = dict(raw_output_directory=str(output), raw_execution_artifacts=raw,
                    execution_code_sha256=plan['code_sha256'],
                    public_artifacts={p.name: dict(sha256=file_sha(p), bytes=p.stat().st_size)
                                      for p in destination.iterdir() if p.is_file()},
                    note='Public JSON/logs use UTF-8 LF. Raw artifact hashes refer to unchanged local execution files.')
    save_json(destination / 'manifest.json', manifest)
    print(f'Published {len(public)} verified clips; {sum(c["frames"] for c in public)} decoded frames')


if __name__ == '__main__':
    main()
