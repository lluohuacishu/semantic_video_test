"""Verify all saved streams/arrays and export numerical evidence without GPU work."""
from __future__ import annotations

import argparse
from pathlib import Path

from role_codec import REPO
from sender_common import file_sha, load_json, save_json
from run_trial import compare_frames


def without_frames(result):
    return {key: value for key, value in result.items() if key not in ('records', 'header')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimal', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    output, minimal, assets = args.output.resolve(), args.minimal.resolve(), args.assets.resolve()
    summary, plan = load_json(output / 'summary.json'), load_json(output / 'plan.json')
    if not summary['complete'] or summary['plan_sha256'] != file_sha(output / 'plan.json'):
        raise ValueError('Incomplete or changed trial')
    for name, digest in plan['code_sha256'].items():
        if file_sha(REPO / name) != digest:
            raise ValueError('Execution code changed: ' + name)
    if file_sha(assets / 'manifest.json') != plan['assets_manifest_sha256']:
        raise ValueError('Asset manifest changed')
    asset_manifest = load_json(assets / 'manifest.json')
    for info in asset_manifest['roles'].values():
        for item in info['files'].values():
            if file_sha(assets / item['file']) != item['sha256']:
                raise ValueError('Role asset changed')
    small, timings, traces = [], [], []
    for report in summary['reports']:
        clip = next(c for c in plan['clips'] if c['id'] == report['clip'])
        if file_sha(clip['source']) != clip['source_sha256']:
            raise ValueError('Source array changed')
        if file_sha(clip['teacher_result']) != clip['teacher_result_sha256']:
            raise ValueError('Original receiver evidence changed')
        teacher = load_json(clip['teacher_result'])
        role, name = report['role'], report['clip']
        value = {key: value for key, value in report.items() if key not in ('runs', 'slim_only_single_probe')}
        value.pop('receiver_rgb_exact' if role == 'sender' else 'sender_entire_stream_exact')
        value['runs'] = []
        for pair, run in enumerate(report['runs'], 1):
            for variant, result in run.items():
                folder = output / name / role / f'pair{pair}_{variant}'
                compare_frames(teacher, result, rgb=role == 'receiver')
                stream = folder / 'stream.glcsnd' if role == 'sender' else Path(clip['stream'])
                if file_sha(stream) != result['stream_sha256']:
                    raise ValueError('Saved stream changed')
                if role == 'sender' and stream.read_bytes() != Path(clip['stream']).read_bytes():
                    raise ValueError('Whole sender stream bytes differ')
                if role == 'receiver' and file_sha(folder / 'decoded_rgb.npy') != result['receiver_rgb_sha256']:
                    raise ValueError('Receiver RGB array changed')
                value['runs'].append(dict(pair=pair, **without_frames(result)))
                timings.append(dict(clip=name, role=role, variant=variant, pair=pair,
                    frames=[{key: row[key] for key in ('index', 'kind', 'seconds')} for row in result['records']]))
            compare_frames(run['original'], run['slim_fast'], rgb=role == 'receiver')
        if 'slim_only_single_probe' in report:
            probe = report['slim_only_single_probe']
            compare_frames(teacher, probe, rgb=role == 'receiver')
            value['slim_only_single_probe'] = without_frames(probe)
        small.append(value)
        trace = dict(clip=name, role=role)
        keys = ('index', 'kind', 'latent_sha256', 'rgb_sha256', 'payload_sha256', 'entropy', 'payload_bytes')
        for variant in ['original', 'slim_fast']:
            trace[variant] = [{key: row[key] for key in keys if key in row} for row in report['runs'][0][variant]['records']]
        traces.append(trace)
    verification = load_json(output / 'verification.json')
    if any(item['returncode'] for item in verification['checks']):
        raise ValueError('CPU regression failed')
    minimal_result = load_json(minimal / 'validation.json')
    if not minimal_result['success'] or not minimal_result['failed_reference_never_published']:
        raise ValueError('Minimal CLI validation failed')
    for name, digest in minimal_result['validation_code_sha256'].items():
        if file_sha(REPO / name) != digest:
            raise ValueError('Minimal CLI validation code changed')
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    public = dict(schema='glc-role-deployment-public-v1', **{key:value for key,value in summary.items() if key!='reports'}, reports=small)
    for name, data in [('summary', public), ('timings', timings), ('verified_frames', traces),
                       ('plan', plan), ('assets_manifest', asset_manifest), ('verification', verification),
                       ('codebook_lookup_float_audit', load_json(output/'codebook_lookup_float_audit.json')),
                       ('minimal_validation', minimal_result)]:
        save_json(destination / (name + '.json'), data)
    for folder in ['sender', 'receiver', 'failed_receiver']:
        for filename in ['result.json', 'failure.json', 'access_audit.json']:
            path = minimal / folder / filename
            if path.exists():
                save_json(destination / ('minimal_' + folder + '_' + filename), load_json(path))
    for item in verification['checks']:
        (destination / item['log']).write_text((output/item['log']).read_text('utf-8'), encoding='utf-8', newline='\n')
    raw = {}
    for root in [output, minimal]:
        raw[root.name] = {path.relative_to(root).as_posix(): dict(sha256=file_sha(path), bytes=path.stat().st_size)
                          for path in root.rglob('*') if path.is_file() and path.suffix in ('.json','.log','.npy','.glcsnd','.mp4')}
    save_json(destination / 'manifest.json', dict(
        raw_execution_artifacts=raw, execution_code_sha256=plan['code_sha256'],
        minimal_code_sha256=minimal_result['validation_code_sha256'],
        public_artifacts={path.name:dict(sha256=file_sha(path), bytes=path.stat().st_size)
                          for path in destination.iterdir() if path.is_file()},
        note='Raw files remain local; public JSON/logs use UTF-8 LF. Non-applicable cross-role flags omitted.'))
    print(f'Exported {len(small)} role/settings, all stream/array checks passed')


if __name__ == '__main__':
    main()
