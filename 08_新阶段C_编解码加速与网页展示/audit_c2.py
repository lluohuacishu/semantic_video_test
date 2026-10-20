"""Read-only C2 evidence audit and small final summary; never starts inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from statistics import median

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SPECS = [
    ('C2_diagnose66_20261019_r1', 'diagnostic', 6, 0),
    ('C2_gate8_20261019_r1', 'gate', 6, 0),
    ('C2_gate66_20261019_r1', 'gate', 6, 0),
    ('C2_compact_pairs_20261019_r1', 'main_c1_vs_native', 16, 3),
    ('C2_detail_pair_20261019_r1', 'main_c1_vs_native', 8, 1),
    ('C2_motion_pair_20261019_r1', 'main_c1_vs_native', 8, 1),
    ('C2_detail_motion_pair_20261019_r1', 'main_c1_vs_native', 8, 1),
    ('C2_second_gate8_20261019_r1', 'gate', 8, 0),
    ('C2_combined_gate8_20261020_r2', 'gate', 8, 0),
    ('C2_second_gate8_20261020_r1', 'gate', 8, 0),
    ('C2_second_pairs_20261020_r1', 'second_c1_vs_native', 16, 3),
    ('C2_combined_pairs_20261020_r1', 'main_legacy_vs_final', 16, 3),
    ('C2_cold_pair_20261020_r1', 'main_legacy_vs_final_cold', 4, 0),
]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def evidence(path):
    return {'path': str(path.resolve()), 'sha256': sha(path), 'data': read(path)}


def audit(root, kind, count, pairs, sources):
    require(read(root / 'stage.json')['state'] == 'complete', f'Incomplete stage: {root}')
    manifest = read(root / 'manifest.json')
    source = Path(manifest['input']['path'])
    if source not in sources:
        sources[source] = sha(source)
    require(sources[source] == manifest['input']['sha256'], f'Input changed: {source}')
    for relative, digest in manifest['code_sha256'].items():
        require(sha(root / 'measurement_code' / relative) == digest, f'Snapshot changed: {root.name}/{relative}')
    if manifest.get('native'):
        require(sha(root / 'glc_rans.dll') == manifest['native']['sha256'], f'DLL changed: {root}')
    target = manifest.get('validation_target', 'b4_main')
    key = 'all_rgb_and_latent_hashes_match_reference' if target == 'second_clip_python_reference' else 'all_rgb_and_latent_hashes_match_b4'
    rows, processes = [], []
    for config_path in sorted(root.glob('*_process_config.json')):
        config = read(config_path)
        process_path = Path(config['process_folder']) / 'process.json'
        process = read(process_path)
        require(process['state'] == 'completed' and process['returncode'] == 0, f'Failed worker: {process_path}')
        processes.append(evidence(process_path))
        for task in config['tasks']:
            folder = Path(task['output'])
            require(folder.parent == root, f'Task escaped experiment: {folder}')
            timing, validation = read(folder / 'timing.json'), read(folder / 'validation.json')
            require(read(folder / 'status.json')['state'] == 'ready', f'Missing ready state: {folder}')
            require(timing['run_id'] == task['id'] and read(folder / 'hotspots.json')['run_id'] == task['id'], f'ID mismatch: {folder}')
            require(validation.get(key) is True, f'Wrong reference or failed validation: {folder}')
            role = config['role']
            result_path = folder / ('encoder/encode.json' if role == 'encode' else 'receiver/decode.json')
            require(sha(result_path) == validation['result_sha256'], f'Result changed: {result_path}')
            result = read(result_path)
            n = task.get('frames') or (125 if task['preset'] in ('compact', 'detail') else 250)
            require(len(result['frames']) == validation['frames_checked'] == n, f'Frame count changed: {folder}')
            if role == 'encode':
                require(sha(folder / 'stream.glcv') == validation['stream_sha256'], f'Stream changed: {folder}')
            else:
                require(validation['independent_decoder'] and validation['source_media_reads'] == [], f'Receiver dependence: {folder}')
            previews = {}
            for name, info in validation['previews'].items():
                path = result_path.parent / name
                require(path.stat().st_size > 0 and int(info['nb_read_frames']) == n, f'Invalid preview evidence: {path}')
                previews[name] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
            rows.append({'id': task['id'], 'role': role, 'backend': timing['entropy_backend'],
                         'implementation': timing.get('implementation', 'c1'), 'preset': task['preset'],
                         'group': task.get('group'), 'pair': task.get('pair'), 'frames': n,
                         'resident': timing['model_was_resident'], 'profile': timing['detailed_instrumentation'],
                         'task_seconds': timing['task_seconds'], 'codec_acquire_seconds': timing['codec_acquire_seconds'],
                         'timing_sha256': sha(folder / 'timing.json'), 'validation_sha256': sha(folder / 'validation.json'),
                         'validation': validation, 'previews_at_audit': previews})
    require(len(rows) == count, f'Task count mismatch: {root}')
    result = {'name': root.name, 'root': str(root), 'kind': kind, 'validation_target': target,
              'manifest': evidence(root / 'manifest.json'), 'processes': processes, 'tasks': rows}
    if pairs:
        summary = read(root / 'paired_summary.json')
        require(summary['complete_pair_count'] == pairs, f'Partial pairs: {root}')
        chosen = {(r['pair'], r['role'], r['backend']): r for r in rows if r['group'] == 'paired'}
        paired, evidence_pairs = [], []
        for p in range(1, pairs + 1):
            for role in ('encode', 'decode', 'sum'):
                roles = ('encode', 'decode') if role == 'sum' else (role,)
                for r in roles:
                    for backend in ('python', 'native'):
                        row = chosen[(p, r, backend)]
                        require(row['resident'] and not row['profile'], f'Invalid warm performance task: {row["id"]}')
                        expected = 'legacy' if kind == 'main_legacy_vs_final' and backend == 'python' else 'c1'
                        require(row['implementation'] == expected, f'Wrong comparison arm: {row["id"]}')
                before = sum(chosen[(p, r, 'python')]['task_seconds']['offline_view_ready_seconds'] for r in roles)
                after = sum(chosen[(p, r, 'native')]['task_seconds']['offline_view_ready_seconds'] for r in roles)
                paired.append({'pair': p, 'role': role, 'before_seconds': before, 'after_seconds': after,
                               'saved_seconds': before - after, 'reduction_percent': 100 * (1 - after / before), 'speedup': before / after})
                eb = sum(chosen[(p, r, 'python')]['task_seconds']['evidence_ready_seconds'] for r in roles)
                ea = sum(chosen[(p, r, 'native')]['task_seconds']['evidence_ready_seconds'] for r in roles)
                evidence_pairs.append({'pair': p, 'role': role, 'before_seconds': eb, 'after_seconds': ea,
                                       'saved_seconds': eb - ea, 'reduction_percent': 100 * (1 - ea / eb), 'speedup': eb / ea})
        require(summary['pairs'] == paired, f'Pair arithmetic differs: {root}')
        for group in summary['groups']:
            values = [r for r in paired if r['role'] == group['role']]
            for metric in ('before_seconds', 'after_seconds', 'saved_seconds', 'reduction_percent', 'speedup'):
                require(group[metric]['median'] == median(r[metric] for r in values), f'Median differs: {root}')
        evidence_groups = []
        for role in ('encode', 'decode', 'sum'):
            values = [r for r in evidence_pairs if r['role'] == role]
            evidence_groups.append({'role': role, 'n': len(values), **{
                metric: {'median': median(r[metric] for r in values), 'min': min(r[metric] for r in values),
                         'max': max(r[metric] for r in values)} for metric in
                ('before_seconds', 'after_seconds', 'saved_seconds', 'reduction_percent', 'speedup')}})
        result['performance'] = {'pairs': paired, 'groups': summary['groups'], 'complete_pair_count': pairs,
                                 'evidence_ready_pairs': evidence_pairs, 'evidence_ready_groups': evidence_groups}
    elif kind.endswith('_cold'):
        cold = read(root / 'cold_summary.json')
        require(cold['complete_pair_count'] == 1 and all(not r['resident'] for r in rows), 'Invalid cold pair')
        for row in cold['records']:
            timing = read(root / row['id'] / 'timing.json')
            dispatch = read(root / (row['id'] + '_dispatch.json'))['dispatch_perf_counter']
            offset = timing['task_timer_origin_perf_counter'] - dispatch
            require(offset >= 0, 'Cold timer ordering changed')
            for endpoint in ('evidence', 'view'):
                key = 'evidence_ready_seconds' if endpoint == 'evidence' else 'offline_view_ready_seconds'
                require(row[f'dispatch_to_{endpoint}_ready_seconds'] == offset + timing['task_seconds'][key], 'Cold endpoint arithmetic differs')
        before = sum(r['dispatch_to_view_ready_seconds'] for r in cold['records'] if r['backend'] == 'python')
        after = sum(r['dispatch_to_view_ready_seconds'] for r in cold['records'] if r['backend'] == 'native')
        require(cold['comparison']['before_seconds'] == before and cold['comparison']['after_seconds'] == after, 'Cold totals differ')
        result['cold'] = evidence(root / 'cold_summary.json')
    elif kind == 'gate':
        require(read(root / 'gate.json')['passed'], f'Gate not passed: {root}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = {}
    stages = [audit(args.data_root.resolve() / name, kind, count, pairs, sources)
              for name, kind, count, pairs in SPECS]
    checks = read(args.checks / 'checks.json')
    require(checks['all_passed'], 'Regression checks failed')
    logs, total_tests = [], 0
    for row in checks['results']:
        path = Path(row['log_path'])
        require(row['returncode'] == 0 and sha(path) == row['log_sha256'], 'Regression log mismatch')
        content = path.read_text(encoding='utf-8')
        n = int(re.search(r'Ran (\d+) tests', content).group(1))
        total_tests += n
        logs.append({'suite': row['suite'], 'tests': n, 'path': str(path), 'sha256': sha(path), 'text': content})
    for relative, digest in checks['test_sources'].items():
        require(sha(REPO / relative) == digest, f'Test changed since regression: {relative}')
    failures = []
    for name in ('C2_second_pairs_20261019_r1', 'C2_combined_gate8_20261020_r1'):
        root = args.data_root / name
        require(read(root / 'stage.json')['state'] == 'interrupted_or_failed', 'Failure evidence was altered')
        failures.append({'name': name, 'stage': evidence(root / 'stage.json'),
                         'processes': [evidence(p) for p in sorted(root.glob('*_process/process.json'))]})
    combination = next(s for s in stages if s['kind'] == 'main_legacy_vs_final')
    combined_group = next(g for g in combination['performance']['groups'] if g['role'] == 'sum')
    evidence_group = next(g for g in combination['performance']['evidence_ready_groups'] if g['role'] == 'sum')
    goals = {'scope': 'Main-clip compact, separate resident encode+decode tasks through MP4 readiness; three paired repeats',
             'reduction_percent_median': combined_group['reduction_percent']['median'],
             'speedup_median': combined_group['speedup']['median'],
             'minimum_20_percent_reduction_met': combined_group['reduction_percent']['median'] >= 20,
             'challenge_2x_met': combined_group['speedup']['median'] >= 2,
             'evidence_ready_reduction_percent_median': evidence_group['reduction_percent']['median'],
             'evidence_ready_minimum_20_percent_reduction_met': evidence_group['reduction_percent']['median'] >= 20,
             'evidence_ready_speedup_median': evidence_group['speedup']['median'],
             'evidence_ready_challenge_2x_met': evidence_group['speedup']['median'] >= 2,
             'source_experiment': combination['name']}
    value = {'created_utc': datetime.now(timezone.utc).isoformat(), 'stage': 'C2_candidate_validated_with_scope_limits',
             'speed_goals': goals,
             'stages': stages, 'regression': {'checks': evidence(args.checks / 'checks.json'), 'tests': total_tests, 'logs': logs},
             'excluded_failed_batches': failures,
             'limitations': ['Primary timings are separate-role resident task sums to MP4, not browser or network latency.',
                             'Other three presets have one pair each; no stable three-pair claim.',
                             'Cold comparison has one pair; OS disk cache is not flushed.',
                             'Second existing clip is distinct content; capture-source independence is not established.',
                             'C1 historical interruption causes remain unknown.', 'C3/C4 have not started.']}
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'audited_stages': len(stages), 'validated_tasks': sum(len(s['tasks']) for s in stages),
                      'regression_tests': total_tests, 'output': str(args.output), 'bytes': args.output.stat().st_size}))


if __name__ == '__main__':
    main()
