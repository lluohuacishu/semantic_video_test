"""One full-video cold pair: separate new processes, no warmup or reused models."""
import argparse
import json
from pathlib import Path
import time
import traceback

from run_c0 import HERE, REPO, read, sha
from run_c1 import task
from run_c2 import manifest, worker
from c2_watch import utc, write


def endpoints(timing, dispatch):
    """Same-machine perf_counter endpoints; do not add model loading twice."""
    startup = timing['task_timer_origin_perf_counter'] - dispatch
    if startup < 0 or timing['model_was_resident']:
        raise ValueError('Cold task must follow dispatch in a fresh process')
    marks = timing['task_seconds']
    return {'dispatch_to_task_start_seconds': startup,
            'dispatch_to_evidence_ready_seconds': startup + marks['evidence_ready_seconds'],
            'dispatch_to_view_ready_seconds': startup + marks['offline_view_ready_seconds'],
            'task_to_view_ready_seconds': marks['offline_view_ready_seconds'],
            'model_acquire_seconds_included_in_task': timing['codec_acquire_seconds']}


def cold_tasks(root):
    # Alternate the reference/candidate order across roles, each with a new process.
    for role, backends in (('encode', ('python', 'native')), ('decode', ('native', 'python'))):
        for backend in backends:
            name = role + '_cold_' + backend
            extra = {'input': str(root / 'encode_cold_python/stream.glcv')} if role == 'decode' else {}
            yield role, backend, task(root, name, role, preset='compact', frames=125,
                implementation='legacy' if backend == 'python' else 'c1',
                backend=backend, group='cold_single_pair', pair=1, profile=False, **extra)


def run(root):
    records = []
    for role, backend, item in cold_tasks(root):
        if (root / 'STOP_REQUESTED').exists():
            break
        dispatch = time.perf_counter()
        write(root / (item['id'] + '_dispatch.json'), {'requested_utc': utc(),
              'dispatch_perf_counter': dispatch, 'id': item['id']})
        worker(root, item['id'] + '_process', role, [item])
        timing = read(Path(item['output']) / 'timing.json')
        records.append({'id': item['id'], 'role': role, 'backend': backend,
                        **endpoints(timing, dispatch),
                        'validation': read(Path(item['output']) / 'validation.json')})
    comparison = None
    if len(records) == 4:
        totals = {backend: sum(r['dispatch_to_view_ready_seconds'] for r in records if r['backend'] == backend)
                  for backend in ('python', 'native')}
        comparison = {'before_seconds': totals['python'], 'after_seconds': totals['native'],
                      'saved_seconds': totals['python'] - totals['native'],
                      'reduction_percent': 100 * (1 - totals['native'] / totals['python']),
                      'speedup': totals['python'] / totals['native']}
    result = {'records': records, 'complete_pair_count': int(comparison is not None), 'comparison': comparison,
              'scope': 'One main-clip compact legacy versus C1+native pair, each task in a fresh Python/CUDA process.',
              'timer': 'External dispatch to exact task evidence/view endpoint; includes supervisor/config setup, imports and model load; excludes post-task GPU snapshot, process shutdown and offline validation.',
              'limitations': ['One pair is a cost example, not a repeated cold-speed estimate.',
                              'OS disk caches are not flushed. No browser/network timing.']}
    write(root / 'cold_summary.json', result)
    print(json.dumps({'complete_pair_count': result['complete_pair_count'], 'comparison': comparison}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-library', type=Path, required=True)
    parser.add_argument('--prior-gate', type=Path, required=True)
    args = parser.parse_args()
    gate = read(args.prior_gate / 'gate.json')
    previous = read(args.prior_gate / 'manifest.json')
    if not gate['passed'] or gate.get('comparison') != 'legacy_vs_c1_native' or gate['frames'] != 8:
        parser.error('Passed combined gate8 is required')
    if previous['native']['sha256'] != sha(args.native_library):
        parser.error('Native library differs from the combined gate')
    for name in ('c2_worker.py', 'c2_native.py', 'c1_worker.py', 'c1_runtime.py'):
        path = HERE / name
        if previous['code_sha256'][str(path.relative_to(REPO))] != sha(path):
            parser.error('Runtime changed since the combined gate: ' + name)
    args.phase, args.frames, args.pairs = 'cold_single_pair', 125, 1
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest(root, args)
    write(root / 'stage.json', {'state': 'running', 'started_utc': utc(), 'prior_gate': str(args.prior_gate.resolve())})
    try:
        run(root)
        write(root / 'stage.json', {'state': 'stopped' if (root / 'STOP_REQUESTED').exists() else 'complete', 'ended_utc': utc()})
    except BaseException:
        write(root / 'stage.json', {'state': 'interrupted_or_failed', 'ended_utc': utc(), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
