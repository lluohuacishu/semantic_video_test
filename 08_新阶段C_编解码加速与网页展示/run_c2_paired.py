"""Finite full-video Python/native AB/BA comparisons, one selected preset at a time."""
import argparse
import json
from pathlib import Path
import shutil
from statistics import median
import traceback

from run_c0 import HERE, REPO, read, sha
from run_c1 import task
from run_c2 import manifest, worker
from c2_watch import utc, write


def summarize(root):
    config = read(root / 'paired_config.json')
    rows, complete_pairs = [], []
    for path in sorted(root.glob('*/timing.json')):
        if not path.with_name('validation.json').exists():
            continue
        value = read(path)
        rows.append({'id': value['run_id'], 'role': value['role'], 'backend': value['entropy_backend'],
                     'group': value['group'], 'pair': value['pair'], 'resident': value['model_was_resident'],
                     'view_seconds': value['task_seconds']['offline_view_ready_seconds'],
                     'evidence_seconds': value['task_seconds']['evidence_ready_seconds'],
                     'codec_acquire_seconds': value['codec_acquire_seconds'],
                     'validation': read(path.with_name('validation.json'))})
    for pair in range(1, config['pairs'] + 1):
        chosen = {(row['role'], row['backend']): row for row in rows
                  if row['group'] == 'paired' and row['pair'] == pair}
        if set(chosen) != {(r, b) for r in ('encode', 'decode') for b in ('python', 'native')}:
            continue
        for role in ('encode', 'decode', 'sum'):
            roles = ('encode', 'decode') if role == 'sum' else (role,)
            before = sum(chosen[(r, 'python')]['view_seconds'] for r in roles)
            after = sum(chosen[(r, 'native')]['view_seconds'] for r in roles)
            complete_pairs.append({'pair': pair, 'role': role, 'before_seconds': before, 'after_seconds': after,
                                   'saved_seconds': before - after, 'reduction_percent': 100 * (1 - after / before),
                                   'speedup': before / after})
    groups = []
    for role in ('encode', 'decode', 'sum'):
        selected = [p for p in complete_pairs if p['role'] == role]
        if selected:
            groups.append({'role': role, 'n': len(selected), **{
                key: {'median': median(p[key] for p in selected), 'min': min(p[key] for p in selected),
                      'max': max(p[key] for p in selected)} for key in
                ('before_seconds', 'after_seconds', 'saved_seconds', 'reduction_percent', 'speedup')}})
    write(root / 'paired_summary.json', {'config': config, 'records': rows, 'pairs': complete_pairs, 'groups': groups,
          'validated_tasks': len(rows), 'complete_pair_count': len(complete_pairs) // 3,
          'scope': 'Same clip/preset/C1 display/CUDA FP32; Python versus native entropy. Separate-role resident workers; warmups excluded.',
          'not_end_to_end': 'Sum of separately executed encode and decode; not network or browser latency.'})
    print(json.dumps({'validated_tasks': len(rows), 'groups': groups}, ensure_ascii=False), flush=True)


def run(root, preset, pairs):
    warmups = [task(root, 'encode_warmup_' + b, 'encode', preset=preset, frames=8,
                    backend=b, group='warmup', profile=False) for b in ('python', 'native')]
    encode = list(warmups)
    for pair in range(1, pairs + 1):
        for backend in (('python', 'native') if pair % 2 else ('native', 'python')):
            encode.append(task(root, f'encode_{pair}_{backend}', 'encode', preset=preset,
                               backend=backend, group='paired', pair=pair, profile=False))
    worker(root, 'encode_process', 'encode', encode)
    if (root / 'STOP_REQUESTED').exists():
        return
    decode = [task(root, 'decode_warmup_' + b, 'decode', preset=preset, frames=8,
                    input=str(Path(warmups[0]['output']) / 'stream.glcv'),
                    backend=b, group='warmup', profile=False) for b in ('native', 'python')]
    for pair in range(1, pairs + 1):
        for backend in (('native', 'python') if pair % 2 else ('python', 'native')):
            decode.append(task(root, f'decode_{pair}_{backend}', 'decode', preset=preset,
                               backend=backend, group='paired', pair=pair, profile=False))
    worker(root, 'decode_process', 'decode', decode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('paired', 'summary'))
    parser.add_argument('--preset', choices=('compact', 'detail', 'motion', 'detail_motion'), default='compact')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--native-library', type=Path)
    parser.add_argument('--prior-gate', type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.phase == 'summary':
        summarize(root)
        return
    if not args.native_library or not args.prior_gate:
        parser.error('--native-library and --prior-gate are required')
    gate = read(args.prior_gate / 'gate.json')
    previous = read(args.prior_gate / 'manifest.json')
    if not gate['passed'] or gate['frames'] != 66 or previous['native']['sha256'] != sha(args.native_library):
        parser.error('Matching 66-frame cross-decode gate is required')
    candidate = HERE / 'c2_native.py'
    if previous['code_sha256'][str(candidate.relative_to(REPO))] != sha(candidate):
        parser.error('Candidate wrapper changed since the 66-frame gate')
    root.mkdir(parents=True, exist_ok=False)
    args.frames = 125 if args.preset in ('compact', 'detail') else 250
    manifest(root, args)
    current = read(root / 'manifest.json')
    current.update(preset=args.preset)
    own = Path(__file__).resolve()
    current['code_sha256'][str(own.relative_to(REPO))] = sha(own)
    shutil.copy2(own, root / 'measurement_code' / own.relative_to(REPO))
    write(root / 'manifest.json', current)
    write(root / 'paired_config.json', {'preset': args.preset, 'pairs': args.pairs, 'frames': args.frames,
                                       'created_utc': utc(), 'prior_gate': str(args.prior_gate.resolve())})
    write(root / 'stage.json', {'state': 'running', 'started_utc': utc()})
    try:
        run(root, args.preset, args.pairs)
        summarize(root)
        write(root / 'stage.json', {'state': 'stopped' if (root / 'STOP_REQUESTED').exists() else 'complete', 'ended_utc': utc()})
    except BaseException:
        summarize(root)
        write(root / 'stage.json', {'state': 'interrupted_or_failed', 'ended_utc': utc(), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
