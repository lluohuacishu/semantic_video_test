"""Finite C2 diagnostic runs; STOP_REQUESTED pauses dispatch at frame boundaries."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import traceback

from run_c0 import HERE, REPO, SCRIPTS, RUNTIME, FFMPEG, SOURCE, INPUT_SHA, sha, read, command_text, gpu_snapshot
from run_c1 import task, validate
from c2_watch import supervise, write, inspect, utc


def manifest(root, args):
    files = list(HERE.glob('c[012]_*.py')) + [HERE / n for n in ('run_c0.py', 'run_c1.py', 'run_c2.py', 'build_c2_native.py', 'test_c2_native.py', 'test_c2_watch.py')]
    files += list((HERE / 'native').glob('*'))
    files += [HERE / name for name in ('run_c2_paired.py', 'recover_c2.py', 'check_c2.py',
                                       'export_c2_checkpoint.py', 'test_c2_results.py')]
    files += [SCRIPTS / n for n in ('glc_video_codec.py', 'glc_video_probe.py', 'glc_entropy.py',
                                  'glc_container.py', 'glc_presets.py', 'glc_presets.json')]
    if sha(SOURCE) != INPUT_SHA:
        raise ValueError('Frozen input changed')
    for file in files:
        target = root / 'measurement_code' / file.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
    native = None
    if args.native_library:
        library = args.native_library.resolve(strict=True)
        build = read(library.with_name('build.json'))
        if (build['returncode'] != 0 or sha(library) != build['library_sha256'] or
            any(sha(HERE / 'native' / name) != digest for name, digest in build['source_sha256'].items())):
            raise ValueError('Native build does not match current sources and library')
        native = {'path': str(library), 'sha256': sha(library), 'build': build}
        shutil.copy2(library, root / library.name)
    write(root / 'manifest.json', {'created_utc': utc(), 'phase': args.phase, 'frames': args.frames,
          'pairs': args.pairs, 'base_commit': command_text(['git', 'rev-parse', 'HEAD']),
          'code_sha256': {str(p.relative_to(REPO)): sha(p) for p in files},
          'input': {'path': str(SOURCE), 'sha256': INPUT_SHA}, 'python': sys.executable,
          'gpu': gpu_snapshot(), 'power_scheme': command_text(['powercfg', '/getactivescheme']),
          'policy': 'Serialized separate-role C1 display workers; cold warmup explicit; profile AB/BA controls; no GPU profiler',
          'native': native, 'limitations': 'Finite C2 evidence; no extrapolated full-preset speedup claim; scopes are CPU wall spans'})


def worker(root, name, role, tasks):
    folder = root / name
    config = {'role': role, 'repo': str(RUNTIME / 'tools/GLC'), 'weights': str(RUNTIME / 'checkpoints/glc'),
              'ffmpeg': str(FFMPEG), 'process_folder': str(folder), 'stop_file': str(root / 'STOP_REQUESTED'),
              'tasks': tasks}
    if (root / 'glc_rans.dll').exists():
        config['native_library'] = str(root / 'glc_rans.dll')
    config_path = root / (name + '_config.json')
    write(config_path, config)
    value = supervise([sys.executable, '-B', str(HERE / 'c2_worker.py'), '--config', str(config_path)],
                      folder, REPO, root / 'STOP_REQUESTED')
    # Validate every completed task even if a later task in this worker failed.
    complete = 0
    for item in tasks:
        output = Path(item['output'])
        ready = all((output / name).is_file() for name in ('timing.json', 'hotspots.json', 'status.json'))
        if ready:
            ready = (read(output / 'timing.json').get('run_id') == item['id'] and
                     read(output / 'hotspots.json').get('run_id') == item['id'] and
                     read(output / 'status.json').get('state') == 'ready')
        if ready:
            validate(item, role)
            complete += 1
            print('C2_VALIDATED ' + item['id'], flush=True)
    if value['returncode'] != 0:
        raise RuntimeError(f"Worker {name}: {value['state']}, exit {value['returncode']}; see {folder}")
    if complete != len(tasks):
        raise RuntimeError(f'Worker returned 0 with missing terminal task evidence: {complete}/{len(tasks)}')


def diagnose(root, frames, pairs):
    short = task(root, 'encode_warmup8', 'encode', frames=8, group='cold_warmup', profile=False)
    encode_tasks = [short]
    for pair in range(1, pairs + 1):
        for profile in ([False, True] if pair % 2 else [True, False]):
            encode_tasks.append(task(root, f'encode_{pair}_profile{int(profile)}', 'encode', frames=frames,
                                     group='diagnostic', pair=pair, profile=profile))
    worker(root, 'encode_process', 'encode', encode_tasks)
    if (root / 'STOP_REQUESTED').exists():
        return
    stream = Path(encode_tasks[1]['output']) / 'stream.glcv'
    decode_tasks = [task(root, 'decode_warmup8', 'decode', frames=8, group='cold_warmup', profile=False,
                         input=str(Path(short['output']) / 'stream.glcv'))]
    for pair in range(1, pairs + 1):
        for profile in ([True, False] if pair % 2 else [False, True]):
            decode_tasks.append(task(root, f'decode_{pair}_profile{int(profile)}', 'decode', frames=frames,
                                     group='diagnostic', pair=pair, profile=profile, input=str(stream)))
    worker(root, 'decode_process', 'decode', decode_tasks)


def gate(root, count):
    encodes = [task(root, 'encode_' + backend, 'encode', frames=count, group='gate',
                    backend=backend, profile=True) for backend in ('python', 'native')]
    worker(root, 'encode_process', 'encode', encodes)
    if (root / 'STOP_REQUESTED').exists():
        return
    if sha(Path(encodes[0]['output']) / 'stream.glcv') != sha(Path(encodes[1]['output']) / 'stream.glcv'):
        raise ValueError('Python and native full streams differ')
    decodes = []
    for encoder in ('python', 'native'):
        for decoder in ('native', 'python'):
            decodes.append(task(root, f'decode_{encoder}_by_{decoder}', 'decode', frames=count,
                                backend=decoder, group='cross_decode', profile=True,
                                input=str(root / ('encode_' + encoder) / 'stream.glcv')))
    worker(root, 'decode_process', 'decode', decodes)
    write(root / 'gate.json', {'passed': True, 'frames': count, 'stream_byte_identical': True,
          'cross_decode_directions': ['python->python', 'python->native', 'native->python', 'native->native']})


def summary(root):
    rows = []
    for path in sorted(root.glob('*/timing.json')):
        if not path.with_name('validation.json').exists():
            continue
        timing = read(path)
        hot = read(path.with_name('hotspots.json'))
        scopes = hot['scopes']
        view = timing['task_seconds']['offline_view_ready_seconds']
        entropy = sum(scopes.get('entropy.' + name, {}).get('inclusive_seconds', 0)
                      for name in ('indexes', 'encode', 'decode')) if hot['enabled'] else None
        rows.append({'id': timing['run_id'], 'role': timing['role'], 'backend': timing['entropy_backend'], 'profile': hot['enabled'],
                     'resident': timing['model_was_resident'], 'view_seconds': view,
                     'entropy_seconds': entropy, 'entropy_fraction': entropy / view if entropy is not None else None,
                     'scopes': scopes, 'validation': read(path.with_name('validation.json'))})
    write(root / 'summary.json', {'records': rows, 'validated_tasks': len(rows),
          'interpretation': 'Diagnostic evidence only. Inclusive frame and entropy scopes overlap; never add them.'})
    print(json.dumps([{k: v for k, v in row.items() if k not in ('scopes', 'validation')} for row in rows]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('diagnose', 'gate8', 'gate66', 'summary', 'inspect'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, choices=(8, 16, 66, 125), default=66)
    parser.add_argument('--pairs', type=int, choices=(1, 2), default=1)
    parser.add_argument('--native-library', type=Path)
    parser.add_argument('--prior-gate', type=Path, help='Completed 8-frame gate required before gate66')
    args = parser.parse_args()
    root = args.output.resolve()
    if args.phase == 'inspect':
        for folder in root.glob('*_process'):
            print(json.dumps(inspect(folder), ensure_ascii=False), flush=True)
        return
    if args.phase == 'summary':
        summary(root)
        return
    if args.phase.startswith('gate'):
        if not args.native_library:
            parser.error('--native-library is required for native gates')
        args.frames = int(args.phase[4:])
        if args.phase == 'gate66':
            if not args.prior_gate or not read(args.prior_gate / 'gate.json')['passed']:
                parser.error('A completed --prior-gate is required')
            previous = read(args.prior_gate / 'manifest.json')
            if previous['native']['sha256'] != sha(args.native_library):
                parser.error('Previous gate used a different native library')
    root.mkdir(parents=True, exist_ok=False)
    manifest(root, args)
    write(root / 'stage.json', {'state': 'running', 'started_utc': utc()})
    try:
        if args.phase.startswith('gate'):
            gate(root, args.frames)
        else:
            diagnose(root, args.frames, args.pairs)
        summary(root)
        write(root / 'stage.json', {'state': 'stopped' if (root / 'STOP_REQUESTED').exists() else 'complete', 'ended_utc': utc()})
    except BaseException:
        write(root / 'stage.json', {'state': 'interrupted_or_failed', 'ended_utc': utc(), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
