"""Offline recovery after supervisor loss; never reruns inference or invents exit codes."""
import argparse
from functools import partial
from pathlib import Path

from c2_watch import inspect, utc, write
from run_c0 import HERE, REPO, read, sha
from run_c1 import validate


def recover(root, report):
    root, report = Path(root).resolve(), Path(report).resolve()
    if report.exists():
        raise FileExistsError('Choose a new recovery report; previous evidence is retained')
    processes = []
    for folder in sorted(root.glob('*_process')):
        result = inspect(folder)
        if result['observed_state'] in ('running', 'orphan_running', 'liveness_unknown'):
            raise RuntimeError(f'Process liveness must be resolved before recovery: {folder}')
        processes.append({'folder': str(folder), **result})
    manifest = read(root / 'manifest.json')
    target = manifest.get('validation_target', 'b4_main')
    if target == 'second_clip_python_reference':
        from c2_holdout import validate as second_clip_validate
        validator = HERE / 'c2_holdout.py'
        validate_task = partial(second_clip_validate, root)
        passed_key = 'all_rgb_and_latent_hashes_match_reference'
    elif target == 'b4_main':
        validator = HERE / 'run_c1.py'
        validate_task = validate
        passed_key = 'all_rgb_and_latent_hashes_match_b4'
    else:
        raise ValueError('Unknown validation target; never guess the reference clip')
    if manifest['code_sha256'][str(validator.relative_to(REPO))] != sha(validator):
        raise RuntimeError('Validator changed; inspect the immutable measurement snapshot before recovery')
    validated, excluded = [], []
    for config_path in sorted(root.glob('*_process_config.json')):
        config = read(config_path)
        for task in config['tasks']:
            folder = Path(task['output'])
            timing_file, hotspot_file, status_file = [folder / name for name in
                                                     ('timing.json', 'hotspots.json', 'status.json')]
            if not all(p.is_file() for p in (timing_file, hotspot_file, status_file)):
                excluded.append({'id': task['id'], 'reason': 'Missing final timing, scopes or status'})
                continue
            timing, scopes, status = read(timing_file), read(hotspot_file), read(status_file)
            if (timing.get('run_id') != task['id'] or scopes.get('run_id') != task['id'] or
                    timing.get('entropy_backend') != task.get('backend', 'python') or status.get('state') != 'ready'):
                excluded.append({'id': task['id'], 'reason': 'Worker did not finish consistent final metadata'})
                continue
            if not (folder / 'validation.json').exists():
                validate_task(task, config['role'])
            validation = read(folder / 'validation.json')
            if not validation.get(passed_key):
                raise ValueError('Existing validation is not passing')
            validated.append({'id': task['id'], 'validation': validation})
    record = {'recovered_utc': utc(), 'root': str(root), 'processes': processes,
              'validated': validated, 'excluded': excluded,
              'policy': 'Offline verification only; original process states and unknown exit codes are preserved. No inference rerun.'}
    report.parent.mkdir(parents=True, exist_ok=True)
    write(report, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    value = recover(args.experiment, args.report)
    print(f"Validated {len(value['validated'])}; excluded {len(value['excluded'])}; report: {args.report}")


if __name__ == '__main__':
    main()
