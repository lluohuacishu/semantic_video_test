"""Export small C2 evidence; raw logs, binaries and media stay in local experiments."""
import argparse
import hashlib
import json
from pathlib import Path

from c2_watch import utc, write


def entry(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'data': json.loads(path.read_text(encoding='utf-8'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', nargs='+', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--checks', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stages = []
    for root in args.experiments:
        evidence = {}
        for name in ('manifest.json', 'stage.json', 'gate.json', 'summary.json', 'paired_summary.json'):
            if (root / name).is_file():
                evidence[name] = entry(root / name)
        processes = [entry(p) for p in sorted(root.glob('*_process/process.json'))]
        stages.append({'root': str(root.resolve()), 'evidence': evidence, 'processes': processes})
    value = {'exported_utc': utc(), 'c2_stage': 'in_progress_not_accepted', 'stages': stages,
             'build': entry(args.build / 'build.json'),
             'limitations': ['C1 speedup evidence is compact-only.',
                            '66-frame hotspot proportions are diagnostic, not all-preset estimates.',
                            'Original C1 interruption causes remain unknown.',
                            'No training, browser or physical-link claims.']}
    for name in ('cpu_tests.log', 'regression_tests.log'):
        path = args.build / name
        if path.is_file():
            value[name] = {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'text': path.read_text(encoding='utf-8')}
    if args.checks:
        value['checks'] = entry(args.checks / 'checks.json')
        value['regression_logs'] = [{
            'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'text': path.read_text(encoding='utf-8')} for path in sorted(args.checks.glob('*_regression.log'))]
    write(args.output, value)
    print(json.dumps({'output': str(args.output), 'experiments': len(stages), 'bytes': args.output.stat().st_size}))


if __name__ == '__main__':
    main()
