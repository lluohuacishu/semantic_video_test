"""Finite paired decoder evaluation after BOTH pre-registered trials finish."""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import subprocess
import sys

from pilot_common import HERE, file_sha, load_json, save_json, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot-python', type=Path, required=True)
    args = parser.parse_args()
    python = settings()['runtime']['python']
    frozen = dict(plan_sha256=file_sha(args.output / 'plan.json'), modes={})
    for mode in ['kd', 'gt']:
        summary = load_json(args.output / f'train_{mode}/summary.json')
        if summary['status'] != 'complete' or summary['completed_steps'] != 5000:
            raise ValueError('Both actual 5000-step trials must complete before final test')
        frozen['modes'][mode] = dict(selected_step=summary['best_validation']['step'],
                                     checkpoint_sha256=file_sha(args.output / f'train_{mode}/best.pt'))
    (args.output / 'evaluation').mkdir(exist_ok=True)
    frozen_path = args.output / 'evaluation/selection_frozen.json'
    if frozen_path.exists():
        raise FileExistsError('Evaluation already started; preserve it and inspect its records')
    save_json(frozen_path, frozen)
    # Alternating pair order; separate fresh processes, same FP32 arithmetic,
    # two warmup frames each. Keep all stalls, no best-of selection.
    for trial, order in enumerate([['teacher', 'kd'], ['kd', 'teacher'], ['teacher', 'kd']]):
        for mode in order:
            command = [str(python), '-B', str(HERE / 'decode_student_pilot.py'), '--output', str(args.output),
                       '--clip', 'v007', '--mode', mode, '--trial', str(trial)]
            if trial == 0:
                command.append('--save-rgb')
            subprocess.run(command, check=True)
    for clip, mode in [('v007', 'gt'), ('v025', 'kd'), ('v025', 'gt')]:
        subprocess.run([str(python), '-B', str(HERE / 'decode_student_pilot.py'), '--output', str(args.output),
                        '--clip', clip, '--mode', mode, '--trial', '0', '--save-rgb'], check=True)
    timing = {'scope': 'Warm full codec core (entropy, reference chain, RGB and checksums), not full application or network latency',
              'arithmetic': 'FP32 for teacher and students; TF32 disabled', 'paired_trials': [],
              'all_stalls_retained': True, 'other_desktop_apps_not_stopped': True}
    for trial in range(3):
        pair = {}
        for mode in ['teacher', 'kd']:
            row = load_json(args.output / f'evaluation/full_decode/v007_{mode}_{trial}/decode.json')
            pair[mode] = {k: row[k] for k in ['core_seconds', 'mean_frame_seconds', 'p_mean_frame_seconds', 'peak_allocated_mib']}
        pair['speedup_teacher_over_kd'] = pair['teacher']['core_seconds'] / pair['kd']['core_seconds']
        timing['paired_trials'].append(pair)
    timing['median_teacher_core_seconds'] = statistics.median(r['teacher']['core_seconds'] for r in timing['paired_trials'])
    timing['median_kd_core_seconds'] = statistics.median(r['kd']['core_seconds'] for r in timing['paired_trials'])
    timing['median_paired_speedup'] = statistics.median(r['speedup_teacher_over_kd'] for r in timing['paired_trials'])
    save_json(args.output / 'evaluation/timings.json', timing)
    subprocess.run([str(python), '-B', str(HERE / 'evaluate_pilot.py'), '--output', str(args.output)], check=True)
    subprocess.run([str(args.plot_python), '-B', str(HERE / 'plot_pilot.py'), '--output', str(args.output)], check=True)
    evidence = dict(plan_sha256=file_sha(args.output / 'plan.json'),
                    script_sha256={p.name: file_sha(p) for p in HERE.glob('*.py')},
                    selection_frozen_sha256=file_sha(frozen_path), quality_sha256=file_sha(args.output / 'evaluation/quality.json'),
                    timing_sha256=file_sha(args.output / 'evaluation/timings.json'))
    save_json(args.output / 'evaluation/evidence_manifest.json', evidence)
    print('Pilot evaluation complete', flush=True)


if __name__ == '__main__':
    main()
