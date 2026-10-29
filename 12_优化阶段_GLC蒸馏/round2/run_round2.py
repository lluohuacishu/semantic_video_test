"""Finite four-case training then frozen-selection independent evaluation."""
from __future__ import annotations
import argparse
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pilot_common import file_sha, load_json, save_json, settings, check_stop


def run(script, output, *extra, python=None):
    check_stop(output)
    command = [str(python or settings()['runtime']['python']), '-B', str(HERE/script), '--output', str(output), *map(str, extra)]
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['train', 'evaluate'], required=True)
    parser.add_argument('--plot-python', type=Path)
    args = parser.parse_args()
    plan = load_json(args.output/'plan.json')
    cases = list(plan['training']['cases'])
    if args.phase=='train':
        if not (args.output/'sampling_pairs.json').exists():
            run('freeze_samples.py', args.output)
        for case in cases:
            summary = args.output/f'train_{case}/summary.json'
            if summary.exists():
                evidence = load_json(summary)
                if evidence['status']=='complete' and evidence['completed_steps']==plan['training']['steps_each']:
                    print(f'skip completed {case}', flush=True)
                    continue
                raise ValueError('Incomplete existing case: use train_round2.py --resume explicitly')
            run('train_round2.py', args.output, '--case', case)
        return
    if args.plot_python is None:
        raise ValueError('Evaluation needs existing CPU matplotlib Python via --plot-python')
    evaluation = args.output/'evaluation'
    evaluation.mkdir(exist_ok=True)
    frozen_path = evaluation/'selection_frozen.json'
    if frozen_path.exists():
        raise FileExistsError('Evaluation already started; preserve and inspect records')
    summaries = {}
    for case in cases:
        summary = load_json(args.output/f'train_{case}/summary.json')
        if summary['status']!='complete' or summary['completed_steps']!=5000:
            raise ValueError('All four 5000-step trials must complete before final holdout')
        summaries[case] = summary
    if len({s['sampling_sha256'] for s in summaries.values()})!=1 or len({s['sampling_manifest_sha256'] for s in summaries.values()})!=1:
        raise ValueError('Cases used different sample schedules or training pairs')
    if len({summaries[c]['initial_model_sha256'] for c in cases if c.startswith('mapped_')})!=1:
        raise ValueError('Mapped cases used different initial weights')
    selected_case = min(cases, key=lambda c: summaries[c]['best_validation']['mse_rgb'])
    frozen = dict(plan_sha256=file_sha(args.output/'plan.json'), selected_case=selected_case,
                  selection='minimum UV96 P-frame original RGB MSE; no final/regression metrics consulted',
                  cases={c: dict(step=summaries[c]['best_validation']['step'], checkpoint_sha256=file_sha(args.output/f'train_{c}/best.pt')) for c in cases})
    save_json(frozen_path, frozen)
    print(f'Frozen best case before final evaluation: {selected_case}', flush=True)
    for clip in [c for c in plan['clips'] if c['role']!='train']:
        for case in cases:
            run('decode_round2.py', args.output, '--clip', clip['id'], '--case', case, '--trial', '0', '--save-rgb')
    # Quality trial0 is excluded from paired timing. Every measured pair below
    # runs consecutively in alternating order, fresh processes and warm FP32.
    for trial, order in [(1, ['teacher', selected_case]), (2, [selected_case, 'teacher']), (3, ['teacher', selected_case])]:
        for case in order:
            run('decode_round2.py', args.output, '--clip', 'uv096', '--case', case, '--trial', str(trial))
    timing = dict(clip='uv096', selected_case=selected_case, precision='FP32, TF32 off, 2-frame warmup',
                  scope='warm codec core; excludes model loading, disk I/O, MP4 and network', pairs=[], all_stalls_retained=True)
    for trial in range(1, 4):
        teacher = load_json(evaluation/'full_decode'/f'uv096_teacher_{trial}/decode.json')
        student = load_json(evaluation/'full_decode'/f'uv096_{selected_case}_{trial}/decode.json')
        timing['pairs'].append(dict(trial=trial, teacher_seconds=teacher['core_seconds'], student_seconds=student['core_seconds'],
                                    speedup=teacher['core_seconds']/student['core_seconds']))
    timing['median_paired_speedup'] = statistics.median(r['speedup'] for r in timing['pairs'])
    timing['median_teacher_seconds'] = statistics.median(r['teacher_seconds'] for r in timing['pairs'])
    timing['median_student_seconds'] = statistics.median(r['student_seconds'] for r in timing['pairs'])
    save_json(evaluation/'timings.json', timing)
    run('evaluate_round2.py', args.output)
    run('plot_round2.py', args.output, python=args.plot_python)
    save_json(evaluation/'execution_manifest.json', dict(plan_sha256=file_sha(args.output/'plan.json'),
              sampling_manifest_sha256=file_sha(args.output/'sampling_pairs.json'),
              source_script_sha256={p.name: file_sha(p) for p in HERE.glob('*.py')},
              quality_sha256=file_sha(evaluation/'quality.json'), timings_sha256=file_sha(evaluation/'timings.json')))
    print('Round2 evaluation complete', flush=True)


if __name__ == '__main__':
    main()
