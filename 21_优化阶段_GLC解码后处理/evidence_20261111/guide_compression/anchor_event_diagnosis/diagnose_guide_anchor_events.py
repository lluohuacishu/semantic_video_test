"""Reaggregate saved S8I scores by anchor and GLC I-frame events; no inference."""

import argparse
import csv
from pathlib import Path
import shutil

import numpy as np

from receiver_structure import KEYS, RAMP, EVENTS, file_sha, load_json, save_json
from run_fullframe_factorial import BOX


def diagnose(run):
    output = run/'anchor_event_diagnosis'
    output.mkdir(exist_ok=False)
    table = run/'evaluation/initial/temporal.csv'
    receipt = run.parent/'source_dense_flow_three_clips_20261109_01/video1/base_receive/result.json'
    alpha_path = run/'video1/S8I/receive/alpha.npy'
    shared_path = run/'video1/S8I/receive/shared.npy'
    rows = list(csv.DictReader(table.open(encoding='utf-8-sig', newline='')))
    header = load_json(receipt)['header']
    fps = header['fps']
    refresh = set(KEYS[1:])
    transition = {k+j for k in refresh for j in range(len(RAMP)+1)}
    intra = set(range(header['intra_period'], header['frames'], header['intra_period']))
    predicates = dict(anchor_exact=lambda i: i in refresh,
        transition_and_exit=lambda i: i in transition,
        outside_anchor_windows=lambda i: i not in transition,
        base_intra=lambda i: i in intra)
    report = {}
    archived = load_json(run/'evaluation/initial/summary.json')['scores']
    for method in ('raw', 'A12', 'S8I'):
        report[method] = {}
        for scope in ('full', 'coral'):
            selected = [r for r in rows if r['method'] == method and r['scope'] == scope]
            def aggregate(values):
                pixels = sum(int(r['pixels']) for r in values)
                return dict(pairs=len(values), pixels=pixels,
                    coarse_MAE=sum(float(r['coarse_sum']) for r in values)/pixels)
            overall = aggregate(selected)
            assert overall['coarse_MAE'] == archived[scope][method]['coarse_mae']
            groups = {name: aggregate([r for r in selected if predicate(int(r['frame']))])
                for name, predicate in predicates.items()}
            peaks = sorted(selected, key=lambda r: float(r['coarse_sum'])/int(r['pixels']), reverse=True)[:12]
            report[method][scope] = dict(overall=overall, groups=groups,
                peak_pairs=[dict(frame=int(r['frame']), seconds=int(r['frame'])/fps,
                    coarse_MAE=float(r['coarse_sum'])/int(r['pixels']),
                    anchor_window=int(r['frame']) in transition,
                    base_intra=int(r['frame']) in intra) for r in peaks])
    alpha = np.load(alpha_path, mmap_mode='r')
    shared = np.load(shared_path, mmap_mode='r')
    x0, y0, x1, y1 = BOX
    coverage = []
    for key in sorted(refresh):
        j = EVENTS.index(key)
        for scope, crop in (('full', np.s_[:, :]), ('coral', np.s_[y0:y1, x0:x1])):
            new = alpha[key][crop] > 0
            overlap = shared[j][crop] > 0
            coverage.append(dict(frame=key, seconds=key/fps, scope=scope,
                new_reliable_fraction=float(new.mean()),
                old_new_shared_fraction=float(overlap.mean()),
                new_reliable_without_shared_fraction=float((new & ~overlap).mean())))
    paths = (table, receipt, alpha_path, shared_path)
    save_json(output/'summary.json', dict(complete=True, read_only_existing_outputs=True,
        new_receiver_or_encoding_experiment=False, profile='S8I', frames=header['frames'], fps=fps,
        anchor_frames=list(KEYS), anchor_change_seconds=[k/fps for k in sorted(refresh)],
        old_reference_transition_frames=len(RAMP), transition_exit_included_in_groups=True,
        GLC_intra_frames=sorted(intra), GLC_intra_seconds=[i/fps for i in sorted(intra)],
        scores=report, anchor_overlap=coverage, saved_overall_scores_exact=True,
        inputs_sha256={str(p): file_sha(p) for p in paths},
        code_sha256=file_sha(__file__),
        interpretation='Temporal association only; no anchor or I-frame counterfactual was run. '
            'A common timestamp does not by itself establish the cause of the user-observed jump. '
            'Shared fractions describe gate eligibility, not a measured decomposition of displayed error.'))
    shutil.copy2(__file__, output/Path(__file__).name)
    print('ANCHOR EVENT DIAGNOSIS COMPLETE: saved CSV reaggregation only', flush=True)
    print('S8I coral groups', report['S8I']['coral']['groups'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    arguments = parser.parse_args()
    diagnose(arguments.run.resolve())
