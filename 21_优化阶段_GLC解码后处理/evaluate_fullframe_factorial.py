"""Score the frozen-motion interventions on fixed external evaluation support."""

import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np

from coral_reference_sidecar import file_sha, load_json, save_json
from compare_deflicker import save_csv
import evaluate_fullframe_reference as full_score
from run_fullframe_factorial import BOX
from run_p0 import preview
from verify_video1_persistent_structure import inspect_preview


def sequences(workspace, run):
    root = workspace/'02_实验数据/优化_GLC后处理'
    batch = root/'source_dense_flow_three_clips_20261109_01/video1'
    prior = root/'fullframe_reference_refined_20261110_01/video1'
    local = root/'video1_persistent_aux_20261110_01/g17q4/receive'
    records = load_json(run/'render_summary.json')
    assert records['complete'] and records['E_rgb_alpha_bit_exact']
    paths = dict(source=batch/'source.npy', raw=batch/'base_receive/decoded_rgb.npy',
                 U=local/'rgb.npy', dense=prior/'offline/rgb.npy', F=prior/'g24q2_F/receive/rgb.npy')
    for name in ('DP', 'DB', 'GP', 'GB', 'native_DP'):
        folder = run/records['cases'][name]['canonical']
        receipt = load_json(folder/'receipt.json')
        for filename, digest in receipt['outputs_sha256'].items():
            assert file_sha(folder/filename) == digest
        paths[name] = folder/'rgb.npy'
    return {k: np.load(p, mmap_mode='r') for k, p in paths.items()}, records, prior


def evaluate(workspace, run, ffmpeg):
    out = run/'evaluation'
    out.mkdir(exist_ok=False)
    seq, records, prior = sequences(workspace, run)
    full_score.SCOPES['coral'] = BOX
    scores, temporal, spatial = full_score.score(seq)
    save_csv(out/'temporal.csv', temporal)
    save_csv(out/'spatial.csv', spatial)
    old = load_json(prior/'evaluation_perframe/summary.json')
    for name, archived in (('raw', 'raw'), ('U', 'local_U'), ('F', 'F'), ('GB', 'E')):
        assert abs(scores['coral'][name]['coarse_mae']-old['coral'][archived]['coarse_temporal_mae']) < 1e-5
    refresh = {k+j for k in records['keys'][1:] for j in range(7)}
    times = {}
    for label, selected in (('refresh_with_exit', refresh), ('non_refresh', set(range(1, 150))-refresh)):
        times[label] = {}
        for name in seq:
            rows = [r for r in temporal if r['scope'] == 'coral' and r['method'] == name and r['frame'] in selected]
            n = sum(r['pixels'] for r in rows)
            times[label][name] = dict(pairs=len(rows), valid_pixels=n,
                coarse_mae=sum(r['coarse_sum'] for r in rows)/n,
                fine_ncc=sum(r['fine_ncc']*r['pixels'] for r in rows)/n)
    effects = {}
    for scope in scores:
        j = {k: s['coarse_mae'] for k, s in scores[scope].items()}
        effects[scope] = dict(block_at_dense=j['DB']-j['DP'], block_at_grid=j['GB']-j['GP'],
                             motion_at_pixels=j['GP']-j['DP'], motion_at_blocks=j['GB']-j['DB'],
                             interaction=(j['GB']-j['GP'])-(j['DB']-j['DP']),
                             common_gate_at_dense=j['DP']-j['native_DP'])
    result = dict(complete=True, source_assisted_offline=True, actual_transmitted_candidate=False,
        frames=150, fps=15, scores=scores, coral_time_groups=times, factorial_effects=effects,
        aliases={'E': 'GB', 'native_GB': 'GB', 'native_GP': 'GP'},
        baseline_scores_reproduced=True, evaluation_support='Fixed external source texture reliability; no output-mask intersection',
        non_refresh_includes_I_reset=True, visual_review='pending', code_sha256=file_sha(__file__))
    save_json(out/'summary.json', result)
    print('CORAL '+str({k: (round(v['coarse_mae'], 6), round(v['fine_ncc'], 6))
                         for k, v in scores['coral'].items()}), flush=True)
    print('EFFECTS '+str(effects['coral']), flush=True)
    labels = {'DP': 'DP / dense motion + pixels [offline]', 'DB': 'DB / dense motion + 8x8 [offline]',
              'GP': 'GP / grid motion + pixels [offline]', 'GB': 'GB = E / grid motion + 8x8',
              'raw': 'Original GLC', 'U': 'Previous local U', 'native_DP': 'Native dense + pixels [offline]',
              'dense': 'Previous dense full-frame [offline]'}
    specs = [('factorial_full.mp4', ('DP', 'DB', 'GP', 'GB'), None),
             ('factorial_coral.mp4', ('DP', 'DB', 'GP', 'GB'), (250, 0, 512, 132)),
             ('grid_pixel_vs_block_full.mp4', ('GP', 'GB'), None),
             ('U_vs_grid_pixel_coral.mp4', ('U', 'GP'), (250, 0, 512, 132))]
    previews = {}
    for name, methods, crop in specs:
        panel = [seq[k] for k in methods]
        selection = None if crop is None else np.s_[crop[1]:crop[3], crop[0]:crop[2]]
        preview(ffmpeg, out/name, panel, 15, 0, [labels[k] for k in methods], crop=selection)
        previews[name] = inspect_preview(ffmpeg, out/name, panel, crop=crop)
    save_json(out/'preview_verification.json', dict(complete=True, previews=previews))
    shutil.copyfile(__file__, run/Path(__file__).name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    evaluate(args.workspace.resolve(), args.run.resolve(), args.ffmpeg.resolve())


if __name__ == '__main__':
    main()
