"""Byte-only rate selection, byte-matched HEVC and independently checked views."""

import argparse
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from cache_hevc_baseline import encode_controls
from coral_reference_sidecar import file_sha, load_json, save_json
import evaluate_fullframe_reference as scoring
from compare_deflicker import save_csv
from run_compact_support import CASES, paths
from run_fullframe_factorial import BOX
from run_p0 import preview
from verify_video1_persistent_structure import inspect_preview


def select(workspace, run):
    batch, _ = paths(workspace)
    records = {}
    driver = Path(__file__).with_name('run_compact_support.py')
    common = [sys.executable, '-B', '-X', 'utf8', str(driver)]
    for case in CASES:
        records[case] = {}
        base_size = (batch/case/'base.glcsnd').stat().st_size
        for fusion in ('E', 'F'):
            attempts = []
            for grid in (24, 20, 16, 12):
                profile = f'S{grid}_B4_T5'
                name = profile+'_'+fusion
                folder = run/case/name
                tail = ['--workspace', str(workspace), '--run', str(run), '--case', case]
                if not (folder/'send.json').exists():
                    subprocess.run(common+['prepare', *tail, '--profiles', profile, '--fusion', fusion], check=True)
                send = load_json(folder/'send.json')
                assert file_sha(folder/'support.glccsp') == send['sha256']
                attempts.append(dict(name=name, bytes=send['bytes']))
                if send['bytes'] < base_size and send['within_absolute_cap']:
                    if not (folder/'receive/receipt.json').exists():
                        subprocess.run(common+['receive', *tail, '--name', name], check=True)
                    receipt = load_json(folder/'receive/receipt.json')
                    assert receipt['rgb_array_sha256'] == send['sender_expected_rgb_sha256']
                    assert receipt['alpha_array_sha256'] == send['sender_expected_alpha_sha256']
                    records[case][fusion] = dict(name=name, bytes=send['bytes'], kbps=send['kbps'],
                         base_bytes=base_size, base_kbps=base_size*8/10000,
                         below_base=True, attempts=attempts, selection_uses_quality=False)
                    break
            else:
                raise ValueError('No below-base candidate for '+case+'/'+fusion)
    save_json(run/'selected.json', dict(complete=True, profiles=records,
              rule='First below-base file in fixed S24,S20,S16,S12 sequence, fixed B4 T5, no score-based selection.'))
    print(records, flush=True)


def baselines(workspace, run, ffmpeg):
    batch, _ = paths(workspace)
    chosen = load_json(run/'selected.json')['profiles']
    for case in CASES:
        source = np.load(batch/case/'source.npy', mmap_mode='r')
        budgets = {'h265_'+fusion: rec['base_bytes']+rec['bytes'] for fusion, rec in chosen[case].items()}
        encode_controls(ffmpeg, run/case/'hevc', CASES[case][0], source, budgets)


def views(workspace, run, ffmpeg):
    batch, prior = paths(workspace)
    chosen = load_json(run/'selected.json')['profiles']
    all_scores = {}
    all_checks = {}
    for case in CASES:
        out = run/case/'final_evaluation'
        out.mkdir(exist_ok=False)
        seq = dict(source=np.load(batch/case/'source.npy', mmap_mode='r'),
                   raw=np.load(batch/case/'base_receive/decoded_rgb.npy', mmap_mode='r'),
                   old_E=np.load(prior/case/'g24q2_E/receive/rgb.npy', mmap_mode='r'),
                   old_F=np.load(prior/case/'g24q2_F/receive/rgb.npy', mmap_mode='r'))
        for fusion in ('E', 'F'):
            folder = run/case/chosen[case][fusion]['name']
            receipt = load_json(folder/'receive/receipt.json')
            for name, digest in receipt['outputs_sha256'].items():
                assert file_sha(folder/'receive'/name) == digest
            seq['L'+fusion] = np.load(folder/'receive/rgb.npy', mmap_mode='r')
            seq['H'+fusion] = np.load(run/case/'hevc'/('h265_'+fusion+'.npy'), mmap_mode='r')
        scoring.SCOPES.pop('coral', None)
        if case == 'video1':
            scoring.SCOPES['coral'] = BOX
            seq['U'] = np.load(run.parent/'video1_persistent_aux_20261110_01/g17q4/receive/rgb.npy', mmap_mode='r')
            seq['GP'] = np.load(run.parent/'fullframe_factorial_20261110_01/native_GP/rgb.npy', mmap_mode='r')
        scores, temporal, spatial = scoring.score(seq)
        save_csv(out/'temporal.csv', temporal)
        save_csv(out/'spatial.csv', spatial)
        hevc = load_json(run/case/'hevc/result.json')
        old_rates = {f: load_json(prior/case/('g24q2_'+f)/'send.json')['kbps'] for f in ('E', 'F')}
        labels = dict(raw='Original GLC', source='Source',
                      old_E=f'E / aux {old_rates["E"]:.1f} kbps', old_F=f'F / aux {old_rates["F"]:.1f} kbps',
                      LE=f'L / aux {chosen[case]["E"]["kbps"]:.1f} kbps',
                      LF=f'LB / aux {chosen[case]["F"]["kbps"]:.1f} kbps',
                      HE=f'H.265 / total {hevc["controls"]["h265_E"]["kbps"]:.1f} kbps',
                      HF=f'H.265 / total {hevc["controls"]["h265_F"]["kbps"]:.1f} kbps',
                      U='Previous local U', GP='Pixel GP / offline')
        specs = [('01_oldF_vs_L_full.mp4', ('old_F', 'LE'), None),
                 ('02_L_vs_LB_full.mp4', ('LE', 'LF'), None),
                 ('03_L_vs_HEVC_full.mp4', ('LE', 'HE'), None)]
        if case == 'video1':
            specs.extend([('04_oldE_vs_L_coral.mp4', ('old_E', 'LE'), (250, 0, 512, 132)),
                          ('05_U_vs_L_coral.mp4', ('U', 'LE'), (250, 0, 512, 132))])
        checks = {}
        for filename, names, crop in specs:
            panels = [seq[name] for name in names]
            selection = None if crop is None else np.s_[crop[1]:crop[3], crop[0]:crop[2]]
            preview(ffmpeg, out/filename, panels, 15, 0, [labels[n] for n in names], crop=selection)
            checks[filename] = inspect_preview(ffmpeg, out/filename, panels, crop=crop)
        save_json(out/'summary.json', dict(complete=True, case=case, scores=scores, rates=chosen[case],
                  hevc=hevc, visual_review='pending', frames=150, fps=15, previews=checks,
                  evaluation_support='Frozen external source texture correspondence; no output gate intersection'))
        all_scores[case] = scores
        all_checks[case] = checks
        print(case, {n: dict(mae=round(s['coarse_mae'],4), ncc=round(s['fine_ncc'],4),
                            psnr=s['psnr_db']) for n,s in scores.get('coral',scores['full']).items()}, flush=True)
    save_json(run/'final_summary.json', dict(complete=True, selected=chosen, scores=all_scores,
              preview_checks=all_checks, code_sha256=file_sha(__file__), visual_review='pending'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('select', 'baselines', 'views'))
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--ffmpeg', type=Path)
    args = p.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    if args.action == 'select':
        select(args.workspace.resolve(), args.run.resolve())
    elif args.action == 'baselines':
        baselines(args.workspace.resolve(), args.run.resolve(), args.ffmpeg.resolve())
    else:
        views(args.workspace.resolve(), args.run.resolve(), args.ffmpeg.resolve())
