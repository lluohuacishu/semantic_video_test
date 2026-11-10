"""Lossless full-frame scores and source-aligned support changes at actual bytes."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from audit_deflicker_texture import source_match
from compare_deflicker import save_csv
from coral_reference_sidecar import file_sha, load_json, save_json
import evaluate_fullframe_reference as scoring
from probe_reef import bands
from run_compact_support import paths
from run_fullframe_factorial import BOX
from temporal_filter import warp


def evaluate(workspace, run, case, names, label):
    out = run/case/label
    out.mkdir(exist_ok=False)
    batch, prior = paths(workspace)
    seq = dict(source=np.load(batch/case/'source.npy', mmap_mode='r'),
               raw=np.load(batch/case/'base_receive/decoded_rgb.npy', mmap_mode='r'),
               E=np.load(prior/case/'g24q2_E/receive/rgb.npy', mmap_mode='r'),
               F=np.load(prior/case/'g24q2_F/receive/rgb.npy', mmap_mode='r'))
    states = {n: np.load(prior/case/('g24q2_'+n)/'receive/alpha.npy', mmap_mode='r') for n in ('E', 'F')}
    rates = {}
    for name in names:
        folder = run/case/name
        send = load_json(folder/'send.json')
        recv = load_json(folder/'receive/receipt.json')
        assert send['sha256'] == recv['aux_sha256'] == file_sha(folder/'support.glccsp')
        assert send['sender_expected_rgb_sha256'] == recv['rgb_array_sha256']
        assert send['sender_expected_alpha_sha256'] == recv['alpha_array_sha256']
        assert recv['source_files_unavailable'] and recv['zero_weight_unchanged'] and send['unsafe_pixels'] == 0
        for filename, digest in recv['outputs_sha256'].items():
            assert file_sha(folder/'receive'/filename) == digest
        seq[name] = np.load(folder/'receive/rgb.npy', mmap_mode='r')
        states[name] = np.load(folder/'receive/alpha.npy', mmap_mode='r')
        rates[name] = {k: send[k] for k in ('bytes', 'kbps', 'base_bytes', 'base_kbps', 'below_base',
                       'decoded_safe_fraction', 'decoded_coral_fraction', 'lookahead_frames_for_support')}
    if case == 'video1':
        scoring.SCOPES['coral'] = BOX
        local = run.parent/'video1_persistent_aux_20261110_01/g17q4/receive'
        gp = run.parent/'fullframe_factorial_20261110_01/native_GP'
        for name, folder in (('U', local), ('GP', gp)):
            seq[name] = np.load(folder/'rgb.npy', mmap_mode='r')
            states[name] = np.load(folder/'alpha.npy', mmap_mode='r')
    scores, temporal, spatial = scoring.score(seq)
    save_csv(out/'temporal.csv', temporal)
    save_csv(out/'spatial.csv', spatial)
    support_rows = []
    last_bands = bands(seq['source'][0])
    scopes = {'full': scoring.SCOPES['full']}
    if case == 'video1':
        scopes['coral'] = BOX
    for i in range(1, 150):
        current_bands = bands(seq['source'][i])
        mx, my, valid = source_match(seq['source'][i-1], seq['source'][i], last_bands,
                                    current_bands, np.s_[:, :])
        for name, alpha in states.items():
            aligned = warp(alpha[i-1], mx, my)
            prev_on = cv2.remap((alpha[i-1] > 0).astype('u1'), mx, my, cv2.INTER_NEAREST).astype(bool)
            on = alpha[i] > 0
            delta = np.abs(alpha[i]-aligned)
            for scope, (x0, y0, x1, y1) in scopes.items():
                region = np.zeros_like(valid)
                region[y0:y1, x0:x1] = True
                area = valid & region
                always = area & on & prev_on
                support_rows.append(dict(frame=i, scope=scope, method=name, pixels=int(area.sum()),
                    switch_pixels=int((area & (on != prev_on)).sum()),
                    alpha_delta_sum=float(delta[area].sum()), always_pixels=int(always.sum()),
                    always_alpha_delta_sum=float(delta[always].sum()),
                    held_pixels=int((area & (alpha[i] >= .99) & (aligned >= .99)).sum())))
        last_bands = current_bands
    save_csv(out/'support.csv', support_rows)
    support = {}
    for scope in scopes:
        support[scope] = {}
        for name in states:
            rr = [r for r in support_rows if r['scope'] == scope and r['method'] == name]
            n = sum(r['pixels'] for r in rr)
            a = sum(r['always_pixels'] for r in rr)
            support[scope][name] = dict(pixels=n, switch_fraction=sum(r['switch_pixels'] for r in rr)/n,
                alpha_change=sum(r['alpha_delta_sum'] for r in rr)/n,
                always_on_alpha_change=sum(r['always_alpha_delta_sum'] for r in rr)/max(a, 1),
                held_fraction=sum(r['held_pixels'] for r in rr)/n)
    save_json(out/'summary.json', dict(complete=True, case=case, frames=150, fps=15, billing_seconds=10,
              rates=rates, scores=scores, support=support, independently_received=True,
              gate_outside_pixel_safe_count=0, evaluation_support='fixed external reliable textured source pixels',
              visual_review='pending', code_sha256=file_sha(__file__)))
    for name in seq:
        s = scores.get('coral', scores['full'])[name]
        print(case, name, 'kbps', rates.get(name, {}).get('kbps'),
              'MAE', round(s['coarse_mae'], 4), 'NCC', round(s['fine_ncc'], 4),
              'full MAE', round(scores['full'][name]['coarse_mae'], 4), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--case', choices=('video1', 'video2', 'video3'), required=True)
    p.add_argument('--names', nargs='+', required=True)
    p.add_argument('--label', default='evaluation')
    args = p.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    evaluate(args.workspace.resolve(), args.run.resolve(), args.case, args.names, args.label)
