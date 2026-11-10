"""Diagnose archived U/E/F outputs without generating a new candidate video."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from audit_deflicker_texture import source_match
from coral_reference_sidecar import file_sha
from fullframe_structure_stream import expanded_motion, unpack as unpack_full
from persistent_structure_stream import CROP, unpack as unpack_local
from probe_reef import bands
from temporal_filter import warp


BOX = (342, 10, 502, 105)
KEYS = (0, 15, 45, 75, 105, 135)
EVENT_PAIRS = {i for key in KEYS[1:] for i in range(key, key + 7)}


def analyze(workspace, output):
    output.mkdir(parents=True, exist_ok=False)
    base = workspace / '02_实验数据/优化_GLC后处理'
    batch = base / 'source_dense_flow_three_clips_20261109_01/video1'
    refined = base / 'fullframe_reference_refined_20261110_01/video1'
    local = base / 'video1_persistent_aux_20261110_01/g17q4'
    rgb_paths = {
        'source': batch / 'source.npy',
        'raw': batch / 'base_receive/decoded_rgb.npy',
        'U': local / 'receive/rgb.npy',
        'dense_offline': refined / 'offline/rgb.npy',
        'E': refined / 'g24q2_E/receive/rgb.npy',
        'F': refined / 'g24q2_F/receive/rgb.npy',
    }
    alpha_paths = {
        'U': local / 'receive/alpha.npy',
        'dense_offline': refined / 'offline/alpha.npy',
        'E': refined / 'g24q2_E/receive/alpha.npy',
        'F': refined / 'g24q2_F/receive/alpha.npy',
    }
    sequences = {k: np.load(p, mmap_mode='r') for k, p in rgb_paths.items()}
    alphas = {k: np.load(p, mmap_mode='r') for k, p in alpha_paths.items()}
    for video in sequences.values():
        assert video.shape == (150, 288, 512, 3) and video.dtype == np.uint8
    for alpha in alphas.values():
        assert alpha.shape == (150, 288, 512)
    assert np.array_equal(alphas['E'] > 0, alphas['F'] > 0)
    x0, y0, x1, y1 = BOX
    roi = np.zeros((288, 512), bool)
    roi[y0:y1, x0:x1] = True
    previous = {k: bands(v[0]) for k, v in sequences.items()}
    rows, turnover_rows = [], []
    for i in range(1, 150):
        current = {k: bands(v[i]) for k, v in sequences.items()}
        mx, my, common = source_match(sequences['source'][i-1], sequences['source'][i],
                                      previous['source'], current['source'], np.s_[:, :])
        valid = common & roi
        assert valid.sum() >= 64
        aligned = {k: warp(v, mx, my) for k, v in previous.items()}
        source_delta = current['source'] - aligned['source']
        error = {k: np.abs(current[k][..., 0] - aligned[k][..., 0] - source_delta[..., 0])
                 for k in sequences if k != 'source'}
        held = {}
        for name, alpha in alphas.items():
            prior = warp(alpha[i-1], mx, my)
            prior_active = cv2.remap((alpha[i-1] > 0).astype(np.uint8), mx, my,
                                     cv2.INTER_NEAREST).astype(bool)
            active = alpha[i] > 0
            held[name] = (alpha[i] >= .99) & (prior >= .99)
            turnover_rows.append(dict(frame=i, method=name,
                timing='refresh_window' if i in EVENT_PAIRS else 'non_refresh',
                pixels=int(valid.sum()), switched=int(((active != prior_active) & valid).sum()),
                alpha_change_sum=float(np.abs(alpha[i] - prior)[valid].sum()),
                held_pixels=int((held[name] & valid).sum())))
        # All compared methods use exactly the same pixels within each group.
        groups = {
            'all': valid,
            'both_U_E_held': valid & held['U'] & held['E'],
            'U_held_E_not': valid & held['U'] & ~held['E'],
            'E_held_U_not': valid & held['E'] & ~held['U'],
            'neither_U_E_held': valid & ~held['U'] & ~held['E'],
        }
        assert sum(int(m.sum()) for k, m in groups.items() if k != 'all') == int(valid.sum())
        for group, mask in groups.items():
            n = int(mask.sum())
            if not n:
                continue
            row = dict(frame=i, timing='refresh_window' if i in EVENT_PAIRS else 'non_refresh',
                       group=group, pixels=n)
            row.update({name + '_coarse_sum': float(e[mask].sum()) for name, e in error.items()})
            rows.append(row)
        previous = current
        if i % 50 == 0:
            print(f'Diagnostic pairs: {i}/149', flush=True)

    groups = {}
    for timing in ('all', 'non_refresh', 'refresh_window'):
        groups[timing] = {}
        for group in ('all', 'both_U_E_held', 'U_held_E_not', 'E_held_U_not', 'neither_U_E_held'):
            subset = [r for r in rows if r['group'] == group and
                      (timing == 'all' or r['timing'] == timing)]
            n = sum(r['pixels'] for r in subset)
            if n:
                groups[timing][group] = dict(pairs=len(subset), pixels=n,
                    coarse_mae={name: sum(r[name + '_coarse_sum'] for r in subset) / n
                                for name in error})
    turnover = {}
    for timing in ('all', 'non_refresh', 'refresh_window'):
        turnover[timing] = {}
        for name in alphas:
            subset = [r for r in turnover_rows if r['method'] == name and
                      (timing == 'all' or r['timing'] == timing)]
            n = sum(r['pixels'] for r in subset)
            turnover[timing][name] = dict(pixels=n,
                switch_fraction=sum(r['switched'] for r in subset)/n,
                mean_alpha_change=sum(r['alpha_change_sum'] for r in subset)/n,
                held_fraction=sum(r['held_pixels'] for r in subset)/n)

    archived = json.loads((refined / 'evaluation_perframe/summary.json').read_text(encoding='utf-8'))
    for method, old_name in (('raw', 'raw'), ('U', 'local_U'), ('E', 'E'), ('F', 'F')):
        actual = groups['all']['all']['coarse_mae'][method]
        expected = archived['coral'][old_name]['coarse_temporal_mae']
        assert abs(actual - expected) < 1e-5, (method, actual, expected)

    digest = file_sha(batch / 'base.glcsnd')
    uh, ua = unpack_local((local / 'structure.glcstruct').read_bytes(), digest)
    fh, fa = unpack_full((refined / 'g24q2_F/fullframe.glcfull').read_bytes(), digest)
    dense = np.load(refined / 'flow/to_anchor.npy', mmap_mode='r')
    assert dense.shape == (150, 288, 512, 2)
    # Dense RAFT is a common approximation reference, not optical-flow ground truth.
    motion_samples = {'U': [], 'E_F': []}
    cx0, cy0, cx1, cy1 = CROP
    for i in range(150):
        if i in KEYS:
            continue
        support = roi & (alphas['U'][i] >= .99) & (alphas['E'][i] >= .99)
        um = cv2.resize(ua['new_motion'][i].astype(np.float32)/uh['motion_scale'],
                        (cx1-cx0, cy1-cy0), interpolation=cv2.INTER_LINEAR)
        fm = expanded_motion(fa['new_motion'][i], fh['scale'])
        ur = np.linalg.norm(um-dense[i, cy0:cy1, cx0:cx1], axis=2)
        fr = np.linalg.norm(fm-dense[i], axis=2)
        motion_samples['U'].append(ur[support[cy0:cy1, cx0:cx1]])
        motion_samples['E_F'].append(fr[support])
    motion = {}
    for name, values in motion_samples.items():
        values = np.concatenate(values)
        assert values.size
        motion[name] = dict(pixels=int(values.size), mean_endpoint_distance=float(values.mean()),
                            p95_endpoint_distance=float(np.percentile(values, 95)))
    result = dict(complete=True, kind='Archived-output diagnostic, no candidate changes',
        frames=150, fps=15, fixed_box=BOX, keys=KEYS,
        classification='Motion-compensated alpha >=0.99 in both frames; U/E common groups.',
        refresh_window='Each nonzero key through key+6 inclusive, including transition exit.',
        non_refresh='Other pairs; includes base I/reset events and is not a P-to-P-only subset.',
        scores=groups, motion_compensated_support=turnover,
        common_dense_flow_approximation=motion,
        caution='Post-hoc subgroup associations are not causal factor swaps. Dense source flow is not ground truth. This diagnostic does not prove new performance.',
        F_section_bytes={k: v['bytes'] for k, v in fh['sections'].items()},
        source_paths={k: str(p) for k, p in rgb_paths.items()},
        source_file_sha256={k: file_sha(p) for k, p in rgb_paths.items()},
        alpha_file_sha256={k: file_sha(p) for k, p in alpha_paths.items()},
        script_sha256=file_sha(__file__), old_full_score_reproduced=True)
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for name, data in (('grouped_temporal.csv', rows), ('support_turnover.csv', turnover_rows)):
        with (output/name).open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps({k: result[k] for k in ('scores', 'motion_compensated_support',
                                            'common_dense_flow_approximation', 'F_section_bytes')}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    analyze(args.workspace.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
