"""Independent saved-output checks for the offline factorial experiment."""

import argparse
import csv
from pathlib import Path
import shutil

import numpy as np

from coral_reference_sidecar import file_sha, load_json, save_json


def verify(run):
    plan = load_json(run/'frozen_plan.json')
    records = load_json(run/'render_summary.json')
    scores = load_json(run/'evaluation/summary.json')
    diagnostics = load_json(run/'diagnostics/summary.json')
    previews = load_json(run/'evaluation/preview_verification.json')
    for spec in plan['frozen_inputs'].values():
        assert file_sha(Path(spec['path'])) == spec['sha256']
    for name, digest in records['gate_file_hashes'].items():
        assert file_sha(run/'gates'/name) == digest
    raw = np.load(plan['frozen_inputs']['base_rgb']['path'], mmap_mode='r')
    unique = {r['canonical'] for r in records['cases'].values()}
    for name in unique:
        folder = run/name
        receipt = load_json(folder/'receipt.json')
        for n, digest in receipt['outputs_sha256'].items():
            assert file_sha(folder/n) == digest
        rgb = np.load(folder/'rgb.npy', mmap_mode='r')
        alpha = np.load(folder/'alpha.npy', mmap_mode='r')
        old = np.load(folder/'old_weight.npy', mmap_mode='r')
        assert rgb.shape == (150, 288, 512, 3) and rgb.dtype == np.uint8
        assert alpha.shape == old.shape == (150, 288, 512)
        assert alpha.dtype == old.dtype == np.float32
        assert np.isfinite(alpha).all() and np.isfinite(old).all()
        for i in range(150):
            assert np.all((0 <= alpha[i]) & (alpha[i] <= 1))
            assert np.all((0 <= old[i]) & (old[i] <= alpha[i]))
            assert np.array_equal(rgb[i][alpha[i] == 0], raw[i][alpha[i] == 0])
    for a, b in (('DP', 'GP'), ('DB', 'GB')):
        for name in ('alpha.npy', 'old_weight.npy'):
            pa = run/records['cases'][a]['canonical']/name
            pb = run/records['cases'][b]['canonical']/name
            assert np.array_equal(np.load(pa, mmap_mode='r'), np.load(pb, mmap_mode='r'))
    source_root = Path(plan['frozen_inputs']['E_packet']['path']).parent
    e_matches = {}
    for name in ('rgb.npy', 'alpha.npy'):
        e_matches[name] = bool(np.array_equal(np.load(source_root/'receive'/name, mmap_mode='r'),
                                            np.load(run/'native_GB'/name, mmap_mode='r')))
        assert e_matches[name]
    refined = source_root.parent
    native_dense_exact = all(np.array_equal(np.load(run/'native_DP'/name, mmap_mode='r'),
                                           np.load(refined/'offline'/name, mmap_mode='r'))
                             for name in ('rgb.npy', 'alpha.npy'))
    assert native_dense_exact
    masks = {kind: {n: np.load(run/'gates'/(kind+'_'+n+'.npy'), mmap_mode='r')
                   for n in ('PD', 'PG', 'C', 'BC', 'BG')} for kind in ('new', 'old')}
    for kind, items in masks.items():
        assert np.array_equal(items['C'], items['PD'] & items['PG'])
        assert np.array_equal(items['C'], items['PG'])
        assert np.array_equal(items['BC'], items['BG'])
        assert not np.any(items['BC'] & ~items['C'])
    rows = list(csv.DictReader((run/'evaluation/temporal.csv').open(encoding='utf-8', newline='')))
    for scope, methods in scores['scores'].items():
        counts = set()
        for method, expected in methods.items():
            selected = [r for r in rows if r['scope'] == scope and r['method'] == method]
            n = sum(int(r['pixels']) for r in selected)
            assert len(selected) == 149 and n == expected['valid_pixels']
            counts.add(tuple(int(r['pixels']) for r in selected))
            mae = sum(float(r['coarse_sum']) for r in selected)/n
            assert abs(mae-expected['coarse_mae']) < 1e-10
        assert len(counts) == 1
    with (run/'diagnostics/rejection_on_evaluation_support.csv').open(encoding='utf-8', newline='') as f:
        cause_rows = list(csv.DictReader(f))
    for row in cause_rows:
        exclusive = sum(int(v) for k, v in row.items() if k.endswith('_exclusive'))
        assert exclusive+int(row['block_safe']) == int(row['pixels'])
    assert diagnostics['rejection_causes']['U_held_GB_not']['pixels'] == 376270
    assert diagnostics['pixel_mask_byte_probe']['over_budget_before_header']
    for name, meta in previews['previews'].items():
        assert file_sha(run/'evaluation'/name) == meta['sha256']
        assert int(meta['probe']['nb_read_frames']) == 150 and meta['probe']['avg_frame_rate'] == '15/1'
    assert scores['complete'] and diagnostics['complete'] and previews['complete']
    stage = Path(__file__).parent
    sources = ('run_fullframe_factorial.py', 'evaluate_fullframe_factorial.py', 'diagnose_fullframe_factorial.py')
    for name in sources:
        assert file_sha(stage/name) == file_sha(run/name)
    result = dict(complete=True, frozen_inputs_unchanged=True, all_saved_outputs_hashed=True,
        unique_rendered_versions=len(unique), E_rgb_alpha_bit_exact=e_matches,
        native_dense_equals_archived_dense=native_dense_exact,
        same_gate_motion_pairs_have_identical_alpha_and_old_weights=True,
        shared_external_evaluation_denominators=True, all_temporal_scores_reconstructed=True,
        reason_partition_exact=True, preview_count=len(previews['previews']),
        actual_transmitted_candidate=False, code_sha256=file_sha(__file__))
    save_json(run/'verification.json', result)
    shutil.copyfile(__file__, run/Path(__file__).name)
    save_json(run/'source_manifest.json', {name: file_sha(stage/name) for name in (*sources, Path(__file__).name)})
    print('PASS: frozen inputs, five unique renders, E/dense reproduction, controlled weights, scores, causes, four previews', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    verify(parser.parse_args().run.resolve())
