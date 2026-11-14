"""Verify, compare and prepare original-speed views for the mild round."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from runtime_speed_common import STAGE, FFMPEG, WEIGHT, file_sha, load_json, save_json, check_inputs
from receiver_structure import render
from receive_guide_texture import texture_gate
from run_generalization_reliability import PROFILES
from finish_generalization_factorial import sample_sheet
from run_p0 import preview
from verify_video1_persistent_structure import inspect_preview
import structure_guide_stream as guide


def verify(run):
    target = run / 'verification.json'
    if target.exists():
        raise FileExistsError(target)
    plan = load_json(run / 'execution_plan.json')
    check_inputs(plan['input_sha256'])
    for name, digest in plan['code_sha256'].items():
        assert file_sha(STAGE / name) == digest
        assert file_sha(run / 'source_snapshot' / name) == digest
    cases, count, baseline_count, replay_count = {}, 0, 0, 0
    for case, spec in plan['cases'].items():
        raw = np.load(spec['raw'], mmap_mode='r')
        base_digest = file_sha(spec['base'])
        per_case = {}
        for quality in ('Q6', 'Q9'):
            root = run / case / quality
            out = root / 'receive'
            send, receipt = load_json(root / 'send.json'), load_json(out / 'receipt.json')
            assert receipt['complete'] and receipt['source_reads_blocked'] and len(receipt['blocked']) == 5
            assert receipt['code_sha256'] == plan['code_sha256']['run_generalization_reliability.py']
            assert receipt['gate_code_sha256'] == plan['code_sha256']['mild_reliability_gate.py']
            # ReadGuard always authorizes the preinstalled RAFT checkpoint in
            # addition to the per-job input list recorded by the worker.
            assert set(receipt['reads']).issubset(set(receipt['allowed']) | {str(WEIGHT.resolve())})
            assert str(Path(spec['source']).resolve()) not in receipt['reads']
            assert receipt['base_sha256'] == base_digest and receipt['base_decode_verified']
            assert receipt['auxiliary_sha256'] == send['sha256'] == file_sha(root / 'guide.glcguide')
            assert send['bytes'] == (root / 'guide.glcguide').stat().st_size
            assert send['bytes'] <= 50000 and Path(spec['base']).stat().st_size <= 50000
            assert abs(send['total_kbps'] - (send['bytes'] + Path(spec['base']).stat().st_size)*8/10000) < 1e-9
            header, _, images = guide.decode((root / 'guide.glcguide').read_bytes(), base_digest, FFMPEG)
            assert header == receipt['guide_header'] and file_sha(out / 'guide_rgb.npy') == receipt['guide_rgb_sha256']
            assert np.array_equal(images, np.load(out / 'guide_rgb.npy', mmap_mode='r'))
            flows = []
            for name in ('to_anchor', 'from_anchor', 'transition_flow'):
                path = Path(receipt['flow_paths'][name])
                assert file_sha(path) == receipt['flow_sha256'][name]
                value = np.load(path, mmap_mode='r')
                assert np.isfinite(value).all() and value.dtype == np.float32
                assert value.shape == ((30, 2, 288, 512, 2) if name == 'transition_flow' else (150, 288, 512, 2))
                flows.append(value)
            assert receipt['flow']['pairs'] == 174
            if quality == 'Q6':
                assert receipt['flow']['reused_local_cache']
                assert file_sha(receipt['flow']['provenance_receipt']) == receipt['flow']['provenance_sha256']
            alpha_base = np.load(out / 'T3/alpha.npy', mmap_mode='r')
            shared_base = np.load(out / 'T3/shared.npy', mmap_mode='r')
            for mode in ('T3', 'R25', 'D25'):
                values = {}
                for name, digest in receipt['variants'][mode]['outputs_sha256'].items():
                    path = out / mode / (name + '.npy')
                    assert file_sha(path) == digest
                    value = np.load(path, mmap_mode='r')
                    assert value.shape == {'rgb': (150, 288, 512, 3), 'alpha': (150, 288, 512), 'shared': (30, 288, 512)}[name]
                    assert value.dtype == (np.uint8 if name == 'rgb' else np.float32)
                    assert np.isfinite(value).all()
                    values[name] = value
                for name, base in (('alpha', alpha_base), ('shared', shared_base)):
                    value = values[name]
                    assert np.all((value >= 0) & (value <= 1))
                    assert np.all(value <= base + 1e-7) and np.all(value >= .75 * base - 1e-7)
                    assert np.array_equal(value > 0, base > 0)
                zero = values['alpha'] == 0
                assert np.array_equal(values['rgb'][zero], raw[zero])
                if mode == 'T3' and quality == 'Q6':
                    for name in values:
                        assert file_sha(out / mode / (name + '.npy')) == file_sha(Path(spec['prior_root']) / 'Q6/receive/T3' / (name + '.npy'))
                    baseline_count += 1
                count += 1
            if quality == 'Q9':
                # Original uncached T3 gate, same independently received bytes and
                # receiver-local flow; no re-estimation or source input needed.
                replay = render(raw, images, *flows, gate_fn=texture_gate(images))
                for name, array in replay.items():
                    assert np.array_equal(array, np.load(out / 'T3' / (name + '.npy'), mmap_mode='r'))
                replay_count += 1
            per_case[quality] = dict(hashes_types_shapes_finite=True, source_free_read_boundary=True,
                                    actual_guide_decode_exact=True, original_support_identical=True,
                                    attenuation_bounded_to_25_percent=True, zero_weight_unchanged=True,
                                    total_kbps=send['total_kbps'], local_flow_reused=quality=='Q6')
        cases[case] = per_case
        print('VERIFIED', case, flush=True)
    assert count == 24 and baseline_count == 4 and replay_count == 4
    save_json(target, dict(complete=True, verified_new_outputs=count, baseline_exact_cases=baseline_count,
        uncached_Q9_T3_gate_replays=replay_count, guide_conditions=8, new_Q9_flow_runs=4,
        reused_Q6_flow_caches=4, frozen_inputs_and_code_unchanged=True, adopted_configuration_unchanged=True,
        cases=cases, verifier_sha256=file_sha(__file__),
        scope='Replay uses the same flow; not an independent RAFT implementation or end-to-end timing test'))
    print('VERIFICATION COMPLETE', flush=True)


def previews(run):
    assert load_json(run / 'verification.json')['complete']
    summary, plan = load_json(run / 'summary.json'), load_json(run / 'execution_plan.json')
    review = load_json(run / 'review_selection.json')
    assert review['thresholds_changed'] is False and review['automatic_adoption'] is False
    assert review['original_numeric_selected'] == summary['selected']
    chosen, other = review['primary'], review['secondary']
    assert chosen in PROFILES[1:] and other in PROFILES[1:] and chosen != other
    candidates = [chosen, other]
    out = run / 'previews'
    out.mkdir(exist_ok=False)
    records = {}
    for case, spec in plan['cases'].items():
        seq = dict(source=np.load(spec['source'], mmap_mode='r'), raw=np.load(spec['raw'], mmap_mode='r'))
        for name in PROFILES:
            q, mode = name.split('_')
            seq[name] = np.load(run / case / q / 'receive' / mode / 'rgb.npy', mmap_mode='r')
        rates = summary['cases'][case]['rates']
        labels = {n:f'{n} / {rates[n]:.2f} kbps' for n in PROFILES}
        labels.update(source='Source RGB', raw='Original GLC')
        for candidate in candidates:
            for suffix, names in [('source_raw_current_' + candidate, ('source', 'raw', 'Q6_T3', candidate)),
                                  ('current_vs_' + candidate, ('Q6_T3', candidate))]:
                path = out / (case + '_' + suffix + '.mp4')
                panels = [seq[n] for n in names]
                preview(FFMPEG, path, panels, 15, 0, [labels[n] for n in names])
                records[path.name] = inspect_preview(FFMPEG, path, panels)
        if case == 'video2':
            path = out / 'video2_all_six_profiles.mp4'
            panels = [seq[n] for n in PROFILES]
            preview(FFMPEG, path, panels, 15, 0, [labels[n] for n in PROFILES])
            records[path.name] = inspect_preview(FFMPEG, path, panels)
        sample_sheet(out / (case + '_contact.png'), case, {n:seq[n] for n in ('source','raw','Q6_T3',chosen)},
                     dict(source='原片', raw='原GLC', Q6_T3='当前方案', **{chosen:'候选 ' + chosen}), (20,64,80,128,149))
        print('PREVIEWS', case, chosen, other, flush=True)
    assert len(records) == 17
    save_json(run / 'preview_verification.json', dict(complete=True, selected_for_viewing=chosen,
        numeric_screen_passed=summary['selected'] is not None, additional_candidate=other,
        candidates=candidates, videos=records, subjective_review='pending', automatic_adoption=False,
        review_selection=review, selection_stage='post-score tradeoff review; frozen numeric screen unchanged',
        layout='Pair: current Q6_T3 left, candidate right. Four-way: source/raw/current/candidate in row-major order.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'previews'))
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    (verify if args.action == 'verify' else previews)(args.run)
