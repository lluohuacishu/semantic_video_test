"""Verify the completed 24-condition round without rerunning inference."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from runtime_speed_common import STAGE, FFMPEG, WEIGHT, file_sha, load_json, save_json
from finish_generalization_factorial import sample_sheet
import structure_guide_stream as guide


def verify(run):
    target = run / 'final_verification.json'
    if target.exists():
        raise FileExistsError(target)
    plan = load_json(run / 'execution_plan.json')
    extra_plan = load_json(run / 'extra_scene/execution_plan.json')
    for path, digest in plan['input_sha256'].items():
        assert file_sha(path) == digest, path
    for name, digest in plan['code_sha256'].items():
        assert file_sha(STAGE / name) == digest, name
        assert file_sha(run / 'source_snapshot' / name) == digest, name
    assert file_sha(STAGE / 'run_generalization_extra.py') == extra_plan['code_sha256']
    assert file_sha(run / 'execution_plan.json') == extra_plan['parent_plan_sha256']
    registration = load_json(run / 'registered_extra_source/registration.json')
    assert file_sha(registration['source_metadata']) == registration['source_metadata_sha256']
    for chunk in registration['chunks']:
        assert file_sha(chunk['path']) == chunk['sha256']
    assert file_sha(run / 'registered_extra_source/source.npy') == registration['source_npy_sha256']

    specs = [(name, run, spec) for name, spec in plan['cases'].items()]
    extra_specs = load_json(run / 'extra_scene/receiver_input_registry.json')
    specs += [(name, run / 'extra_scene', spec) for name, spec in extra_specs.items()]
    records = {}
    variants_checked = flows_checked = exact_baselines = 0
    for case, parent, spec in specs:
        raw = np.load(spec['raw'], mmap_mode='r', allow_pickle=False)
        base_sha = file_sha(spec['base'])
        base_receipt = load_json(spec['base_receipt'])
        assert base_receipt['stream_sha256'] == base_sha
        assert base_receipt['decoded_rgb_sha256'] == file_sha(spec['raw'])
        assert base_receipt['reference_verified'] and base_receipt['source_media_reads_forbidden']
        case_record = {}
        for quality in ('Q6', 'Q12'):
            folder = parent / case / quality
            receive = folder / 'receive'
            receipt = load_json(receive / 'receipt.json')
            send = load_json(folder / 'send.json')
            assert receipt['complete'] and receipt['source_reads_blocked']
            assert len(receipt['blocked']) == 5 and receipt['no_per_clip_parameters']
            assert receipt['code_sha256'] == plan['code_sha256']['run_generalization_factorial.py']
            assert receipt['base_sha256'] == base_sha and receipt['base_decode_verified']
            allowed = {str(Path(p).resolve()) for p in
                       [spec['base'], spec['raw'], spec['base_receipt'], folder / 'guide.glcguide', WEIGHT]}
            assert set(receipt['reads']).issubset(allowed), receipt['reads']
            assert str(Path(spec['source']).resolve()) not in receipt['reads']
            assert file_sha(folder / 'guide.glcguide') == send['sha256'] == receipt['auxiliary_sha256']
            assert send['bytes'] == (folder / 'guide.glcguide').stat().st_size
            assert send['bytes'] <= 50000 and Path(spec['base']).stat().st_size <= 50000
            assert abs(send['total_kbps'] - (send['bytes'] + Path(spec['base']).stat().st_size) * 8 / 10000) < 1e-9
            header, _, full = guide.decode((folder / 'guide.glcguide').read_bytes(), base_sha, FFMPEG)
            assert header == receipt['guide_header'] and full.shape == (150, 288, 512, 3)
            assert np.array_equal(full, np.load(receive / 'guide_rgb.npy', mmap_mode='r'))
            assert receipt['flow']['pairs'] == 174
            for name, digest in receipt['flow_sha256'].items():
                path = receive / (name + '.npy')
                assert file_sha(path) == digest and np.isfinite(np.load(path, mmap_mode='r')).all()
            flows_checked += 1
            alpha_base = np.load(receive / 'T3/alpha.npy', mmap_mode='r')
            shared_base = np.load(receive / 'T3/shared.npy', mmap_mode='r')
            for mode in ('T3', 'F8', 'C8'):
                arrays = {}
                for name, digest in receipt['variants'][mode]['outputs_sha256'].items():
                    path = receive / mode / (name + '.npy')
                    assert file_sha(path) == digest
                    value = np.load(path, mmap_mode='r', allow_pickle=False)
                    assert value.shape == {'rgb': (150, 288, 512, 3), 'alpha': (150, 288, 512), 'shared': (30, 288, 512)}[name]
                    assert value.dtype == (np.uint8 if name == 'rgb' else np.float32)
                    assert np.isfinite(value).all()
                    arrays[name] = value
                assert np.all((arrays['alpha'] >= 0) & (arrays['alpha'] <= alpha_base + 1e-7))
                assert np.all((arrays['shared'] >= 0) & (arrays['shared'] <= shared_base + 1e-7))
                assert np.max(arrays['alpha']) <= 1 and np.max(arrays['shared']) <= 1
                zero = arrays['alpha'] == 0
                assert np.array_equal(arrays['rgb'][zero], raw[zero])
                if quality == 'Q6' and mode == 'T3':
                    original = Path(spec['current']) if 'current' in spec else parent / case / 'original_i12'
                    for name in arrays:
                        assert file_sha(receive / mode / (name + '.npy')) == file_sha(original / (name + '.npy'))
                    exact_baselines += 1
                variants_checked += 1
            case_record[quality] = dict(output_hashes_shapes_dtypes_finite=True, guide_decode_exact=True,
                                        read_allowlist_and_denials=True, base_binding=True,
                                        support_and_zero_alpha_preserved=True, actual_total_kbps=send['total_kbps'])
        records[case] = case_record
        print('VERIFIED', case, flush=True)

    main_previews = load_json(run / 'preview_verification.json')
    extra_summary = load_json(run / 'extra_scene/summary.json')
    assert main_previews['complete']
    videos = [(run / 'previews' / name, record) for name, record in main_previews['videos'].items()]
    videos += [(run / 'extra_scene/archive82_first10s/evaluation' / name, record)
               for name, record in extra_summary['previews'].items()]
    for path, record in videos:
        assert file_sha(path) == record['sha256']
        probe = record['probe']
        assert probe['avg_frame_rate'] == '15/1' and int(probe['nb_read_frames']) == 150
        assert abs(float(probe['duration']) - 10) < 1e-6
        assert len(record['frame_indices']) == 3
    spec = extra_specs['archive82_first10s']
    extra_root = run / 'extra_scene/archive82_first10s'
    contact = extra_root / 'evaluation/archive82_contact.png'
    if contact.exists():
        raise FileExistsError(contact)
    sequences = {'source': np.load(spec['source'], mmap_mode='r'), 'raw': np.load(spec['raw'], mmap_mode='r'),
                 'Q6_T3': np.load(extra_root / 'Q6/receive/T3/rgb.npy', mmap_mode='r'),
                 'Q12_T3': np.load(extra_root / 'Q12/receive/T3/rgb.npy', mmap_mode='r')}
    sample_sheet(contact, 'archive82', sequences,
                 dict(source='原片', raw='原GLC', Q6_T3='当前方案', Q12_T3='候选 Q12_T3'), (20, 64, 80, 128, 149))
    assert variants_checked == 24 and flows_checked == 8 and exact_baselines == 4 and len(videos) == 15
    save_json(target, dict(complete=True, date='2026-11-12', verified_variants=variants_checked,
                          verified_guide_flows=flows_checked, exact_current_baselines=exact_baselines,
                          verified_preview_files=len(videos), frozen_inputs_and_code_unchanged=True,
                          adopted_configuration_unchanged=True, cases=records,
                          subjective_review='pending; static contact sheets do not establish temporal visual quality',
                          verifier_sha256=file_sha(__file__)))
    print('FINAL VERIFIED: 24 variants, 8 guide flows, 4 exact baselines, 15 previews', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    verify(args.run.resolve())
