"""Independent receive, safety checks and fixed-support evaluation of a byte probe."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np

import compact_support_stream as base
import predictive_support_stream as previous
import mixed_support_stream as codec
from cache_motion_inputs import base_frames
from compare_deflicker import save_csv
from coral_reference_sidecar import file_sha, load_json, save_json
from experiment_common import array_sha
import evaluate_fullframe_reference as scoring
from fullframe_structure_stream import unpack as unpack_full
from run_compact_support import paths
from run_dense_batch import CASES
from run_fullframe_factorial import BOX
from run_p0 import preview
from verify_compact_support import altered_header
from verify_video1_persistent_structure import inspect_preview


def receive(workspace, run, case, name):
    batch, _ = paths(workspace)
    inp = batch / case
    folder = run / case / name
    packet = folder / 'support.glccsp3'
    out = folder / 'receive'; out.mkdir(exist_ok=False)
    allowed = {p.resolve() for p in (inp/'base.glcsnd', inp/'base_cut.json',
        inp/'base_receive/decoded_rgb.npy', inp/'base_receive/result.json', packet)}
    observed = set()
    protected = {'.json', '.npy', '.npz', '.png', '.jpg', '.mp4', '.mkv', '.glcsnd',
                 '.glcfull', '.glccsp', '.glccsp2', '.glccsp3', '.glcflow', '.glcgate', '.glcstruct'}
    def audit(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes)):
            p = Path(os.fsdecode(args[0])).resolve()
            if p.suffix.lower() in protected:
                if p not in allowed and not p.is_relative_to(out):
                    raise RuntimeError('Untransmitted receiver input: ' + str(p))
                if p in allowed:
                    observed.add(str(p))
    sys.addaudithook(audit)
    blocked = []
    for p in (inp/'source.npy', folder/'probe.json', run/'source_motion.npy', run/'source_support.npy'):
        try:
            p.read_bytes()
        except RuntimeError:
            blocked.append(str(p))
        else:
            raise ValueError('Receiver input guard failed')
    raw, digest, size = base_frames(inp, CASES[case][0])
    h, a = codec.unpack(packet.read_bytes(), digest)
    rgb, alpha = codec.render(raw, h, a)
    np.save(out/'rgb.npy', rgb, allow_pickle=False)
    np.save(out/'alpha.npy', alpha, allow_pickle=False)
    save_json(out/'receipt.json', dict(complete=True, source_files_unavailable=True,
        independently_verified_base=True, reads=sorted(observed), blocked_inputs=blocked,
        base_sha256=digest, base_bytes=size, aux_bytes=packet.stat().st_size,
        aux_sha256=file_sha(packet), rgb_sha256=array_sha(rgb), alpha_sha256=array_sha(alpha),
        zero_weight_unchanged=bool(np.array_equal(rgb[alpha == 0], raw[alpha == 0])),
        outputs_sha256={p.name:file_sha(p) for p in out.glob('*.npy')},
        code_sha256=file_sha(__file__), codec_sha256=file_sha(Path(__file__).with_name('mixed_support_stream.py'))))
    print(case, name, 'independent receiver PASS', flush=True)


def verify_case(workspace, run, case, names):
    batch, prior = paths(workspace)
    old = run.parent/'support_refinement_20261111_01'/case
    digest = file_sha(batch/case/'base.glcsnd')
    kh, ka = previous.unpack((old/'K/support.glccsp2').read_bytes(), digest)
    _, na = previous.unpack((old/'N/support.glccsp2').read_bytes(), digest)
    _, full = unpack_full((prior/case/'g24q2_E/fullframe.glcfull').read_bytes(), digest)
    frozen = np.concatenate([full['new_motion'], full['old_motion']])
    motion = np.stack([np.rint(cv2.resize(q.astype(np.float32), tuple(kh['grid']), interpolation=cv2.INTER_LINEAR)).astype('<i2') for q in frozen])
    raw = np.load(batch/case/'base_receive/decoded_rgb.npy', mmap_mode='r')
    kfine = np.repeat(np.repeat(ka['support'], 2, 1), 2, 2)
    records = {}
    for name in names:
        folder = run/case/name
        data = (folder/'support.glccsp3').read_bytes()
        send = load_json(folder/'probe.json'); receipt = load_json(folder/'receive/receipt.json')
        assert file_sha(folder/'support.glccsp3') == send['sha256'] == receipt['aux_sha256']
        assert len(data) == send['bytes'] == receipt['aux_bytes'] <= 50000
        assert receipt['source_files_unavailable'] and len(receipt['reads']) == 5 and len(receipt['blocked_inputs']) == 4
        for file, digest_file in receipt['outputs_sha256'].items():
            assert file_sha(folder/'receive'/file) == digest_file
        h, a = codec.unpack(data, digest)
        assert h == send['header']
        assert np.all(a['support'][kfine]) and not np.any(a['support'] & ~na['support'])
        needed = base.required_nodes(a['support'], 5, 2, tuple(kh['grid']), 'E')
        assert np.array_equal(a['motion'][needed], motion[needed])
        expected, expected_alpha = codec.render(raw, h, dict(motion=motion, support=a['support']))
        actual = np.load(folder/'receive/rgb.npy', mmap_mode='r')
        alpha = np.load(folder/'receive/alpha.npy', mmap_mode='r')
        assert np.array_equal(actual, expected) and np.array_equal(alpha, expected_alpha)
        assert np.array_equal(actual[alpha == 0], raw[alpha == 0])
        assert np.all(alpha >= np.load(old/'K/receive/alpha.npy', mmap_mode='r') - 1e-6)
        assert np.all(alpha <= np.load(old/'N/receive/alpha.npy', mmap_mode='r') + 1e-6)
        damaged = bytearray(data); damaged[-8] ^= 1
        bad = [('CRC', damaged, digest), ('truncated', data[:-1], digest),
               ('wrong_base', data, '0'*64), ('over_cap', data + bytes(50000), digest)]
        edits = dict(grid=lambda x:x.update(grid=[25,14]), schedule=lambda x:x.update(keys=[0,30]),
            resolution=lambda x:x.update(mask_step=4), group=lambda x:x.update(group_span=3),
            support_length=lambda x:x['sections']['support'].update(raw_bytes=1),
            predictor=lambda x:x['sections']['motion'].update(codec='invalid'))
        if h['sections']['support']['codec'] == 'mixed':
            edits.update(patch_count=lambda x:x['sections']['support'].update(patch_count=-1),
                patch_layout=lambda x:x['sections']['support'].update(patch_layout='invalid'))
        bad.extend((label, altered_header(data, edit), digest) for label, edit in edits.items())
        checks = {}
        for label, payload, binding in bad:
            try:
                codec.unpack(payload, binding)
            except ValueError:
                checks[label] = True
            else:
                raise ValueError('Malformed packet accepted: ' + label)
        records[name] = dict(complete=True, bytes=len(data), safe_subset_of_N=True,
            all_K_support_preserved=True, needed_motion_exact=True, unpruned_RGB_alpha_exact=True,
            independent_receive=True, zero_weight_unchanged=True, negative_packet_checks=checks)
    return records


def evaluate_case(workspace, run, case, names, ffmpeg):
    batch, _ = paths(workspace)
    old = run.parent/'support_refinement_20261111_01'/case
    out = run/case/'evaluation'; out.mkdir(exist_ok=False)
    sequences = dict(source=np.load(batch/case/'source.npy', mmap_mode='r'),
        raw=np.load(batch/case/'base_receive/decoded_rgb.npy', mmap_mode='r'),
        K=np.load(old/'K/receive/rgb.npy', mmap_mode='r'),
        N=np.load(old/'N/receive/rgb.npy', mmap_mode='r'))
    for name in names:
        sequences[name] = np.load(run/case/name/'receive/rgb.npy', mmap_mode='r')
    scoring.SCOPES.pop('coral', None)
    if case == 'video1':
        scoring.SCOPES['coral'] = BOX
        sequences['U'] = np.load(run.parent/'video1_persistent_aux_20261110_01/g17q4/receive/rgb.npy', mmap_mode='r')
    scores, temporal, spatial = scoring.score(sequences)
    archived = load_json(old/'evaluation/summary.json')['scores']
    for scope in scores:
        for name in ('raw', 'K', 'N'):
            for field in ('coarse_mae', 'fine_ncc', 'psnr_db', 'valid_pixels'):
                assert abs(scores[scope][name][field] - archived[scope][name][field]) < 1e-9
    save_csv(out/'temporal.csv', temporal); save_csv(out/'spatial.csv', spatial)
    changes = {}
    for name in names:
        difference = sequences[name].astype(np.float32) - sequences['K']
        changes[name] = dict(changed_pixel_fraction=float(np.any(difference != 0, axis=3).mean()),
            rgb_abs_mean=float(np.abs(difference).mean()), rgb_abs_max=float(np.abs(difference).max()))
    rates = {name:load_json(run/case/name/'probe.json') for name in names}
    labels = {'K':f'K / aux {load_json(old/"K/send.json")["kbps"]:.2f} kbps'}
    labels.update({name:f'{name} / aux {rates[name]["kbps"]:.2f} kbps' for name in names})
    specs = [(f'K_vs_{name}_full.mp4', ('K', name), None) for name in names]
    if case == 'video1':
        specs += [(f'K_vs_{name}_coral.mp4', ('K', name), (250,0,512,132)) for name in names]
    previews = {}
    for filename, pair, crop in specs:
        panels = [sequences[n] for n in pair]
        region = None if crop is None else np.s_[crop[1]:crop[3],crop[0]:crop[2]]
        preview(ffmpeg, out/filename, panels, 15, 0, [labels[n] for n in pair], crop=region)
        previews[filename] = inspect_preview(ffmpeg, out/filename, panels, crop=crop)
    result = dict(complete=True, scores=scores, baseline_scores_exact=True, changes_from_K=changes,
        rates={k:{field:v[field] for field in ('bytes','kbps','delta_K_bytes','delta_K_percent','base_bytes')} for k,v in rates.items()},
        previews=previews, visual_review='pending', source_correspondence_only_for_evaluation=True)
    save_json(out/'summary.json', result)
    scope = 'coral' if case == 'video1' else 'full'
    print(case, scope, {k:(round(v['coarse_mae'],6),round(v['fine_ncc'],6)) for k,v in scores[scope].items()}, flush=True)
    return result


def finish(workspace, run, ffmpeg):
    choice = load_json(run/'render_selection.json')
    names = choice['variants']
    if len(names) > 2:
        raise ValueError('Frozen render limit')
    records = {}; results = {}
    for case in ('video1','video2','video3'):
        for name in names:
            subprocess.run([sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'receive',
                '--workspace',str(workspace),'--run',str(run),'--case',case,'--name',name],check=True)
        records[case] = verify_case(workspace,run,case,names)
        save_json(run/'verification_progress.json',records)
        results[case] = evaluate_case(workspace,run,case,names,ffmpeg)
        save_json(run/'evaluation_progress.json',results)
    save_json(run/'verification.json',dict(complete=True,cases=records))
    save_json(run/'final_summary.json',dict(complete=True,cases=results,visual_review='pending',
        no_default_change=True,no_equal_budget_claim=True))
    snap=run/'source_final';snap.mkdir()
    for name in ('finish_mixed_support.py','probe_mixed_support.py','mixed_support_stream.py',
                 'predictive_support_stream.py','compact_support_stream.py','fullframe_structure_stream.py',
                 'evaluate_fullframe_reference.py'):
        shutil.copyfile(Path(__file__).with_name(name),snap/name)
    save_json(snap/'manifest.json',{p.name:file_sha(p) for p in snap.iterdir()})


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('receive','finish'))
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--case',choices=tuple(CASES));p.add_argument('--name')
    p.add_argument('--ffmpeg',type=Path)
    a=p.parse_args()
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    if a.action=='receive':
        if not a.case or not a.name:p.error('receive requires case and name')
        receive(a.workspace.resolve(),a.run.resolve(),a.case,a.name)
    else:
        if not a.ffmpeg:p.error('finish requires ffmpeg')
        finish(a.workspace.resolve(),a.run.resolve(),a.ffmpeg.resolve())
