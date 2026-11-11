"""Finite low-rate support experiment with frozen GLC references and safe exits."""

import argparse
import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

from cache_motion_inputs import base_frames
from compact_support_stream import (KEYS, LIMIT, expand_support, pack, reduce_support, render, unpack, required_nodes)
from coral_reference_sidecar import file_sha, load_json, save_json
from experiment_common import array_sha
from fullframe_structure_stream import expanded_motion, unpack as old_unpack
from run_dense_batch import CASES
from run_fullframe_factorial import pixel_gate


PACKET = 'support.glccsp'
PROFILES = {
    'M24_B8_T1': ((24, 14), 8, 1),  # Lossless repacking of E.
    'M24_B4_T3': ((24, 14), 4, 3),  # Gate-only mechanism control.
    'M16_B8_T1': ((16, 9), 8, 1),
    'M16_B4_T3': ((16, 9), 4, 3),
    'M12_B4_T3': ((12, 7), 4, 3),
    'M16_B4_T5': ((16, 9), 4, 5),
    'M12_B4_T5': ((12, 7), 4, 5),
    'S24_B4_T3': ((24, 14), 4, 3),
    'S24_B4_T5': ((24, 14), 4, 5),
    'S20_B4_T5': ((20, 12), 4, 5),
    'S16_B4_T5': ((16, 9), 4, 5),
    'S12_B4_T5': ((12, 7), 4, 5),
}


def paths(workspace):
    root = workspace/'02_实验数据/优化_GLC后处理'
    return root/'source_dense_flow_three_clips_20261109_01', root/'fullframe_reference_refined_20261110_01'


def safety(source, prior, motion):
    dense = np.load(prior/'flow/to_anchor.npy', mmap_mode='r')
    old = np.load(prior/'transition_flow.npy', mmap_mode='r')
    initial = np.load(prior/'offline/alpha.npy', mmap_mode='r')
    shared = np.load(prior/'offline/shared.npy', mmap_mode='r')
    frame_ids = list(range(150))+[k+j for k in KEYS[1:] for j in range(6)]
    result = []
    for j, i in enumerate(frame_ids):
        key = KEYS[np.searchsorted(KEYS, i, side='right')-1] if j < 150 else KEYS[(j-150)//6]
        d = dense[j] if j < 150 else old[j-150, 0]
        support = initial[j] if j < 150 else shared[j-150]
        safe, _ = pixel_gate(source, i, key, d, expanded_motion(motion[j], 2), support,
                             np.zeros((288, 512), np.uint16))
        result.append(safe)
    return np.stack(result)


def prepare(workspace, run, case, profiles, fusion):
    batch, prior_root = paths(workspace)
    base, prior = batch/case, prior_root/case
    raw, digest, size = base_frames(base, CASES[case][0])
    source = np.load(base/'source.npy', mmap_mode='r')
    previous = prior/'g24q2_E/fullframe.glcfull'
    h, a = old_unpack(previous.read_bytes(), digest)
    frozen = np.concatenate([a['new_motion'], a['old_motion']])
    if h['keys'] != list(KEYS) or h['scale'] != 2:
        raise ValueError('Frozen motion profile')
    cache = {}
    case_dir = run/case
    case_dir.mkdir(exist_ok=True)
    records = {}
    inputs = {str(p): file_sha(p) for p in (base/'source.npy', base/'base.glcsnd',
              base/'base_receive/decoded_rgb.npy', previous, prior/'flow/to_anchor.npy',
              prior/'transition_flow.npy', prior/'offline/alpha.npy', prior/'offline/shared.npy')}
    for profile in profiles:
        grid, step, span = PROFILES[profile]
        if grid not in cache:
            motion = np.stack([np.rint(cv2.resize(q.astype(np.float32), grid,
                      interpolation=cv2.INTER_LINEAR)).astype('<i2') for q in frozen])
            safe = safety(source, prior, motion)
            if grid == (24, 14):
                assert np.array_equal(motion, frozen)
                if case == 'video1':
                    factorial = run.parent/'fullframe_factorial_20261110_01/gates'
                    expected = np.concatenate([np.load(factorial/(kind+'_PG.npy')) for kind in ('new', 'old')])
                    assert np.array_equal(safe, expected)
            cache[grid] = motion, safe
        motion, safe = cache[grid]
        support = reduce_support(safe, step, span)
        sparse = profile.startswith('S')
        payload, header = pack(motion, support, digest, step=step, span=span, fusion=fusion, sparse=sparse)
        name = profile+'_'+fusion
        folder = case_dir/name
        folder.mkdir(exist_ok=False)
        with (folder/PACKET).open('xb') as f:
            f.write(payload)
        info = dict(profile=name, bytes=len(payload), kbps=len(payload)*8/10000,
                    base_bytes=size, base_kbps=size*8/10000, below_base=len(payload)<size,
                    within_absolute_cap=len(payload)<=LIMIT, header=header,
                    sha256=file_sha(folder/PACKET), inputs_sha256=inputs,
                    safety_policy='All pixels in each spatial cell and all frames of each group must pass the represented-motion safety gate.',
                    source_assistance_at_sender=True, no_hand_edit_ROI=True,
                    lookahead_frames_for_support=span-1, motion_every_frame=True)
        if len(payload) <= LIMIT:
            dh, da = unpack(payload, digest)
            if sparse:
                needed = required_nodes(support, span, step, grid, fusion)
                assert np.array_equal(da['motion'][needed], motion[needed])
                info.update(needed_node_fraction=float(needed.mean()), unused_nodes_omitted=True)
            else:
                assert np.array_equal(da['motion'], motion)
            assert np.array_equal(da['support'], support)
            decoded = expand_support(da['support'], span).astype(bool)
            decoded = np.repeat(np.repeat(decoded, step, axis=1), step, axis=2)
            violations = int((decoded & ~safe).sum())
            assert violations == 0
            result, alpha = render(raw, dh, da)
            if sparse:
                expected, expected_alpha = render(raw, dh, dict(motion=motion, support=support))
                assert np.array_equal(result, expected) and np.array_equal(alpha, expected_alpha)
                info['sparse_vs_full_grid_rgb_alpha_exact'] = True
            info.update(sender_expected_rgb_sha256=array_sha(result), sender_expected_alpha_sha256=array_sha(alpha),
                        unsafe_pixels=violations, pixel_safe_fraction=float(safe.mean()),
                        decoded_safe_fraction=float(decoded.mean()),
                        decoded_coral_fraction=float(decoded[:150, 10:105, 342:502].mean()),
                        required_motion_roundtrip_exact=True, support_roundtrip_exact=True)
            if profile == 'M24_B8_T1':
                original = prior/('g24q2_'+fusion)/'receive'
                assert np.array_equal(result, np.load(original/'rgb.npy', mmap_mode='r'))
                assert np.array_equal(alpha, np.load(original/'alpha.npy', mmap_mode='r'))
                info['original_'+fusion+'_rgb_alpha_exact'] = True
            damaged = bytearray(payload)
            damaged[-8] ^= 1
            checks = {}
            for label, data, binding in (('CRC', damaged, digest), ('wrong_base', payload, '0'*64),
                   ('truncated', payload[:-1], digest), ('over_cap', payload+b'0'*LIMIT, digest)):
                try:
                    unpack(data, binding)
                except ValueError:
                    checks[label] = True
                else:
                    raise ValueError('Malformed packet accepted: '+label)
            info['negative_checks'] = checks
        save_json(folder/'send.json', info)
        records[name] = info
        print(case, name, 'bytes', len(payload), 'below_base', len(payload)<size,
              'safe', info.get('decoded_safe_fraction'), flush=True)
    save_json(case_dir/('prepare_'+fusion+'_'+profiles[0]+'.json'), dict(complete=True, profiles=records))


def receive(workspace, run, case, name):
    batch, _ = paths(workspace)
    base, folder = batch/case, run/case/name
    output = folder/'receive'
    output.mkdir(exist_ok=False)
    allowed = {base/'base.glcsnd', base/'base_cut.json', base/'base_receive/decoded_rgb.npy',
               base/'base_receive/result.json', folder/PACKET}
    allowed = {p.resolve() for p in allowed}
    observed = set()
    protected = {'.json', '.npy', '.npz', '.png', '.jpg', '.mp4', '.mkv', '.glcsnd',
                 '.glcfull', '.glccsp', '.glcflow', '.glcgate', '.glcstruct'}
    def audit(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.suffix.lower() in protected:
                if path not in allowed and not path.is_relative_to(output):
                    raise RuntimeError('Untransmitted receiver input: '+str(path))
                if path in allowed:
                    observed.add(str(path))
    sys.addaudithook(audit)
    blocked = []
    for path in (base/'source.npy', folder/'send.json', run/'untransmitted_flow.npy', run/'untransmitted_mask.npy'):
        try:
            path.read_bytes()
        except RuntimeError:
            blocked.append(str(path))
        else:
            raise ValueError('Receiver guard failed')
    raw, digest, size = base_frames(base, CASES[case][0])
    h, a = unpack((folder/PACKET).read_bytes(), digest)
    result, alpha = render(raw, h, a)
    np.save(output/'rgb.npy', result, allow_pickle=False)
    np.save(output/'alpha.npy', alpha, allow_pickle=False)
    save_json(output/'receipt.json', dict(complete=True, reads=sorted(observed), blocked_inputs=blocked,
              source_files_unavailable=True, independently_verified_base=True, base_sha256=digest,
              base_bytes=size, aux_bytes=(folder/PACKET).stat().st_size,
              aux_sha256=file_sha(folder/PACKET), rgb_array_sha256=array_sha(result),
              alpha_array_sha256=array_sha(alpha), outputs_sha256={p.name:file_sha(p) for p in output.glob('*.npy')},
              zero_weight_unchanged=bool(np.array_equal(result[alpha==0],raw[alpha==0])),
              code_sha256=file_sha(__file__), codec_sha256=file_sha(Path(__file__).with_name('compact_support_stream.py'))))
    print(case, name, 'independent receive PASS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'receive'))
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--case', choices=CASES, required=True)
    p.add_argument('--profiles', nargs='+', choices=PROFILES, default=list(PROFILES))
    p.add_argument('--fusion', choices=('E', 'F'), default='E')
    p.add_argument('--name')
    args = p.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    if args.action == 'prepare':
        prepare(args.workspace.resolve(), args.run.resolve(), args.case, args.profiles, args.fusion)
    else:
        if not args.name:
            p.error('--name required')
        receive(args.workspace.resolve(), args.run.resolve(), args.case, args.name)
