"""Independent read-back of long-reference and transmitted-reference outputs."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from cache_motion_inputs import REPO, base_frames
from coral_reference_sidecar import file_sha, load_json, save_json
from run_structure_memory import CLIP, LONG_ANCHORS
from run_video1_persistent_structure import VIEW_BOX, load_inputs
from run_video1_reference_transition import RAMP, compose
from verify_video1_persistent_structure import inspect_preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--long-run', type=Path, required=True)
    parser.add_argument('--aux-run', type=Path)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    root = args.long_run.resolve()
    _, raw, s, back, _, inputs = load_inputs(args.workspace.resolve())
    receipts = {name: load_json(root/name/'summary.json')
        for name in ('result', 'direct_result', 'transition_result')}
    arrays = {}
    for folder, receipt in receipts.items():
        for name, digest in receipt['outputs_sha256'].items():
            path = root/folder/name
            if file_sha(path) != digest:
                raise ValueError('Long-run array changed: '+str(path))
            a = np.load(path, mmap_mode='r', allow_pickle=False)
            arrays[(folder, name)] = a
            if name.endswith('_rgb.npy') and (a.shape != raw.shape or a.dtype != np.uint8):
                raise ValueError('Wrong output RGB geometry')
    for method in ('source', 'raw', 'S'):
        if not all(r['scores']['fixed'][method] == receipts['result']['scores']['fixed'][method]
                   for r in receipts.values()):
            raise ValueError('Fixed ROI controls changed across evaluations')
    n2 = arrays['direct_result', 'N2_rgb.npy']
    t = arrays['transition_result', 'T_rgb.npy']
    weight = arrays['transition_result', 'transition_weights.npy']
    alpha = arrays['direct_result', 'N2_alpha.npy']
    c2 = np.load(args.workspace/'02_实验数据/优化_GLC后处理/video1_persistent_structure_20261110_01/expanded/C2_rgb.npy')
    if not np.array_equal(c2, n2[105:128]):
        raise ValueError('N2 did not reproduce accepted C2')
    affected = [key+j for key in LONG_ANCHORS[1:] for j in range(len(RAMP))]
    unchanged = [i for i in range(150) if i not in affected]
    if not np.array_equal(t[unchanged], n2[unchanged]) or not np.array_equal(t[weight == 0], n2[weight == 0]):
        raise ValueError('T changed pixels outside its declared transition')
    if not np.array_equal(t[alpha == 0], raw[alpha == 0]):
        raise ValueError('T changed unselected pixels')
    fish = arrays['result', 'fish_locks.npy'].astype(bool)
    if not np.array_equal(t[fish], raw[fish]):
        raise ValueError('T changed protected fish pixels')
    bridge = np.load(root/'bridges/bridge_flow.npy', allow_pickle=False)
    if file_sha(root/'bridges/bridge_flow.npy') != load_json(root/'bridges/flow.json')['flow_sha256']:
        raise ValueError('Bridge changed')
    yy, xx = np.mgrid[:288, :512].astype(np.float32)
    for slot, (old, new) in enumerate(zip(LONG_ANCHORS[:-1], LONG_ANCHORS[1:], strict=True)):
        for j, fade in enumerate(RAMP):
            i = new+j
            if (weight[i] > alpha[i]*fade+1e-6).any():
                raise ValueError('Transition exceeds shared support')
            flow, _ = compose(back[i], bridge[slot, 0], xx, yy)
            old_rgb = cv2.remap(raw[old].astype(np.float32), xx+flow[..., 0], yy+flow[..., 1], cv2.INTER_CUBIC)
            new_rgb = cv2.remap(raw[new].astype(np.float32), xx+back[i, ..., 0], yy+back[i, ..., 1], cv2.INTER_CUBIC)
            expected = np.rint(np.clip(n2[i].astype(np.float32)+weight[i, ..., None]*(old_rgb-new_rgb), 0, 255)).astype(np.uint8)
            if not np.array_equal(t[i], expected):
                raise ValueError('T failed independent original-reference reconstruction')
    preview_checks = (
        ('transition_result/transition_coral_pair_10s.mp4', [raw, t], VIEW_BOX),
        ('transition_result/transition_full_pair_10s.mp4', [raw, t], None),
        ('transition_result/transition_ablation_10s.mp4', [raw, s, n2, t], VIEW_BOX),
        ('direct_result/fresh_reference_pair_10s.mp4', [raw, n2], VIEW_BOX),
        ('result/memory_coral_pair_10s.mp4', [raw, arrays['result', 'M_rgb.npy']], VIEW_BOX),
        ('direct_result/direct_reference_pair_10s.mp4', [raw, arrays['direct_result', 'F_rgb.npy']], VIEW_BOX),
    )
    previews = {name: inspect_preview(args.ffmpeg, root/name, panels, crop)
        for name, panels, crop in preview_checks}
    save_json(root/'verification.json', dict(complete=True, all_saved_array_hashes_checked=True,
        frozen_input_hashes=inputs['files'], fixed_control_scores_exact=True, C2_interval_reproduced=True,
        transition_formula_reproduced_bit_exact=True, other120_frames_unchanged=True,
        zero_weight_pixels_unchanged=True, protected_fish_pixels_unchanged=True,
        previews=previews, code_sha256=file_sha(__file__)))
    print('PASS: long arrays, frozen controls, C2 reproduction, transitions and six previews', flush=True)
    if args.aux_run:
        aux = args.aux_run.resolve()
        summary = load_json(aux/'evaluation/summary.json')
        base, _, size = base_frames(aux, CLIP)
        if not np.array_equal(base, raw):
            raise ValueError('Fresh base decoder differs from frozen baseline')
        for name, receipt in summary['receipts'].items():
            folder = aux/name
            for filename, digest in receipt['output_sha256'].items():
                if file_sha(folder/'receive'/filename) != digest:
                    raise ValueError('Auxiliary output changed')
            packet = folder/'structure.glcstruct'
            if (file_sha(packet) != receipt['auxiliary_sha256'] or packet.stat().st_size != receipt['auxiliary_bytes']
                    or size > 50000 or packet.stat().st_size > 50000):
                raise ValueError('Actual stream integrity/budget failed')
        chosen = summary['selected_development_profile']
        standalone = aux/chosen/'standalone_receive'
        standalone_receipt = load_json(standalone/'receipt.json')
        if not (standalone_receipt['sender_receipt_read_blocked']
                and standalone_receipt['only_base_aux_and_locally_decoded_inputs']):
            raise ValueError('Standalone receiver input isolation failed')
        for filename in ('rgb.npy', 'alpha.npy'):
            actual_hash = file_sha(standalone/filename)
            if (actual_hash != standalone_receipt['output_sha256'][filename]
                    or actual_hash != file_sha(aux/chosen/'receive'/filename)):
                raise ValueError('Standalone replay differs from scored U')
        save_json(aux/'standalone_verification.json', dict(complete=True,
            RGB_and_alpha_file_hashes_exact=True, original_scored_receiver=str(aux/chosen/'receive'),
            standalone_receiver=str(standalone), receipt=standalone_receipt,
            code_sha256=file_sha(__file__)))
        u = np.load(aux/chosen/'receive/rgb.npy', mmap_mode='r')
        alpha_u = np.load(aux/chosen/'receive/alpha.npy', mmap_mode='r')
        if not np.array_equal(u[fish], raw[fish]) or not np.array_equal(u[alpha_u == 0], raw[alpha_u == 0]):
            raise ValueError('Auxiliary preservation failed')
        for method in ('source', 'raw', 'S', 'T'):
            if summary['scores']['fixed'][method] != receipts['transition_result']['scores']['fixed'][method]:
                raise ValueError('Auxiliary evaluation controls changed')
        previews_u = {name: inspect_preview(args.ffmpeg, aux/'evaluation'/name, panels, crop)
            for name, panels, crop in (
                ('auxiliary_coral_pair_10s.mp4', [raw, u], VIEW_BOX),
                ('auxiliary_full_pair_10s.mp4', [raw, u], None),
                ('offline_vs_auxiliary_10s.mp4', [t, u], VIEW_BOX))}
        save_json(aux/'verification.json', dict(complete=True, all_saved_output_hashes_checked=True,
            actual_base_and_aux_each_within_50000_bytes=True, bill_seconds=10,
            fresh_base_decode_bit_exact=True, shared_fixed_roi_controls_exact=True,
            standalone_replay_bit_exact=True, sender_receipt_unavailable_to_standalone=True,
            protected_fish_pixels_unchanged=True, zero_alpha_pixels_unchanged=True,
            receiver_independent_receipts=summary['receipts'], previews=previews_u,
            code_sha256=file_sha(__file__)))
        print('PASS: actual files, independent decode, protected pixels and three auxiliary previews', flush=True)


if __name__ == '__main__':
    main()
