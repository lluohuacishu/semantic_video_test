"""Offline video1 controls for a coarser structure band and longer GLC anchors."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from cache_motion_inputs import base_frames
from compare_source_dense_flow import load_flow
from coral_reference_sidecar import file_sha, load_json, save_json
from experiment_common import array_sha
from run_dense_batch import CASES
from run_dense_transfer import source_input
from run_gate_ablation import EDIT_WINDOWS, editable, render, source_gate
from source_raft_flow import infer


CASE = 'video1'
CLIP, _ = CASES[CASE]
LONG_ANCHORS = (0, 15, 45, 75, 105, 135)


def fit_long(base, run, upstream):
    source = source_input(base, CLIP)
    periodic = load_flow(base/'flow', source, CLIP)
    infer(source, CLIP, upstream, run/'long_flow', anchors=LONG_ANCHORS, periodic_flows=periodic)


def render_structure(base, ablation, run):
    raw, base_sha, _ = base_frames(base, CLIP)
    source = source_input(base, CLIP)
    flow, reverse = load_flow(base/'flow', source, CLIP)
    allowed = editable(CASE)
    gate = source_gate(source, flow, reverse)
    original = load_json(ablation/'upper/result.json')
    prior, alpha = render(raw, flow, gate, allowed)
    if (array_sha(prior) != original['rgb_sha256'] or array_sha(alpha) != original['alpha_sha256']
            or original['base_sha256'] != base_sha or not original['offline_upper']):
        raise ValueError('Could not reproduce the existing O reference bit for bit')
    del prior
    result, structure_alpha = render(raw, flow, gate, allowed, structure_windows=EDIT_WINDOWS[CASE])
    if not np.array_equal(alpha, structure_alpha) or not np.array_equal(result[alpha==0], raw[alpha==0]):
        raise ValueError('The structure control changed the reliability weights or protected pixels')
    out = run/'structure'
    out.mkdir(parents=True, exist_ok=False)
    np.save(out/'rgb.npy', result, allow_pickle=False)
    save_json(out/'result.json', dict(offline_source_oracle=True, base_sha256=base_sha,
        rgb_sha256=array_sha(result), alpha_sha256=array_sha(alpha), original_o_bit_exact=True,
        structure_sigma=(3, 12), protection_windows=EDIT_WINDOWS[CASE], code_sha256=file_sha(__file__)))
    print('O reproduced bit-exactly; S used exactly the same weights and anchors', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('long-flow', 'structure'))
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--ablation', type=Path)
    parser.add_argument('--upstream', type=Path)
    args = parser.parse_args()
    base, run = args.batch.resolve()/CASE, args.run.resolve()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    if args.action == 'long-flow':
        if args.upstream is None:
            parser.error('long-flow requires --upstream')
        fit_long(base, run, args.upstream.resolve())
    else:
        if args.ablation is None:
            parser.error('structure requires --ablation')
        render_structure(base, args.ablation.resolve()/CASE, run)


if __name__ == '__main__':
    main()
