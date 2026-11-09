"""Run the same affine and source RAFT controls on the frozen shipwreck clip."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from cache_motion_inputs import FPS, base_frames
from coral_reference_sidecar import file_sha, load_json, save_json
from dense_transfer_inputs import CLIP
from experiment_common import array_sha
from run_cache_motion import PACKET, estimate, pack, render, unpack
from source_raft_flow import infer


def source_input(run):
    receipt = load_json(run/'inputs.json')
    path = run/'source.npy'
    source = np.load(path, allow_pickle=False)
    if (file_sha(path) != receipt['source_npy_sha256']
            or array_sha(source) != receipt['source_rgb_sha256']
            or receipt['source_range_15fps'] != [CLIP.start, CLIP.stop]
            or tuple(receipt['background_roi']) != CLIP.coral):
        raise ValueError('The frozen transfer input changed')
    return source


def evaluate(run, ffmpeg, source):
    from compare_source_dense_flow import compare_clip

    raw, base_sha, _ = base_frames(run, CLIP)
    encoded = pack(CLIP, base_sha, estimate(source, use_history=True))
    wire = run/PACKET
    with wire.open('xb') as stream:
        stream.write(encoded)
    records = unpack(wire.read_bytes(), CLIP, base_sha)
    c, mask = render(raw, records)
    output = run/'flow'
    compare_clip(CLIP, dict(source=source, raw=raw, c=c), records, mask, output, ffmpeg,
                 base_sha, region='ship')
    size = wire.stat().st_size
    save_json(run/'affine_control.json', dict(
        motion_bytes=size, motion_kbps=size*8*FPS/CLIP.count/1000,
        motion_sha256=file_sha(wire), accepted_steps=int(records['valid'].sum()),
        total_steps=len(records), rgb_sha256=array_sha(c), alpha_sha256=array_sha(mask),
        inputs_sha256=file_sha(run/'inputs.json'), code_sha256=file_sha(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('flow', 'compare'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--ffmpeg', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    source = source_input(run)
    if args.action == 'flow':
        if args.upstream is None:
            parser.error('--upstream is required to compute RAFT flow')
        infer(source, CLIP, args.upstream.resolve(), run/'flow')
    else:
        if args.ffmpeg is None:
            parser.error('--ffmpeg is required to render the comparisons')
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        cv2.setRNGSeed(20261109)
        evaluate(run, args.ffmpeg.resolve(), source)


if __name__ == '__main__':
    main()
