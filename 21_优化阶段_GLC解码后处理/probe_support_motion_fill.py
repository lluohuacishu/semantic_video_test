"""Conservative bilinear-node dependency pruning, measured by exact output."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from compact_support_stream import expand_support, pack, reduce_support
from coral_reference_sidecar import file_sha, save_json
from fullframe_structure_stream import unpack


def dependencies(mask, grid, tone=False):
    """All four resize contributors for every potentially used output pixel.

    F tone matching blurs remapped RGB: include its complete OpenCV Gaussian
    kernel neighborhood (sigma=12 has 97 taps for float32) before mapping.
    """
    if tone:
        mask = cv2.dilate(mask.astype('u1'), np.ones((97, 97), 'u1')).astype(bool)
    y, x = np.nonzero(mask)
    fy = (y+.5)*grid[1]/288-.5
    fx = (x+.5)*grid[0]/512-.5
    y0, x0 = np.floor(fy).astype(int), np.floor(fx).astype(int)
    needed = np.zeros((grid[1], grid[0]), bool)
    for dy in (0, 1):
        for dx in (0, 1):
            needed[(y0+dy).clip(0, grid[1]-1), (x0+dx).clip(0, grid[0]-1)] = True
    return needed


def fill_motion(motion, masks, *, mode='affine', tone=False):
    result = motion.copy()
    yy, xx = np.mgrid[:motion.shape[1], :motion.shape[2]]
    basis = np.stack([xx, yy, np.ones_like(xx)], axis=-1).astype(np.float64)
    required = []
    for j, (grid, mask) in enumerate(zip(motion, masks, strict=True)):
        need = dependencies(mask, (motion.shape[2], motion.shape[1]), tone)
        required.append(need)
        if mode == 'zero' or need.sum() < 3:
            value = np.zeros_like(grid)
        else:
            fit = np.linalg.lstsq(basis[need], grid[need].astype(np.float64), rcond=None)[0]
            value = np.rint(basis@fit).clip(-4096, 4096).astype('<i2')
        result[j][~need] = value[~need]
    return result, np.stack(required)


def run(workspace, output):
    root = workspace/'02_实验数据/优化_GLC后处理'
    digest = file_sha(root/'source_dense_flow_three_clips_20261109_01/video1/base.glcsnd')
    h, a = unpack((root/'fullframe_reference_refined_20261110_01/video1/g24q2_E/fullframe.glcfull').read_bytes(), digest)
    motion = np.concatenate([a['new_motion'], a['old_motion']])
    safe = np.concatenate([np.load(root/'fullframe_factorial_20261110_01/gates'/f'{k}_PG.npy') for k in ('new', 'old')])
    rows = []
    for step, span in ((8, 1), (4, 3), (4, 5)):
        support = reduce_support(safe, step, span)
        masks = np.repeat(np.repeat(expand_support(support, span), step, 1), step, 2).astype(bool)
        for mode in ('zero', 'affine'):
            for tone in (False, True):
                filled, needed = fill_motion(motion, masks, mode=mode, tone=tone)
                payload, header = pack(filled, support, digest, step=step, span=span, fusion='F' if tone else 'E')
                row = dict(step=step, span=span, fill=mode, fusion='F' if tone else 'E', bytes=len(payload),
                           motion_bytes=header['sections']['motion']['bytes'], needed_fraction=float(needed.mean()))
                rows.append(row)
                print(row, flush=True)
    save_json(output, dict(screening_only=True, profiles=rows, code_sha256=file_sha(__file__)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cv2.setNumThreads(1)
    run(args.workspace.resolve(), args.output.resolve())
