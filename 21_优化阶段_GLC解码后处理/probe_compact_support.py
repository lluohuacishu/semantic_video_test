"""Byte/coverage screening only: conservative supports and lossless motion coding."""

import argparse
import lzma
from pathlib import Path

import cv2
import numpy as np

from coral_reference_sidecar import file_sha, save_json
from fullframe_structure_stream import unpack


def coded(value, axes, layout):
    residual = value.copy()
    for axis in axes:
        residual = np.diff(residual, axis=axis,
                           prepend=np.zeros_like(np.take(residual, [0], axis=axis)))
    if layout == 'i16':
        data = residual.astype('<i2').tobytes()
    elif layout == 'bytesplit':
        data = residual.astype('<i2').view('u1').reshape(-1, 2).T.tobytes()
    elif layout == 'zigzag':
        r = residual.astype(np.int32)
        zz = ((r << 1) ^ (r >> 31)).astype('<u2')
        data = zz.view('u1').reshape(-1, 2).T.tobytes()
    else:
        raise ValueError(layout)
    return lzma.compress(data)


def mask_sizes(cells):
    choices = []
    for axes in ((0, 1, 2), (1, 2, 0), (2, 1, 0)):
        a = cells.transpose(axes)
        for delta in (False, True):
            v = a.copy()
            if delta:
                v[..., 1:] ^= a[..., :-1]
            payload = lzma.compress(np.packbits(v, axis=-1).tobytes())
            choices.append(dict(bytes=len(payload), transpose=axes, xor_last=delta))
    return min(choices, key=lambda r: r['bytes']), choices


def grouped_masks(safe, step, span, keys):
    """A mask is sent once per group and is safe on every covered frame.

    New-anchor groups and each six-frame old-anchor transition are separate.
    This uses at most span-1 frames of sender lookahead, with no delayed exits.
    """
    cells = safe.reshape(180, 288//step, step, 512//step, step).all(axis=(2, 4))
    intervals = list(zip(keys, list(keys[1:])+[150]))+[(150+i, 156+i) for i in range(0, 30, 6)]
    groups = [(start, min(start+span, stop)) for begin, stop in intervals
              for start in range(begin, stop, span)]
    compact = np.stack([cells[a:b].all(axis=0) for a, b in groups])
    expanded = np.empty_like(cells)
    for mask, (a, b) in zip(compact, groups):
        expanded[a:b] = mask
    return compact, np.repeat(np.repeat(expanded, step, axis=1), step, axis=2), groups


def disks(safe, step, levels):
    """Union of disks strictly inside the safe set; every rendered pixel checked."""
    codes = np.zeros((len(safe), 288//step, 512//step), np.uint8)
    decoded = np.zeros_like(safe)
    yy, xx = np.mgrid[step//2:288:step, step//2:512:step]
    for i, mask in enumerate(safe):
        # Border padding makes distances conservative even for all-safe images.
        d = cv2.distanceTransform(np.pad(mask, 1).astype('u1'), cv2.DIST_L2,
                                  cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        radii = np.minimum(np.floor(d[yy, xx]-1e-4), levels[-1])
        code = np.searchsorted(levels, radii, side='right').clip(0, len(levels))
        codes[i] = code
        canvas = np.zeros(mask.shape, 'u1')
        for y, x, c in zip(yy.ravel(), xx.ravel(), code.ravel()):
            if c:
                cv2.circle(canvas, (int(x), int(y)), int(levels[c-1]), 1, -1)
        decoded[i] = canvas.astype(bool)
        assert not np.any(decoded[i] & ~mask)
    choices = []
    for axes in ((0, 1, 2), (1, 2, 0)):
        a = codes.transpose(axes)
        choices.append(dict(bytes=len(lzma.compress(a.tobytes())), transpose=axes))
    return decoded, min(choices, key=lambda r: r['bytes'])


def run(workspace, output):
    output.mkdir(parents=True, exist_ok=False)
    root = workspace/'02_实验数据/优化_GLC后处理'
    prior = root/'fullframe_reference_refined_20261110_01/video1/g24q2_E/fullframe.glcfull'
    digest = file_sha(root/'source_dense_flow_three_clips_20261109_01/video1/base.glcsnd')
    h, arrays = unpack(prior.read_bytes(), digest)
    new, old = arrays['new_motion'], arrays['old_motion']
    events = [k+j for k in h['keys'][1:] for j in range(6)]
    motion = {}
    for name, values in (('new', new), ('old', old), ('old_minus_new', old-new[events])):
        choices = []
        for axes in ((), (0,), (0, 0), (0, 2), (1, 2), (0, 1, 2), (0, 0, 2)):
            for layout in ('i16', 'bytesplit', 'zigzag'):
                p = coded(values, axes, layout)
                choices.append(dict(bytes=len(p), axes=axes, layout=layout))
        motion[name] = dict(best=min(choices, key=lambda r: r['bytes']), choices=choices)
        print('MOTION', name, motion[name]['best'], flush=True)
    safe = np.concatenate([np.load(root/'fullframe_factorial_20261110_01/gates'/f'{k}_PG.npy')
                           for k in ('new', 'old')])
    masks = {}
    for step in (8, 4, 2):
        cells = safe.reshape(180, 288//step, step, 512//step, step).all(axis=(2, 4))
        best, choices = mask_sizes(cells)
        decoded = np.repeat(np.repeat(cells, step, axis=1), step, axis=2)
        masks['block'+str(step)] = dict(best=best, choices=choices, coverage=float(decoded.mean()),
                                       coral_coverage=float(decoded[:150, 10:105, 342:502].mean()))
        print('MASK', 'block'+str(step), masks['block'+str(step)], flush=True)
    disk_results = {}
    for step, levels in ((16, (3, 7, 11, 15, 23, 31, 47)), (8, (1, 3, 5, 7, 11, 15, 23))):
        decoded, spec = disks(safe, step, levels)
        disk_results[str(step)] = dict(best=spec, levels=levels, coverage=float(decoded.mean()),
                                      coral_coverage=float(decoded[:150, 10:105, 342:502].mean()))
        print('DISKS', step, disk_results[str(step)], flush=True)
    save_json(output/'probe.json', dict(complete=True, motion=motion, masks=masks, disks=disk_results,
              pixel_coverage=float(safe.mean()), packet_bytes=None, screening_only=True,
              input_packet_sha256=file_sha(prior), code_sha256=file_sha(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    run(args.workspace.resolve(), args.output.resolve())
