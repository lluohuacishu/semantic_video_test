"""Expand the accepted fixed-structure pilot to the whole visible upper coral.

The source polygon and source-colour fish exclusions remain offline assistance.
Keep this separate from the initial pilot so its masks and results stay frozen.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from cache_motion_inputs import FPS, FRAME_SHAPE, REPO
from compare_deflicker import save_csv
from coral_reference_sidecar import file_sha, save_json
from run_p0 import preview, roi
from run_video1_persistent_structure import START, STOP, ANCHOR, VIEW_BOX, load_inputs, score
from temporal_filter import correspondence


# Source-105 silhouette with a narrow water/shadow margin. No result-dependent fitting.
OUTLINE = ((345, 0), (511, 0), (511, 57), (490, 70), (456, 85),
           (420, 98), (380, 110), (340, 123), (310, 132), (292, 131),
           (277, 123), (276, 107), (285, 89), (297, 71), (314, 47), (330, 23))


def orange_fish(rgb):
    """Conservative source-only exclusions for the two small orange passing fish."""
    r, g, b = rgb.astype(np.float32).transpose(2, 0, 1)
    selected = (r > 1.22*g) & (r-b > 25) & (r > 55)
    selected[110:] = False
    selected[:, :380] = False
    count, labels, stats, _ = cv2.connectedComponentsWithStats(selected.astype(np.uint8))
    selected[:] = False
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= 3:
            selected |= labels == label
    return cv2.dilate(selected.astype(np.uint8), np.ones((7, 7), np.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan = dict(start=START, stop_exclusive=STOP, anchor=ANCHOR, fps=FPS,
        offline_source_assisted=True, outline=OUTLINE, feather_pixels=2., erosion_pixels=1,
        source_cycle_limit=1.5, source_low_rgb_difference_limit=16.,
        orange_fish_exclusion='source R>1.22G, R-B>25, R>55; x>=380,y<110; >=3px components; dilate7',
        change_from_pilot='Expand to full coral and place feather outside its textured face; retain direct full RGB reference.',
        feedback='User saw a clear reduction in C, but only one central area was clear and the outside looked blurred.',
        limitations=['Existing source RAFT and source masks are untransmitted.',
            'Unknown/newly revealed and occluded pixels fall back to raw GLC.',
            'This pilot does not solve long-term reference refresh or entry/exit.'])
    save_json(out/'plan.json', plan)
    source, raw, _, back, forward, inputs = load_inputs(args.workspace.resolve())
    save_json(out/'inputs.json', inputs)
    anchor_mask = np.zeros(FRAME_SHAPE[:2], np.uint8)
    cv2.fillPoly(anchor_mask, [np.asarray(OUTLINE, np.int32)], 1)
    anchor_fish = orange_fish(source[ANCHOR])
    reference_support = anchor_mask*(1-anchor_fish)
    c, r = raw[START:STOP].copy(), raw[START:STOP].copy()
    alpha = np.zeros(raw[START:STOP].shape[:3], np.float32)
    tracked, locks = np.zeros_like(alpha, np.uint8), np.zeros_like(alpha, np.uint8)
    old_alpha = np.load(args.pilot/'alpha.npy', allow_pickle=False)
    old_track = np.load(args.pilot/'tracked.npy', allow_pickle=False)
    old_c = np.load(args.pilot/'C_rgb.npy', allow_pickle=False)
    anchor_low = cv2.GaussianBlur(source[ANCHOR].astype(np.float32), (0, 0), 3)
    coverage = []
    for j, i in enumerate(range(START, STOP)):
        mx, my, inside, cycle = correspondence(back[i], forward[i])
        tracked[j] = cv2.remap(anchor_mask, mx, my, cv2.INTER_NEAREST)*inside
        support = cv2.remap(reference_support, mx, my, cv2.INTER_NEAREST).astype(bool)
        locks[j] = orange_fish(source[i])
        low = cv2.GaussianBlur(source[i].astype(np.float32), (0, 0), 3)
        aligned = cv2.remap(anchor_low, mx, my, cv2.INTER_LINEAR)
        diff = np.max(np.abs(aligned-low), axis=2)
        safe = support & inside & (cycle < 1.5) & (diff < 16) & ~locks[j].astype(bool)
        safe = cv2.erode(safe.astype(np.uint8), np.ones((3, 3), np.uint8))
        alpha[j] = np.minimum(cv2.distanceTransform(safe, cv2.DIST_L2, 5)/2., 1.)
        a = alpha[j, ..., None]
        for frames, anchor in ((c, raw[ANCHOR]), (r, source[ANCHOR])):
            ref = cv2.remap(anchor.astype(np.float32), mx, my, cv2.INTER_CUBIC)
            frames[j] = np.rint(np.clip(raw[i]*(1-a)+ref*a, 0, 255)).astype(np.uint8)
        area = tracked[j].astype(bool)
        coverage.append(dict(frame=i, target_pixels=int(area.sum()),
            old_active=int((old_alpha[j] > 0).sum()), new_active=int((alpha[j] > 0).sum()),
            old_full_weight=int((old_alpha[j] == 1).sum()), new_full_weight=int((alpha[j] == 1).sum()),
            old_fractional=int(((old_alpha[j] > 0)&(old_alpha[j] < 1)).sum()),
            new_fractional=int(((alpha[j] > 0)&(alpha[j] < 1)).sum()),
            fish_locked=int((locks[j].astype(bool)&area).sum()),
            new_mean_weight=float(alpha[j][area].mean())))
    sequences = dict(source=source[START:STOP], raw=raw[START:STOP], C=old_c, C2=c, R2=r)
    primary, temporal, spatial = score(sequences, tracked)
    old_scope, _, _ = score(sequences, old_track)
    outer = tracked*(1-old_track)
    outer_scope, outer_rows, _ = score(sequences, outer)
    save_csv(out/'temporal.csv', temporal)
    save_csv(out/'spatial.csv', spatial)
    save_csv(out/'outer_temporal.csv', outer_rows)
    save_csv(out/'coverage.csv', coverage)
    for name, value in dict(C2_rgb=c, R2_rgb=r, alpha=alpha, tracked=tracked, fish_locks=locks).items():
        np.save(out/f'{name}.npy', value, allow_pickle=False)
    crop = roi(VIEW_BOX, FRAME_SHAPE)
    labels = dict(source='Source RGB', raw='GLC raw', C='C / small patch',
                  C2='C2 / full coral reference', R2='R2 / full SOURCE reference')
    previews = {}
    for filename, names, view in (
        ('expanded_coral_4way.mp4', ('raw', 'C', 'C2', 'R2'), crop),
        ('expanded_coral_pair.mp4', ('raw', 'C2'), crop),
        ('expanded_full_pair.mp4', ('raw', 'C2'), None)):
        previews[filename] = preview(args.ffmpeg, out/filename, [sequences[n] for n in names],
            FPS, START, [labels[n] for n in names], view)
    samples = (105, 110, 115, 120, 127)
    canvas = Image.new('RGB', (524*len(samples), 4*288), (24, 35, 48))
    draw = ImageDraw.Draw(canvas)
    for col, i in enumerate(samples):
        for row, name in enumerate(('raw', 'C', 'C2', 'R2')):
            x, y = 524*col, 288*row
            draw.text((x+6, y+4), f'{labels[name]} | f{i}', fill='white')
            canvas.paste(Image.fromarray(sequences[name][i-START][crop]).resize((524, 264)), (x, y+24))
    canvas.save(out/'expanded_comparison.png')
    canvas = Image.new('RGB', (512*len(samples), 312), (24, 35, 48))
    draw = ImageDraw.Draw(canvas)
    for col, i in enumerate(samples):
        j = i-START
        a = alpha[j, ..., None]*.45
        overlay = source[i]*(1-a)+np.array([0, 255, 170])*a
        overlay[locks[j].astype(bool)] = [255, 70, 50]
        canvas.paste(Image.fromarray(np.rint(overlay).astype(np.uint8)), (512*col, 24))
        draw.text((512*col+6, 4), f'f{i} | green active / red fish lock', fill='white')
    canvas.save(out/'expanded_support.png')
    checks = dict(zero_alpha_bit_exact=bool(np.array_equal(c[alpha == 0], raw[START:STOP][alpha == 0])),
        source_version_zero_alpha_bit_exact=bool(np.array_equal(r[alpha == 0], raw[START:STOP][alpha == 0])),
        fish_lock_bit_exact=bool(np.array_equal(c[locks.astype(bool)], raw[START:STOP][locks.astype(bool)])),
        anchor105_bit_exact=bool(np.array_equal(c[0], raw[105])),
        unchanged_inputs=all(file_sha(Path(p)) == h for p, h in inputs['files'].items()))
    if not all(checks.values()):
        raise ValueError('Expanded-render preservation failed')
    save_json(out/'summary.json', dict(complete=True, plan=plan, scores=primary,
        original_pilot_tracked_scores=old_scope['tracked'], newly_covered_scores=outer_scope['tracked'],
        coverage_means={key: float(np.mean([r[key] for r in coverage])) for key in coverage[0] if key != 'frame'},
        previews=previews, verification=checks, code_sha256=file_sha(__file__),
        helper_code_sha256=file_sha(Path(__file__).with_name('run_video1_persistent_structure.py')),
        output_sha256={p.name: file_sha(p) for p in out.glob('*.npy')}))
    print('PASS', flush=True)
    for name, values in outer_scope['tracked'].items():
        print(name, 'outer coarse', values['coarse_temporal_mae'], 'outer fine NCC', values['fine_ncc'], flush=True)


if __name__ == '__main__':
    main()
