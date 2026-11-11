"""Offline, fixed-anchor RGB controls on video1 ordinary P frames.

This is a source-assisted diagnostic, not a transmitted auxiliary stream.
Rendering uses existing direct source RAFT. Scoring uses adjacent source
Farneback, shared masks, and lossless uint8 RGB. No codec state is modified.
"""

import argparse
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from audit_deflicker_texture import normalized, source_match
from cache_motion_inputs import FPS, FRAME_SHAPE, REPO, base_frames
from compare_boundary_codec import frame_kind
from compare_deflicker import save_csv
from compare_source_dense_flow import load_flow
from coral_reference_sidecar import file_sha, gray, load_json, save_json
from experiment_common import array_sha
from probe_reef import bands
from run_dense_transfer import source_input
from run_gate_ablation import EDIT_WINDOWS, editable, render, source_gate
from run_p0 import preview, roi
from run_structure_memory import CLIP, LONG_ANCHORS
from temporal_filter import correspondence, warp


START, STOP, ANCHOR = 105, 128, 105
FIXED_BOX = (342, 10, 502, 105)
VIEW_BOX = (250, 0, 512, 132)
# Chosen only from source frame 105, before generating any candidate result.
# The polygon covers the central/left coral face, excluding the fish at right.
POLYGON = ((354, 4), (433, 4), (441, 53), (426, 73), (380, 89),
           (335, 106), (301, 116), (292, 108), (305, 83), (325, 54), (345, 25))
LABELS = dict(source='Source RGB', raw='GLC raw', S='S / previous structure',
              SL='S+L / legacy blend', C='C / fixed GLC structure',
              R='R / fixed SOURCE structure', B0='B0 / spatial blur')


def plan():
    return dict(case='video1', start=START, stop_exclusive=STOP, anchor=ANCHOR,
        fps=FPS, frames=STOP-START, source_assisted_offline=True,
        base_codec=dict(q=0, gop=64, reset=32, backend='FP32 eager'),
        source_anchor_polygon=POLYGON, fixed_evaluation_box=FIXED_BOX,
        view_box=VIEW_BOX, cycle_limit_px=1.5, source_low_rgb_limit=16.,
        erosion_px=1, inward_feather_px=4., interpolation='one direct cubic warp',
        rendering='full RGB anchor replacement; unit weight in reliable interior',
        illumination_correction=False, recurrent_resampling=False,
        texture_amplitude_gate=False, periodic_refresh=False,
        gate='source sigma3 RGB difference and bidirectional source RAFT',
        evaluation='independent adjacent source Farneback; same masks for every method',
        scopes=['original fixed box', 'tracked polygon, independent of rendering reliability'],
        additional_control='S structure band combined with L long anchors, original blend and windows',
        blur_control='one spatial sigma matching C tracked-region fine amplitude',
        context=[102, 133],
        limitations=['Source motion, source visibility, and manual source polygon are untransmitted.',
                    'R also reads one source RGB anchor; it is not a deployable codec.',
                    'The isolated pilot does not solve entry, exit, refresh, or newly exposed regions.',
                    'Legacy S/SL have different editing support; C versus R is the controlled anchor comparison.'])


def image_bands(rgb):
    b = bands(rgb)
    low12 = cv2.GaussianBlur(b[..., 3], (0, 0), 12)
    return np.concatenate((b, (b[..., 0]-low12)[..., None]), axis=2)


def load_inputs(workspace):
    data = workspace/'02_实验数据/优化_GLC后处理'
    base = data/'source_dense_flow_three_clips_20261109_01/video1'
    prior = data/'structure_memory_video1_20261109_01'
    source = source_input(base, CLIP)
    raw, base_sha, base_size = base_frames(base, CLIP)
    s = np.load(prior/'structure/rgb.npy', allow_pickle=False)
    record = load_json(prior/'structure/result.json')
    if array_sha(s) != record['rgb_sha256'] or record['base_sha256'] != base_sha:
        raise ValueError('Existing S result changed')
    back, forward = load_flow(prior/'long_flow', source, CLIP, anchors=LONG_ANCHORS)
    if not all(frame_kind(i, 64, 32) == 'ordinary_p' for i in range(START, STOP)):
        raise ValueError('Primary pilot must contain only ordinary P frames')
    files = [base/'source.npy', base/'base.glcsnd', base/'base_receive/decoded_rgb.npy',
             prior/'structure/rgb.npy', prior/'long_flow/to_anchor.npy',
             prior/'long_flow/from_anchor.npy']
    inputs = dict(files={str(p): file_sha(p) for p in files}, base_sha256=base_sha,
        base_bytes=base_size, base_kbps_10s=base_size*8/10/1000,
        source_rgb_sha256=array_sha(source), raw_rgb_sha256=array_sha(raw),
        legacy_s_rgb_sha256=array_sha(s), long_flow_receipt=load_json(prior/'long_flow/flow.json'))
    return source, raw, s, back, forward, inputs


def fixed_anchor(source, raw, back, forward):
    count = STOP-START
    anchor_mask = np.zeros(FRAME_SHAPE[:2], np.uint8)
    cv2.fillPoly(anchor_mask, [np.asarray(POLYGON, np.int32)], 1)
    tracked = np.zeros((count, *FRAME_SHAPE[:2]), np.uint8)
    alpha = np.zeros_like(tracked, np.float32)
    c, r = raw[START:STOP].copy(), raw[START:STOP].copy()
    anchor_low = cv2.GaussianBlur(source[ANCHOR].astype(np.float32), (0, 0), 3)
    rows = []
    for j, i in enumerate(range(START, STOP)):
        mx, my, inside, cycle = correspondence(back[i], forward[i])
        support = cv2.remap(anchor_mask, mx, my, cv2.INTER_NEAREST).astype(bool) & inside
        tracked[j] = support
        low = cv2.GaussianBlur(source[i].astype(np.float32), (0, 0), 3)
        predicted_low = cv2.remap(anchor_low, mx, my, cv2.INTER_LINEAR)
        difference = np.max(np.abs(predicted_low-low), axis=2)
        reliable = support & (cycle < 1.5) & (difference < 16)
        reliable = cv2.erode(reliable.astype(np.uint8), np.ones((3, 3), np.uint8))
        distance = cv2.distanceTransform(reliable, cv2.DIST_L2, 5)
        alpha[j] = np.minimum(distance / 4., 1.)
        for frames, ref in ((c, raw[ANCHOR]), (r, source[ANCHOR])):
            projected = cv2.remap(ref.astype(np.float32), mx, my, cv2.INTER_CUBIC)
            a = alpha[j, ..., None]
            frames[j] = np.rint(np.clip(raw[i]*(1-a) + projected*a, 0, 255)).astype(np.uint8)
        count_support = int(support.sum())
        rows.append(dict(frame=i, tracked_pixels=count_support,
            changed_support_pixels=int((alpha[j] > 0).sum()),
            full_replacement_pixels=int((alpha[j] == 1).sum()),
            mean_alpha_on_tracked=float(alpha[j][support].mean()),
            cycle_rejected_fraction=float((cycle[support] >= 1.5).mean()),
            source_photo_rejected_fraction=float((difference[support] >= 16).mean())))
    return c, r, alpha, tracked, anchor_mask, rows


def fine_amplitude(frames, masks):
    numerator, denominator = 0., 0
    for frame, mask in zip(frames, masks, strict=True):
        values = np.abs(bands(frame)[..., 2])[mask.astype(bool)]
        numerator += float(values.sum())
        denominator += len(values)
    return numerator / denominator


def spatial_control(raw, target, alpha, tracked):
    def make(sigma):
        blurred = raw if sigma == 0 else np.stack([cv2.GaussianBlur(f, (0, 0), sigma) for f in raw])
        a = alpha[..., None]
        return np.rint(raw*(1-a) + blurred*a).astype(np.uint8)
    desired = fine_amplitude(target, tracked)
    low, high = 0., 2.
    result = make(0.)
    best = (abs(fine_amplitude(result, tracked)-desired), 0., result)
    for _ in range(12):
        sigma = (low+high)/2
        candidate = make(sigma)
        energy = fine_amplitude(candidate, tracked)
        if abs(energy-desired) < best[0]:
            best = (abs(energy-desired), sigma, candidate)
        if energy > desired:
            low = sigma
        else:
            high = sigma
    return best[2], dict(sigma=best[1], target_fine_amplitude=desired,
        blur_fine_amplitude=fine_amplitude(best[2], tracked), absolute_mismatch=best[0])


def score(sequences, tracked, start=START):
    fixed = np.zeros(FRAME_SHAPE[:2], bool)
    fixed[roi(FIXED_BOX, FRAME_SHAPE)] = True
    rows, spatial = [], []
    previous = None
    for j in range(len(sequences['source'])):
        current = {k: image_bands(v[j]) for k, v in sequences.items()}
        scope_masks = dict(fixed=fixed, tracked=tracked[j].astype(bool))
        for scope, area in scope_masks.items():
            for name, video in sequences.items():
                error = video[j].astype(np.float32)-sequences['source'][j].astype(np.float32)
                spatial.append(dict(frame=start+j, scope=scope, method=name, pixels=int(area.sum()),
                    rgb_sse=float(np.square(error[area]).sum()),
                    structure_error_sum=float(np.abs(current[name][..., 4]-current['source'][..., 4])[area].sum()),
                    fine_amplitude_sum=float(np.abs(current[name][..., 2])[area].sum())))
        if j:
            mx, my, common = source_match(sequences['source'][j-1], sequences['source'][j],
                previous['source'][..., :4], current['source'][..., :4], np.s_[:, :])
            aligned = {k: warp(v, mx, my) for k, v in previous.items()}
            delta = current['source']-aligned['source']
            for scope, area in scope_masks.items():
                mask = common & area
                if scope == 'tracked':
                    mask &= warp(tracked[j-1].astype(np.float32), mx, my) >= .999
                pixels = int(mask.sum())
                if pixels < 64:
                    raise ValueError(f'Insufficient common {scope} evaluation at frame {start+j}')
                for name in sequences:
                    a, b = aligned[name][mask], current[name][mask]
                    error = np.abs(b-a-delta[mask])
                    correlations = (normalized(a[:, [2, 4]])*normalized(b[:, [2, 4]])).mean(axis=0)
                    rows.append(dict(frame=start+j, scope=scope, method=name, valid_pixels=pixels,
                        coarse_sum=float(error[:, 0].sum()), fine_sum=float(error[:, 2].sum()),
                        full_sum=float(error[:, 3].sum()), structure_sum=float(error[:, 4].sum()),
                        fine_ncc=float(correlations[0]), structure_ncc=float(correlations[1])))
        previous = current
    summary = {}
    for scope in ('fixed', 'tracked'):
        summary[scope] = {}
        for method in sequences:
            pairs = [r for r in rows if r['scope'] == scope and r['method'] == method]
            images = [r for r in spatial if r['scope'] == scope and r['method'] == method]
            n, nimage = sum(r['valid_pixels'] for r in pairs), sum(r['pixels'] for r in images)
            mse = sum(r['rgb_sse'] for r in images)/(3*nimage)
            stats = dict(pairs=len(pairs), valid_pixels=n, spatial_pixels=nimage,
                psnr_db=None if mse == 0 else float(10*np.log10(255**2/mse)),
                source_structure_mae=sum(r['structure_error_sum'] for r in images)/nimage,
                fine_amplitude=sum(r['fine_amplitude_sum'] for r in images)/nimage)
            for band in ('coarse', 'fine', 'full', 'structure'):
                stats[f'{band}_temporal_mae'] = sum(r[f'{band}_sum'] for r in pairs)/n
            for band in ('fine', 'structure'):
                stats[f'{band}_ncc'] = sum(r[f'{band}_ncc']*r['valid_pixels'] for r in pairs)/n
            summary[scope][method] = stats
    return summary, rows, spatial


def diagnostic_images(output, sequences, tracked, alpha, back, forward):
    sample_indices = (105, 110, 115, 120, 127)
    crop = roi(VIEW_BOX, FRAME_SHAPE)
    height, width = 264, 524
    names = ('source', 'raw', 'S', 'SL', 'C', 'R')
    canvas = Image.new('RGB', (width*len(sample_indices), (height+24)*len(names)), (24, 35, 48))
    draw = ImageDraw.Draw(canvas)
    for col, index in enumerate(sample_indices):
        j = index-START
        for row, name in enumerate(names):
            x, y = col*width, row*(height+24)
            draw.text((x+7, y+5), f'{LABELS[name]} | f{index}', fill='white')
            canvas.paste(Image.fromarray(sequences[name][j][crop]).resize((width, height)), (x, y+24))
    canvas.save(output/'unaligned_comparison.png')
    # Canonical-coordinate diagnostic: all frames are projected onto anchor 105.
    yy, xx = np.mgrid[:FRAME_SHAPE[0], :FRAME_SHAPE[1]].astype(np.float32)
    for col, index in enumerate(sample_indices):
        mx, my = xx+forward[index, ..., 0], yy+forward[index, ..., 1]
        for row, name in enumerate(names):
            x, y = col*width, row*(height+24)
            aligned = cv2.remap(sequences[name][index-START], mx, my, cv2.INTER_LINEAR)
            canvas.paste(Image.fromarray(aligned[crop]).resize((width, height)), (x, y+24))
    canvas.save(output/'aligned_to_anchor105.png')
    support = Image.new('RGB', (512*len(sample_indices), 288+24), (24, 35, 48))
    draw = ImageDraw.Draw(support)
    for col, index in enumerate(sample_indices):
        j = index-START
        frame = sequences['source'][j].astype(np.float32)
        a = alpha[j, ..., None]*.45
        frame = np.rint(frame*(1-a)+np.array([0, 255, 170])*a).astype(np.uint8)
        support.paste(Image.fromarray(frame), (512*col, 24))
        draw.text((512*col+7, 5), f'f{index} | green: actual replacement', fill='white')
    support.save(output/'tracking_support.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/'plan.json', plan())
    tick = time.perf_counter()
    source, raw, s, back, forward, inputs = load_inputs(args.workspace.resolve())
    save_json(output/'inputs.json', inputs)
    print('Verified source, original base, S, and existing long-anchor flows', flush=True)
    c, r, alpha, tracked, anchor_mask, coverage = fixed_anchor(source, raw, back, forward)
    gate = source_gate(source, back, forward, anchors=LONG_ANCHORS)
    sl, sl_alpha = render(raw, back, gate, editable('video1'), anchors=LONG_ANCHORS,
                          structure_windows=EDIT_WINDOWS['video1'])
    b0, blur_record = spatial_control(raw[START:STOP], c, alpha, tracked)
    sequences = dict(source=source[START:STOP], raw=raw[START:STOP], S=s[START:STOP],
                     SL=sl[START:STOP], C=c, R=r, B0=b0)
    checks = dict(ordinary_p_frames=STOP-START, original_glc_anchor_exact=bool(np.array_equal(c[0], raw[START])),
                  unchanged_where_alpha_zero={}, shared_alpha_for_C_and_R=True)
    for name in ('C', 'R', 'B0'):
        checks['unchanged_where_alpha_zero'][name] = bool(np.array_equal(
            sequences[name][alpha == 0], raw[START:STOP][alpha == 0]))
    if not checks['original_glc_anchor_exact'] or not all(checks['unchanged_where_alpha_zero'].values()):
        raise ValueError('Reference or zero-mask preservation failed')
    for name, frames in sequences.items():
        if name not in ('source', 'raw', 'S'):
            np.save(output/f'{name}_rgb.npy', frames, allow_pickle=False)
    for name, value in dict(alpha=alpha, tracked=tracked, SL_alpha=sl_alpha[START:STOP]).items():
        np.save(output/f'{name}.npy', value, allow_pickle=False)
    Image.fromarray(anchor_mask*255).save(output/'anchor_mask.png')
    save_csv(output/'coverage.csv', coverage)
    scores, temporal, spatial = score(sequences, tracked)
    save_csv(output/'temporal.csv', temporal)
    save_csv(output/'spatial.csv', spatial)
    diagnostic_images(output, sequences, tracked, alpha, back, forward)
    print('Lossless metrics complete; writing original-speed previews', flush=True)
    previews = {}
    for filename, names, crop in (
        ('pilot_coral_4way.mp4', ('raw', 'S', 'C', 'R'), roi(VIEW_BOX, FRAME_SHAPE)),
        ('pilot_full_4way.mp4', ('raw', 'S', 'C', 'R'), None),
        ('pilot_blur_control.mp4', ('source', 'raw', 'B0', 'C'), roi(VIEW_BOX, FRAME_SHAPE)),
        ('pilot_source_pair.mp4', ('source', 'R'), roi(VIEW_BOX, FRAME_SHAPE)),
        ('pilot_legacy_SL.mp4', ('raw', 'S', 'SL', 'C'), roi(VIEW_BOX, FRAME_SHAPE))):
        previews[filename] = preview(args.ffmpeg, output/filename, [sequences[k] for k in names],
            FPS, START, [LABELS[k] for k in names], crop)
    left, right = 102, 133
    contexts = [raw[left:right].copy(), s[left:right].copy(), raw[left:right].copy(), raw[left:right].copy()]
    contexts[2][START-left:STOP-left] = c
    contexts[3][START-left:STOP-left] = r
    previews['context_with_entry_exit.mp4'] = preview(args.ffmpeg, output/'context_with_entry_exit.mp4',
        contexts, FPS, left, [LABELS[k] for k in ('raw', 'S', 'C', 'R')], roi(VIEW_BOX, FRAME_SHAPE))
    context_masks = np.zeros((right-left, *FRAME_SHAPE[:2]), np.uint8)
    context_masks[START-left:STOP-left] = tracked
    # The fixed ROI is meaningful at the entry/exit even when the pilot mask is absent.
    fixed_mask = np.zeros(FRAME_SHAPE[:2], np.uint8)
    fixed_mask[roi(FIXED_BOX, FRAME_SHAPE)] = 1
    context_scores, context_rows, _ = score(dict(source=source[left:right], raw=contexts[0],
        S=contexts[1], C=contexts[2], R=contexts[3]), np.broadcast_to(fixed_mask, context_masks.shape), start=left)
    save_csv(output/'context_temporal.csv', context_rows)
    checks['input_hashes_unchanged'] = all(file_sha(Path(p)) == h for p, h in inputs['files'].items())
    if not checks['input_hashes_unchanged']:
        raise ValueError('An input file changed')
    hashes = {p.name: file_sha(p) for p in output.glob('*.npy')}
    summary = dict(complete=True, plan=plan(), code_sha256=file_sha(__file__),
        input_receipt_sha256=file_sha(output/'inputs.json'), scores=scores, blur=blur_record,
        coverage=dict(mean_pixels=float(np.mean([v['tracked_pixels'] for v in coverage])),
            mean_alpha_on_tracked=float(np.mean([v['mean_alpha_on_tracked'] for v in coverage])),
            mean_full_replacement_pixels=float(np.mean([v['full_replacement_pixels'] for v in coverage]))),
        previews=previews, outputs_sha256=hashes, verification=checks,
        context_fixed_scores=context_scores['fixed'],
        entry_exit_rows=[v for v in context_rows if v['scope'] == 'fixed' and v['frame'] in (105, 128)],
        cpu_wall_seconds=time.perf_counter()-tick)
    save_json(output/'summary.json', summary)
    print('PASS', flush=True)
    for name, values in scores['tracked'].items():
        print(name, 'coarse', round(values['coarse_temporal_mae'], 4),
              'fine_NCC', round(values['fine_ncc'], 4), 'PSNR', values['psnr_db'], flush=True)


if __name__ == '__main__':
    main()
