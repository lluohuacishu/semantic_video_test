"""Replace shared affine motion with source-only dense flow in the fixed GLC cache."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from audit_deflicker_texture import fine_energy, matched_blur, measure
from cache_motion_inputs import FPS, FRAME_SHAPE, HEIGHT, REPO, WIDTH, base_frames, source_frames
from compare_cache_motion import read_rgb, score_scope
from compare_deflicker import save_csv
from compare_seedvr2 import source_detail
from coral_reference_sidecar import SCALES, detail, file_sha, gray, load_json, save_json
from experiment_common import array_sha
from run_cache_motion import (
    CACHE_FRAMES, PACKET, blend_cached_texture, homogeneous, low_frequency_difference, render, unpack,
)
from run_p0 import preview, roi
from source_raft_flow import CLIP
from temporal_filter import correspondence


VARIANTS = ('raft_glc_gate', 'raft_source_gate')


def load_flow(output, source, clip=CLIP):
    receipt = load_json(output/'flow.json')
    shape = (clip.count, HEIGHT, WIDTH, 2)
    anchors = list(range(clip.start, clip.stop, CACHE_FRAMES))
    if (receipt['input_source_rgb_sha256'] != array_sha(source) or tuple(receipt['flow_shape']) != shape
            or receipt['source_range'] != [clip.start, clip.stop] or receipt['fps'] != FPS
            or receipt['anchors'] != anchors or not receipt['offline_source_oracle']):
        raise ValueError('Expected the source-only RAFT flow for this timeline and cache schedule')
    flows = []
    for filename in ('to_anchor.npy', 'from_anchor.npy'):
        path = output/filename
        flow = np.load(path, mmap_mode='r', allow_pickle=False)
        if (file_sha(path) != receipt['flow_file_sha256'][filename] or flow.shape != shape
                or flow.dtype != np.float32 or flow[::CACHE_FRAMES].any()):
            raise ValueError(f'Invalid flow array: {filename}')
        flows.append(flow)
    return flows


def render_dense(raw, source, backward, forward, affine_records, crop, clip=CLIP):
    videos = {name: raw.copy() for name in VARIANTS}
    masks = {name: np.zeros(raw.shape[:3], np.uint8) for name in VARIANTS}
    support = np.ones(raw.shape[1:3], np.float32)
    geometry = []
    for i, rgb in enumerate(raw):
        coarse = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), 3)
        texture = detail(gray(rgb), 3)
        source_low = cv2.GaussianBlur(source[i].astype(np.float32), (0, 0), 3)
        if i % CACHE_FRAMES == 0:
            anchor = i
            ref_low, ref_texture, ref_source_low = coarse, texture, source_low
            source_anchor_luma = gray(source[i]).astype(np.float32)
            transform = np.eye(3)
            continue

        mx, my, inside, cycle = correspondence(backward[i], forward[i])
        dense_visible = cv2.remap(support, mx, my, cv2.INTER_LINEAR) >= .999
        projected_low = cv2.remap(ref_low, mx, my, cv2.INTER_LINEAR)
        projected_texture = cv2.remap(ref_texture, mx, my, cv2.INTER_CUBIC)
        projected_source = cv2.remap(ref_source_low, mx, my, cv2.INTER_LINEAR)
        source_difference = low_frequency_difference(projected_source, source_low)
        consistent = inside & (cycle < 1.5)
        gates = {
            'raft_glc_gate': (dense_visible, low_frequency_difference(projected_low, coarse)),
            'raft_source_gate': (consistent, source_difference),
        }
        for name, (visible, difference) in gates.items():
            videos[name][i], masks[name][i] = blend_cached_texture(
                rgb, texture, projected_texture, visible, difference, i, anchor)

        # Only compare warps that still refer to the same periodic anchor.
        if transform is None or not affine_records[i-1]['valid']:
            transform = None
            continue
        transform = homogeneous(affine_records[i-1]['affine'] / SCALES) @ transform
        affine = transform[:2]
        affine_visible = cv2.warpAffine(support, affine, (WIDTH, HEIGHT)) >= .999
        shared = (dense_visible & affine_visible)[crop]
        if not shared.any():
            continue
        target = gray(source[i])[crop][shared]
        dense = cv2.remap(source_anchor_luma, mx, my, cv2.INTER_LINEAR)[crop][shared]
        affine_warp = cv2.warpAffine(source_anchor_luma, affine, (WIDTH, HEIGHT))[crop][shared]
        geometry.append(dict(frame=clip.start+i, anchor_frame=clip.start+anchor, gap=i-anchor,
            shared_roi_pixels=int(shared.sum()), dense_abs_sum=float(np.abs(target-dense).sum()),
            affine_abs_sum=float(np.abs(target-affine_warp).sum()),
            dense_cycle_consistent_fraction=float(consistent[crop].mean())))
    return videos, masks, geometry


def geometry_scores(rows):
    if not rows:
        return None
    count = sum(row['shared_roi_pixels'] for row in rows)
    return dict(frames=len(rows), mean_shared_pixels=count/len(rows),
        affine_luma_mae=sum(row['affine_abs_sum'] for row in rows)/count,
        raft_luma_mae=sum(row['dense_abs_sum'] for row in rows)/count,
        raft_cycle_consistent_roi_fraction=float(np.mean([row['dense_cycle_consistent_fraction']
                                                         for row in rows])))


def evaluate(workspace, cache, output, ffmpeg):
    folder = cache/'moving_reef'
    raw, base_sha, _ = base_frames(folder, CLIP)
    source = source_frames(workspace, CLIP)
    records = unpack((folder/PACKET).read_bytes(), CLIP, base_sha)
    if not records['valid'].all():
        raise ValueError('Affine control must use the same periodic anchors as RAFT')
    b2_result = load_json(folder/'autonomous/result.json')['variants']['b2']
    c_result = load_json(folder/'assisted/result.json')['variants']['c']
    b2 = read_rgb(folder/'autonomous', 'b2', b2_result['rgb_sha256'], CLIP)
    c = read_rgb(folder/'assisted', 'c', c_result['rgb_sha256'], CLIP)
    replay, c_mask = render(raw, records)
    if not np.array_equal(c, replay) or array_sha(c_mask) != c_result['alpha_sha256']:
        raise ValueError('Shared renderer changed the prior affine C result')
    del replay
    old = load_json(cache/'summary.json')['clips']['moving_reef']['fixed_scopes']
    compare_clip(CLIP, dict(source=source, raw=raw, b2=b2, c=c), records, c_mask,
                 output, ffmpeg, base_sha, expected_scopes=old)


def compare_clip(clip, sequences, records, c_mask, output, ffmpeg, base_sha, *,
                 region='coral', expected_scopes=None):
    raw, source = sequences['raw'], sequences['source']
    backward, forward = load_flow(output, source, clip)
    background = roi(clip.coral, FRAME_SHAPE)
    videos, masks, geometry = render_dense(raw, source, backward, forward, records, background, clip)
    sequences = {**sequences, **videos}

    result = output/'comparison'
    result.mkdir(exist_ok=False)
    geometry_late = [row for row in geometry if row['frame'] >= clip.stop-2*FPS]
    if geometry:
        save_csv(result/'anchor_alignment.csv', geometry)
    alpha = {'c': c_mask, **masks}
    diagnostics, blur = {}, {}
    for name, mask in alpha.items():
        active = mask[:, background[0], background[1]]
        if not np.array_equal(sequences[name][mask == 0], raw[mask == 0]):
            raise ValueError('RGB changed outside the allowed fusion support')
        diagnostics[name] = dict(
            roi_mean_alpha=float(active.mean()/255), roi_late_mean_alpha=float(active[-2*FPS:].mean()/255),
            roi_coverage=float((active > 0).mean()), roi_late_coverage=float((active[-2*FPS:] > 0).mean()))
        if name != 'c':
            np.save(result/f'{name}.npy', videos[name], allow_pickle=False)
            np.save(result/f'{name}_alpha.npy', mask, allow_pickle=False)
            diagnostics[name].update(rgb_sha256=array_sha(videos[name]), alpha_sha256=array_sha(mask))
            sigma, control = matched_blur(raw, videos[name], background)
            target = fine_energy(videos[name], background)
            energy = fine_energy(control, background)
            sequences[f'b0_{name}'] = control
            blur[name] = dict(sigma=sigma, fine_energy=energy, target_energy=target,
                              relative_error=abs(energy-target)/target)

    scopes = {}
    for name, crop in (('full', np.s_[:, :]), (region, background)):
        temporal = measure(sequences, crop, start=clip.start)
        spatial = source_detail(source, sequences, crop, start=clip.start)
        save_csv(result/f'{name}_temporal.csv', temporal)
        save_csv(result/f'{name}_source_detail.csv', spatial)
        scopes[name] = score_scope(temporal, spatial, sequences, crop)
        scopes[name+'_last2'] = score_scope(temporal[-2*FPS:], spatial[-2*FPS:],
            {key: video[-2*FPS:] for key, video in sequences.items()}, crop)
    if expected_scopes is not None:
        for scope, scores in scopes.items():
            for method in ('source', 'raw', 'b2', 'c'):
                if scores['scores'][method] != expected_scopes[scope]['scores'][method]:
                    raise ValueError(f'Fixed evaluation drifted from the prior control: {scope}/{method}')

    labels = dict(raw='A / GLC', c='C / Source affine',
        raft_glc_gate='RAFT / GLC gate', raft_source_gate='RAFT / Source gate')
    previews = {}
    for name, keys, crop in (
            ('full_comparison', ('raw', 'c', *VARIANTS), None),
            (f'{region}_closeup', ('raw', 'c', *VARIANTS), background)):
        previews[name] = preview(ffmpeg, result/f'{name}.mp4', [sequences[key] for key in keys],
            FPS, clip.start, [labels[key] for key in keys], crop)
    summary = dict(offline_source_oracle=True,
        source_range=[clip.start, clip.stop], fps=FPS, base_sha256=base_sha,
        flow_receipt_sha256=file_sha(output/'flow.json'),
        fixed_scopes=scopes,
        alignment={region: geometry_scores(geometry), region+'_last2': geometry_scores(geometry_late)},
        gate_diagnostics=diagnostics, blur_controls=blur, previews=previews,
        code_sha256=file_sha(__file__), shared_blend_sha256=file_sha(Path(__file__).with_name('run_cache_motion.py')))
    summary[f'{region}_roi'] = clip.coral
    if expected_scopes is not None:
        summary['affine_replay_bit_exact'] = True
    save_json(output/'summary.json', summary)
    print(f'Verified {len(previews)} original-speed dense flow comparisons', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--cache-run', type=Path, required=True)
    parser.add_argument('--flow-run', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    evaluate(args.workspace.resolve(), args.cache_run.resolve(), args.flow_run.resolve(), args.ffmpeg.resolve())


if __name__ == '__main__':
    main()
