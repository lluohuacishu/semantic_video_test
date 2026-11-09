"""Evaluate cache controls on fixed regions and render synchronized 15 FPS comparisons."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from audit_deflicker_texture import measure, summarize
from cache_motion_inputs import CLIPS, FPS, FRAME_SHAPE, LANE_LIMIT, REPO, base_frames, source_frames
from compare_deflicker import save_csv
from compare_multi_coral import prior_result
from compare_seedvr2 import source_detail
from coral_reference_sidecar import file_sha, load_json, save_json
from experiment_common import array_sha
from run_cache_motion import C_FAIR_BUDGET, PACKET, unpack
from run_p0 import preview, psnr, roi


LABELS = dict(source='Source RGB', raw='A / GLC', b0_b2='B0 / matched blur',
              b1='B1 / adjacent', b2='B2 / history', c='C / sent motion', hybrid='Prior hybrid')


def read_rgb(folder, method, digest, clip):
    video = np.load(folder/f'{method}.npy', mmap_mode='r', allow_pickle=False)
    if video.shape != (clip.count, *FRAME_SHAPE) or video.dtype != np.uint8 or array_sha(video) != digest:
        raise ValueError(f'Unverified RGB input: {method}')
    return video


def load_controls(folder, clip, source, raw, base_sha):
    packets = load_json(folder/'motion_send.json')
    packet = folder/PACKET
    motion_bytes = packet.stat().st_size
    motion_sha = file_sha(packet)
    source_sha = array_sha(source)
    records = unpack(packet.read_bytes(), clip, base_sha)
    if (packets['base_sha256'] != base_sha or packets['source_rgb_sha256'] != source_sha
            or packets['sha256'] != motion_sha or packets['bytes'] != motion_bytes
            or packets['accepted_steps'] != int(records['valid'].sum())):
        raise ValueError('The sent motion file differs from its receipt')

    sequences = dict(source=source, raw=raw)
    variants = {}
    for directory in ('autonomous', 'assisted'):
        case = folder/directory
        receipt = load_json(case/'result.json')
        expected_motion = motion_sha if directory == 'assisted' else None
        expected_methods = {'c'} if directory == 'assisted' else {'b1', 'b2'}
        if (receipt['base_sha256'] != base_sha or receipt['frames'] != clip.count
                or receipt['motion_sha256'] != expected_motion or not receipt['source_media_reads_forbidden']
                or receipt['code_sha256'] != packets['code_sha256']
                or receipt['variants'].keys() != expected_methods):
            raise ValueError(f'Mismatched sender/receiver provenance: {case}')
        for name, record in receipt['variants'].items():
            rgb = read_rgb(case, name, record['rgb_sha256'], clip)
            mask = np.load(case/f'{name}_alpha.npy', mmap_mode='r', allow_pickle=False)
            if (mask.shape != raw.shape[:3] or mask.dtype != np.uint8
                    or array_sha(mask) != record['alpha_sha256']
                    or not np.array_equal(rgb[mask == 0], raw[mask == 0])):
                raise ValueError(f'Invalid cache mask: {name}')
            sequences[name] = rgb
            if 'b0' in record:
                control = f'b0_{name}'
                sequences[control] = read_rgb(case, control, record['b0']['rgb_sha256'], clip)
            variants[name] = dict(accepted_steps=record['accepted_steps'],
                mean_effective_fraction=float((mask > 0).mean()),
                late_effective_fraction=float((mask[-2*FPS:] > 0).mean()),
                b0=record.get('b0'), rgb_sha256=record['rgb_sha256'])

    hevc = load_json(folder/'hevc/result.json')
    if hevc['source_rgb_sha256'] != source_sha:
        raise ValueError('The H.265 input differs from the evaluation source')
    for name, record in hevc['controls'].items():
        path = folder/'hevc'/f'{name}.mp4'
        if path.stat().st_size != record['bytes'] or file_sha(path) != record['sha256']:
            raise ValueError(f'Changed H.265 bitstream: {name}')
        sequences[name] = read_rgb(path.parent, name, record['rgb_sha256'], clip)
    return sequences, motion_bytes, variants, hevc['controls']


def score_scope(temporal, spatial, sequences, crop):
    result = summarize(temporal, sequences, crop)
    total = sum(row['valid_pixels'] for row in spatial)
    for name, frames in sequences.items():
        for band in ('middle', 'fine'):
            key = f'{band}_ncc_to_source'
            result[name][key] = sum(row['valid_pixels'] * row[f'{name}_{key}'] for row in spatial) / total
        if name != 'source':
            mse = np.mean([np.mean((a[crop].astype(np.float32) - b[crop]) ** 2)
                           for a, b in zip(frames, sequences['source'], strict=True)])
            result[name]['psnr_db'] = psnr(mse)
    return dict(pairs=len(temporal),
        mean_source_valid_pixels=float(np.mean([row['valid_pixels'] for row in temporal])),
        scores=result)


def second_deltas(temporal, clip):
    result = []
    for offset in range(0, clip.count, FPS):
        start = clip.start + offset
        rows = [row for row in temporal if start <= row['frame'] < start + FPS]
        weight = sum(row['valid_pixels'] for row in rows)
        deltas = {}
        for before, after in (('raw', 'b1'), ('raw', 'b2'), ('b2', 'c')):
            deltas[f'{after}_minus_{before}'] = sum(
                row['valid_pixels'] * (row[f'{after}_fine_ncc'] - row[f'{before}_fine_ncc'])
                for row in rows) / weight
        result.append(dict(source_start_frame=start, pairs=len(rows), **deltas))
    return result


def evaluate_clip(folder, clip, workspace, ffmpeg, hybrid_folder):
    raw, base_sha, base_size = base_frames(folder, clip)
    source = source_frames(workspace, clip)
    sequences, motion_bytes, variants, hevc = load_controls(folder, clip, source, raw, base_sha)
    hybrid_bytes = None
    if hybrid_folder is not None:
        sequences['hybrid'] = prior_result(hybrid_folder, base_sha, 'hybrid')
        receipt = load_json(hybrid_folder/'received/result.json')
        packet = hybrid_folder/'coral_multi.glcsem'
        hybrid_bytes = packet.stat().st_size
        if (sequences['hybrid'].shape != raw.shape or sequences['hybrid'].dtype != raw.dtype
                or file_sha(packet) != receipt['semantic_sha256']
                or base_size + hybrid_bytes != hevc['h265_dual']['target_bytes']):
            raise ValueError('Prior hybrid run is not the same-byte reference')

    out = folder/'comparison'
    out.mkdir(exist_ok=False)
    regions = {'full': np.s_[:, :], 'coral': roi(clip.coral, FRAME_SHAPE)}
    scopes, seconds = {}, {}
    for name, crop in regions.items():
        temporal = measure(sequences, crop, start=clip.start)
        spatial = source_detail(source, sequences, crop, start=clip.start)
        save_csv(out/f'{name}_temporal.csv', temporal)
        save_csv(out/f'{name}_source_detail.csv', spatial)
        scopes[name] = score_scope(temporal, spatial, sequences, crop)
        scopes[name+'_last2'] = score_scope(temporal[-2*FPS:], spatial[-2*FPS:],
            {key: video[-2*FPS:] for key, video in sequences.items()}, crop)
        seconds[name] = second_deltas(temporal, clip)

    rate_labels = {name: f'H.265 / {record["kbps"]:.2f} kbps' for name, record in hevc.items()}
    labels = {**LABELS, **rate_labels}
    videos = {
        'cache_comparison': (('source', 'raw', 'b2', 'c'), None),
        'coral_closeup': (('source', 'raw', 'b2', 'c'), regions['coral']),
        'blur_control': (('raw', 'b0_b2', 'b1', 'b2'), None),
        'h265_rates': (('source', 'h265_base', 'h265_40', 'h265_dual'), None),
    }
    if 'hybrid' in sequences:
        videos['prior_hybrid_vs_h265'] = (('source', 'raw', 'hybrid', 'h265_dual'), None)
    previews = {}
    for name, (methods, crop) in videos.items():
        previews[name] = preview(ffmpeg, out/f'{name}.mp4', [sequences[key] for key in methods],
                                  FPS, clip.start, [labels[key] for key in methods], crop)

    rates = dict(base_bytes=base_size, semantic_bytes=motion_bytes,
        base_kbps=base_size*8*FPS/clip.count/1000, semantic_kbps=motion_bytes*8*FPS/clip.count/1000)
    if hybrid_bytes is not None:
        rates['prior_hybrid_bytes'] = hybrid_bytes
        rates['prior_hybrid_combined_kbps'] = (base_size+hybrid_bytes)*8*FPS/clip.count/1000
    print(f'{folder.name}: {len(previews)} original-speed comparisons verified', flush=True)
    return dict(source_range=[clip.start, clip.stop], base_start=clip.base_start,
        fixed_coral_roi=clip.coral, rates=rates, base_sha256=base_sha, rgb_sha256=array_sha(raw),
        hevc=hevc, variants=variants, fixed_scopes=scopes, fine_ncc_deltas_by_second=seconds,
        previews=previews)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    parser.add_argument('--prior-hybrid', type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    run = args.run.resolve()
    results = {name: evaluate_clip(run/name, clip, args.workspace.resolve(), args.ffmpeg.resolve(),
                                   args.prior_hybrid if name == 'moving_reef' else None)
               for name, clip in CLIPS.items()}
    save_json(run/'summary.json', dict(fps=FPS, frame_shape=FRAME_SHAPE, lane_limit_kbps=LANE_LIMIT/1000,
        motion_test_cap_bytes=C_FAIR_BUDGET, clips=results, code_sha256=file_sha(__file__)))


if __name__ == '__main__':
    main()
