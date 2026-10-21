"""Read existing D1 masks to separate branch, dilation and history contributions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from d1_protection import seed_masks, hard_protection


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def run(args):
    if args.output.exists():
        raise FileExistsError('Choose a new diagnostic output; existing results are preserved')
    start = time.perf_counter()
    plan = read_json(args.run/'plan.json')
    config = read_json(args.run/'frozen_protection.json')['config']
    inputs = read_json(args.run/'inputs.json')['inputs']
    manifest = read_json(args.corpus/'ready_manifest.json')
    truths = {f'sfishtrack_{s["video_id"]:03d}/{f["original_file_name"]}': f
              for s in manifest['fish_sequences'] if s['role'] == 'development'
              for f in s['frames']}
    groups = {}
    previous = None
    previous_video = None
    count = 0
    for index, source in enumerate(inputs):
        video = source.get('video_id')
        if video is None:
            continue
        if video not in (1, 2, 3):
            raise ValueError('Only the existing fish development clips are in scope')
        if video != previous_video:
            previous = None
            previous_video = video
        gt = np.asarray(Image.open(args.corpus/truths[source['id']]['mask_path'])) != 0
        packed = np.asarray(Image.open(args.run/'packed'/f'{index:04d}.png'))
        fish, diver = seed_masks(packed, plan, config)
        # Dilation distributes over union: dilate(current | previous) equals
        # dilate(current) | dilate(previous). Reuse the previous dilated seed.
        spatial_config = dict(config, previous_seed_frames=0)
        spatial = [hard_protection(seed, None, spatial_config) for seed in (fish, diver)]
        temporal = spatial if previous is None else [now | old for now, old in zip(spatial, previous)]
        stages = {'raw': (fish, diver), 'dilated': spatial, 'dilated_hold1': temporal}
        group = groups.setdefault(f'fish_{video:03d}', {'frames': 0, 'frame_pixels': 0,
                 'gt_pixels': 0, 'stages': {}})
        group['frames'] += 1
        group['frame_pixels'] += int(gt.size)
        group['gt_pixels'] += int(gt.sum())
        for stage_name, (f, d) in stages.items():
            masks = {'fish': f, 'diver': d, 'union': f | d,
                     'diver_only_addition': d & ~f, 'fish_only_addition': f & ~d}
            results = group['stages'].setdefault(stage_name, {})
            for branch, mask in masks.items():
                item = results.setdefault(branch, {'selected_pixels': 0, 'fish_gt_overlap_pixels': 0,
                                                  'frames_with_no_fish_gt_overlap': 0})
                overlap = int(np.count_nonzero(mask & gt))
                item['selected_pixels'] += int(np.count_nonzero(mask))
                item['fish_gt_overlap_pixels'] += overlap
                item['frames_with_no_fish_gt_overlap'] += int(overlap == 0)
        previous = spatial
        count += 1
        if count % 64 == 0:
            print(f'Analyzed {count}/192 cached development masks', flush=True)
    for group in groups.values():
        group['mean_labeled_fish_frame_fraction'] = group['gt_pixels']/group['frame_pixels']
        for results in group['stages'].values():
            for item in results.values():
                item['mean_frame_fraction'] = item['selected_pixels']/group['frame_pixels']
                item['pooled_fish_gt_recall'] = item['fish_gt_overlap_pixels']/group['gt_pixels']
                item['outside_labeled_fish_frame_fraction'] = (
                    item['selected_pixels']-item['fish_gt_overlap_pixels'])/group['frame_pixels']
        hard = group['stages']['dilated_hold1']
        group['diver_exclusive_fraction_of_union_area'] = (
            hard['diver_only_addition']['selected_pixels']/hard['union']['selected_pixels'])
    windows = {}
    for tile, stride in ((512, 384), (256, 192)):
        xs = sorted(set(list(range(0, max(1, 1920-tile), stride)) + [max(0, 1920-tile)]))
        ys = sorted(set(list(range(0, max(1, 1080-tile), stride)) + [max(0, 1080-tile)]))
        windows[str(tile)] = {'stride': stride, 'x_count': len(xs), 'y_count': len(ys),
                             'windows_per_1080p_frame': len(xs)*len(ys),
                             'total_input_pixels': len(xs)*len(ys)*tile*tile}
    result = {'schema': 'd1-cached-branch-contributions-v1', 'date': '2026-10-21',
              'source_run': str(args.run.resolve()), 'profile': 'base_tiles512',
              'config': config, 'input_count': count, 'groups': groups,
              'window_count_from_existing_anchored_grid_rule': windows,
              'seconds': time.perf_counter()-start,
              'limitations': [
                  'Diver predictions are evaluated only against fish GT; there is no diver/background GT for these clips.',
                  'Outside labeled fish is not necessarily confirmed background or a false detection.',
                  'Hard masks are measured before soft edges and full-frame fallback.',
                  'Raw, dilation and one-frame history are a sequential decomposition, not additive independent causal effects.',
                  'No parameter selection, new inference, model load, training, codec run or integrity rescan.'
              ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for name, group in groups.items():
        print(name, flush=True)
        for stage, branches in group['stages'].items():
            print(stage, {branch: {'fish_coverage': round(values['pooled_fish_gt_recall']*100, 3),
                                 'frame_area': round(values['mean_frame_fraction']*100, 3)}
                          for branch, values in branches.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
