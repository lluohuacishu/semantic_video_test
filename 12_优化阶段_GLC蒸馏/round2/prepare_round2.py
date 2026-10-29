"""Expand to 49 SFishTrack sequences; keep UV monitoring/holdout separate."""
from __future__ import annotations
import argparse
from collections import Counter
import io
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot_common import file_sha, load_json, save_json, settings, sha


def contiguous_window(entries, count):
    indexed = sorted((int(Path(e.filename).stem.replace('frame_', '')), e) for e in entries)
    if len({i for i, _ in indexed}) != len(indexed):
        raise ValueError('Ambiguous duplicate source frame numbers')
    runs = [[]]
    for item in indexed:
        if runs[-1] and item[0] != runs[-1][-1][0] + 1:
            runs.append([])
        runs[-1].append(item)
    run = max(runs, key=lambda r: (len(r), -r[0][0]))
    if len(run) < count:
        raise ValueError('No sufficiently long consecutive RGB window')
    start = (len(run) - count) // 2
    return run[start:start+count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-preparation', action='store_true')
    args = parser.parse_args()
    if (args.output / 'plan.json').exists() and not args.resume_preparation:
        raise FileExistsError('Preparation plan already exists; preserve this run')
    workspace = settings()['workspace']
    old_run = workspace / '02_实验数据/优化_GLC蒸馏/pilot_20261029_01'
    inventory_path = args.output / 'sfish_audit/inventory.json'
    inventory = load_json(inventory_path)
    train_ids = [i for i in range(1, 55) if i not in {7, 12, 25, 26, 27}]
    uv_root = Path('C:/Users/lenovo/Desktop/works/work/semantic-video-mlvc/data/phase_c_20261013')
    uv_meta = {v['video']: v for v in load_json(uv_root / 'manifest.json')['videos']}
    plan = dict(schema='glc-distillation-round2-v1', complete=False, date='2026-10-29',
                architecture=dict(width=64, res_blocks=3, parameters=10985155, group_norm_groups=32),
                codec=dict(width=512, height=288, padded_height=320, q=12, intra_period=64, reset_interval=32),
                training=dict(steps_each=5000, seed=29, learning_rate=1e-4, final_learning_rate=1e-5,
                              warmup_steps=100, batch=1, precision='FP32 parameters; BF16 training; FP32 validation; TF32 off',
                              selection='minimum uv096 P-frame aggregate MSE to original',
                              cases={'random_mixed': {'initialization': 'random', 'loss': 'mixed'},
                                     'mapped_mixed': {'initialization': 'mapped', 'loss': 'mixed'},
                                     'mapped_teacher': {'initialization': 'mapped', 'loss': 'teacher'},
                                     'mapped_gt': {'initialization': 'mapped', 'loss': 'gt'}},
                              losses={'mixed': '0.5*L1(S,T_uint8)+0.5*L1(S,G)', 'teacher': 'L1(S,T_uint8)', 'gt': 'L1(S,G)'},
                              all_cases_same_sample_order=True, test_used_for_selection=False),
                sfish_inventory_sha256=file_sha(inventory_path), train_sfish_ids=train_ids,
                sfish_window_rule='centered 64-frame window in longest consecutive RGB run; no model/GT-score selection',
                validation=['uv096'], final_holdouts=['uv108', 'sf026'], regression=['v007', 'v025'],
                source_independence_established=False,
                source_caveats=['SFishTrack all54 metadata conflicts with old metadata; many similar/rotated/letterboxed views',
                                'Training SFishTrack only; UV96/108 excluded from training, capture identity unverified',
                                'SF26 shares conservative capture group with previously viewed SF25; not an untouched independent-source test',
                                'UV96/108 used in older project evaluations; held out from this student experiment, not never-seen benchmarks',
                                'Published frame order used; no verified full original capture cadence or bitrate/time claim'],
                clips=[])
    if args.resume_preparation:
        previous = load_json(args.output / 'plan.json')
        if previous['complete'] or any(previous[k] != plan[k] for k in ['architecture', 'codec', 'training', 'train_sfish_ids']):
            raise ValueError('Cannot resume a completed or different preparation protocol')
        plan['clips'] = previous['clips']
    save_json(args.output / 'plan.json', plan)
    corpus = workspace / '02_实验数据/D_语义目标增强/D0_20261021_r1'
    ready = load_json(corpus / 'ready_manifest.json')
    with zipfile.ZipFile(ready['archive_path']) as archive:
        names = [r for r in archive.infolist() if not r.is_dir()]
        for video_id in train_ids + [26]:
            cid = f'sf{video_id:03d}'
            role = 'train' if video_id in train_ids else 'final_holdout'
            folder = args.output / cid
            completed = next((c for c in plan['clips'] if c['id'] == cid), None)
            if completed:
                if file_sha(folder / 'source.npy') != completed['source_array_sha256']:
                    raise ValueError('Completed source array changed')
                continue
            folder.mkdir(exist_ok=args.resume_preparation)
            prefix = f'SFISHTRACK/Frames/video_{video_id:03d}/'
            entries = [r for r in names if r.filename.startswith(prefix) and Path(r.filename).suffix.lower() in {'.jpg', '.png', '.jpeg'}]
            chosen = contiguous_window(entries, 64)
            source = np.lib.format.open_memmap(folder / 'source.npy', mode='w+', dtype='uint8', shape=(64, 288, 512, 3))
            records = []
            for i, (source_index, entry) in enumerate(chosen):
                raw = archive.read(entry)
                image = Image.open(io.BytesIO(raw)).convert('RGB')
                if image.width * 9 != image.height * 16:
                    raise ValueError('Unexpected non-16:9 source; require explicit aspect policy')
                rgb = np.asarray(image.resize((512, 288), Image.Resampling.LANCZOS))
                source[i] = rgb
                records.append(dict(index=i, source_index=source_index, member=entry.filename,
                                    source_file_sha256=sha(raw), resized_rgb_sha256=sha(rgb.tobytes())))
            source.flush()
            clip = dict(id=cid, role=role, dataset='SFishTrack', video_id=video_id, frames=64, fps=None,
                        timing_status='consecutive published RGB indices; original cadence not audited',
                        source_start_index=chosen[0][0], source_end_index=chosen[-1][0],
                        source_array_sha256=file_sha(folder / 'source.npy'))
            save_json(folder / 'source_manifest.json', {**clip, 'records': records})
            plan['clips'].append(clip)
            save_json(args.output / 'plan.json', plan)
            print(f'prepared {cid} {role} source {chosen[0][0]}..{chosen[-1][0]}', flush=True)
    for video, role in [('video96', 'validation'), ('video108', 'final_holdout')]:
        meta = uv_meta[video]
        cid = f'uv{int(video[5:]):03d}'
        folder = args.output / cid
        if any(c['id'] == cid for c in plan['clips']):
            continue
        folder.mkdir(exist_ok=args.resume_preparation)
        frames = 128
        source = np.lib.format.open_memmap(folder / 'source.npy', mode='w+', dtype='uint8', shape=(frames, 288, 512, 3))
        masks = np.lib.format.open_memmap(folder / 'fish_mask.npy', mode='w+', dtype='uint8', shape=(frames, 288, 512))
        available = np.zeros(frames, dtype='bool')
        labels = {r['frame_index']: r for r in meta['labels']}
        records = []
        for i in range(frames):
            path = Path(meta['sequence_dir']) / f'{i:05d}.jpg'
            image = Image.open(path).convert('RGB')
            if sha(np.asarray(image).tobytes()) != meta['source_frame_rgb_hashes'][i]:
                raise ValueError('UV source pixels changed from audited manifest')
            rgb = np.asarray(image.resize((512, 288), Image.Resampling.LANCZOS))
            source[i] = rgb
            masks[i] = 0  # Storage placeholder only; availability gate is mandatory.
            if i in labels:
                label = labels[i]
                if file_sha(label['published_mask']) != label['mask_sha256']:
                    raise ValueError('UV published mask changed')
                published = np.asarray(Image.open(label['published_mask']).convert('RGB'))
                binary = np.all(published == [0, 0, 128], axis=2)
                if not np.array_equal(binary, np.asarray(Image.open(label['fish_mask']).convert('L')) > 0):
                    raise ValueError('UV stored fish mask differs from published RGB(0,0,128) fish label')
                mask = Image.fromarray(binary.astype('uint8'))
                masks[i] = np.asarray(mask.resize((512, 288), Image.Resampling.NEAREST)) > 0
                available[i] = True
            records.append(dict(index=i, source_index=i, source_path=str(path), source_file_sha256=file_sha(path),
                                resized_rgb_sha256=sha(rgb.tobytes()), gt_available=bool(available[i]),
                                fish_mask_sha256=sha(masks[i].tobytes()) if available[i] else None))
        source.flush()
        masks.flush()
        np.save(folder / 'gt_available.npy', available)
        clip = dict(id=cid, role=role, dataset='UV-Multi', frames=frames, fps=None,
                    official_split=meta['official_split'], original_video_id=video,
                    timing_status='published frame order only; per-clip original capture time unverified',
                    source_array_sha256=file_sha(folder / 'source.npy'), fish_mask_array_sha256=file_sha(folder / 'fish_mask.npy'),
                    gt_available_array_sha256=file_sha(folder / 'gt_available.npy'), labeled_frames=int(available.sum()))
        save_json(folder / 'source_manifest.json', {**clip, 'records': records})
        plan['clips'].append(clip)
        save_json(args.output / 'plan.json', plan)
        print(f'prepared {cid} {role} {frames} frames, {int(available.sum())} labeled', flush=True)
    # Dense/sparse SF26 fish GT is for evaluation only, not sample selection.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '09_新阶段D_语义目标增强'))
    from prepare_d0 import indexed_annotations, decode_union, validate_supplied_mask
    sf26 = next(c for c in plan['clips'] if c['id'] == 'sf026')
    sf_folder = args.output / 'sf026'
    sf_rows = load_json(sf_folder / 'source_manifest.json')['records']
    sf_masks = np.lib.format.open_memmap(sf_folder / 'fish_mask.npy', mode='w+', dtype='uint8', shape=(64, 288, 512))
    sf_available = np.zeros(64, dtype='bool')
    with zipfile.ZipFile(ready['archive_path']) as archive:
        import json
        annotation = json.loads(archive.read('SFISHTRACK/Annotations/video026.json'))
        instances = indexed_annotations(annotation)
        frames_by_index = {int(Path(r['file_name']).stem): r for r in annotation['images']}
        for i, row in enumerate(sf_rows):
            sf_masks[i] = 0
            frame = frames_by_index.get(row['source_index'])
            if frame is not None:
                union = decode_union(frame, instances[frame['id']])
                validate_supplied_mask(archive.read('SFISHTRACK/Masks/masks_026/' + frame['file_name']), union)
                sf_masks[i] = np.asarray(Image.fromarray(union.astype('uint8')).resize((512, 288), Image.Resampling.NEAREST))
                sf_available[i] = True
    sf_masks.flush()
    np.save(sf_folder / 'gt_available.npy', sf_available)
    sf26.update(fish_mask_array_sha256=file_sha(sf_folder / 'fish_mask.npy'), gt_available_array_sha256=file_sha(sf_folder / 'gt_available.npy'), labeled_frames=int(sf_available.sum()))
    previous_plan = load_json(old_run / 'plan.json')
    for cid in plan['regression']:
        original = next(c for c in previous_plan['clips'] if c['id'] == cid)
        folder = args.output / cid
        folder.mkdir()
        for name in ['source.npy', 'teacher.npy', 'latent.npy', 'fish_mask.npy', 'stream.glcv', 'source_manifest.json', 'encode_cache.json', 'decode_cache.json']:
            shutil.copyfile(old_run / cid / name, folder / name)
        clip = {**original, 'role': 'regression', 'dataset': 'SFishTrack', 'reused_original_run': str(old_run)}
        np.save(folder / 'gt_available.npy', np.ones(clip['frames'], dtype='bool'))
        clip['fish_mask_array_sha256'] = file_sha(folder / 'fish_mask.npy')
        clip['gt_available_array_sha256'] = file_sha(folder / 'gt_available.npy')
        plan['clips'].append(clip)
    # Exact duplicates do not establish source independence, but cross-split
    # identity would invalidate selection. Related historical regressions exempt.
    seen, duplicates = {}, []
    for clip in plan['clips']:
        rows = load_json(args.output / clip['id'] / 'source_manifest.json')['records']
        for row in rows:
            digest = row['resized_rgb_sha256']
            if digest in seen:
                old = seen[digest]
                duplicates.append(dict(first=old, second=[clip['id'], clip['role'], row['index']]))
                if old[1] != clip['role'] and 'regression' not in [old[1], clip['role']]:
                    raise ValueError('Exact RGB duplicate crosses strict train/validation/final split')
            else:
                seen[digest] = [clip['id'], clip['role'], row['index']]
    save_json(args.output / 'input_audit.json', dict(unique_resized_rgb_frames=len(seen), exact_duplicate_pairs=duplicates,
              source_independence_established=False, total_frames=sum(c['frames'] for c in plan['clips']),
              train_clips=len(train_ids), train_p_frames=len(train_ids)*63,
              original_sfish_clips=inventory['total_clips'], original_sfish_rgb_frames=inventory['total_rgb_frames']))
    plan['complete'] = True
    save_json(args.output / 'plan.json', plan)
    save_json(args.output / 'plan_identity.json', {'sha256': file_sha(args.output / 'plan.json')})
    print(f'frozen: {len(train_ids)} training clips, {len(train_ids)*63} P frames', flush=True)


if __name__ == '__main__':
    main()
