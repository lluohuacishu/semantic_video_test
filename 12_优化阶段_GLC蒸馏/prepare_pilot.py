"""Freeze sequence split and prepare a small local corpus; never download data."""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image

from pilot_common import REPO, file_sha, load_json, save_json, settings, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-preparation', action='store_true', help='Reuse completed, hash-verified preparation clips only')
    args = parser.parse_args()
    workspace = settings()['workspace']
    corpus = workspace / '02_实验数据/D_语义目标增强/D0_20261021_r1'
    manifest_path = corpus / 'ready_manifest.json'
    old = load_json(manifest_path)
    sys.path.insert(0, str(REPO / '09_新阶段D_语义目标增强'))
    from prepare_d0 import indexed_annotations, decode_union, validate_supplied_mask
    from audit_d0_timing import audit
    if args.output.exists() and any(args.output.iterdir()) and not args.resume_preparation:
        raise FileExistsError('Use a new or empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    chosen = [(i, 'train', 64) for i in range(1, 7)] + [(7, 'validation', 96), (25, 'test_static', 64)]
    assert not set(range(1, 7)) & {7, 25}
    plan = {
        'schema': 'glc-distillation-pilot-v1', 'date': '2026-10-29',
        'purpose': '5000-step P-frame generator KD versus same-initialization GT-only pilot',
        'train_video_ids': list(range(1, 7)), 'validation_video_ids': [7], 'test_video_ids': [25],
        'group_basis': old['selection']['group_basis'],
        'holdout_claim': old['selection']['holdout_claim'],
        'source_independence_established': False,
        'ready_manifest_sha256': file_sha(manifest_path), 'archive_size': old['archive_size'],
        'archive_identity_note': old['archive_identity_note'],
        'codec': dict(width=512, height=288, padded_height=320, q=12, intra_period=64, reset_interval=32),
        'training': dict(steps_each=5000, seed=29, batch=1, learning_rate=1e-4, final_learning_rate=1e-5,
                         warmup_steps=100, optimizer='AdamW', weight_decay=0.0, gradient_clip_norm=1.0,
                         precision='FP32 parameters with BF16 autocast; validation FP32; TF32 disabled',
                         loss_kd='0.5*L1(student, uint8 teacher) + 0.5*L1(student, original)',
                         loss_gt='L1(student, original)', loss_space='RGB normalized to [-1,1]; cropped valid 512x288',
                         selection='minimum validation P-frame mean squared RGB error to original',
                         initialization='same random initialization; no teacher weight transplant',
                         sampling='identical seeded frame sampling with replacement, P frames only',
                         perceptual_loss=False, temporal_loss=False),
        'clips': [], 'complete': False,
    }
    old_by_id = {s['video_id']: s for s in old['fish_sequences']}
    if args.resume_preparation:
        previous_plan = load_json(args.output / 'plan.json')
        if previous_plan['complete'] or previous_plan['ready_manifest_sha256'] != plan['ready_manifest_sha256']:
            raise ValueError('Only this incomplete preparation can be resumed')
        if previous_plan['codec'] != plan['codec'] or previous_plan['training'] != plan['training']:
            raise ValueError('Cannot change the protocol while resuming preparation')
        plan['clips'] = previous_plan['clips']
    save_json(args.output / 'plan.json', plan)
    with zipfile.ZipFile(old['archive_path']) as archive:
        if Path(old['archive_path']).stat().st_size != old['archive_size']:
            raise ValueError('Archive size changed')
        for video_id, role, count in chosen:
            folder = args.output / f'v{video_id:03d}'
            completed = next((c for c in plan['clips'] if c['video_id'] == video_id), None)
            if completed:
                if completed['role'] != role or completed['frames'] != count or file_sha(folder / 'source.npy') != completed['source_array_sha256']:
                    raise ValueError('Previously prepared clip changed')
                print(f'reuse verified preparation {completed["id"]}', flush=True)
                continue
            # Only unfinished arrays from this explicitly resumed preparation may
            # be regenerated; completed clip arrays above are never overwritten.
            folder.mkdir(exist_ok=args.resume_preparation)
            data = __import__('json').loads(archive.read(f'SFISHTRACK/Annotations/video{video_id:03d}.json'))
            by_image = indexed_annotations(data)
            frames = sorted(data['images'], key=lambda r: int(Path(r['file_name']).stem))[:count]
            if len(frames) != count:
                raise ValueError('Not enough annotated source frames')
            sources = np.lib.format.open_memmap(folder / 'source.npy', mode='w+', dtype='uint8', shape=(count, 288, 512, 3))
            masks = np.lib.format.open_memmap(folder / 'fish_mask.npy', mode='w+', dtype='uint8', shape=(count, 288, 512)) if role != 'train' else None
            records = []
            audit_frames = []
            if video_id == 7:
                (folder / 'timing_rgb').mkdir(exist_ok=args.resume_preparation)
            for i, frame in enumerate(frames):
                original_index = int(Path(frame['file_name']).stem)
                if original_index != i:
                    raise ValueError('Source frame sequence has a gap')
                member = f'SFISHTRACK/Frames/video_{video_id:03d}/frame_{i:06d}.jpg'
                raw = archive.read(member)  # ZIP CRC checked by ZipFile.read.
                raw_sha = sha(raw)
                original = Image.open(io.BytesIO(raw)).convert('RGB')
                if original.size != (frame['width'], frame['height']):
                    raise ValueError('RGB dimensions differ from annotation')
                previous = old_by_id.get(video_id)
                if previous and i < len(previous['frames']):
                    if previous['frames'][i]['source_file_sha256'] != raw_sha:
                        raise ValueError('Source bytes changed from verified ready corpus')
                rgb = np.asarray(original.resize((512, 288), Image.Resampling.LANCZOS))
                sources[i] = rgb
                row = dict(index=i, kind='I' if i % 64 == 0 else 'P', source_member=member,
                           source_file_sha256=raw_sha, resized_rgb_sha256=sha(rgb.tobytes()))
                if previous and i < len(previous['frames']):
                    row['source_pts'] = previous['frames'][i].get('source_pts')
                    row['source_time_base'] = previous['frames'][i].get('source_time_base')
                if masks is not None:
                    union = decode_union(frame, by_image[frame['id']])
                    mask_member = f'SFISHTRACK/Masks/masks_{video_id:03d}/{frame["file_name"]}'
                    validate_supplied_mask(archive.read(mask_member), union)
                    masks[i] = np.asarray(Image.fromarray(union.astype('uint8')).resize((512, 288), Image.Resampling.NEAREST))
                    row['fish_mask_sha256'] = sha(masks[i].tobytes())
                if video_id == 7:
                    rgb_path = folder / 'timing_rgb' / f'{i:06d}.jpg'
                    rgb_path.write_bytes(raw)
                    mask_path = folder / 'timing_rgb' / f'{i:06d}_mask.png'
                    Image.fromarray(union.astype('uint8')).save(mask_path)
                    audit_frames.append(dict(index=i, original_file_name=frame['file_name'], annotation_frame_id=frame.get('frame_id'),
                                             rgb_path=str(rgb_path), mask_path=str(mask_path), source_file_sha256=raw_sha,
                                             mask_file_sha256=file_sha(mask_path)))
                records.append(row)
            sources.flush()
            if masks is not None:
                masks.flush()
            clip = dict(id=f'v{video_id:03d}', video_id=video_id, role=role, frames=count, records=records,
                        fps=old_by_id.get(video_id, {}).get('fps'),
                        timing_status=old_by_id.get(video_id, {}).get('timing_status', 'index_order_only_no_timing_claim'))
            if video_id == 7:
                runtime = settings()['runtime']
                seq = old_by_id[7]
                timing = audit(dict(video_id=7, role=role, video_path=seq['video_path'], video_sha256=seq['video_sha256'],
                                    frames=audit_frames), corpus, runtime['ffmpeg'], runtime['ffmpeg'].with_name('ffprobe.exe'))
                save_json(folder / 'timing_audit.json', timing)
                clip['dense_video_comparison_ready'] = timing['ready_for_dense_video_comparison']
                clip['fps'] = timing['fps_if_constant'] if timing['ready_for_dense_video_comparison'] else None
                clip['timing_status'] = ('verified_constant_rate' if timing['ready_for_dense_video_comparison']
                                         else 'extended_annotation_sequence_one_or_more_timing_matches_ambiguous_static_metrics_only')
                for row, matched in zip(records, timing['matches']):
                    row['source_pts'] = matched['source_pts_ticks']
                    row['source_time_base'] = timing['source_stream']['time_base']
            clip['source_array_sha256'] = file_sha(folder / 'source.npy')
            save_json(folder / 'source_manifest.json', clip)
            plan['clips'].append({k: v for k, v in clip.items() if k != 'records'})
            save_json(args.output / 'plan.json', plan)
            print(f'prepared {clip["id"]} {role} {count} frames', flush=True)
    plan['complete'] = True
    save_json(args.output / 'plan.json', plan)
    save_json(args.output / 'plan_identity.json', {'sha256': file_sha(args.output / 'plan.json')})


if __name__ == '__main__':
    main()
