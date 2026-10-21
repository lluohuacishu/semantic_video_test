"""Prepare a small auditable D0 corpus on CPU. Never import model code.

Dependencies: numpy, Pillow, pycocotools (available in the old MLVC environment).
Only selected ZIP members are read; originals are never modified. Output must be new.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as coco_mask


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def frame_number(name: str) -> int:
    path = PurePosixPath(name)
    if path.name != name or not path.stem.isdecimal():
        raise ValueError(f'Expected a plain numeric annotation filename: {name}')
    return int(path.stem)


def resolve_rgb(names: set[str], video_id: int, file_name: str) -> str:
    frame_number(file_name)
    stem = PurePosixPath(file_name).stem
    folder = f'SFISHTRACK/Frames/video_{video_id:03d}'
    candidates = {f'{folder}/{file_name}', f'{folder}/frame_{stem}.jpg'} & names
    if len(candidates) != 1:
        raise ValueError(f'Unresolved or ambiguous RGB: video{video_id:03d}/{file_name}: {candidates}')
    return candidates.pop()


def indexed_annotations(data: dict):
    image_ids = [row['id'] for row in data['images']]
    ann_ids = [row['id'] for row in data['annotations']]
    if len(set(image_ids)) != len(image_ids) or len(set(ann_ids)) != len(ann_ids):
        raise ValueError('Duplicate image or instance IDs within sequence')
    if len({row['file_name'] for row in data['images']}) != len(image_ids):
        raise ValueError('Duplicate image filenames within sequence')
    if len(data['categories']) != 1 or data['categories'][0]['id'] != 1 or data['categories'][0]['name'] not in {'fish', 'object'}:
        raise ValueError('Unsupported fish category alias')
    by_image = defaultdict(list)
    known = set(image_ids)
    for row in data['annotations']:
        if row['image_id'] not in known or row['category_id'] != 1 or row.get('iscrowd', 0):
            raise ValueError('Unbound annotation, unsupported category or crowd')
        by_image[row['image_id']].append(row)
    return by_image


def decode_union(frame: dict, instances: list[dict]) -> np.ndarray:
    union = np.zeros((frame['height'], frame['width']), dtype=bool)
    for row in instances:
        rle = row['segmentation']
        if tuple(rle['size']) != union.shape:
            raise ValueError('RLE dimensions differ from RGB declaration')
        decoded = coco_mask.decode(rle).astype(bool)
        if decoded.shape != union.shape or int(decoded.sum()) != row['area']:
            raise ValueError('RLE area/shape differs from annotation')
        actual_bbox = coco_mask.toBbox(rle).tolist()
        # This archive stores [xmin, ymin, xmax-xmin, ymax-ymin]. COCO's
        # toBbox counts the inclusive last pixel, hence width/height +1.
        x, y, w, h = row['bbox']
        if actual_bbox != [x, y, w + 1, h + 1]:
            raise ValueError(f'RLE bbox differs from declared max-minus-min box: {actual_bbox} vs {row["bbox"]}')
        union |= decoded
    return union


def validate_supplied_mask(data: bytes, union: np.ndarray) -> None:
    supplied = np.asarray(Image.open(io.BytesIO(data)).convert('RGB'))
    foreground = np.any(supplied != 0, axis=2)
    if foreground.shape != union.shape or not np.array_equal(foreground, union):
        raise ValueError('Supplied mask foreground differs from decoded RLE union')


def validate_split(config: dict) -> None:
    dev, reserved, quarantine = (set(config[k]) for k in (
        'development_group_video_ids', 'reserved_group_video_ids', 'quarantined_video_ids'))
    if dev & reserved or dev & quarantine or reserved & quarantine or dev | reserved | quarantine != set(range(1, 55)):
        raise ValueError('Video split overlaps or does not cover all 54 sequences')
    if not set(config['fish_development_video_ids']) <= dev or not set(config['fish_reserved_video_ids']) <= reserved:
        raise ValueError('Selected video outside its split')
    historical = set(config['historical_codec_evaluation_video_ids'] + config['historical_alignment_only_video_ids'])
    if set(config['fish_reserved_video_ids']) & historical:
        raise ValueError('Historical evaluation/alignment sample selected as fresh D holdout')


def audit_archive(archive: zipfile.ZipFile, old_root: Path, output: Path):
    members = [info for info in archive.infolist() if not info.is_dir()]
    counts = Counter(info.filename for info in members)
    if any(n != 1 for n in counts.values()):
        raise ValueError('Duplicate archive member name')
    names = set(counts)
    sequences, mapping, data_by_video = [], [], {}
    for video_id in range(1, 55):
        entry = f'SFISHTRACK/Annotations/video{video_id:03d}.json'
        raw = archive.read(entry)
        data = json.loads(raw)
        indexed_annotations(data)
        data_by_video[video_id] = data
        aliases = 0
        for frame in data['images']:
            rgb_entry = resolve_rgb(names, video_id, frame['file_name'])
            mask_entry = f'SFISHTRACK/Masks/masks_{video_id:03d}/{frame["file_name"]}'
            if mask_entry not in names:
                raise ValueError(f'Missing mask member: {mask_entry}')
            aliases += rgb_entry.endswith('.jpg')
            mapping.append({'video_id': video_id, 'image_id': frame['id'], 'annotation_frame_id': frame['frame_id'],
                            'file_name': frame['file_name'], 'rgb_member': rgb_entry, 'mask_member': mask_entry})
        metadata_raw = archive.read(f'SFISHTRACK/metadata/video_{video_id:03d}.json')
        old_annotation = old_root / f'Annotations/video{video_id:03d}.json'
        history = None
        if old_annotation.exists():
            old = load_json(old_annotation)
            stripped_old = dict(old)
            stripped_old['annotations'] = [{k: v for k, v in row.items() if k != 'frame_number'} for row in old['annotations']]
            history = {'old_sha256': file_sha(old_annotation), 'same_after_removing_old_frame_number': stripped_old == data,
                       'old_frame_number_field_count': sum('frame_number' in row for row in old['annotations'])}
        sequence = {'video_id': video_id, 'image_count': len(data['images']), 'annotation_count': len(data['annotations']),
                    'category_alias': data['categories'][0]['name'], 'jpg_name_alias_count': aliases,
                    'annotation_sha256': sha(raw), 'metadata_sha256': sha(metadata_raw), 'metadata': json.loads(metadata_raw),
                    'historical_annotation_comparison': history}
        sequences.append(sequence)
    save_json(output / 'all_frame_mapping.json', mapping)
    audit = {'schema': 'd0-archive-audit-v1', 'scope': 'all names/JSON; content validation only for selected samples',
             'non_directory_members': len(members), 'annotation_images': len(mapping), 'rgb_mapping_missing_or_ambiguous': 0,
             'missing_supplied_mask_names': 0, 'sequences': sequences, 'full_archive_crc_scan': False}
    save_json(output / 'archive_audit.json', audit)
    return names, data_by_video, audit


def prepare_fish(archive, names, data_by_video, config, output, old_root):
    sequences = []
    for role, key in [('development', 'fish_development_video_ids'), ('reserved_sequence_holdout', 'fish_reserved_video_ids')]:
        for video_id in config[key]:
            data = data_by_video[video_id]
            by_image = indexed_annotations(data)
            frames = sorted(data['images'], key=lambda row: frame_number(row['file_name']))[:config['frames_per_video']]
            folder = output / 'fish' / f'video_{video_id:03d}'
            (folder / 'rgb').mkdir(parents=True)
            (folder / 'gt_fish').mkdir()
            records = []
            for index, frame in enumerate(frames):
                rgb_entry = resolve_rgb(names, video_id, frame['file_name'])
                rgb_raw = archive.read(rgb_entry)
                rgb = Image.open(io.BytesIO(rgb_raw)).convert('RGB')
                if rgb.size != (frame['width'], frame['height']):
                    raise ValueError('RGB dimensions differ from annotation')
                instances = by_image[frame['id']]
                union = decode_union(frame, instances)
                mask_entry = f'SFISHTRACK/Masks/masks_{video_id:03d}/{frame["file_name"]}'
                mask_raw = archive.read(mask_entry)
                validate_supplied_mask(mask_raw, union)
                rgb_path = folder / 'rgb' / PurePosixPath(rgb_entry).name
                rgb_path.write_bytes(rgb_raw)
                mask_path = folder / 'gt_fish' / f'{index:06d}.png'
                Image.fromarray(union.astype(np.uint8)).save(mask_path)
                old_rgb = old_root / Path(rgb_entry).relative_to('SFISHTRACK')
                records.append({'index': index, 'coco_image_id': frame['id'], 'annotation_frame_id': frame['frame_id'],
                                'original_file_name': frame['file_name'], 'source_filename_index': frame_number(frame['file_name']),
                                'archive_rgb_member': rgb_entry, 'archive_mask_member': mask_entry,
                                'rgb_path': rgb_path.relative_to(output).as_posix(), 'mask_path': mask_path.relative_to(output).as_posix(),
                                'width': rgb.width, 'height': rgb.height, 'source_file_sha256': sha(rgb_raw),
                                'rgb_pixels_sha256': sha(rgb.tobytes()), 'mask_file_sha256': file_sha(mask_path),
                                'supplied_mask_sha256': sha(mask_raw), 'fish_pixels': int(union.sum()),
                                'instances': [{'id': r['id'], 'track_id': r['track_id'], 'bbox': r['bbox'], 'area': r['area']} for r in instances],
                                'historical_rgb_identical': file_sha(old_rgb) == sha(rgb_raw) if old_rgb.exists() else None,
                                'source_pts': None})
            video_entry = f'SFISHTRACK/Videos/video_{video_id:03d}.mp4'
            video_path = folder / f'video_{video_id:03d}.mp4'
            with archive.open(video_entry) as source, video_path.open('xb') as target:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    target.write(block)
            record = {'video_id': video_id, 'role': role, 'frame_count': len(records), 'reference_kind': 'unaltered publisher RGB',
                      'video_path': video_path.relative_to(output).as_posix(), 'video_sha256': file_sha(video_path),
                      'annotation_sha256': sha(archive.read(f'SFISHTRACK/Annotations/video{video_id:03d}.json')),
                      'bbox_convention': 'xmin,ymin,xmax-xmin,ymax-ymin; inclusive COCO width/height are +1',
                      'timing_status': 'unverified_until_separate_pixel_and_pts_audit', 'original_resolution_preserved': True, 'frames': records}
            sequences.append(record)
            print(f'Prepared fish {video_id:03d}: {len(records)} RGB/RLE/supplied-mask pairs ({role})', flush=True)
    hashes = defaultdict(set)
    for seq in sequences:
        for row in seq['frames']:
            hashes[row['rgb_pixels_sha256']].add(seq['role'])
    if any(len(roles) > 1 for roles in hashes.values()):
        raise ValueError('Pixel-identical fish frame crosses development/reserved split')
    return sequences


def prepare_suim(root: Path, config: dict, output: Path):
    manifest = load_json(root / 'manifest.json')
    candidates = manifest['splits'][config['suim_existing_split']]
    positives = sorted((r for r in candidates if r['class_pixels'][1] > 0),
                       key=lambda r: (r['class_pixels'][1] / (r['width'] * r['height']), r['id']))
    negatives = sorted((r for r in candidates if r['class_pixels'][1] == r['class_pixels'][6] == 0),
                       key=lambda r: sha(r['id'].encode()))
    count = config['suim_diver_count']
    if len(positives) < count or len(negatives) < config['suim_no_fish_or_diver_count']:
        raise ValueError('Insufficient SUIM candidates')
    ranks = np.linspace(0, len(positives) - 1, count).round().astype(int).tolist()
    selected = [positives[i] for i in ranks] + negatives[:config['suim_no_fish_or_diver_count']]
    forbidden = {r['rgb_sha256'] for split in ('train', 'test') for r in manifest['splits'][split]}
    if len({r['rgb_sha256'] for r in selected}) != len(selected) or any(r['rgb_sha256'] in forbidden for r in selected):
        raise ValueError('Duplicate SUIM RGB across selected images or existing splits')
    folder = output / 'suim'
    (folder / 'gt_target').mkdir(parents=True)
    records = []
    for index, row in enumerate(selected):
        rgb_path, labels_path = root / row['image'], root / row['mask_index']
        rgb = Image.open(rgb_path).convert('RGB')
        labels = np.asarray(Image.open(labels_path))
        if rgb.size != (row['width'], row['height']) or labels.shape != (rgb.height, rgb.width):
            raise ValueError('SUIM dimensions changed')
        if sha(str(rgb.size).encode() + rgb.tobytes()) != row['rgb_sha256']:
            raise ValueError('SUIM decoded RGB digest differs from old manifest')
        if not set(np.unique(labels)) <= set(range(8)) | {255}:
            raise ValueError('Unexpected SUIM index labels')
        class_pixels = [int(np.count_nonzero(labels == i)) for i in range(8)]
        if class_pixels != row['class_pixels']:
            raise ValueError('SUIM class counts differ from old manifest')
        target_path = folder / 'gt_target' / f'{index:04d}.png'
        Image.fromarray(np.isin(labels, [1, 6]).astype(np.uint8)).save(target_path)
        records.append({'index': index, 'id': row['id'], 'role': 'development_regression',
                        'sample_kind': 'diver_positive' if row['class_pixels'][1] else 'no_fish_or_diver',
                        'rgb_path': str(rgb_path), 'eight_class_mask_path': str(labels_path),
                        'target_mask_path': target_path.relative_to(output).as_posix(),
                        'rgb_file_sha256': file_sha(rgb_path), 'rgb_pixels_sha256': sha(rgb.tobytes()),
                        'old_manifest_rgb_digest': row['rgb_sha256'], 'eight_class_mask_sha256': file_sha(labels_path),
                        'target_mask_sha256': file_sha(target_path), 'width': rgb.width, 'height': rgb.height,
                        'class_pixels': class_pixels, 'ignore_pixels': int(np.count_nonzero(labels == 255))})
    print(f'Prepared SUIM: {count} diver + {len(selected) - count} no-target images', flush=True)
    return {'source_manifest': str(root / 'manifest.json'), 'source_manifest_sha256': file_sha(root / 'manifest.json'),
            'source_split': config['suim_existing_split'], 'diver_candidate_count': len(positives),
            'negative_candidate_count': len(negatives), 'positive_area_rank_indices': ranks, 'images': records}


def contact_sheet(items, output: Path, filename: str):
    width, height = 480, 295
    canvas = Image.new('RGB', (width * 3, height * ((len(items) + 2) // 3)), '#151e2b')
    for index, (rgb_path, mask_path, label) in enumerate(items):
        rgb = Image.open(rgb_path).convert('RGB')
        mask = np.asarray(Image.open(mask_path)) > 0
        array = np.asarray(rgb).copy()
        array[mask] = (array[mask] * 0.55 + np.array([0, 255, 110]) * 0.45).astype(np.uint8)
        panel = Image.fromarray(array)
        panel.thumbnail((480, 270))
        x, y = (index % 3) * width, (index // 3) * height
        canvas.paste(panel, (x + (width - panel.width) // 2, y + 25))
        ImageDraw.Draw(canvas).text((x + 6, y + 6), label, fill='white')
    path = output / filename
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--old-sfish-root', type=Path, required=True)
    parser.add_argument('--suim-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection', type=Path, default=Path(__file__).with_name('d0_selection.json'))
    args = parser.parse_args()
    config = load_json(args.selection)
    validate_split(config)
    args.output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(args.archive) as archive:
        names, data_by_video, audit = audit_archive(archive, args.old_sfish_root, args.output)
        fish = prepare_fish(archive, names, data_by_video, config, args.output, args.old_sfish_root)
    suim = prepare_suim(args.suim_root, config, args.output)
    manifest = {'schema': 'd0-corpus-v1', 'archive_path': str(args.archive), 'archive_size': args.archive.stat().st_size,
                'archive_full_sha256': None, 'archive_identity_note': 'size and per-member SHA/ZIP CRC; no whole 27 GB SHA scan',
                'selection': config, 'selection_sha256': file_sha(args.selection), 'preparation_script_sha256': file_sha(Path(__file__)),
                'model_loaded': False, 'gpu_used': False, 'fish_sequences': fish, 'suim': suim}
    save_json(args.output / 'manifest.json', manifest)
    fish_items = []
    for seq in fish:
        if seq['role'] != 'development':
            continue
        for index in (0, len(seq['frames']) // 2, len(seq['frames']) - 1):
            frame = seq['frames'][index]
            fish_items.append((args.output / frame['rgb_path'], args.output / frame['mask_path'],
                               f'video{seq["video_id"]:03d} frame {frame["original_file_name"]} GT fish'))
    contact_sheet(fish_items, args.output, 'fish_development_overlays.jpg')
    suim_items = [(Path(row['rgb_path']), args.output / row['target_mask_path'], f'{row["index"]:02d} {row["id"]}')
                  for row in suim['images']]
    contact_sheet(suim_items, args.output, 'suim_development_overlays.jpg')
    summary = {'fish_development_frames': sum(s['frame_count'] for s in fish if s['role'] == 'development'),
               'fish_reserved_frames': sum(s['frame_count'] for s in fish if s['role'] != 'development'),
               'validated_instance_masks': sum(len(f['instances']) for s in fish for f in s['frames']),
               'suim_images': len(suim['images']), 'all_annotation_rgb_names_resolved': audit['annotation_images'],
               'fish_pixel_duplicates_across_splits': 0, 'suim_selected_duplicates_or_cross_split_hashes': 0,
               'original_dimensions_preserved': True, 'timing_audit_pending': True,
               'output': str(args.output), 'manifest_sha256': file_sha(args.output / 'manifest.json')}
    save_json(args.output / 'preparation_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
