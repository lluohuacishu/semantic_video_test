"""Inventory all 54 existing SFishTrack clips using prior audit and ZIP names.

Reads three representative JPEGs per clip, not the entire 27GB archive. Source
metadata conflicts remain explicit; screenshots cannot prove source identity.
"""
from __future__ import annotations
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot_common import file_sha, load_json, save_json, settings, sha


def group(video_id, old):
    if video_id == 12:
        return 'quarantine_invalid_date'
    date = old['Date of recording'].replace(' ', '')
    if date.endswith('2025'):
        return 'old_metadata_2025'
    if date.endswith('2022'):
        return 'old_metadata_2022'
    if date.endswith('2023'):
        return 'old_metadata_2023_ambiguous_day_month'
    raise ValueError('Unclassified old metadata date')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    workspace = settings()['workspace']
    corpus = workspace / '02_实验数据/D_语义目标增强/D0_20261021_r1'
    ready = load_json(corpus / 'ready_manifest.json')
    prior = load_json(corpus / 'archive_audit.json')
    old_root = Path('C:/Users/lenovo/Desktop/works/work/semantic-video-mlvc/data/sfishtrack_official/selected/SFISHTRACK/metadata')
    prior_by_id = {r['video_id']: r for r in prior['sequences']}
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    pages = [Image.new('RGB', (1440, 6 * 116), '#101820') for _ in range(3)]
    rows, counts, sizes = [], Counter(), Counter()
    with zipfile.ZipFile(ready['archive_path']) as archive:
        catalog = [r for r in archive.infolist() if not r.is_dir()]
        for entry in catalog:
            category = entry.filename.split('/')[1]
            counts[category] += 1
            sizes[category] += entry.compress_size
        for video_id in range(1, 55):
            old_path = old_root / f'video_{video_id:03d}.json'
            old = load_json(old_path)
            prefix = f'SFISHTRACK/Frames/video_{video_id:03d}/'
            entries = sorted([r for r in catalog if r.filename.startswith(prefix) and Path(r.filename).suffix.lower() in {'.jpg', '.jpeg', '.png'}],
                             key=lambda r: int(Path(r.filename).stem.replace('frame_', '')))
            if not entries:
                raise ValueError(f'No source RGB frames video{video_id:03d}')
            indices = [int(Path(r.filename).stem.replace('frame_', '')) for r in entries]
            duplicates = [i for i, n in Counter(indices).items() if n > 1]
            gaps = sorted(set(range(min(indices), max(indices)+1)) - set(indices))
            sample_indices = [0, len(entries)//2, len(entries)-1]
            page = pages[(video_id - 1) // 18]
            cell = (video_id - 1) % 18
            x, y = (cell % 3) * 480, (cell // 3) * 116
            ImageDraw.Draw(page).text((x+4, y+1), f'{video_id:03d}  RGB {len(entries)}  {group(video_id, old)}', font=font, fill='white')
            samples = []
            for j, index in enumerate(sample_indices):
                raw = archive.read(entries[index])
                image = Image.open(io.BytesIO(raw)).convert('RGB')
                rgb = np.asarray(image)
                page.paste(image.resize((160, 90), Image.Resampling.BILINEAR), (x+j*160, y+26))
                samples.append(dict(index=indices[index], member=entries[index].filename, file_sha256=sha(raw), rgb_sha256=sha(rgb.tobytes())))
            rows.append(dict(video_id=video_id, rgb_frames=len(entries), annotated_frames=prior_by_id[video_id]['image_count'],
                             first_frame_index=min(indices), last_frame_index=max(indices), rgb_index_gaps=gaps, duplicate_frame_indices=duplicates,
                             conservative_group=group(video_id, old), old_metadata=old, old_metadata_sha256=file_sha(old_path),
                             current_archive_metadata=prior_by_id[video_id]['metadata'], source_identity_verified=False,
                             samples=samples, rgb_member_bytes=sum(r.compress_size for r in entries)))
    for i, page in enumerate(pages, start=1):
        page.save(args.output / f'sfish_all54_page{i}.jpg', quality=92)
    groups = {}
    for row in rows:
        groups.setdefault(row['conservative_group'], []).append(row['video_id'])
    report = dict(schema='distill-round2-all54-inventory-v1', archive_bytes=ready['archive_size'],
                  total_clips=len(rows), total_rgb_frames=sum(r['rgb_frames'] for r in rows),
                  total_annotated_frames=sum(r['annotated_frames'] for r in rows),
                  zip_member_counts=dict(counts), compressed_bytes_by_category=dict(sizes), groups=groups, clips=rows,
                  prior_audit_sha256=file_sha(corpus / 'archive_audit.json'),
                  scope='all ZIP names + existing all54 JSON audit + 3 JPEGs per clip; no whole archive rehash or full CRC rescan',
                  source_independence_established=False,
                  metadata_conflict='New ZIP metadata is identical across clips and conflicts with old selected metadata; old dates are conservative hints only')
    save_json(args.output / 'inventory.json', report)
    print(json.dumps({k: report[k] for k in ['total_clips', 'total_rgb_frames', 'total_annotated_frames', 'groups', 'compressed_bytes_by_category']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
