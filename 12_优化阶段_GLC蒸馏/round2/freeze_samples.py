"""Freeze training-only exact (latent, original, teacher) supervision dedup."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot_common import load_json, save_json, file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    target = args.output / 'sampling_pairs.json'
    if target.exists():
        raise FileExistsError(target)
    plan = load_json(args.output / 'plan.json')
    seen, pairs, duplicates = {}, [], []
    for clip in [c for c in plan['clips'] if c['role'] == 'train']:
        folder = args.output / clip['id']
        cached = load_json(folder / 'decode_cache.json')
        source = load_json(folder / 'source_manifest.json')
        for row, rgb in zip(cached['records'], source['records']):
            if row['kind'] == 0:
                continue
            key = (row['latent_input_sha256'], rgb['resized_rgb_sha256'], row['teacher_rgb_sha256'])
            pair = [clip['id'], row['index']]
            if key in seen:
                duplicates.append(dict(kept=seen[key], removed=pair))
            else:
                seen[key] = pair
                pairs.append(pair)
    value = dict(plan_sha256=file_sha(args.output / 'plan.json'), pairs=pairs,
                 candidate_p_frames=sum(c['frames']-((c['frames']+63)//64) for c in plan['clips'] if c['role'] == 'train'),
                 unique_supervision_pairs=len(pairs), exact_duplicate_supervision=duplicates,
                 method='exact latent input + original RGB + teacher RGB hashes; training split only; fixed before optimizer updates')
    save_json(target, value)
    print(f'Frozen sampling: {len(pairs)} unique P supervision pairs; removed {len(duplicates)} exact repeats', flush=True)


if __name__ == '__main__':
    main()
