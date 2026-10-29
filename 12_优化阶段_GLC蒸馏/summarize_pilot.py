"""Publish small numerical evidence; never copy datasets, videos or weights to Git."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from pilot_common import HERE, file_sha, load_json, save_json


def finalize_publication(destination):
    # Git stores this repository's text as LF. Hash that exact representation,
    # while leaving all original execution artifacts and their raw hashes alone.
    for path in destination.iterdir():
        if path.suffix not in {'.json', '.svg'}:
            continue
        content = path.read_text('utf-8')
        if path.suffix == '.svg':
            content = '\n'.join(line.rstrip() for line in content.splitlines()) + '\n'
        with path.open('w', encoding='utf-8', newline='\n') as stream:
            stream.write(content)
    manifest = dict(
        representation='UTF-8 LF; SVG trailing whitespace removed. evidence_manifest.json records original local execution-artifact hashes, before publication normalization.',
        published_script_sha256={p.name: file_sha(p) for p in HERE.glob('*.py')},
        files={p.name: file_sha(p) for p in destination.iterdir() if p.is_file() and p.name != 'publication_manifest.json'})
    with (destination / 'publication_manifest.json').open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination = HERE / 'evidence_pilot_20261029'
    destination.mkdir(exist_ok=False)
    plan = load_json(args.output / 'plan.json')
    quality = load_json(args.output / 'evaluation/quality.json')
    timing = load_json(args.output / 'evaluation/timings.json')
    audit = load_json(args.output / 'input_audit.json')
    training = {}
    provenance = {}
    for mode in ['kd', 'gt']:
        folder = args.output / f'train_{mode}'
        summary = load_json(folder / 'summary.json')
        rows = [json.loads(line) for line in (folder / 'loss.jsonl').read_text('utf-8').splitlines()]
        if summary['status'] != 'complete' or [r['step'] for r in rows] != list(range(1, 5001)):
            raise ValueError('Only complete contiguous 5000-step logs can be published by this pilot summarizer')
        if any(r['learning_rate'] <= 0 or r['frame_index'] % 64 == 0 for r in rows):
            raise ValueError('Zero-learning-rate or I-frame training unexpectedly found')
        for r in rows:
            if r['clip'] not in [f'v{i:03d}' for i in plan['train_video_ids']]:
                raise ValueError('Non-training source in actual optimizer log')
        summary['last100_mean_l1_original'] = float(np.mean([r['l1_original'] for r in rows[-100:]]))
        summary['maximum_training_step_seconds'] = max(r['seconds'] for r in rows)
        summary['steps_over_half_second'] = sum(r['seconds'] > .5 for r in rows)
        summary['minimum_nonzero_learning_rate'] = min(r['learning_rate'] for r in rows)
        training[mode] = summary
        save_json(destination / f'validation_{mode}.json', load_json(folder / 'validation.json'))
        blocks = []
        for start in range(0, len(rows), 100):
            subset = rows[start:start+100]
            blocks.append(dict(steps=[subset[0]['step'], subset[-1]['step']],
                               mean_l1_original=float(np.mean([r['l1_original'] for r in subset])),
                               mean_optimized_loss=float(np.mean([r['loss'] for r in subset])),
                               mean_seconds=float(np.mean([r['seconds'] for r in subset]))))
        save_json(destination / f'loss_blocks_{mode}.json', blocks)
        provenance[f'{mode}_raw_loss_jsonl_sha256'] = file_sha(folder / 'loss.jsonl')
        provenance[f'{mode}_raw_summary_sha256'] = file_sha(folder / 'summary.json')
    if training['kd']['initial_model_sha256'] != training['gt']['initial_model_sha256'] or training['kd']['sampling_sha256'] != training['gt']['sampling_sha256']:
        raise ValueError('Control pairing failed')
    clips = []
    for clip in quality['clips']:
        clips.append({**{k: v for k, v in clip.items() if k != 'models'},
                      'models': {name: {k: v for k, v in value.items() if k != 'frames'} for name, value in clip['models'].items()}})
    exports = [{k: v for k, v in row.items() if k != 'path'} for row in quality['exports']]
    cache = []
    codec_identity = None
    for clip in plan['clips']:
        folder = args.output / clip['id']
        enc, dec = load_json(folder / 'encode_cache.json'), load_json(folder / 'decode_cache.json')
        if codec_identity is None:
            codec_identity = {k: enc['header'][k] for k in ['revision', 'entropy', 'table_sha256', 'weights', 'arithmetic']}
        cache.append(dict(id=clip['id'], encode_seconds=enc['seconds'], decode_and_replay_seconds=dec['seconds'],
                          exact_replay_matches=dec['cache_replay_exact_matches'], stream_sha256=dec['stream_sha256']))
    summary = dict(date='2026-10-29', pilot_only=True, production_replaced=False, orin_tested=False,
                   plan=plan, codec_identity=codec_identity, input_audit=audit, cache=cache, training=training, quality_clips=clips,
                   exports=exports, timing=timing, limitations=quality['limitations'], raw_log_provenance=provenance)
    attention_path = args.output / 'train_eval_attention_audit.json'
    if attention_path.exists():
        summary['attention_mode_diagnostic'] = load_json(attention_path)
    save_json(destination / 'summary.json', summary)
    for name in ['learning_curves.png', 'learning_curves.svg', 'frame_metrics_v007.png', 'frame_metrics_v007.svg', 'evidence_manifest.json', 'selection_frozen.json']:
        shutil.copyfile(args.output / 'evaluation' / name, destination / name)
    finalize_publication(destination)


if __name__ == '__main__':
    main()
