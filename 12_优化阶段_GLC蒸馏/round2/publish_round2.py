"""Publish numerical results/plots, never source images, streams or weights."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pilot_common import file_sha, load_json, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination = HERE/'evidence_20261029'
    destination.mkdir(exist_ok=False)
    plan = load_json(args.output/'plan.json')
    sampling = load_json(args.output/'sampling_pairs.json')
    quality = load_json(args.output/'evaluation/quality.json')
    timing = load_json(args.output/'evaluation/timings.json')
    selection = load_json(args.output/'evaluation/selection_frozen.json')
    audit = load_json(args.output/'sfish_audit/inventory.json')
    transfer = load_json(args.output/'initialization/transfer_audit.json')
    cases = list(plan['training']['cases'])
    training, provenance = {}, {}
    for case in cases:
        folder = args.output/f'train_{case}'
        summary = load_json(folder/'summary.json')
        rows = [json.loads(x) for x in (folder/'loss.jsonl').read_text('utf-8').splitlines()]
        if summary['status']!='complete' or [r['step'] for r in rows]!=list(range(1,5001)):
            raise ValueError('Only complete contiguous four-case evidence may be published')
        if any(r['learning_rate']<=0 or r['frame_index']%64==0 for r in rows):
            raise ValueError('Unexpected zero LR or I training')
        expected_ids = {c['id'] for c in plan['clips'] if c['role']=='train'}
        if any(r['clip'] not in expected_ids for r in rows):
            raise ValueError('Non-training clip entered optimizer log')
        summary['last100_original_l1'] = float(np.mean([r['l1_original'] for r in rows[-100:]]))
        summary['last100_teacher_l1'] = float(np.mean([r['l1_teacher'] for r in rows[-100:]]))
        training[case] = summary
        save_json(destination/f'validation_{case}.json', load_json(folder/'validation.json'))
        blocks = []
        for start in range(0,len(rows),100):
            subset = rows[start:start+100]
            blocks.append(dict(steps=[subset[0]['step'],subset[-1]['step']],
                               mean_original_l1=float(np.mean([r['l1_original'] for r in subset])),
                               mean_teacher_l1=float(np.mean([r['l1_teacher'] for r in subset])),
                               mean_seconds=float(np.mean([r['seconds'] for r in subset]))))
        save_json(destination/f'loss_blocks_{case}.json',blocks)
        provenance[f'{case}/loss.jsonl'] = file_sha(folder/'loss.jsonl')
        provenance[f'{case}/best.pt'] = file_sha(folder/'best.pt')
    small_quality = []
    for clip in quality['clips']:
        small_quality.append({**{k:v for k,v in clip.items() if k!='models'},
                              'models':{name:{k:v for k,v in value.items() if k!='frames'} for name,value in clip['models'].items()}})
    cache_summary = []
    for clip in plan['clips']:
        path = args.output/clip['id']
        dec, enc = load_json(path/'decode_cache.json'), load_json(path/'encode_cache.json')
        if dec['cache_replay_exact_matches']!=clip['frames']:
            raise ValueError('Incomplete exact replay')
        cache_summary.append(dict(id=clip['id'],role=clip['role'],frames=clip['frames'],
                                  stream_sha256=dec['stream_sha256'],encode_seconds=enc['seconds'],
                                  decode_and_replay_seconds=dec['seconds'],exact_replay_matches=dec['cache_replay_exact_matches'],
                                  reused_previous_run=bool(clip.get('reused_original_run'))))
    public_plan = {**plan,'clips':[{k:v for k,v in clip.items() if k!='reused_original_run'} for clip in plan['clips']]}
    report = dict(schema='glc-distillation-round2-public-v1',date='2026-10-29',pilot_only=True,
                  production_replaced=False,orin_tested=False,plan=public_plan,
                  source_inventory={k:audit[k] for k in ['total_clips','total_rgb_frames','total_annotated_frames','archive_bytes','compressed_bytes_by_category','groups','metadata_conflict']},
                  sampling={k:v for k,v in sampling.items() if k not in ['pairs','exact_duplicate_supervision']},
                  removed_duplicate_supervision_pairs=len(sampling['exact_duplicate_supervision']),
                  training=training,selection=selection,quality_clips=small_quality,timing=timing,
                  initialization=transfer,cache=cache_summary,exports=quality['exports'],raw_artifact_sha256=provenance,
                  source_independence_established=False,source_caveats=plan['source_caveats'])
    save_json(destination/'summary.json',report)
    for name in ['learning_curves_round2.png','learning_curves_round2.svg','quality_comparison_round2.png','quality_comparison_round2.svg','selection_frozen.json','execution_manifest.json']:
        shutil.copyfile(args.output/'evaluation'/name,destination/name)
    # Hash Git's LF representation; original execution hashes stay explicitly raw.
    for path in destination.iterdir():
        if path.suffix not in ['.json','.svg']:
            continue
        content = path.read_text('utf-8')
        if path.suffix=='.svg':
            content='\n'.join(line.rstrip() for line in content.splitlines())+'\n'
        with path.open('w',encoding='utf-8',newline='\n') as stream:
            stream.write(content)
    scripts = list(HERE.glob('*.py'))+[HERE.parent/n for n in ['pilot_common.py','cache_pilot.py','train_pilot.py','evaluate_pilot.py']]
    manifest = dict(representation='UTF-8 LF; SVG trailing whitespace removed. Raw artifact hashes refer to original local execution files.',
                    published_script_sha256={str(p.relative_to(HERE.parent)).replace('\\','/'):file_sha(p) for p in scripts},
                    files={p.name:file_sha(p) for p in destination.iterdir() if p.is_file()})
    with (destination/'publication_manifest.json').open('w',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print('Round2 numerical publication prepared',flush=True)


if __name__=='__main__':
    main()
