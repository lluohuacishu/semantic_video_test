"""CPU-only publication of completed candidate and current-source evidence."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics

HERE=Path(__file__).resolve().parent
REPO=HERE.parent


def load(p):
    return json.loads(p.read_text('utf-8-sig'))


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw',type=Path,required=True)
    ap.add_argument('--destination',type=Path,required=True)
    args=ap.parse_args()
    dest=args.destination
    dest.mkdir(parents=True,exist_ok=False)
    raw=args.raw
    receipts={}

    def read(relative):
        p=raw/relative
        receipts[relative]=dict(bytes=p.stat().st_size,sha256=sha(p))
        return load(p)

    screens=[]
    for folder in ('screen_layernorm','screen_tf32','screen_bf16_r1','screen_uniform_bf16_r2'):
        value=read(folder+'/result.json')
        screens.append({k:value[k] for k in ('mode','frames','baseline_ms','candidate_ms','speedup',
            'max_rgb_difference','mean_abs_rgb_difference','candidate','baseline_engine_sha256','source_tensor_unchanged')})
    short=read('full66/summary.json')
    long=read('full900/summary.json')
    for label,campaign in [('full66',short),('full900',long)]:
        if not campaign['complete'] or not campaign['all_reference_states_equal'] or not campaign['all_entropy_symbols_and_cdf_equal']:
            raise ValueError('Incomplete validation campaign')
        runs=[read(f'{label}/{run["name"]}/result.json') for run in campaign['runs']]
        first=runs[0]
        for d in runs:
            if not d['reference_verified'] or not d['source_media_reads_forbidden']:
                raise ValueError('Independent recovery validation missing')
            for key in ('stream_sha256','reference_checksums','entropy_symbols_and_cdf_sha256'):
                if d[key]!=first[key]:raise ValueError('Independent campaign differs')
            for p,value in d['frozen_code_sha256'].items():
                if sha(REPO/p)!=value:raise ValueError('Frozen source changed')
        for a,b in [(runs[0],runs[3]),(runs[1],runs[2])]:
            if [r['rgb_sha256'] for r in a['records']]!=[r['rgb_sha256'] for r in b['records']]:
                raise ValueError('Per-backend RGB did not repeat')
    current=read('current900_trt/summary.json')
    sender=read('current900_trt/sender/result.json')
    receiver=read('current900_trt/receiver/result.json')
    if not current['complete'] or sender['reference_checksums']!=receiver['reference_checksums']:
        raise ValueError('New accelerated stream not independently recovered')
    quality20=read('quality_screen20/summary.json')
    quality66=read('quality66/summary.json')
    quality900=read('quality900/summary.json')
    if abs(quality900['summaries']['eager']['lpips_alex']-.139663)>1e-6 or abs(quality900['summaries']['eager']['dists']-.149056)>1e-6:
        raise ValueError('Historical 900-frame perceptual point did not reproduce')
    rows=list(csv.DictReader((raw/'quality900/per_frame.csv').open(encoding='utf-8-sig')))
    if len(rows)!=3600:raise ValueError('Expected four complete 900-frame evaluations')
    indexed={}
    for r in rows:indexed.setdefault(r['name'],{})[int(r['index'])]=r
    deltas={}
    for name,group in indexed.items():
        if sorted(group)!=list(range(900)):raise ValueError('Missing/duplicate quality frame')
        deltas[name]={}
        for metric in ('lpips_alex','dists'):
            values=[float(group[i][metric])-float(indexed['eager'][i][metric]) for i in range(900)]
            deltas[name][metric]=dict(mean=statistics.mean(values),minimum=min(values),maximum=max(values),
                maximum_index=max(range(900),key=values.__getitem__))
    summary=dict(date='2026-11-05',screens=screens,full66=short,full900=long,current900_trt=current,
        quality20=quality20['summaries'],quality66=quality66['summaries'],quality900=quality900['summaries'],
        quality900_by_kind=quality900['summaries_by_kind'],quality900_deltas_vs_eager=deltas,
        new_candidate_adopted=False,production_changed=False,training_steps=0,
        conclusion='The new normalization+Graph candidate shows only about 1.4% median core throughput gain on 900 frames; not adopted. Existing TRT encoding/decoding combination independently validated on the current source.')
    (dest/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for folder in ('quality_screen20','quality66','quality900'):
        p=raw/folder/'per_frame.csv'
        receipts[str(p.relative_to(raw))]=dict(sha256=sha(p),bytes=p.stat().st_size)
        shutil.copy2(p,dest/(folder+'_per_frame.csv'))
    executed=dest/'executed_code'
    executed.mkdir()
    for p in HERE.glob('*.py'):shutil.copy2(p,executed/p.name)
    failures=[]
    logroot=raw.parent
    for name,note in [
        ('bf16_screen_20261105_01.log','Conv-17 ONNX type contract rejected BF16; corrected through full opset22 conversion.'),
        ('uniform_bf16_screen_20261105_01.log','Pinned parser rejected BF16 InstanceNormalization; corrected by explicit original FP32 normalization.'),
        ('uniform_bf16_screen_20261105_r1.log','Pinned builder rejected BF16 Resize; corrected by FP32 nearest Resize before BF16 convolution.')]:
        p=logroot/name
        failures.append(dict(log=str(p),sha256=sha(p),note=note,counted_as_success=False))
    manifest=dict(raw_directory=str(raw),receipts=receipts,failures=failures,
        producer_sha256=sha(Path(__file__)),final_code_sha256={p.name:sha(p) for p in HERE.glob('*.py')},
        earlier_executed_code_snapshots=['full66/executed_code','full900/executed_code',
            'candidate_backend_v1.py','candidate_backend_v2.py','uniform_bf16_backend_v1.py','uniform_bf16_backend_v2.py'],
        verification=dict(quality_rows=3600,historical_eager_perceptual_point_reproduced=True,
            independent_references_and_entropy_equal=True,per_backend_rgb_repeatable=True,
            original_weights_and_production_unchanged=True))
    (dest/'execution_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PASS publication: completed screens, two ABBA campaigns, new independent stream, 3600 quality records')


if __name__=='__main__':
    main()
