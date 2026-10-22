"""Bounded D2.1 size-only selection with old streams reused in place."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import numpy as np
from prepare_d2 import read_json, save_json
from run_d2_lowrate import dispatch, point, next_q


def candidate_q(records, ceiling, prior):
    valid=[r for r in records if r['file_bytes']<=ceiling]
    if not valid:return None
    best=max(valid,key=lambda p:(p['file_bytes'],-p['q']))
    if (ceiling-best['file_bytes'])/ceiling<=.02:return None
    low=max(r['q'] for r in valid)
    high=min((r['q'] for r in records if r['file_bytes']>ceiling and r['q']>low),default=64)
    available=[q for q in range(low+1,high) if q not in {r['q'] for r in records}]
    if not available:return None
    ordered=sorted(records,key=lambda r:r['q']);left,right=ordered[0],ordered[-1]
    slope=prior
    if right['q']>left['q'] and right['file_bytes']>left['file_bytes']:
        slope=math.log(right['file_bytes']/left['file_bytes'])/(right['q']-left['q'])
    estimate=best['q']+math.log(ceiling/best['file_bytes'])/slope
    return min(available,key=lambda q:(abs(q-estimate),q))


def run(args):
    root=args.run.resolve();source=args.source.resolve();plan=read_json(args.plan)
    if (root/'codec_plan.json').exists():raise FileExistsError('D2.1 already dispatched; inspect its checkpoint')
    diagnosis=read_json(root/'cache_diagnosis.json')
    if not diagnosis['cached_C_alpha_and_input_exactly_reproduced'] or diagnosis['frames']!=128:raise ValueError('Cache diagnosis incomplete')
    if diagnosis['clips']['11']['controls']['F']['fallback_counts']!={'protection_exceeds_half_frame':64}:raise ValueError('Unexpected011 fish-only behavior')
    save_json(root/'codec_plan.json',plan)
    gate=read_json(root/'rate_gate.json');clips={str(c['video_id']):c for c in read_json(root/'inputs.json')['clips']}
    runtime=read_json(root/'runtime.json');old=read_json(source/'rate_selection.json')
    for key in clips:
        if gate['anchors'][key]['S0']!=plan['frozen_S0'][key] or gate['anchors'][key]['ceiling']!=plan['effective_caps'][key]:raise ValueError('Frozen budget mismatch')
    selected={k:{m:dict(v['selected']) for m,v in methods.items()} for k,methods in old.items()}
    registry={k:{m:list(v['tested']) for m,v in methods.items()} for k,methods in old.items()}
    new_encodes=[];decodes=[]
    def encode(key,method,q):
        if len(new_encodes)>=plan['max_new_encodes']:raise RuntimeError('Registered encode limit reached')
        clip=clips[key];width,height=gate['size'];label=f'v{int(key):03d}_{width}x{height}_{method}_q{q:02d}'
        if method=='A':frames=[f['resized'][f'{width}x{height}']['rgb'] for f in clip['frames']]
        else:
            base=source if method=='B' else root
            frames=[str(base/'controls'/f'v{int(key):03d}'/method/f'{f["index"]:04d}.png') for f in clip['frames']]
        task={'id':label,'video_id':int(key),'method':method,'q':q,'width':width,'height':height,'fps':clip['fps'],
              'frames':frames,'output':str(root/'points'/label/'encoder')}
        dispatch(root,runtime,'encode',[task],label+'_encode')
        result=point(root,task);new_encodes.append(result);save_json(root/'new_encodes.json',new_encodes)
        print(f'BYTES {label} {result["file_bytes"]}',flush=True)
        return result
    for group in plan['comparator_refinement']['groups']:
        key=str(group['video_id']);method=group['method'];records=registry[key][method];cap=gate['anchors'][key]['ceiling']
        for _ in range(plan['comparator_refinement']['max_new_points_each']):
            q=next_q(records,cap)
            if q is None:break
            records.append(encode(key,method,q))
        selected[key][method]=max((r for r in records if r['file_bytes']<=cap),key=lambda r:(r['file_bytes'],-r['q']))
        save_json(root/'selection_progress.json',{'selected':selected,'registry':registry})
    matching={}
    for group in plan['new_methods']:
        key=str(group['video_id']);method=group['method'];cap=gate['anchors'][key]['ceiling']
        target=min(selected[key][m]['file_bytes'] for m in 'AB')
        b=registry[key]['B'];prior=float(np.polyfit([p['q'] for p in b],np.log([p['file_bytes'] for p in b]),1)[0])
        if prior<=0:raise ValueError('Nonpositive registered size prior')
        records=[encode(key,method,0)];registry[key][method]=records
        while len(records)<plan['candidate_selection']['max_points_each']:
            q=candidate_q(records,target,prior)
            if q is None:break
            records.append(encode(key,method,q))
        paired=[p for p in records if p['file_bytes']<=target];eligible=[p for p in records if p['file_bytes']<=cap]
        best=max(paired,key=lambda p:(p['file_bytes'],-p['q'])) if paired else min(eligible,key=lambda p:(p['file_bytes'],p['q'])) if eligible else None
        if best:selected[key][method]=best
        matching[f'{key}_{method}']={'target_bytes':target,'hard_cap':cap,'selected':best,'quality_used':False,'no_compliant_output':not eligible}
        save_json(root/'selection_progress.json',{'selected':selected,'registry':registry,'matching':matching})
    # Exact fallback identity was established over all64 source frames before codec execution.
    selected['11']['F']={**selected['11']['C'],'method':'F','alias_of':'C','input_identical_all_frames':True}
    entries=[]
    def entry(key,method,p,category):
        if not p:return
        if p['file_bytes']>gate['anchors'][key]['ceiling']:return
        entries.append({'video_id':int(key),'method':method,'category':category,'point':dict(p)})
    for key,methods in selected.items():
        for method,p in methods.items():entry(key,method,p,'selected')
    for key in clips:
        for method in 'AB':entry(key,method,next(p for p in registry[key][method] if p['q']==0),'q0')
        c0=next(p for p in registry[key]['C'] if p['q']==0)
        entry(key,'C',c0,'q0')
        if key=='11':entry(key,'F',{**c0,'method':'F','alias_of':'C','input_identical_all_frames':True},'q0')
        else:entry(key,'F',next(p for p in registry[key]['F'] if p['q']==0),'q0')
        entry(key,'O',next(p for p in registry[key]['O'] if p['q']==0),'q0')
    # D2's011 C@q0 and A@q0 are the same input/codec. Reuse A's decoder for both.
    for e in entries:
        p=e['point']
        if e['video_id']==11 and p['q']==0 and e['method'] in ('C','F'):
            e['point']={**next(p for p in registry['11']['A'] if p['q']==0),'alias_of':e['method'],'method':e['method']}
    unique={p['output']:p for p in (e['point'] for e in entries)}
    for output,p in unique.items():
        encoder=Path(output);old_decoder=encoder.parent/'decoder'
        if (old_decoder/'decode.json').exists():decoder=old_decoder
        else:
            decoder=root/'decodes'/p['id']
            decodes.append({'id':p['id'],'input':str(encoder/'stream.glcv'),'output':str(decoder)})
        for e in entries:
            if e['point']['output']==output:e['decoder']=str(decoder)
    if len(decodes)>plan['max_new_independent_decodes']:raise RuntimeError('Registered decode limit exceeded')
    save_json(root/'selection.json',{'selected':selected,'matching':matching,'entries':entries,'registry':registry,
                                  'new_encodes':len(new_encodes),'new_decodes':len(decodes),'quality_used':False})
    if decodes:dispatch(root,runtime,'decode',decodes,'selected_and_q0_independent_decode')
    save_json(root/'progress.json',{'state':'codec_complete','new_encodes':len(new_encodes),'new_decodes':len(decodes),
                                 'unique_evaluated_streams':len(unique),'comparison_entries':len(entries)})
    print('D2.1 bounded codec completed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--plan',type=Path,default=Path(__file__).with_name('d21_codec_plan_v1.json'));args=p.parse_args()
    try:run(args)
    except BaseException as exc:
        save_json(args.run/'d21_codec_failure.json',{'type':type(exc).__name__,'message':str(exc)})
        raise
