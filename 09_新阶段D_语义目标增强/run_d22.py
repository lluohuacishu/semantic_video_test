"""One target-smoothing parameter, fixed masks, cached comparators and bounded q."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from prepare_d2 import read_json,save_json,blend
from run_d21_codec import candidate_q
from run_d2_lowrate import dispatch,point
from finish_d2_lowrate import stats
from evaluate_d2 import region


def prepare(args):
    root=args.run.resolve();d21=args.d21.resolve();d2=args.d2.resolve();plan=read_json(args.plan)
    root.mkdir(parents=True,exist_ok=False);save_json(root/'plan.json',plan)
    save_json(root/'sources.json',{'d21':str(d21),'d2':str(d2)})
    for name in ('inputs.json','rate_gate.json','runtime.json'):save_json(root/name,read_json(d21/name))
    gate=read_json(root/'rate_gate.json');clips=read_json(root/'inputs.json')['clips']
    if gate['size']!=plan['size'] or [c['video_id'] for c in clips]!=plan['clips']:raise ValueError('Registered scope mismatch')
    crows={(r['video_id'],r['index']):r for r in read_json(d2/'controls.json')['records']}
    orows={(r['video_id'],r['index']):r for r in read_json(d21/'cache_frame_metrics.json')}
    records=[];start=time.perf_counter()
    for clip in clips:
        vid=clip['video_id'];key=str(vid)
        if len(clip['frames'])!=64 or clip['fps']!=plan['fps'] or gate['anchors'][key]['S0']!=plan['frozen_S0'][key] or gate['anchors'][key]['ceiling']!=plan['effective_caps'][key]:raise ValueError('Timeline or frozen budget changed')
        for method in ('W','P','G'):(root/'controls'/f'v{vid:03d}'/method).mkdir(parents=True)
        for f in clip['frames']:
            if (root/'STOP').exists():raise RuntimeError('Stopped at preparation frame boundary')
            index=f['index'];name=f'{index:04d}.png';folder=d2/'controls'/f'v{vid:03d}'
            rgb=np.asarray(Image.open(f['resized']['512x288']['rgb']).convert('RGB'));gt=np.asarray(Image.open(f['resized']['512x288']['gt']))!=0
            background=np.asarray(Image.open(folder/'B'/name).convert('RGB'))
            weak=np.rint(gaussian_filter(rgb.astype(np.float32),sigma=(plan['target_sigma'],plan['target_sigma'],0),mode='reflect',truncate=4)).clip(0,255).astype(np.uint8)
            images={'W':weak};fallbacks={'W':None};alphas={}
            for method,mask_path,fallback in [('P',folder/'alpha'/name,crows[(vid,index)]['fallback']),
                ('G',d21/'controls'/f'v{vid:03d}'/'alpha_O'/name,orows[(vid,index)]['controls']['O']['fallback'])]:
                alpha=np.asarray(Image.open(mask_path));alphas[method]=alpha
                transformed=rgb.copy() if fallback else blend(weak,background,alpha)
                if fallback:
                    original_control=np.asarray(Image.open(folder/'C'/name).convert('RGB')) if method=='P' else np.asarray(Image.open(d21/'controls'/f'v{vid:03d}'/'O'/name).convert('RGB'))
                    if not np.array_equal(transformed,original_control):raise ValueError('Fallback semantics changed')
                elif not np.array_equal(transformed[alpha==255],weak[alpha==255]):raise ValueError('Target filter mismatch')
                if not fallback and not np.array_equal(transformed[alpha==0],background[alpha==0]):raise ValueError('Background changed')
                images[method]=transformed;fallbacks[method]=fallback
            row={'video_id':vid,'index':index,'methods':{}}
            for method,im in images.items():
                Image.fromarray(im).save(root/'controls'/f'v{vid:03d}'/method/name,compress_level=1)
                row['methods'][method]={'fallback':fallbacks[method],'preprocess_error':{k:{'sse':v[0],'samples':v[1]} for k,v in stats(rgb,im,gt).items()}}
            records.append(row)
        print(f'PREPARED {vid:03d} 64 frames',flush=True)
    summary={}
    for vid in plan['clips']:
        rows=[r for r in records if r['video_id']==vid];summary[str(vid)]={}
        for method in ('W','P','G'):
            summary[str(vid)][method]={'fallback_frames':sum(r['methods'][method]['fallback'] is not None for r in rows),
                'preprocess_error':{k:region(sum(r['methods'][method]['preprocess_error'][k]['sse'] for r in rows),sum(r['methods'][method]['preprocess_error'][k]['samples'] for r in rows)) for k in ('fish','rest','all')}}
    save_json(root/'preparation.json',{'state':'completed','frames':128,'seconds':time.perf_counter()-start,'summary':summary,
        'masks_reused':True,'background_zero_alpha_identical':True,'fallback_passthrough_preserved':True,'new_inference':False})
    save_json(root/'preprocess_frame_metrics.json',records)


def codec(args):
    root=args.run.resolve();plan=read_json(root/'plan.json');sources=read_json(root/'sources.json');d21=Path(sources['d21'])
    if read_json(root/'preparation.json')['state']!='completed':raise ValueError('Preparation incomplete')
    if (root/'codec_started.json').exists():raise FileExistsError('Already started; retain checkpoints')
    save_json(root/'codec_started.json',{'plan':'plan.json','quality_not_read':True})
    prior=read_json(d21/'selection.json');runtime=read_json(root/'runtime.json');gate=read_json(root/'rate_gate.json')
    clips={str(c['video_id']):c for c in read_json(root/'inputs.json')['clips']};registry={};new=[];matching={}
    selected={k:{m:dict(v) for m,v in group.items() if m in ('A','B','C','O')} for k,group in prior['selected'].items()}
    def encode(key,method,q):
        if len(new)>=plan['codec']['max_new_encodes']:raise RuntimeError('Encode bound reached')
        clip=clips[key];label=f'v{int(key):03d}_512x288_{method}_q{q:02d}'
        task={'id':label,'video_id':int(key),'method':method,'q':q,'width':512,'height':288,'fps':clip['fps'],
            'frames':[str(root/'controls'/f'v{int(key):03d}'/method/f'{f["index"]:04d}.png') for f in clip['frames']],
            'output':str(root/'points'/label/'encoder')}
        dispatch(root,runtime,'encode',[task],label+'_encode');p=point(root,task);new.append(p);save_json(root/'new_encodes.json',new)
        print(f'BYTES {label} {p["file_bytes"]}',flush=True);return p
    for group in plan['new_groups']:
        key=str(group['video_id']);method=group['method'];target=min(selected[key][m]['file_bytes'] for m in 'AB')
        b=prior['registry'][key]['B'];slope=float(np.polyfit([p['q'] for p in b],np.log([p['file_bytes'] for p in b]),1)[0])
        if slope<=0:raise ValueError('Nonpositive size prior')
        records=[encode(key,method,0)]
        while len(records)<plan['codec']['max_points_per_group']:
            q=candidate_q(records,target,slope)
            if q is None:break
            records.append(encode(key,method,q))
        eligible=[p for p in records if p['file_bytes']<=gate['anchors'][key]['ceiling']]
        matched=[p for p in eligible if 0<=(target-p['file_bytes'])/target<=.02]
        best=max(matched,key=lambda p:(p['file_bytes'],-p['q'])) if matched else min(eligible,key=lambda p:(abs(p['file_bytes']-target),p['file_bytes'],p['q'])) if eligible else None
        registry.setdefault(key,{})[method]=records
        matching[f'{key}_{method}']={'target_bytes':target,'selected':best,'strict_target_match':bool(matched),'no_compliant_output':not eligible,'quality_used':False}
        if best:selected[key][method]=best
        save_json(root/'selection_progress.json',{'registry':registry,'selected':selected,'matching':matching})
    entries=[dict(e) for e in prior['entries'] if e['method'] in ('A','B','C','O')]
    for category in ('selected','q0'):
        old_c=next(e for e in entries if e['category']==category and e['video_id']==11 and e['method']=='C')
        entries.append({**old_c,'method':'P','fallback_alias_of_C':True})
    selected['11']['P']={**selected['11']['C'],'method':'P','alias_of':'C'}
    for group in plan['new_groups']:
        key=str(group['video_id']);method=group['method']
        for category,p in [('selected',selected[key].get(method)),('q0',registry[key][method][0])]:
            if p is not None and p['file_bytes']<=gate['anchors'][key]['ceiling']:
                entries.append({'video_id':int(key),'method':method,'category':category,'point':p,'decoder':str(root/'decodes'/p['id'])})
    tasks=[];seen=set()
    for e in entries:
        decoder=e['decoder']
        if decoder in seen:continue
        seen.add(decoder)
        if not (Path(decoder)/'decode.json').exists():tasks.append({'id':e['point']['id'],'input':str(Path(e['point']['output'])/'stream.glcv'),'output':decoder})
    if len(tasks)>plan['codec']['max_new_independent_decodes']:raise RuntimeError('Decode bound exceeded')
    save_json(root/'selection.json',{'entries':entries,'registry':registry,'selected':selected,'matching':matching,'new_encodes':len(new),'new_decodes':len(tasks),'quality_used':False})
    if tasks:dispatch(root,runtime,'decode',tasks,'selected_and_q0_independent_decode')
    save_json(root/'progress.json',{'state':'codec_complete','new_encodes':len(new),'new_decodes':len(tasks),
        'unique_evaluated_streams':len({e['point']['output'] for e in entries})})


def evaluate(root):
    from finish_d21 import evaluate as base_evaluate
    result=base_evaluate(root)
    for category in ('selected','q0'):
        for vid in (11,21):
            group={p['method']:p for p in result['points'] if p['category']==category and p['video_id']==vid}
            for method in ('W','P','G'):
                if method not in group:continue
                for base in ('A','B','W','C' if method=='P' else 'O'):
                    if base==method or base not in group:continue
                    p=group[method];b=group[base];gap=(b['file_bytes']-p['file_bytes'])/b['file_bytes']
                    result['comparisons'].append({'category':category,'video_id':vid,'candidate':method,'comparator':base,
                        'byte_change_fraction':p['file_bytes']/b['file_bytes']-1,'near_equal_no_larger_2pct':0<=gap<=.02,
                        'fish_delta_db':p['metrics']['delivery']['fish']['psnr_db']-b['metrics']['delivery']['fish']['psnr_db'],
                        'rest_delta_db':p['metrics']['delivery']['rest']['psnr_db']-b['metrics']['delivery']['rest']['psnr_db'],
                        'original_fish_delta_db':p['metrics']['original_upsampled']['fish']['psnr_db']-b['metrics']['original_upsampled']['fish']['psnr_db']})
    result.update(schema='d22-evaluation-v1',single_target_sigma=0.6,background_sigma=1.2,GT_methods=['O','G'],
        masks_and_fallback_unchanged=True,training=False,new_semantic_inference=False)
    save_json(root/'evaluation.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','codec','evaluate'])
    p.add_argument('--run',type=Path,required=True);p.add_argument('--d21',type=Path);p.add_argument('--d2',type=Path)
    p.add_argument('--plan',type=Path,default=Path(__file__).with_name('d22_plan_v1.json'));a=p.parse_args()
    if a.phase=='prepare':prepare(a)
    elif a.phase=='codec':codec(a)
    else:evaluate(a.run.resolve())
