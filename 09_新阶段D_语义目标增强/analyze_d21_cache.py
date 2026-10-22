"""Diagnose D2.1 from cached masks; never load a predictor or touch the old run."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import time
import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont
from d1_protection import hard_protection, alpha_protection
from prepare_d2 import read_json, save_json, blend
from finish_d2_lowrate import stats
from evaluate_d2 import region


def small(mask, size):
    return np.asarray(Image.fromarray(mask).resize(tuple(size), Image.Resampling.NEAREST))


def mask_stats(mask, gt):
    overlap = int(np.count_nonzero(mask & gt))
    selected = int(mask.sum())
    target = int(gt.sum())
    return dict(pixels=mask.size, selected=selected, target=target, overlap=overlap,
                area=selected/mask.size, fish_recall=overlap/target if target else None,
                fish_precision=overlap/selected if selected else None,
                nonfish_selected=selected-overlap, exceeds_half=2*selected>mask.size)


def overlay(rgb, mask, gt):
    out=rgb.astype(np.float32).copy()
    out[mask]=out[mask]*.55+np.array([255,99,76])*.45
    edge=gt & ~ndimage.binary_erosion(gt)
    out[edge]=[30,255,140]
    return Image.fromarray(np.rint(out).astype(np.uint8))


def panel(path, video, frame, source, gt, fish, diver, union_hard, alphas, fallbacks):
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
    note=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    cells=[('原图 · 绿色为鱼 GT',Image.fromarray(source)),
           ('鱼预测种子',overlay(source,fish,gt)),
           ('潜水员预测种子',overlay(source,diver,gt)),
           ('合并＋历史＋扩张（回退前）',overlay(source,union_hard,gt))]
    for label,title in [('C','原鱼＋潜水员保护'),('F','仅鱼保护'),('O','GT 鱼保护 · 仅诊断')]:
        cells.append((title+(' · 回退' if fallbacks[label] else ''),overlay(source,alphas[label]>=128,gt)))
    heat=np.repeat(alphas['C'][...,None],3,axis=2)
    cells.append(('原 C 的 alpha（白色保留）',Image.fromarray(heat)))
    canvas=Image.new('RGB',(4*512,2*334+60),'#101820');draw=ImageDraw.Draw(canvas)
    draw.text((12,8),f'D2.1 缓存诊断 · 片段 {video:03d} · 帧 {frame+1}/64',font=font,fill='white')
    draw.text((12,35),'红色：该掩码；绿色：鱼 GT 边界。非鱼 GT 区域不等于已确认背景。',font=note,fill='#c8d6e3')
    for i,(title,im) in enumerate(cells):
        x=(i%4)*512;y=(i//4)*334+60
        draw.text((x+10,y+7),title,font=font,fill='white');canvas.paste(im,(x,y+40))
    canvas.save(path)


def aggregate(rows):
    out={}
    for branch in ('fish','diver','union'):
        out[branch]={}
        for stage in ('raw','dilation_only','history_only','dilation_and_history'):
            values=[r['stages'][branch][stage] for r in rows]
            sums={k:sum(v[k] for v in values) for k in ('pixels','selected','target','overlap','nonfish_selected')}
            out[branch][stage]={**sums,'mean_frame_fraction':sums['selected']/sums['pixels'],
                'pooled_fish_recall':sums['overlap']/sums['target'],
                'pooled_fish_precision':sums['overlap']/sums['selected'] if sums['selected'] else None,
                'frames_exceeding_half':sum(v['exceeds_half'] for v in values)}
    return out


def run(args):
    root=args.output.resolve();source=args.source.resolve();plan=read_json(args.plan)
    if root==source: raise ValueError('New run required')
    root.mkdir(parents=True,exist_ok=False)
    save_json(root/'cache_plan.json',plan)
    inputs=read_json(source/'inputs.json');gate=read_json(source/'rate_gate.json')
    if gate['size']!=plan['size'] or [c['video_id'] for c in inputs['clips']]!=plan['clips']:
        raise ValueError('Scope differs from registered cached experiment')
    save_json(root/'inputs.json',inputs);save_json(root/'rate_gate.json',gate)
    save_json(root/'runtime.json',read_json(source/'runtime.json'))
    (root/'diagnostics').mkdir();records=[];summaries={};start=time.perf_counter()
    cfg=plan['protection'];size=plan['size'];key=f'{size[0]}x{size[1]}'
    for clip in inputs['clips']:
        vid=clip['video_id'];previous={k:None for k in ('fish','diver','union','gt')};rows=[]
        if len(clip['frames'])!=64 or clip['fps']!=plan['fps']:raise ValueError('Timeline mismatch')
        for method in ('F','O','alpha_F','alpha_O'):(root/'controls'/f'v{vid:03d}'/method).mkdir(parents=True)
        for frame in clip['frames']:
            if (root/'STOP').exists():raise RuntimeError('Stopped at cache-frame boundary')
            index=frame['index'];name=f'{index:04d}.png';folder=source/'controls'/f'v{vid:03d}'
            packed=np.asarray(Image.open(folder/'seed_original'/name))
            if np.any(packed>3):raise ValueError('Unexpected packed mask bits')
            masks={'fish':(packed&1)!=0,'diver':(packed&2)!=0}
            masks['union']=masks['fish']|masks['diver']
            gt=np.asarray(Image.open(frame['original_gt']))!=0
            if gt.shape!=packed.shape:raise ValueError('Original GT alignment mismatch')
            source_rgb=np.asarray(Image.open(frame['resized'][key]['rgb']).convert('RGB'))
            gt_small=np.asarray(Image.open(frame['resized'][key]['gt']))!=0
            blurred=np.asarray(Image.open(folder/'B'/name).convert('RGB'))
            row={'video_id':vid,'index':index,'stages':{},'controls':{}}
            hard_masks={}
            for branch,raw in masks.items():
                history=raw.copy() if previous[branch] is None else raw|previous[branch]
                dilated=hard_protection(raw,None,cfg);hard=hard_protection(raw,previous[branch],cfg)
                stages={'raw':raw,'dilation_only':dilated,'history_only':history,'dilation_and_history':hard}
                row['stages'][branch]={k:mask_stats(v,gt) for k,v in stages.items()}
                hard_masks[branch]=hard
            row['raw_overlap_fraction']=float(np.mean(masks['fish']&masks['diver']))
            row['raw_diver_only_fraction']=float(np.mean(masks['diver']&~masks['fish']))
            row['raw_fish_only_fraction']=float(np.mean(masks['fish']&~masks['diver']))
            alphas={};fallbacks={}
            inner_edge=gt_small & ~ndimage.binary_erosion(gt_small,iterations=1)
            gt_hard=hard_protection(gt,previous['gt'],cfg)
            for method,raw,hard in [('C',masks['union'],hard_masks['union']),('F',masks['fish'],hard_masks['fish']),('O',gt,gt_hard)]:
                alpha,fallback=alpha_protection(raw,hard);alpha=small(alpha,size)
                transformed=blend(source_rgb,blurred,alpha)
                if method=='C':
                    if not np.array_equal(alpha,np.asarray(Image.open(folder/'alpha'/name))):raise ValueError('Cached alpha reconstruction mismatch')
                    if not np.array_equal(transformed,np.asarray(Image.open(folder/'C'/name).convert('RGB'))):raise ValueError('Cached C input mismatch')
                else:
                    Image.fromarray(alpha).save(root/'controls'/f'v{vid:03d}'/f'alpha_{method}'/name,compress_level=1)
                    Image.fromarray(transformed).save(root/'controls'/f'v{vid:03d}'/method/name,compress_level=1)
                if not np.array_equal(transformed[alpha==255],source_rgb[alpha==255]):raise ValueError('Protected core changed')
                errors=stats(source_rgb,transformed,gt_small)
                row['controls'][method]={'fallback':fallback,'mean_alpha':float(alpha.mean()/255),
                    'hard_core_fraction':float(np.mean(alpha==255)),
                    'fish_pixels':int(gt_small.sum()),'fish_core_pixels':int(np.count_nonzero((alpha==255)&gt_small)),
                    'fish_alpha_sum':int(alpha[gt_small].sum()),'fish_edge_pixels':int(inner_edge.sum()),
                    'fish_edge_core_pixels':int(np.count_nonzero((alpha==255)&inner_edge)),
                    'preprocess_error':{k:{'sse':v[0],'samples':v[1]} for k,v in errors.items()}}
                alphas[method]=alpha;fallbacks[method]=fallback
            row['controls']['B']={'preprocess_error':{k:{'sse':v[0],'samples':v[1]} for k,v in stats(source_rgb,blurred,gt_small).items()}}
            if index in plan['fixed_visual_frame_indices']:
                panel(root/'diagnostics'/f'v{vid:03d}_{index:04d}.png',vid,index,source_rgb,gt_small,
                      small(masks['fish'],size),small(masks['diver'],size),small(hard_masks['union'],size),alphas,fallbacks)
            previous={**masks,'gt':gt};rows.append(row);records.append(row)
            save_json(root/'cache_progress.json',{'state':'running','completed':len(records),'total':128,'last_video':vid})
            if (index+1)%16==0:print(f'CACHE {vid:03d} {index+1}/64',flush=True)
        control_summary={}
        for method in ('B','C','F','O'):
            values=[r['controls'][method] for r in rows]
            value={'preprocess_error':{k:region(sum(v['preprocess_error'][k]['sse'] for v in values),sum(v['preprocess_error'][k]['samples'] for v in values)) for k in ('fish','rest','all')}}
            if method!='B':
                count=sum(v['fish_pixels'] for v in values);edge=sum(v['fish_edge_pixels'] for v in values)
                value.update(fallback_counts=dict(Counter(v['fallback'] or 'none' for v in values)),
                    mean_alpha=float(np.mean([v['mean_alpha'] for v in values])),
                    mean_hard_core_fraction=float(np.mean([v['hard_core_fraction'] for v in values])),
                    fish_core_recall=sum(v['fish_core_pixels'] for v in values)/count,
                    fish_alpha_mean=sum(v['fish_alpha_sum'] for v in values)/(255*count),
                    fish_edge_core_recall=sum(v['fish_edge_core_pixels'] for v in values)/edge)
            control_summary[method]=value
        summaries[str(vid)]={'frames':len(rows),'stages':aggregate(rows),'controls':control_summary,
            'mean_raw_overlap_fraction':float(np.mean([r['raw_overlap_fraction'] for r in rows])),
            'mean_raw_diver_only_fraction':float(np.mean([r['raw_diver_only_fraction'] for r in rows])),
            'mean_raw_fish_only_fraction':float(np.mean([r['raw_fish_only_fraction'] for r in rows]))}
    result={'schema':'d21-cache-results-v1','source_run':str(source),'run':str(root),'clips':summaries,
        'frames':len(records),'seconds':time.perf_counter()-start,'cached_C_alpha_and_input_exactly_reproduced':True,
        'limitations':['Fish GT is not exhaustive diver/background annotation.','Stage/branch differences overlap; do not add them as independent causal effects.',
                      'GT protection is an offline diagnostic, not deployable or a guaranteed upper bound.','Preprocessing preservation does not imply post-codec fidelity.'],
        'semantic_inference':False,'training':False,'integrity_rescan':False}
    save_json(root/'cache_diagnosis.json',result);save_json(root/'cache_frame_metrics.json',records)
    save_json(root/'cache_progress.json',{'state':'completed','completed':128,'seconds':time.perf_counter()-start})
    print('Cache diagnosis completed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path,default=Path(__file__).with_name('d21_cache_plan_v1.json'))
    run(parser.parse_args())
