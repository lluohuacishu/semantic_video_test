"""Bounded native-mask cleanup diagnostics; never import or run a predictor/codec."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from d1_protection import hard_protection, alpha_protection
from analyze_d21_cache import small, mask_stats, overlay
from prepare_d2 import read_json, save_json


def components(raw, gt, bounds):
    labels, count = ndimage.label(raw, structure=np.ones((3, 3), dtype=bool))
    areas = np.bincount(labels.ravel(), minlength=count+1)
    overlaps = np.bincount(labels[gt].ravel(), minlength=count+1)
    sizes, hits = areas[1:], overlaps[1:]
    bins = []
    lower = 1
    for upper in bounds:
        include = (sizes >= lower) & (sizes <= upper)
        bins.append({'min': lower, 'max': upper, 'components': int(include.sum()),
                     'pixels': int(sizes[include].sum()), 'fish_GT_pixels': int(hits[include].sum())})
        lower = upper+1
    top = []
    slices = ndimage.find_objects(labels)
    for offset in np.argsort(-sizes, kind='stable')[:3]:
        idx = int(offset)+1; box = slices[idx-1]
        top.append({'label': idx, 'pixels': int(areas[idx]), 'fish_GT_pixels': int(overlaps[idx]),
                    'bbox_xyxy': [box[1].start, box[0].start, box[1].stop, box[0].stop]})
    # Only native predicted component area enters filtering; GT never enters this operation.
    keep = areas >= 64; keep[0] = False
    cleaned = keep[labels]
    return cleaned, {'count': int(count), 'pixels': int(sizes.sum()),
                     'fish_GT_pixels': int(hits.sum()), 'bins': bins, 'largest3': top}


def boundary_stats(hard, gt):
    edge = gt & ~ndimage.binary_erosion(gt, iterations=1)
    return {'fish_pixels': int(gt.sum()), 'fish_protected': int(np.count_nonzero(gt & hard)),
            'edge_pixels': int(edge.sum()), 'edge_protected': int(np.count_nonzero(edge & hard)),
            'fish_recall': float(np.count_nonzero(gt & hard)/gt.sum()),
            'edge_recall': float(np.count_nonzero(edge & hard)/edge.sum())}


def temporal_stats(current, previous):
    if previous is None:
        return None
    union = int(np.count_nonzero(current | previous))
    return {'xor_fraction': float(np.mean(current != previous)),
            'iou': float(np.count_nonzero(current & previous)/union) if union else 1.0}


def panel(path, vid, index, rgb, gt, seeds, hard_masks, fallbacks):
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
    smallfont = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    cells = [('原图 · 绿色为鱼GT', Image.fromarray(rgb)),
             ('原鱼种子', overlay(rgb, small(seeds['fish'], (512,288)), gt)),
             ('去极小块后的鱼种子', overlay(rgb, small(seeds['cleaned'], (512,288)), gt))]
    for method in ('C', 'K16', 'K8'):
        hard = small(hard_masks[method], (512,288)); stats = boundary_stats(hard, gt)
        title = f'{method} 回退前 · 面积{hard.mean():.1%} / 鱼{stats["fish_recall"]:.1%}'
        title += ' · 回退' if fallbacks[method] else ''
        cells.append((title, overlay(rgb, hard, gt)))
    canvas = Image.new('RGB', (1536, 748), '#101820'); draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f'D2.3 缓存修复筛选 · 片段{vid:03d} · 帧{index+1}/64', font=font, fill='white')
    draw.text((12, 37), '红色为预测/回退前保护；绿色为鱼GT。最终全图回退不计作定位成功。K16/K8保留潜水员分支。', font=smallfont, fill='#c5d4e2')
    for i, (title, im) in enumerate(cells):
        x=(i%3)*512; y=72+(i//3)*334
        draw.text((x+8,y+8),title,font=smallfont,fill='white');canvas.paste(im,(x,y+38))
    canvas.save(path)


def summarize(rows, configs):
    out = {}
    for method in configs:
        values = [r['methods'][method] for r in rows]
        native = [v['original_prefallback'] for v in values]
        delivery = [v['delivery_prefallback'] for v in values]
        def coverage(key, denominator):
            return sum(v[key] for v in delivery)/sum(v[denominator] for v in delivery)
        out[method] = {
            'mean_original_hard_area': float(np.mean([v['area'] for v in native])),
            'max_original_hard_area': max(v['area'] for v in native),
            'original_prefallback_fish_recall': sum(v['overlap'] for v in native)/sum(v['target'] for v in native),
            'delivery_prefallback_fish_recall': coverage('fish_protected','fish_pixels'),
            'delivery_prefallback_edge_recall': coverage('edge_protected','edge_pixels'),
            'min_frame_delivery_fish_recall': min(v['fish_recall'] for v in delivery),
            'min_frame_delivery_edge_recall': min(v['edge_recall'] for v in delivery),
            'fallback_counts': dict(Counter(v['fallback'] or 'none' for v in values)),
            'fallback_frames': sum(v['fallback'] is not None for v in values),
            'final_alpha_mean': float(np.mean([v['final_alpha_mean'] for v in values])),
            'final_fish_core_recall': sum(v['final_fish_core_pixels'] for v in values)/sum(v['fish_pixels'] for v in delivery),
            'final_edge_core_recall': sum(v['final_edge_core_pixels'] for v in values)/sum(v['edge_pixels'] for v in delivery),
            'hard_subset_of_C_all_frames': all(v['hard_subset_of_C'] for v in values),
            'mean_consecutive_hard_xor': float(np.mean([v['temporal']['xor_fraction'] for v in values if v['temporal'] is not None])),
            'mean_consecutive_hard_iou': float(np.mean([v['temporal']['iou'] for v in values if v['temporal'] is not None]))}
    return out


def assess(summary, records, plan):
    gate=plan['screening_gate']; decisions={}
    for cfg in plan['candidates']:
        method=cfg['method']; clips={}
        for vid in plan['clips']:
            s=summary[str(vid)]['methods'][method]; checks={
                'original_fish_coverage':s['original_prefallback_fish_recall']>=gate['original_prefallback_fish_pixel_coverage_min'],
                'mean_area':s['mean_original_hard_area']<=gate['mean_prefallback_hard_area_max'],
                'max_area':s['max_original_hard_area']<=gate['max_prefallback_hard_area'],
                'fallback':s['fallback_frames']<=gate['fallback_frames_max'],
                'delivery_fish_coverage':s['delivery_prefallback_fish_recall']>=gate['delivery_prefallback_fish_coverage_min'],
                'delivery_edge_coverage':s['delivery_prefallback_edge_recall']>=gate['delivery_prefallback_fish_inner_edge_coverage_min']}
            changes={}
            if vid==21:
                baseline=summary[str(vid)]['methods']['C']
                for key in ('fish','edge'):
                    delta=s[f'delivery_prefallback_{key}_recall']-baseline[f'delivery_prefallback_{key}_recall']
                    worst=min(r['methods'][method]['delivery_prefallback'][f'{key}_recall']-r['methods']['C']['delivery_prefallback'][f'{key}_recall'] for r in records if r['video_id']==vid)
                    changes[key]={'pooled_delta':delta,'worst_frame_delta':worst}
                    checks[f'021_{key}_pooled_nonregression']=delta>=-gate['clip021_pooled_fish_and_edge_drop_vs_C_max']
                    checks[f'021_{key}_frame_nonregression']=worst>=-gate['clip021_per_frame_fish_and_edge_drop_vs_C_max']
            clips[str(vid)]={'passed':all(checks.values()),'checks':checks,'failed':[k for k,v in checks.items() if not v], 'changes_vs_C':changes}
        decisions[method]={'passed':all(v['passed'] for v in clips.values()),'clips':clips}
    return decisions


def run(args):
    root=args.output.resolve();source=args.source.resolve();plan=read_json(args.plan)
    if root==source:raise ValueError('New output directory required')
    root.mkdir(parents=True,exist_ok=False);(root/'diagnostics').mkdir()
    save_json(root/'plan.json',plan);save_json(root/'sources.json',{'d2':str(source),'d21':str(args.d21.resolve()),'d22':str(args.d22.resolve())})
    for name in ('inputs.json','rate_gate.json','runtime.json'):save_json(root/name,read_json(source/name))
    clips=read_json(root/'inputs.json')['clips'];gate=read_json(root/'rate_gate.json')
    if [c['video_id'] for c in clips]!=plan['clips'] or gate['size']!=plan['size']:raise ValueError('Plan/input mismatch')
    if [c['method'] for c in plan['candidates']]!=['K16','K8'] or any(c['fish_min_component_pixels']!=64 for c in plan['candidates']):raise ValueError('Unsupported preregistered candidates')
    cfgs={'C':plan['baseline'],**{c['method']:c for c in plan['candidates']}}
    records=[];summaries={};start=time.perf_counter()
    for clip in clips:
        vid=clip['video_id'];rows=[];previous={k:None for k in ('fish','cleaned','diver')};previous_hard={m:None for m in cfgs}
        if len(clip['frames'])!=64 or clip['fps']!=plan['fps']:raise ValueError('Timeline mismatch')
        for method in cfgs:(root/'masks'/f'v{vid:03d}'/method).mkdir(parents=True)
        for frame in clip['frames']:
            if (root/'STOP').exists():raise RuntimeError('Stopped at cache frame boundary')
            index=frame['index'];name=f'{index:04d}.png';folder=source/'controls'/f'v{vid:03d}'
            packed=np.asarray(Image.open(folder/'seed_original'/name))
            gt=np.asarray(Image.open(frame['original_gt']))!=0
            if packed.shape!=gt.shape or packed.shape!=(1080,1920) or np.any(packed>3):raise ValueError('Source alignment/seed mismatch')
            gt_small=np.asarray(Image.open(frame['resized']['512x288']['gt']))!=0
            fish=(packed&1)!=0;diver=(packed&2)!=0
            cleaned,fish_comps=components(fish,gt,plan['component_diagnosis']['area_bin_upper_bounds'])
            _,diver_comps=components(diver,gt,plan['component_diagnosis']['area_bin_upper_bounds'])
            if np.any(cleaned & ~fish):raise ValueError('Cleanup grew a seed')
            row={'video_id':vid,'index':index,'components':{'fish':fish_comps,'diver':diver_comps},
                 'cleanup':{'removed_pixels':int(np.count_nonzero(fish & ~cleaned)), 'removed_fish_GT_pixels':int(np.count_nonzero(gt & fish & ~cleaned))},'methods':{}}
            hard_masks={};fallbacks={}
            for method,cfg in cfgs.items():
                seed_f=fish if method=='C' else cleaned;prev_f=previous['fish' if method=='C' else 'cleaned']
                fish_hard=hard_protection(seed_f,prev_f,{'margin_at_1920':cfg['fish_margin'],'previous_seed_frames':1})
                diver_hard=hard_protection(diver,previous['diver'],{'margin_at_1920':cfg['diver_margin'],'previous_seed_frames':1})
                hard=fish_hard|diver_hard;hard_masks[method]=hard
                alpha,fallback=alpha_protection(seed_f|diver,hard);alpha=small(alpha,plan['size'])
                if method=='C' and not np.array_equal(alpha,np.asarray(Image.open(folder/'alpha'/name))):raise ValueError('Original C alpha mismatch')
                is_subset=not np.any(hard & ~hard_masks['C'])
                if not is_subset:raise ValueError('Registered subset relation violated')
                hard_small=small(hard,plan['size']);metrics=boundary_stats(hard_small,gt_small)
                final_core=alpha==255;final_stats=boundary_stats(final_core,gt_small)
                row['methods'][method]={'original_prefallback':mask_stats(hard,gt), 'delivery_prefallback':metrics,
                    'fallback':fallback,'final_alpha_mean':float(alpha.mean()/255),
                    'final_fish_core_pixels':final_stats['fish_protected'],'final_edge_core_pixels':final_stats['edge_protected'],
                    'hard_subset_of_C':is_subset,'temporal':temporal_stats(hard,previous_hard[method])}
                # PNG packs pre-fallback hard as bit0 and final full-protection core as bit1.
                Image.fromarray(hard_small.astype(np.uint8)+2*final_core.astype(np.uint8)).save(root/'masks'/f'v{vid:03d}'/method/name,compress_level=1)
                fallbacks[method]=fallback;previous_hard[method]=hard
            if index in plan['fixed_visual_frame_indices']:
                rgb=np.asarray(Image.open(frame['resized']['512x288']['rgb']).convert('RGB'))
                panel(root/'diagnostics'/f'v{vid:03d}_{index:04d}.png',vid,index,rgb,gt_small,{'fish':fish,'cleaned':cleaned},hard_masks,fallbacks)
            previous={'fish':fish,'cleaned':cleaned,'diver':diver};rows.append(row);records.append(row)
            save_json(root/'progress.json',{'state':'running_cache','completed':len(records),'total':128})
            if (index+1)%16==0:print(f'CACHE {vid:03d} {index+1}/64',flush=True)
        comp_summary={}
        for branch in ('fish','diver'):
            values=[r['components'][branch] for r in rows]
            bins=[{**values[0]['bins'][i],**{k:sum(v['bins'][i][k] for v in values) for k in ('components','pixels','fish_GT_pixels')}} for i in range(len(values[0]['bins']))]
            comp_summary[branch]={'component_frame_count':sum(v['count'] for v in values),
                'total_seed_pixels':sum(v['pixels'] for v in values),'seed_overlap_fish_GT':sum(v['fish_GT_pixels'] for v in values),'bins':bins,
                'largest_component_seed_fraction':sum(v['largest3'][0]['pixels'] if v['largest3'] else 0 for v in values)/sum(v['pixels'] for v in values),
                'largest_component_GT_overlap_fraction_of_seed_overlap':sum(v['largest3'][0]['fish_GT_pixels'] if v['largest3'] else 0 for v in values)/max(1,sum(v['fish_GT_pixels'] for v in values))}
        summaries[str(vid)]={'frames':len(rows),'methods':summarize(rows,cfgs),'components':comp_summary,
            'cleanup':{'removed_pixels':sum(r['cleanup']['removed_pixels'] for r in rows),'removed_fish_GT_pixels':sum(r['cleanup']['removed_fish_GT_pixels'] for r in rows)}}
    decisions=assess(summaries,records,plan);eligible=[m for m,v in decisions.items() if v['passed']]
    result={'schema':'d23-cache-results-v1','date':'2026-10-22','frames':len(records),'candidate_frame_checks':len(records)*len(plan['candidates']),
            'clips':summaries,'screening':decisions,'eligible_candidates':eligible,'new_semantic_inference':False,'training':False,'new_encodes':0,'new_decodes':0,
            'original_C_alpha_reproduced_frames':len(records),'all_candidate_hard_masks_subset_of_C':True,'confidence_cache_available':False,
            'subset_bound':'For these candidates, each pre-fallback hard mask is a subset of old C. Pixel removal or smaller dilation cannot cover any GT pixel outside old C, regardless of component thresholds. This does not bound new inference or other mask-growing methods.',
            'source_independence_proven':False,'full_D1_or_diver_gate_revalidated':False,'subjective_acceptance':False,'seconds':time.perf_counter()-start}
    save_json(root/'frame_diagnostics.json',records);save_json(root/'cache_results.json',result)
    state='cache_passed_awaiting_bounded_codec_manifest' if eligible else 'completed_no_candidate_for_codec'
    save_json(root/'progress.json',{'state':state,'completed':128,'eligible_candidates':eligible,'new_encodes':0,'new_decodes':0,'new_semantic_inference':False})
    save_json(root/'code_provenance.json',{'files':[{'path':name,'sha256':hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()} for name in ('analyze_d23_cache.py','d23_cache_repair_plan_v1.json','d1_protection.py','analyze_d21_cache.py')],'no_model_or_dataset_rescan':True})
    print('RESULT '+state+' eligible='+str(eligible),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--d21',type=Path,required=True);p.add_argument('--d22',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path,default=Path(__file__).with_name('d23_cache_repair_plan_v1.json'));args=p.parse_args()
    try:run(args)
    except BaseException as exc:
        if args.output.is_dir() and not (args.output/'cache_results.json').exists():save_json(args.output/'failure.json',{'type':type(exc).__name__,'message':str(exc)})
        raise
