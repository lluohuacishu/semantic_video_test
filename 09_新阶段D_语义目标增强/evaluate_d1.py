"""Evaluate cached D1 predictions on development GT and freeze one simple rule."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

from d1_protection import seed_masks, hard_protection, alpha_protection


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n', encoding='utf-8')


def ratio(numerator, denominator):
    return numerator/denominator if denominator else None


def mean(values):
    usable = [v for v in values if v is not None]
    return float(np.mean(usable)) if usable else None


def stats(mask, gt, valid=None):
    if valid is not None:
        mask, gt = mask & valid, gt & valid
    predicted, support = int(mask.sum()), int(gt.sum())
    tp = int(np.count_nonzero(mask & gt))
    return {'tp': tp, 'predicted': predicted, 'support': support,
            'precision': ratio(tp, predicted), 'recall': ratio(tp, support),
            'iou': ratio(tp, predicted+support-tp),
            'frame_fraction': predicted / mask.size,
            'outside_gt_frame_fraction': (predicted-tp)/mask.size}


def aggregate(rows):
    tp = sum(r['tp'] for r in rows)
    predicted = sum(r['predicted'] for r in rows)
    support = sum(r['support'] for r in rows)
    return {'frame_count':len(rows), 'tp':tp, 'predicted':predicted, 'support':support,
            'pooled_precision':ratio(tp,predicted), 'pooled_recall':ratio(tp,support),
            'pooled_iou':ratio(tp,predicted+support-tp),
            'mean_image_recall':mean([r['recall'] for r in rows]),
            'images_at_least_90pct_covered':sum(r['recall'] is not None and r['recall'] >= .9 for r in rows),
            'mean_protection_fraction':mean([r['frame_fraction'] for r in rows]),
            'max_protection_fraction':max((r['frame_fraction'] for r in rows), default=0),
            'mean_outside_gt_frame_fraction':mean([r['outside_gt_frame_fraction'] for r in rows]),
            'zero_overlap_positive_frames':sum(r['support']>0 and r['tp']==0 for r in rows)}


def instance_summary(rows):
    return {'instance_frame_count':len(rows),
            'at_least_90pct_covered':sum(r['covered90'] for r in rows),
            'covered90_fraction':ratio(sum(r['covered90'] for r in rows),len(rows)),
            'zero_overlap_count':sum(r['tp']==0 for r in rows),
            'mean_coverage':mean([r['coverage'] for r in rows]),
            'raw_target_zero_overlap_count':sum(r['raw_tp']==0 for r in rows),
            'raw_target_covered90_fraction':ratio(sum(10*r['raw_tp']>=9*r['pixels'] for r in rows),len(rows))}


def track_summary(rows):
    tracks = defaultdict(list)
    for r in rows:
        tracks[(r['video_id'],r['track_id'])].append(r)
    results = []
    for (video_id, track_id), records in sorted(tracks.items()):
        records.sort(key=lambda r:r['index'])
        zero_run = below_run = max_zero = max_below = toggles = 0
        last = None
        for r in records:
            if last is None or r['index'] != last['index']+1:
                zero_run = below_run = 0
            else:
                toggles += r['covered90'] != last['covered90']
            zero_run = zero_run+1 if r['tp']==0 else 0
            below_run = below_run+1 if not r['covered90'] else 0
            max_zero, max_below = max(max_zero,zero_run), max(max_below,below_run)
            last = r
        results.append({'video_id':video_id,'track_id':track_id, 'visible_frames':len(records),
                        'mean_coverage':mean([r['coverage'] for r in records]),
                        'zero_overlap_frames':sum(r['tp']==0 for r in records),
                        'longest_zero_overlap_run':max_zero, 'longest_below90_run':max_below,
                        'covered90_toggles_between_visible_frames':toggles})
    return results


def build_development_gt(manifest, corpus):
    result = {}
    for seq in manifest['fish_sequences']:
        if seq['role'] != 'development':
            continue
        for frame in seq['frames']:
            key = f"sfishtrack_{seq['video_id']:03d}/{frame['original_file_name']}"
            result[key] = {'kind':'fish', 'video_id':seq['video_id'], **frame}
    for row in manifest['suim']['images']:
        result['suim/'+row['id']] = {'kind':row['sample_kind'], **row}
    return result


def gate_results(summary, plan):
    g = plan['evaluation']['gate']
    fish = [summary['groups'][f'fish_{v:03d}'] for v in (1,2,3)]
    diver, negative = summary['groups']['diver_positive'], summary['groups']['no_target']
    checks = {
        'fish_each_clip_pixel_coverage': all(r['pooled_recall'] >= g['fish_each_clip_pixel_coverage_min'] for r in fish),
        'fish_instance_90pct_coverage': summary['instances']['all']['covered90_fraction'] >= g['fish_instance_90pct_covered_fraction_min'],
        'diver_mean_image_coverage': diver['mean_image_recall'] >= g['diver_mean_image_coverage_min'],
        'diver_images_at_least_90pct_covered': diver['images_at_least_90pct_covered'] >= g['diver_images_at_least_90pct_covered_min'],
        'fish_mean_area': all(r['mean_protection_fraction'] <= g['fish_each_clip_mean_protection_fraction_max'] for r in fish),
        'fish_max_area': all(r['max_protection_fraction'] <= g['fish_frame_protection_fraction_max'] for r in fish),
        'suim_positive_excess_area': diver['all_labeled_targets_protection']['mean_outside_gt_frame_fraction'] <= g['suim_positive_mean_protection_outside_target_fraction_max'],
        'negative_mean_area': negative['mean_protection_fraction'] <= g['negative_mean_raw_protection_fraction_max'],
        'negative_max_area': negative['max_protection_fraction'] <= g['negative_max_raw_protection_fraction_max'],
    }
    return {'passed': all(checks.values()), 'checks': checks,
            'failed_checks':[k for k,v in checks.items() if not v],
            'selection_min_coverage':min([r['pooled_recall'] for r in fish]+[diver['mean_image_recall']])}


def run(args):
    started = time.perf_counter()
    plan = read_json(args.run/'plan.json')
    inputs = read_json(args.run/'inputs.json')['inputs']
    predictions = [json.loads(s) for s in (args.run/'predictions.jsonl').read_text('utf-8').splitlines()]
    if read_json(args.run/'inference_summary.json')['status'] != 'completed' or len(predictions)!=len(inputs):
        raise ValueError('Inference must complete before development evaluation')
    manifest = read_json(args.corpus/'ready_manifest.json')
    gt_rows = build_development_gt(manifest,args.corpus)
    instances_by_image = {}
    # Read only the three development annotation JSON members, no archive scan/extraction.
    with zipfile.ZipFile(manifest['archive_path']) as archive:
        for video_id in (1,2,3):
            annotations = json.loads(archive.read(f'SFISHTRACK/Annotations/video{video_id:03d}.json'))
            by_image = defaultdict(list)
            for obj in annotations['annotations']:
                by_image[obj['image_id']].append(obj)
            instances_by_image[video_id] = by_image
    configurations = [plan['diagnostic_baseline']]+plan['candidates']
    metrics, instance_rows, temporal = [], defaultdict(list), defaultdict(list)
    previous_seed, previous_hard = {}, {}
    confusion = np.zeros((8,8), dtype=np.int64)
    last_group = None
    for source, prediction in zip(inputs,predictions):
        if source['id'] != prediction['id']:
            raise ValueError('Prediction order differs from RGB input order')
        truth = gt_rows[source['id']]
        kind = truth['kind']
        video_id = source.get('video_id')
        group = f'fish_{video_id:03d}' if video_id else ('diver_positive' if kind=='diver_positive' else 'no_target')
        packed = np.asarray(Image.open(args.run/prediction['packed_path']))
        if video_id:
            gt_fish = np.asarray(Image.open(args.corpus/truth['mask_path'])) != 0
            gt_diver = None
            gt_target = gt_fish
            valid = None
            object_masks = []
            for obj in instances_by_image[video_id][truth['coco_image_id']]:
                decoded = coco_mask.decode(obj['segmentation']).astype(bool)
                pixels = int(decoded.sum())
                fraction = pixels/decoded.size
                object_masks.append((obj,decoded,pixels,'small' if fraction<.001 else 'medium' if fraction<.01 else 'large'))
        else:
            indexed = np.asarray(Image.open(truth['eight_class_mask_path']))
            valid = indexed < 8
            gt_fish, gt_diver = indexed==6, indexed==1
            gt_target = gt_fish | gt_diver
            base_labels = np.asarray(Image.open(args.run/prediction['base_labels_path']))
            confusion += np.bincount(8*indexed[valid].astype(np.int64)+base_labels[valid].astype(np.int64),minlength=64).reshape(8,8)
            object_masks = []
        if not video_id or group != last_group:
            previous_seed, previous_hard = {}, {}
        frame = {'id':source['id'],'input_index':prediction['input_index'], 'group':group,
                 'video_id':video_id, 'index':source.get('index',truth['index']),
                 'kind':kind, 'candidates':{}}
        for cfg in configurations:
            name = cfg['name']
            fish, diver = seed_masks(packed,plan,cfg)
            current = fish|diver
            hard = hard_protection(current,previous_seed.get(name),cfg)
            fish_stats = stats(fish,gt_fish,valid)
            protection = stats(hard,gt_fish if video_id else gt_diver if group=='diver_positive' else gt_target,valid)
            fallback = 'empty_current_prediction' if not current.any() else 'protection_exceeds_half_frame' if int(hard.sum())*2>hard.size else None
            frame['candidates'][name] = {
                'fish_raw':fish_stats,
                'diver_raw':stats(diver,gt_diver,valid) if gt_diver is not None else None,
                'protection':protection,
                'all_labeled_targets_protection':stats(hard,gt_target,valid),
                'predicted_diver_fraction':float(diver.mean()),
                'fallback':fallback,
            }
            for obj, mask, pixels, size_bin in object_masks:
                overlap = int(np.count_nonzero(mask & hard))
                raw_overlap = int(np.count_nonzero(mask & fish))
                instance_rows[name].append({'video_id':video_id,'index':frame['index'],
                    'annotation_id':obj['id'],'track_id':obj['track_id'],'pixels':pixels,
                    'size_bin':size_bin,'tp':overlap,'coverage':overlap/pixels,
                    'covered90':10*overlap>=9*pixels,
                    'raw_tp':raw_overlap,'raw_coverage':raw_overlap/pixels})
            if video_id and name in previous_hard:
                old = previous_hard[name]
                temporal[name].append({'video_id':video_id,'index':frame['index'],
                    'mask_iou':ratio(int((hard&old).sum()),int((hard|old).sum())),
                    'xor_frame_fraction':float((hard^old).mean()),
                    'absolute_area_change_fraction':abs(int(hard.sum())-int(old.sum()))/hard.size})
            previous_seed[name],previous_hard[name] = current,hard
        metrics.append(frame)
        last_group = group
        if len(metrics)%32==0 or len(metrics)==len(inputs):
            print(f'Evaluated {len(metrics)}/{len(inputs)} development inputs',flush=True)
    summaries = {}
    for cfg in configurations:
        name = cfg['name']
        groups = {}
        for group in sorted({r['group'] for r in metrics}):
            matching = [r['candidates'][name] for r in metrics if r['group']==group]
            protection = aggregate([r['protection'] for r in matching])
            protection['fish_raw'] = aggregate([r['fish_raw'] for r in matching])
            protection['diver_raw'] = aggregate([r['diver_raw'] for r in matching if r['diver_raw'] is not None])
            protection['all_labeled_targets_protection'] = aggregate([r['all_labeled_targets_protection'] for r in matching])
            protection['fallback_frames'] = sum(r['fallback'] is not None for r in matching)
            protection['fallback_reasons'] = {reason:sum(r['fallback']==reason for r in matching) for reason in ('empty_current_prediction','protection_exceeds_half_frame')}
            groups[group] = protection
        instances = {'all':instance_summary(instance_rows[name])}
        for size_bin in ('small','medium','large'):
            instances[size_bin] = instance_summary([r for r in instance_rows[name] if r['size_bin']==size_bin])
        for video_id in (1,2,3):
            instances[f'fish_{video_id:03d}'] = instance_summary([r for r in instance_rows[name] if r['video_id']==video_id])
        summary = {'config':cfg,'groups':groups,'instances':instances,
                   'mean_all_frame_protection_fraction':mean([r['candidates'][name]['protection']['frame_fraction'] for r in metrics]),
                   'tracks':track_summary(instance_rows[name]),
                   'temporal':{f'fish_{video_id:03d}':{
                       'transition_count':sum(r['video_id']==video_id for r in temporal[name]),
                       'mean_mask_iou':mean([r['mask_iou'] for r in temporal[name] if r['video_id']==video_id]),
                       'mean_xor_frame_fraction':mean([r['xor_frame_fraction'] for r in temporal[name] if r['video_id']==video_id]),
                       'mean_absolute_area_change_fraction':mean([r['absolute_area_change_fraction'] for r in temporal[name] if r['video_id']==video_id]),
                   } for video_id in (1,2,3)}}
        summary['gate'] = gate_results(summary,plan)
        summaries[name] = summary
    names = [c['name'] for c in plan['candidates']]
    passing = [n for n in names if summaries[n]['gate']['passed']]
    if passing:
        chosen = min(passing,key=lambda n:(summaries[n]['mean_all_frame_protection_fraction'],names.index(n)))
    else:
        chosen = min(names,key=lambda n:(-summaries[n]['gate']['selection_min_coverage'],summaries[n]['mean_all_frame_protection_fraction'],names.index(n)))
    regression = []
    for class_id in range(8):
        tp=int(confusion[class_id,class_id]); support=int(confusion[class_id].sum()); predicted=int(confusion[:,class_id].sum())
        regression.append({'class_id':class_id,'tp':tp,'support':support,'predicted':predicted,
                           'iou':ratio(tp,support+predicted-tp),'recall':ratio(tp,support)})
    selected = summaries[chosen]
    frozen = {'schema':'d1-frozen-protection-v1','date':'2026-10-21',
              'status':'pilot_gate_passed' if passing else 'diagnostic_only_localization_gate_failed',
              'plan':'d1_run_plan_v1.json','budget_protocol':'d1_d2_protocol_v2.json',
              'selected_name':chosen,'config':selected['config'],'models':plan['models'],
              'protection':plan['protection'],'failed_checks':selected['gate']['failed_checks'],
              'can_claim_deployable_target_protection':bool(passing),
              'selection_used_development_GT':True,'holdout_used':False,
              'training_performed':False,'GLC_experiment_performed':False,
              'semantic_quality_gain_proven':False,
              'integrity_recheck_performed':False}
    (args.run/'selected_hard').mkdir(exist_ok=False)
    (args.run/'selected_alpha').mkdir(exist_ok=False)
    selected_outputs = []
    previous, last_group = None,None
    for source,prediction,record in zip(inputs,predictions,metrics):
        if not source.get('video_id') or record['group'] != last_group:
            previous = None
        packed=np.asarray(Image.open(args.run/prediction['packed_path']))
        fish,diver=seed_masks(packed,plan,selected['config'])
        current=fish|diver
        hard=hard_protection(current,previous,selected['config'])
        alpha,reason=alpha_protection(current,hard)
        filename=f"{prediction['input_index']:04d}.png"
        Image.fromarray(hard.astype(np.uint8)*255).save(args.run/'selected_hard'/filename)
        Image.fromarray(alpha).save(args.run/'selected_alpha'/filename)
        selected_outputs.append({'id':source['id'],'rgb_path':source['rgb_path'],
                                 'hard_path':'selected_hard/'+filename,'alpha_path':'selected_alpha/'+filename,
                                 'fallback':reason,'effective_alpha_mean':float(alpha.mean()/255)})
        previous,last_group=current,record['group']
    result={'schema':'d1-evaluation-v1','date':'2026-10-21', 'input_count':len(inputs),
            'development_fish_instance_frames':len(instance_rows[chosen]),
            'candidate_results':summaries,'selection':frozen,
            'base_suim_regression':{'sample_count':24,'confusion_matrix':confusion.tolist(),'per_class':regression,
                                    'mean_iou_present_union':mean([r['iou'] for r in regression]),
                                    'reused_from_D1':bool(plan.get('SUIM_base_regression_reused_not_profile_specific',False)),
                                    'measures_the_compared_fish_profile':False,
                                    'not_comparable_to_full_SUIM_validation_mIoU':True},
            'evaluator_seconds':time.perf_counter()-started,
            'integrity_recheck_performed':False,
            'limitations':['Development only; three fish clips share a conservative source group.',
                          'SFISHTRACK precision uses incomplete target labels; complement is not confirmed background.',
                          'Diver temporal behavior untested; SUIM images are static.',
                          'Mask XOR includes true motion and is not a pure flicker measure.',
                          'Coverage is measured before full-frame fallback; no codec quality or bitrate result.']}
    save_json(args.run/'evaluation.json',result)
    save_json(args.run/'frame_metrics.json',metrics)
    save_json(args.run/'instance_metrics.json',dict(instance_rows))
    save_json(args.run/'temporal_metrics.json',dict(temporal))
    save_json(args.run/'frozen_protection.json',frozen)
    save_json(args.run/'selected_outputs.json',{'selected_name':chosen,'outputs':selected_outputs})
    print(json.dumps({'selected':chosen,'status':frozen['status'],'failed_checks':frozen['failed_checks'],
                      'seconds':result['evaluator_seconds']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    run(parser.parse_args())
