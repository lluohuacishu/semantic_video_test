"""Collect all four fixed D1 profile results and visualize paired target crops."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from evaluate_d1 import read_json, save_json
from d1_protection import seed_masks


def percent(value):
    return f'{value:.2%}' if value is not None else 'NA'


def run(args):
    plan=read_json(args.run/'comparison_plan.json')
    runs={k:Path(v) for k,v in read_json(args.run/'profile_runs.json').items()}
    profiles={}
    names=[p['name'] for p in plan['profiles']]
    for definition in plan['profiles']:
        name=definition['name']
        result=read_json(runs[name]/'evaluation.json')
        selected=result['candidate_results']['recall_hold1']
        mean_iou=float(np.mean([selected['groups'][f'fish_{v:03d}']['fish_raw']['pooled_iou'] for v in (1,2,3)]))
        profiles[name]={'definition':definition,'groups':selected['groups'],
                        'instances':selected['instances'],'tracks':selected['tracks'],
                        'temporal':selected['temporal'],'gate':selected['gate'],
                        'mean_all_frame_protection_fraction':selected['mean_all_frame_protection_fraction'],
                        'minimum_fish_clip_protection_recall':min(selected['groups'][f'fish_{v:03d}']['pooled_recall'] for v in (1,2,3)),
                        'mean_raw_fish_clip_iou':mean_iou}
    passing=[n for n in names if profiles[n]['gate']['passed']]
    if passing:
        chosen=min(passing,key=lambda n:(profiles[n]['mean_all_frame_protection_fraction'],names.index(n)))
    else:
        chosen=min(names,key=lambda n:(-profiles[n]['minimum_fish_clip_protection_recall'],
                                      -profiles[n]['mean_raw_fish_clip_iou'],
                                      profiles[n]['mean_all_frame_protection_fraction'],names.index(n)))
    contrasts=[]
    for left,right,factor in [
        ('adapted_tiles512','base_tiles512','checkpoint_at_tiles512'),
        ('adapted_letterbox256','base_letterbox256','checkpoint_at_letterbox256'),
        ('base_tiles512','base_letterbox256','preprocessing_with_base_checkpoint'),
        ('adapted_tiles512','adapted_letterbox256','preprocessing_with_adapted_checkpoint'),
    ]:
        contrasts.append({'factor':factor,'from':left,'to':right,
           'raw_fish_recall_delta_by_clip':[profiles[right]['groups'][f'fish_{v:03d}']['fish_raw']['pooled_recall']-profiles[left]['groups'][f'fish_{v:03d}']['fish_raw']['pooled_recall'] for v in (1,2,3)],
           'raw_fish_iou_delta_by_clip':[profiles[right]['groups'][f'fish_{v:03d}']['fish_raw']['pooled_iou']-profiles[left]['groups'][f'fish_{v:03d}']['fish_raw']['pooled_iou'] for v in (1,2,3)],
           'protection_recall_delta_by_clip':[profiles[right]['groups'][f'fish_{v:03d}']['pooled_recall']-profiles[left]['groups'][f'fish_{v:03d}']['pooled_recall'] for v in (1,2,3)]})
    selection={'selected_profile':chosen,'fixed_protection':'recall_hold1',
               'status':'pilot_gate_passed' if passing else 'diagnostic_only_localization_gate_failed',
               'failed_checks':profiles[chosen]['gate']['failed_checks'],
               'production_default_changed':False,'holdout_evaluated':False,'training_performed':False,
               'GLC_experiment_performed':False,'complete_file_growth_measured':False}
    summary={'schema':'d1-model-comparison-results-v1','date':'2026-10-21',
             'profiles':profiles,'paired_contrasts':contrasts,'selection':selection,
             'inference':read_json(args.run/'inference_summary.json'),
             'limitations':[
                'Three development clips share one conservative source group; these are paired diagnostics, not new independent validation.',
                'The diver branch and the context-only eight-class SUIM regression matrix are reused from D1.',
                'All profiles use the same 0.5 fish probability threshold and same protection rule; other thresholds are not tuned.',
                'SFISHTRACK complement may contain unannotated fish; precision is label-relative.',
                'Only localization is evaluated; no compression, transmitted-file-size or reconstructed-video-quality conclusion.'
             ]}
    save_json(args.run/'comparison_summary.json',summary)
    save_json(args.run/'comparison_selection.json',selection)
    # Fixed middle frame from every clip, plus the already documented D1 worst frame.
    inputs=read_json(args.run/'inputs.json')['inputs']
    old_frames=read_json(runs['adapted_tiles512']/'frame_metrics.json')
    examples=[]
    for v in (1,2,3):
        group=list(range((v-1)*64,v*64))
        worst=min(group,key=lambda i:old_frames[i]['candidates']['recall_hold1']['protection']['recall'])
        examples.append((v-1)*64+31)
        if worst not in examples: examples.append(worst)
    manifest=read_json(args.corpus/'ready_manifest.json')
    truths={f'sfishtrack_{s["video_id"]:03d}/{f["original_file_name"]}':f for s in manifest['fish_sequences'] if s['role']=='development' for f in s['frames']}
    canvas=Image.new('RGB',(1600,90+len(examples)*235),(17,24,34))
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    titlefont=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',23)
    draw.text((16,10),'D1 follow-up: same threshold / protection, different checkpoint and input',font=titlefont,fill='white')
    draw.text((16,40),'Green: GT boundary   Orange: fish prediction   Cyan: combined protection',font=font,fill='white')
    frame_metrics={n:read_json(runs[n]/'frame_metrics.json') for n in names}
    for col,label in enumerate(['Original + GT']+names):
        draw.text((col*320+10,65),label,font=font,fill='white')
    for row,i in enumerate(examples):
        source=inputs[i]
        image=Image.open(source['rgb_path']).convert('RGB')
        gt=np.asarray(Image.open(args.corpus/truths[source['id']]['mask_path']))!=0
        yy,xx=np.where(gt)
        box=(max(0,int(xx.min())-64),max(0,int(yy.min())-64),min(image.width,int(xx.max())+65),min(image.height,int(yy.max())+65))
        boundary=ndimage.maximum_filter(gt,size=3)^ndimage.minimum_filter(gt,size=3)
        for col,name in enumerate([None]+names):
            rgb=np.asarray(image).copy()
            if name:
                packed=np.asarray(Image.open(runs[name]/'packed'/f'{i:04d}.png'))
                fish,_=seed_masks(packed,read_json(runs[name]/'plan.json'),{'fish_threshold':.5,'diver_threshold':.5})
                hard=np.asarray(Image.open(runs[name]/'selected_hard'/f'{i:04d}.png'))!=0
                rgb[hard]=(rgb[hard]*.82+np.array([20,190,255])*.18).astype(np.uint8)
                rgb[ndimage.maximum_filter(hard,size=3)^ndimage.minimum_filter(hard,size=3)]=[20,190,255]
                rgb[ndimage.maximum_filter(fish,size=3)^ndimage.minimum_filter(fish,size=3)]=[250,130,40]
            rgb[boundary]=[90,255,90]
            crop=Image.fromarray(rgb).crop(box)
            crop.thumbnail((316,200),Image.Resampling.LANCZOS)
            x=col*320+(320-crop.width)//2; y=95+row*235+(200-crop.height)//2
            canvas.paste(crop,(x,y))
            if name:
                cov=frame_metrics[name][i]['candidates']['recall_hold1']['protection']['recall']
                text=f'protected {cov:.1%}'
            else: text=f'Video {source["video_id"]:03d}, frame {source["index"]}'
            draw.text((col*320+10,298+row*235),text,font=font,fill='white')
    canvas.save(args.run/'model_comparison_crops.jpg',quality=95)
    save_json(args.run/'visualization_index.json',{'examples':examples,'sheet':'model_comparison_crops.jpg',
               'selection_rule':'Middle frame plus lowest coverage from original D1, not selected on new-model results'})
    print(json.dumps(selection),flush=True)
    for name,p in profiles.items():
        print(name,'raw recall',[percent(p['groups'][f'fish_{v:03d}']['fish_raw']['pooled_recall']) for v in (1,2,3)],
              'protected',[percent(p['groups'][f'fish_{v:03d}']['pooled_recall']) for v in (1,2,3)],
              'failed',p['gate']['failed_checks'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--corpus',type=Path,required=True)
    run(parser.parse_args())
