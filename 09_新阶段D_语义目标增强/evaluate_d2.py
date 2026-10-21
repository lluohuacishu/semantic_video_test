"""Evaluate independently decoded RGB against original RGB; select by bytes only."""
from __future__ import annotations
import argparse
from fractions import Fraction
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
from prepare_d2 import read_json, save_json


def region(sse, count):
    if not count:
        return {'sse':int(sse),'samples':0,'mse':None,'psnr_db':None,'lossless':False}
    mse=sse/count
    return {'sse':int(sse),'samples':int(count),'mse':mse,
            'psnr_db':10*math.log10(255**2/mse) if mse else None,'lossless':mse==0}


def compare(left,right):
    gap=(right['file_bytes']-left['file_bytes'])/right['file_bytes']
    def delta(key):
        a=left['regions'][key]['psnr_db']; b=right['regions'][key]['psnr_db']
        return None if a is None or b is None else a-b
    return {'candidate':left['id'],'comparator':right['id'],
            'candidate_bytes':left['file_bytes'],'comparator_bytes':right['file_bytes'],
            'relative_byte_increase_vs_comparator':-gap,
            'near_equal_no_larger_2pct':0<=gap<=.02,
            'target_psnr_delta_db':delta('fish'),'rest_psnr_delta_db':delta('rest'),
            'global_psnr_delta_db':delta('all'),
            'target_mse_change_fraction':left['regions']['fish']['mse']/right['regions']['fish']['mse']-1
                if right['regions']['fish']['mse'] else None}


def run(args):
    root=args.run
    if read_json(root/'progress.json')['state']!='codec_complete':
        raise ValueError('Finish independent codec phases before evaluating quality')
    inputs=read_json(root/'inputs.json'); matrix=read_json(root/'matrix.json')
    anchors=read_json(root/'budget_anchors.json')
    points={}; frame_records=[]
    started=time.perf_counter()
    for clip in inputs['clips']:
        tasks=[t for t in matrix['tasks'] if t['video_id']==clip['video_id']]
        sums={t['id']:{'fish_sse':0,'all_sse':0,'fish_n':0,'all_n':0} for t in tasks}
        for task in tasks:
            folder=root/'points'/task['id']
            encoded=read_json(folder/'encoder/encode.json')
            decoded=read_json(folder/'decoder/decode.json')
            if not decoded['all_frame_checksums_match'] or decoded['source_media_reads'] or len(decoded['frames'])!=64:
                raise ValueError('Independent decoding did not meet the existing codec requirements')
            if not decoded['independent_decoder'] or not decoded['encoder_modules_disabled']:
                raise ValueError('Receiver must disable source-frame encoding modules')
            verification=decoded['reconstruction_verification']
            if not all(verification[key] for key in ('verified','reference_verified','rgb_verified','complete_sequence')):
                raise ValueError('Receiver reconstruction was not fully verified')
            if decoded['input_sha256']!=encoded['file_sha256']:
                raise ValueError('Receiver decoded a different stream from the measured encoding')
            if (folder/'encoder/stream.glcv').stat().st_size!=encoded['file_bytes']:
                raise ValueError('Measured stream size changed after encoding')
            if decoded['pid']==read_json(root/('encode_anchors_summary.json' if task['method']=='A' and task['q']==32 else 'encode_remaining_summary.json'))['pid']:
                raise ValueError('Sender and receiver must be separate processes')
            points[task['id']]={'id':task['id'],'video_id':clip['video_id'],'method':task['method'],'q':task['q'],
                  'file_bytes':encoded['file_bytes'],'file_sha256':encoded['file_sha256'],
                  'side_information_bytes':0,'file_kbps':8*encoded['file_bytes']/float(Fraction(clip['duration_exact']))/1000,
                  'duration_exact':clip['duration_exact'],'encoder_seconds':encoded['seconds'],
                  'decoder_seconds':decoded['seconds'],'independent_decode':True,'reconstruction_verified':True,
                  'growth_vs_S0':encoded['file_bytes']/anchors[str(clip['video_id'])]['S0']-1,
                  'within_B10':10*encoded['file_bytes']<=11*anchors[str(clip['video_id'])]['S0']}
        for frame in clip['records']:
            original=np.asarray(Image.open(frame['paths']['A']).convert('RGB')).astype(np.int16)
            gt=np.asarray(Image.open(frame['gt_path']))!=0
            fish_n=3*int(gt.sum()); all_n=original.size
            for task in tasks:
                decoded_path=root/'points'/task['id']/'decoder/receiver/frames'/f'im{frame["index"]+1:05d}.png'
                restored=np.asarray(Image.open(decoded_path).convert('RGB')).astype(np.int16)
                if restored.shape!=original.shape: raise ValueError('Reconstruction dimensions differ from reference')
                squared=(original-restored).astype(np.int32)**2
                all_sse=int(squared.sum(dtype=np.int64)); fish_sse=int(squared[gt].sum(dtype=np.int64))
                acc=sums[task['id']]
                acc['all_sse']+=all_sse; acc['fish_sse']+=fish_sse
                acc['all_n']+=all_n; acc['fish_n']+=fish_n
                frame_records.append({'task':task['id'],'index':frame['index'],
                    'fish':region(fish_sse,fish_n),'rest':region(all_sse-fish_sse,all_n-fish_n)})
            if (frame['index']+1)%16==0: print(f'Evaluated video {clip["video_id"]:03d} frame {frame["index"]+1}/64 across 12 points',flush=True)
        for task in tasks:
            acc=sums[task['id']]
            points[task['id']]['regions']={'fish':region(acc['fish_sse'],acc['fish_n']),
                 'rest':region(acc['all_sse']-acc['fish_sse'],acc['all_n']-acc['fish_n']),
                 'all':region(acc['all_sse'],acc['all_n'])}
    budgets={}; comparisons=[]; same_q=[]
    for clip in inputs['clips']:
        v=str(clip['video_id']); budgets[v]={}
        candidates=[p for p in points.values() if str(p['video_id'])==v]
        for name,ceiling in [('B0',anchors[v]['S0']),('B10',anchors[v]['B10'])]:
            chosen={}
            for method in ('A','B','C','O'):
                allowed=[p for p in candidates if p['method']==method and p['file_bytes']<=ceiling]
                chosen[method]=max(allowed,key=lambda p:(p['file_bytes'],-p['q']))['id'] if allowed else None
            budgets[v][name]={'ceiling':ceiling,'selected':chosen,
                             'unused_bytes':{k:ceiling-points[t]['file_bytes'] if t else None for k,t in chosen.items()}}
            for method in ('C','O'):
                for comparator in ('A','B'):
                    if chosen[method] and chosen[comparator]:
                        comparisons.append(dict(video_id=int(v),budget=name,**compare(points[chosen[method]],points[chosen[comparator]])))
        for q in (0,32,63):
            indexed={p['method']:p for p in candidates if p['q']==q}
            for method in ('B','C','O'):
                same_q.append(dict(video_id=int(v),q=q,**compare(indexed[method],indexed['A'])))
    result={'schema':'d2-two-clip-results-v1','date':'2026-10-21','scope':'development mechanism diagnostic',
            'points':list(points.values()),'budget_anchors':anchors,'budget_selections':budgets,
            'budget_comparisons':comparisons,'same_q_vs_original':same_q,
            'primary_reference':'Unprocessed source RGB; original fish GT including prediction misses',
            'points_evaluated':len(points),'frame_point_metrics':len(frame_records),
            'quality_evaluation_seconds':time.perf_counter()-started,
            'full_D_round_gate_evaluated':False,'continuous_visual_validation_completed':False,
            'independent_source_generalization_proven':False,'training':False,'holdout_used':False,
            'limitations':['Only two development clips in one source group; no diver-video result.',
                          'Same ceiling is not equal actual bytes. Report all sizes and gaps.',
                          'O is a fish-GT diagnostic reference, not a guaranteed upper bound.',
                          'Decoded PNGs provide quality metrics; lossy previews do not.',
                          'Three q points only; no interpolated or padded streams.']}
    save_json(root/'evaluation.json',result)
    save_json(root/'frame_metrics.json',frame_records)
    for v,levels in budgets.items():
        for name,selection in levels.items():
            print(v,name,{k:(t,points[t]['file_bytes'],round(points[t]['regions']['fish']['psnr_db'],3)) if t else None
                          for k,t in selection['selected'].items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    run(parser.parse_args())
