"""Evaluate the four already-encoded clip001 q32 controls after a user time limit."""
from __future__ import annotations
import argparse
from fractions import Fraction
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from evaluate_d2 import region, compare
from prepare_d2 import read_json, save_json


def run(args):
    root=args.run
    box=root/'timeboxed_20261021'
    plan=read_json(box/'plan.json')
    print('Waiting for the three registered timeboxed decodes.',flush=True)
    while True:
        state=read_json(box/'process.json')['state']
        if state=='completed': break
        if state=='failed': raise RuntimeError('Timeboxed decode failed; retain evidence')
        time.sleep(2)
    matrix=read_json(root/'matrix.json')
    inputs=read_json(root/'inputs.json')
    clip=next(c for c in inputs['clips'] if c['video_id']==1)
    tasks=[t for t in matrix['tasks'] if t['video_id']==1 and t['q']==32]
    if len(tasks)!=4: raise ValueError('Expected exactly A/B/C/O at q32')
    anchor=read_json(root/'budget_anchors.json')['1']
    points={}; sums={}; audit=[]; frames=[]
    started=time.perf_counter()
    for task in tasks:
        p=root/'points'/task['id']
        enc=read_json(p/'encoder/encode.json'); dec=read_json(p/'decoder/decode.json')
        if not dec['all_frame_checksums_match'] or dec['source_media_reads'] or len(dec['frames'])!=64:
            raise ValueError('Receiver verification did not pass')
        if not dec['independent_decoder'] or not dec['encoder_modules_disabled']:
            raise ValueError('Receiver is not independent')
        checks=dec['reconstruction_verification']
        if not all(checks[k] for k in ('verified','reference_verified','rgb_verified','complete_sequence')):
            raise ValueError('Receiver did not fully verify its output')
        if dec['input_sha256']!=enc['file_sha256'] or (p/'encoder/stream.glcv').stat().st_size!=enc['file_bytes']:
            raise ValueError('Measured and received streams differ')
        sender_pid=(read_json(root/'encode_anchors_summary.json')['pid'] if task['method']=='A'
                    else plan['encoder_remaining_pid'])
        if sender_pid==dec['pid']: raise ValueError('Sender and receiver shared a process')
        points[task['id']]={'id':task['id'],'video_id':1,'method':task['method'],'q':32,
            'file_bytes':enc['file_bytes'],'file_sha256':enc['file_sha256'],
            'file_kbps':8*enc['file_bytes']/float(Fraction(clip['duration_exact']))/1000,
            'growth_vs_S0':enc['file_bytes']/anchor['S0']-1,
            'within_B10':10*enc['file_bytes']<=11*anchor['S0'],
            'side_information_bytes':0,'encoder_seconds':enc['seconds'],'decoder_seconds':dec['seconds']}
        sums[task['id']]={'fish_sse':0,'all_sse':0,'fish_n':0,'all_n':0}
        audit.append({'id':task['id'],'sender_pid':sender_pid,'receiver_pid':dec['pid'],
            'source_media_reads':dec['source_media_reads'],'all_frame_checksums_match':True,
            'frame_count':64,'input_sha256':dec['input_sha256'],
            'reconstruction_verification':checks,'header':dec['header']})
    for frame in clip['records']:
        original=np.asarray(Image.open(frame['paths']['A']).convert('RGB')).astype(np.int16)
        gt=np.asarray(Image.open(frame['gt_path']))!=0
        fish_n=3*int(gt.sum()); all_n=original.size
        for task in tasks:
            path=root/'points'/task['id']/'decoder/receiver/frames'/f'im{frame["index"]+1:05d}.png'
            restored=np.asarray(Image.open(path).convert('RGB')).astype(np.int16)
            if restored.shape!=original.shape: raise ValueError('Dimensions changed')
            squared=(original-restored).astype(np.int32)**2
            all_sse=int(squared.sum(dtype=np.int64)); fish_sse=int(squared[gt].sum(dtype=np.int64))
            acc=sums[task['id']]
            acc['all_sse']+=all_sse; acc['fish_sse']+=fish_sse
            acc['all_n']+=all_n; acc['fish_n']+=fish_n
            frames.append({'task':task['id'],'index':frame['index'],
                'fish':region(fish_sse,fish_n),'rest':region(all_sse-fish_sse,all_n-fish_n)})
        if (frame['index']+1)%16==0: print(f'Evaluated {frame["index"]+1}/64 across four controls',flush=True)
    for task in tasks:
        acc=sums[task['id']]
        points[task['id']]['regions']={'fish':region(acc['fish_sse'],acc['fish_n']),
            'rest':region(acc['all_sse']-acc['fish_sse'],acc['all_n']-acc['fish_n']),
            'all':region(acc['all_sse'],acc['all_n'])}
    chosen={p['method']:p['id'] for p in points.values()}
    comparisons=[compare(points[chosen[m]],points[chosen[c]]) for m in ('B','C','O') for c in ('A','B') if m!=c]
    result={'schema':'d2-timeboxed-q32-results-v1','date':'2026-10-21','points_root':str(root),
        'scope':'one development clip, fixed q32, user-imposed ten-minute cutoff',
        'full_registered_matrix_complete':False,'same_q_comparison_only':True,
        'points':list(points.values()),'comparisons':comparisons,'anchor':anchor,
        'budget_selections':{'1':{'q32':{'selected':chosen}}},
        'quality_evaluation_seconds':time.perf_counter()-started,
        'continuous_visual_validation_completed':False,'full_D_round_gate_evaluated':False,
        'training':False,'holdout_used':False,'new_semantic_inference':False,
        'reference':'Original unprocessed RGB; all original fish GT including prediction misses',
        'limitations':['Fixed q is not equal actual bytes.','One development clip cannot establish generalization.',
                       'No complete q-based budget selection is claimed.','O is a diagnostic reference, not a guaranteed upper bound.']}
    save_json(box/'evaluation.json',result); save_json(box/'frame_metrics.json',frames)
    save_json(box/'decoder_audit.json',audit)
    save_json(box/'inputs.json',{'clips':[clip]})
    save_json(box/'matrix.json',{'runtime':matrix['runtime'],'tasks':tasks})
    save_json(box/'preparation.json',read_json(root/'preparation.json'))
    print(json_summary(result),flush=True)
    subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('visualize_d2.py')),'--run',str(box)],check=True)
    save_json(box/'completion.json',{'state':'completed','evaluated_points':4,'frames_each':64,
        'full_matrix_complete':False,'GPU_work_finished':True})


def json_summary(result):
    import json
    return json.dumps({'points':[{'method':p['method'],'bytes':p['file_bytes'],'kbps':p['file_kbps'],
        'fish_psnr':p['regions']['fish']['psnr_db'],'rest_psnr':p['regions']['rest']['psnr_db']}
        for p in result['points']],'comparisons':result['comparisons']},ensure_ascii=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    run(parser.parse_args())
