"""Frozen spatial-cell x temporal-intersection controls following low-rate L.

Offline mechanism experiment. Source correspondence is used only for evaluation.
No new transmitted candidate is claimed by this script.
"""

import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np

from audit_deflicker_texture import source_match
from compact_support_stream import KEYS, RAMP, reduce_support, expand_support
from compare_deflicker import save_csv
from coral_reference_sidecar import file_sha, load_json, save_json
from diagnose_fullframe_factorial import coefficients
import evaluate_fullframe_reference as scoring
from fullframe_structure_stream import expanded_motion, unpack
from probe_reef import bands
from run_fullframe_factorial import BOX, render
from run_p0 import preview
from temporal_filter import warp
from verify_video1_persistent_structure import inspect_preview


CASES = {'P1': (1, 1), 'B1': (4, 1), 'P5': (1, 5), 'B5': (4, 5)}
REFRESH = {k+j for k in KEYS[1:] for j in range(7)}


def timing(i):
    if i in REFRESH:
        return 'refresh_with_exit'
    return 'group_update' if i % 5 == 0 else 'group_interior'


def effects(scores):
    j = {k: v['coarse_mae'] for k, v in scores.items()}
    return dict(spatial_at_T1=j['B1']-j['P1'], spatial_at_T5=j['B5']-j['P5'],
                temporal_at_pixels=j['P5']-j['P1'], temporal_at_B4=j['B5']-j['B1'],
                interaction=j['B5']-j['B1']-j['P5']+j['P1'])


def run(workspace, output, ffmpeg):
    stage = Path(__file__).parent
    root = workspace/'02_实验数据/优化_GLC后处理'
    batch = root/'source_dense_flow_three_clips_20261109_01/video1'
    prior = root/'fullframe_reference_refined_20261110_01/video1/g24q2_E'
    factorial = root/'fullframe_factorial_20261110_01'
    lowrate = root/'compact_support_20261110_01/video1/S24_B4_T5_E/receive'
    local = root/'video1_persistent_aux_20261110_01/g17q4/receive'
    plan = stage/'plan_lowrate_next_20261110.json'
    assert load_json(plan)['status'] == 'planned_not_run'
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(plan, output/'frozen_previous_plan.json')
    inputs = [batch/'source.npy', batch/'base.glcsnd', batch/'base_receive/decoded_rgb.npy',
              prior/'fullframe.glcfull', factorial/'gates/new_PG.npy', factorial/'gates/old_PG.npy',
              factorial/'native_GP/rgb.npy', factorial/'native_GP/alpha.npy',
              lowrate/'rgb.npy', lowrate/'alpha.npy']
    save_json(output/'execution_plan.json', dict(date='2026-11-11', status='frozen_before_render',
        authorization='User asked to begin the next round after the retrospective.',
        cases=CASES, parent_plan_sha256=file_sha(plan), source_assisted_offline=True,
        new_transmitted_candidate=False, input_sha256={str(p):file_sha(p) for p in inputs},
        motion='Identical frozen 24x14 half-pixel motion, no fitting',
        scopes='Fixed external source correspondence for all methods, plus actual coverage',
        old_reference='Same treatment as new support; groups never cross reference identities'))
    raw = np.load(batch/'base_receive/decoded_rgb.npy', mmap_mode='r')
    source = np.load(batch/'source.npy', mmap_mode='r')
    h, a = unpack((prior/'fullframe.glcfull').read_bytes(), file_sha(batch/'base.glcsnd'))
    assert h['grid'] == [24, 14] and h['scale'] == 2 and h['keys'] == list(KEYS)
    motion = {k:np.stack([expanded_motion(q, 2) for q in a[k+'_motion']]) for k in ('new','old')}
    safe = np.concatenate([np.load(factorial/'gates'/(k+'_PG.npy')) for k in ('new','old')])
    receipts, coverage = {}, {}
    seq = dict(source=source, raw=raw, U=np.load(local/'rgb.npy', mmap_mode='r'))
    states = {}
    for name, (step, span) in CASES.items():
        compact = reduce_support(safe, step, span)
        masks = expand_support(compact, span).astype(bool)
        masks = np.repeat(np.repeat(masks, step, 1), step, 2)
        assert not np.any(masks & ~safe)
        folder = output/name
        receipts[name] = render(raw, motion['new'], motion['old'], masks[:150], masks[150:], KEYS, folder)
        save_json(folder/'receipt.json', receipts[name])
        coverage[name] = {}
        for kind, sl in (('new',np.s_[:150]),('old',np.s_[150:])):
            coverage[name][kind] = {scope:dict(pixel_frames=int(masks[sl,y0:y1,x0:x1].size),
                safe_pixel_frames=int(masks[sl,y0:y1,x0:x1].sum()))
                for scope,(x0,y0,x1,y1) in {'full':(0,0,512,288),'coral':BOX}.items()}
        seq[name] = np.load(folder/'rgb.npy', mmap_mode='r')
        states[name] = tuple(np.load(folder/(v+'.npy'), mmap_mode='r') for v in ('alpha','old_weight'))
        if name in ('P1','B5'):
            archived = factorial/'native_GP' if name == 'P1' else lowrate
            for field, value in (('rgb',seq[name]),('alpha',states[name][0])):
                assert np.array_equal(value, np.load(archived/(field+'.npy'), mmap_mode='r')), (name,field)
            receipts[name]['archived_RGB_alpha_bit_exact'] = True
        print('RENDER',name,'step',step,'span',span,'safe fraction',float(masks[:150].mean()),flush=True)
    scoring.SCOPES['coral'] = BOX
    scores, temporal, spatial = scoring.score(seq)
    save_csv(output/'temporal.csv',temporal)
    save_csv(output/'spatial.csv',spatial)
    expected = {'raw':4.55809282879483,'U':1.4255694384159472,'P1':1.971967780149087,'B5':2.6778268570075356}
    for name, value in expected.items():
        assert abs(scores['coral'][name]['coarse_mae']-value) < 1e-5, name
    print('CORAL', {k:(v['coarse_mae'],v['fine_ncc']) for k,v in scores['coral'].items()},flush=True)
    print('EFFECTS', effects(scores['coral']),flush=True)
    times = {}
    for sensitivity, excluded in (('all',set()),('exclude_I',{64,65,128,129}),
                                  ('exclude_period32',{32,33,64,65,96,97,128,129})):
        times[sensitivity] = {}
        for category in ('refresh_with_exit','group_update','group_interior'):
            times[sensitivity][category] = {}
            for name in seq:
                rr=[r for r in temporal if r['scope']=='coral' and r['method']==name
                    and timing(r['frame'])==category and r['frame'] not in excluded]
                n=sum(r['pixels'] for r in rr)
                times[sensitivity][category][name]=dict(pairs=len(rr),pixels=n,
                    coarse_mae=sum(r['coarse_sum'] for r in rr)/n,
                    fine_ncc=sum(r['fine_ncc']*r['pixels'] for r in rr)/n)
    region=np.zeros((288,512),bool)
    x0,y0,x1,y1=BOX
    region[y0:y1,x0:x1]=True
    previous=bands(source[0])
    previous_coeff={k:coefficients(a[0],w[0],0,KEYS) for k,(a,w) in states.items()}
    rows=[]
    for i in range(1,150):
        current=bands(source[i])
        mx,my,valid=source_match(source[i-1],source[i],previous,current,np.s_[:,:])
        for name,(alpha,old_weight) in states.items():
            last=warp(alpha[i-1],mx,my)
            last_on=cv2.remap((alpha[i-1]>0).astype('u1'),mx,my,cv2.INTER_NEAREST).astype(bool)
            on=alpha[i]>0
            delta=np.abs(alpha[i]-last)
            coeff=coefficients(alpha[i],old_weight[i],i,KEYS)
            tv=np.abs(coeff-warp(previous_coeff[name],mx,my)).sum(axis=2)/2
            for scope,area in (('full',valid),('coral',valid & region)):
                always=area & on & last_on
                rows.append(dict(frame=i,timing=timing(i),method=name,scope=scope,pixels=int(area.sum()),
                    switch_pixels=int((area & (on!=last_on)).sum()),
                    held_pixels=int((area & (alpha[i]>=.99) & (last>=.99)).sum()),
                    alpha_delta_sum=float(delta[area].sum()),always_pixels=int(always.sum()),
                    always_alpha_delta_sum=float(delta[always].sum()),reference_tv_sum=float(tv[area].sum())))
            previous_coeff[name]=coeff
        previous=current
    save_csv(output/'support.csv',rows)
    support={}
    for scope in ('full','coral'):
        support[scope]={}
        for name in states:
            rr=[r for r in rows if r['scope']==scope and r['method']==name]
            n=sum(r['pixels'] for r in rr)
            always=sum(r['always_pixels'] for r in rr)
            support[scope][name]=dict(pixels=n,switch_fraction=sum(r['switch_pixels'] for r in rr)/n,
                held_fraction=sum(r['held_pixels'] for r in rr)/n,
                alpha_delta=sum(r['alpha_delta_sum'] for r in rr)/n,
                reference_tv=sum(r['reference_tv_sum'] for r in rr)/n,
                always_on_alpha_delta=sum(r['always_alpha_delta_sum'] for r in rr)/max(always,1))
    labels={'P1':'P1: pixel / every frame [offline]', 'B1':'B1: 4x4 / every frame [offline]',
            'P5':'P5: pixel / 5-frame intersection [offline]', 'B5':'B5 = current L: 4x4 / 5-frame intersection'}
    previews={}
    for filename,methods,crop in [('factorial_full.mp4',tuple(CASES),None),
                                 ('L_vs_B1_full.mp4',('B5','B1'),None),
                                 ('L_vs_B1_coral.mp4',('B5','B1'),(250,0,512,132))]:
        panels=[seq[k] for k in methods]
        selection=None if crop is None else np.s_[crop[1]:crop[3],crop[0]:crop[2]]
        preview(ffmpeg,output/filename,panels,15,0,[labels[k] for k in methods],crop=selection)
        previews[filename]=inspect_preview(ffmpeg,output/filename,panels,crop=crop)
    files=['run_support_factorial.py','run_fullframe_factorial.py','compact_support_stream.py',
           'fullframe_structure_stream.py','evaluate_fullframe_reference.py','diagnose_fullframe_factorial.py',
           'audit_deflicker_texture.py','probe_reef.py','temporal_filter.py']
    snapshot=output/'source'
    snapshot.mkdir()
    for name in files:shutil.copyfile(stage/name,snapshot/name)
    save_json(output/'source_manifest.json',{p.name:file_sha(p) for p in snapshot.iterdir()})
    save_json(output/'summary.json',dict(complete=True,date='2026-11-11',offline_controls=True,
        actual_transmitted_candidate=False, cases=CASES,scores=scores,factorial_effects={k:effects(v) for k,v in scores.items()},
        coral_time_groups=times,coverage=coverage,support=support,receipts=receipts,
        GP_L_RGB_alpha_reproduced=True,archived_scores_reproduced=True,safety_violations=0,
        previews=previews,visual_review='pending',evaluation_support='Fixed external source texture reliability',
        period32_caveat='Schedule sensitivity, not proof of actual reset at each marked boundary'))
    print('COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ffmpeg',type=Path,required=True)
    args=p.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    run(args.workspace.resolve(),args.output.resolve(),args.ffmpeg.resolve())
