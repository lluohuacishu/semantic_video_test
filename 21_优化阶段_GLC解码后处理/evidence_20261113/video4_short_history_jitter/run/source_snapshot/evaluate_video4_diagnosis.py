"""Length-aware scoring with historical 150-frame metric regression."""
import argparse
from pathlib import Path
import shutil
import cv2
import numpy as np
from runtime_speed_common import STAGE,file_sha,load_json,save_json
from audit_deflicker_texture import normalized,source_match
from probe_reef import bands
from temporal_filter import warp
from compare_deflicker import save_csv
from evaluate_fullframe_reference import SCOPES
from run_video4_diagnosis import PRIOR

def score(seq,scopes):
    n=len(seq['source']);tr=[];sr=[];missing=[];previous=None
    assert all(a.shape==seq['source'].shape for a in seq.values())
    for i in range(n):
        current={k:bands(v[i]) for k,v in seq.items()}
        if i:
            mx,my,common=source_match(seq['source'][i-1],seq['source'][i],previous['source'],current['source'],np.s_[:,:])
            aligned={k:warp(v,mx,my) for k,v in previous.items()}
            delta=current['source']-aligned['source']
        for scope,(x0,y0,x1,y1) in scopes.items():
            crop=np.s_[y0:y1,x0:x1]
            for name,video in seq.items():
                error=video[i][crop].astype(np.float32)-seq['source'][i][crop]
                sr.append(dict(frame=i,scope=scope,method=name,pixels=(x1-x0)*(y1-y0),
                    rgb_sse=float(np.square(error).sum()),rgb_abs_sum=float(np.abs(error).sum()),
                    fine_sum=float(np.abs(current[name][crop][...,2]).sum()),
                    coarse_spatial_sum=float(np.abs(current[name][crop][...,0]-current['source'][crop][...,0]).sum()),
                    fine_spatial_sum=float(np.abs(current[name][crop][...,2]-current['source'][crop][...,2]).sum())))
            if i:
                mask=common[crop];count=int(mask.sum())
                if count<64:
                    missing.append(dict(frame=i,scope=scope,pixels=count,reason='Unchanged common-support minimum of 64 pixels'))
                    continue
                for name in seq:
                    a=aligned[name][crop][mask];b=current[name][crop][mask]
                    error=np.abs(b-a-delta[crop][mask])
                    tr.append(dict(frame=i,scope=scope,method=name,pixels=count,
                        coarse_sum=float(error[:,0].sum()),fine_sum=float(error[:,2].sum()),
                        full_sum=float(error[:,3].sum()),
                        fine_ncc=float((normalized(a[:,2:3])*normalized(b[:,2:3])).mean())))
        previous=current
    result={}
    for scope in scopes:
        result[scope]={}
        for name in seq:
            t=[r for r in tr if r['scope']==scope and r['method']==name]
            s=[r for r in sr if r['scope']==scope and r['method']==name]
            count=sum(r['pixels'] for r in t);sn=sum(r['pixels'] for r in s)
            mse=sum(r['rgb_sse'] for r in s)/(3*sn)
            result[scope][name]=dict(pairs=len(t),excluded_pairs=n-1-len(t),valid_pixels=count,
                coarse_mae=sum(r['coarse_sum'] for r in t)/count if count else None,
                fine_mae=sum(r['fine_sum'] for r in t)/count if count else None,
                full_mae=sum(r['full_sum'] for r in t)/count if count else None,
                fine_ncc=sum(r['fine_ncc']*r['pixels'] for r in t)/count if count else None,
                fine_amplitude=sum(r['fine_sum'] for r in s)/sn,
                psnr_db=float(10*np.log10(255**2/mse)) if mse else None,
                rgb_mae=sum(r['rgb_abs_sum'] for r in s)/(3*sn),
                coarse_spatial_mae=sum(r['coarse_spatial_sum'] for r in s)/sn,
                fine_spatial_mae=sum(r['fine_spatial_sum'] for r in s)/sn)
    return result,tr,sr,missing

def dynamic(seq):
    rows=[]
    for i in range(1,len(seq['source'])):
        source_delta=seq['source'][i].astype(np.float32)-seq['source'][i-1]
        magnitude=np.max(np.abs(source_delta),axis=2)
        mask=magnitude>=max(12,float(np.quantile(magnitude,.9)))
        n=int(mask.sum())
        for name,a in seq.items():
            change=a[i].astype(np.float32)-a[i-1]
            rows.append(dict(frame=i,method=name,pixels=n,
                delta_error_sum=float(np.abs(change-source_delta)[mask].sum()),
                change_sum=float(np.abs(change)[mask].sum()),
                source_change_sum=float(np.abs(source_delta)[mask].sum())))
    result={}
    for name in seq:
        r=[v for v in rows if v['method']==name];n=sum(v['pixels'] for v in r)
        result[name]=dict(pixels=n,RGB_delta_error=sum(v['delta_error_sum'] for v in r)/(3*n),
            change_amplitude_ratio=sum(v['change_sum'] for v in r)/sum(v['source_change_sum'] for v in r))
    return result,rows

def sequences(run):
    return dict(source=np.load(run/'source.npy',mmap_mode='r'),
        raw=np.load(run/'video4/base_receive/decoded_rgb.npy',mmap_mode='r'),
        guide=np.load(run/'receive/guide_rgb.npy',mmap_mode='r'),
        Q9_R25=np.load(run/'receive/Q9_R25/rgb.npy',mmap_mode='r'),
        GS25=np.load(run/'receive/GS25/rgb.npy',mmap_mode='r'))

def baseline_regression(run):
    s=load_json(PRIOR/'case_registration.json')['cases']['video1']
    seq={n:np.load(s[n],mmap_mode='r') for n in ('source','raw')}
    seq.update(guide=np.load(PRIOR/'video1/receive/received_guide.npy',mmap_mode='r'),
        Q9_R25=np.load(PRIOR/'video1/receive/Q9_R25/rgb.npy',mmap_mode='r'),
        GS25=np.load(PRIOR/'video1/receive/GS25/rgb.npy',mmap_mode='r'))
    scopes=dict(SCOPES);scopes['reef']=tuple(s['reef'])
    scores,_,_,missing=score(seq,scopes);dyn,_=dynamic(seq)
    old=load_json(PRIOR/'video1/evaluation/summary.json')
    for scope,methods in scores.items():
        for name,v in methods.items():
            for key,w in old['scores'][scope][name].items():
                assert v[key]==w,(scope,name,key,v[key],w)
    for name in seq:assert dyn[name]==old['dynamic'][name]
    assert missing==old['low_support']
    save_json(run/'regression/evaluation150_verification.json',dict(complete=True,
        original_video1_all_scopes_all_inherited_metrics_exact=True,dynamic_exact=True,
        methods=list(seq),source_summary_sha256=file_sha(PRIOR/'video1/evaluation/summary.json'),
        evaluator_sha256=file_sha(__file__)))
    print('150 FRAME EVALUATOR REGRESSION EXACT',flush=True)

def main(run):
    assert load_json(run/'receive/receipt.json')['complete']
    out=run/'evaluation';out.mkdir(exist_ok=False)
    plan=load_json(run/'execution_plan.json')
    save_json(run/'evaluation_plan.json',dict(frozen_before_scoring=True,date='2026-11-13',
        inference_plan_sha256=file_sha(run/'execution_plan.json'),frames=104,
        shared_support_threshold=64,all_methods_use_same_support=True,
        scopes=dict(SCOPES,**plan['regions']),new_metrics=['rgb_mae','coarse_spatial_mae','fine_spatial_mae'],
        event_lists=dict(Q9_anchor_start=[15,45,75],base_reset=[32,64,96],base_intra=[64]),
        event_interpretation='Diagnostic association only; coincident events do not prove causation.',
        code_sha256={n:file_sha(STAGE/n) for n in ('evaluate_video4_diagnosis.py','probe_reef.py',
            'audit_deflicker_texture.py','evaluate_fullframe_reference.py','evaluate_structure_guide.py')}))
    snap=run/'evaluation_source';snap.mkdir()
    for n in load_json(run/'evaluation_plan.json')['code_sha256']:shutil.copy2(STAGE/n,snap/n)
    baseline_regression(run)
    seq=sequences(run);scopes=dict(SCOPES,**plan['regions'])
    scores,tr,sr,missing=score(seq,scopes);dyn,dr=dynamic(seq)
    save_csv(out/'temporal.csv',tr);save_csv(out/'spatial.csv',sr);save_csv(out/'dynamic.csv',dr)
    save_json(out/'missing_support.json',missing)
    events={}
    for label,ids in load_json(run/'evaluation_plan.json')['event_lists'].items():
        events[label]={}
        for name in seq:
            r=[v for v in tr if v['scope']=='full' and v['method']==name and v['frame'] in ids]
            n=sum(v['pixels'] for v in r)
            events[label][name]=dict(frames=[v['frame'] for v in r],coarse_mae=sum(v['coarse_sum'] for v in r)/n if n else None)
    peaks={}
    for name in ('raw','Q9_R25','GS25'):
        rows=[dict(frame=v['frame'],seconds=v['frame']/15,coarse_mae=v['coarse_sum']/v['pixels'])
              for v in tr if v['scope']=='full' and v['method']==name]
        peaks[name]=sorted(rows,key=lambda v:v['coarse_mae'],reverse=True)[:5]
    result=dict(complete=True,case='video4',frames=104,fps=15,billing_seconds=104/15,
        scores=scores,dynamic=dyn,low_support=missing,events=events,largest_full_coarse_peaks=peaks,
        rates=load_json(run/'guide/send.json'),automatic_selection=None,subjective_review='pending',
        development_clip=True,independent_holdout=False)
    save_json(run/'summary.json',result)
    print('VIDEO4 EVALUATED', {n:round(v['coarse_mae'],6) if v['coarse_mae'] is not None else None
                              for n,v in scores['full'].items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);main(a.run)

