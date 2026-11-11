"""Measure transmitted K/N support and reference weights along source motion."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from audit_deflicker_texture import source_match
from compact_support_stream import KEYS,RAMP,expand_support
from compare_deflicker import save_csv
from coral_reference_sidecar import file_sha,save_json
from diagnose_fullframe_factorial import coefficients
from predictive_support_stream import unpack
from probe_reef import bands
from run_fullframe_factorial import BOX,feather
from run_support_factorial import timing
from temporal_filter import warp


def analyze(workspace,run):
    batch=workspace/'02_实验数据/优化_GLC后处理/source_dense_flow_three_clips_20261109_01'
    result={}
    for case in ('video1','video2','video3'):
        out=run/case/'states'
        out.mkdir(exist_ok=False)
        source=np.load(batch/case/'source.npy',mmap_mode='r')
        states={};coverage={};previous_coeff={}
        digest=file_sha(batch/case/'base.glcsnd')
        for name in ('K','N'):
            alpha=np.load(run/case/name/'receive/alpha.npy',mmap_mode='r')
            h,a=unpack((run/case/name/'support.glccsp2').read_bytes(),digest)
            mask=expand_support(a['support'],5)
            mask=np.repeat(np.repeat(mask,h['mask_step'],1),h['mask_step'],2)
            old_weights={}
            for j,i in enumerate(k+d for k in KEYS[1:] for d in range(6)):
                old_weights[i]=np.minimum(alpha[i],feather(mask[150+j]))*RAMP[j%6]
            states[name]=(alpha,old_weights)
            coverage[name]={scope:dict(active_fraction=float((alpha[:,y0:y1,x0:x1]>0).mean()),
                full_weight_fraction=float((alpha[:,y0:y1,x0:x1]>=.99).mean()))
                for scope,(x0,y0,x1,y1) in {'full':(0,0,512,288),**({'coral':BOX} if case=='video1' else {})}.items()}
            previous_coeff[name]=coefficients(alpha[0],np.zeros((288,512),np.float32),0,KEYS)
        assert np.all(states['K'][0]<=states['N'][0]+1e-6)
        previous=bands(source[0]);rows=[]
        region=np.zeros((288,512),bool);x0,y0,x1,y1=BOX;region[y0:y1,x0:x1]=True
        for i in range(1,150):
            current=bands(source[i])
            mx,my,valid=source_match(source[i-1],source[i],previous,current,np.s_[:,:])
            for name,(alpha,weights) in states.items():
                aligned=warp(alpha[i-1],mx,my)
                before=cv2.remap((alpha[i-1]>0).astype('u1'),mx,my,cv2.INTER_NEAREST).astype(bool)
                now=alpha[i]>0
                delta=np.abs(alpha[i]-aligned)
                coeff=coefficients(alpha[i],weights.get(i,np.zeros((288,512),np.float32)),i,KEYS)
                tv=np.abs(coeff-warp(previous_coeff[name],mx,my)).sum(axis=2)/2
                areas={'full':valid}
                if case=='video1':areas['coral']=valid & region
                for scope,area in areas.items():
                    always=area & now & before
                    rows.append(dict(frame=i,scope=scope,timing=timing(i),method=name,pixels=int(area.sum()),
                        switch_pixels=int((area & (now!=before)).sum()),
                        alpha_delta_sum=float(delta[area].sum()),reference_tv_sum=float(tv[area].sum()),
                        always_pixels=int(always.sum()),always_alpha_delta_sum=float(delta[always].sum()),
                        held_pixels=int((area & (alpha[i]>=.99) & (aligned>=.99)).sum())))
                previous_coeff[name]=coeff
            previous=current
        save_csv(out/'support.csv',rows)
        support={}
        for scope in areas:
            support[scope]={}
            for category in ('all','refresh_with_exit','group_update','group_interior'):
                support[scope][category]={}
                for name in states:
                    rr=[r for r in rows if r['scope']==scope and r['method']==name and (category=='all' or r['timing']==category)]
                    n=sum(r['pixels'] for r in rr);always=sum(r['always_pixels'] for r in rr)
                    support[scope][category][name]=dict(pixels=n,held_fraction=sum(r['held_pixels'] for r in rr)/n,
                        switch_fraction=sum(r['switch_pixels'] for r in rr)/n,
                        alpha_delta=sum(r['alpha_delta_sum'] for r in rr)/n,
                        reference_tv=sum(r['reference_tv_sum'] for r in rr)/n,
                        always_on_alpha_delta=sum(r['always_alpha_delta_sum'] for r in rr)/max(always,1))
        result[case]=dict(coverage=coverage,support=support,N_support_contains_K=True)
        save_json(out/'summary.json',result[case])
        print(case,'source-aligned support measured',flush=True)
    save_json(run/'state_summary.json',dict(complete=True,cases=result,source_correspondence_only_for_evaluation=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    analyze(args.workspace.resolve(),args.run.resolve())
