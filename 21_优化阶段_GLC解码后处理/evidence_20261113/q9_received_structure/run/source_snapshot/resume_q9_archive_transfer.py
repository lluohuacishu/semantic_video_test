"""Resume fixed Q9 cases; explicitly exclude insufficient-support temporal cells."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np

from runtime_speed_common import STAGE,REPO,FFMPEG,file_sha,load_json,save_json
from run_complete_runtime_benchmark import Service
from run_generalization_extra import encode_packet
from evaluate_fullframe_reference import SCOPES
from audit_deflicker_texture import normalized,source_match
from temporal_filter import warp
from probe_reef import bands
from compare_deflicker import save_csv
from evaluate_structure_guide import motion_change_diagnostic
from finish_generalization_factorial import sample_sheet
from finish_q9_long_transfer import write_preview


def score_valid_support(seq):
    temporal=[];spatial=[];missing=[];previous=None
    for i in range(150):
        current={k:bands(v[i]) for k,v in seq.items()}
        if i:
            mx,my,common=source_match(seq['source'][i-1],seq['source'][i],previous['source'],current['source'],np.s_[:,:])
            aligned={k:warp(v,mx,my) for k,v in previous.items()};delta=current['source']-aligned['source']
        for scope,(x0,y0,x1,y1) in SCOPES.items():
            crop=np.s_[y0:y1,x0:x1]
            for name,video in seq.items():
                error=video[i][crop].astype(np.float32)-seq['source'][i][crop]
                spatial.append(dict(frame=i,scope=scope,method=name,pixels=(x1-x0)*(y1-y0),rgb_sse=float(np.square(error).sum()),
                    fine_sum=float(np.abs(current[name][crop][...,2]).sum())))
            if i:
                mask=common[crop];count=int(mask.sum())
                if count<64:
                    missing.append(dict(frame=i,scope=scope,pixels=count,reason='Below unchanged 64-pixel shared-support minimum'))
                    continue
                for name in seq:
                    a=aligned[name][crop][mask];b=current[name][crop][mask];error=np.abs(b-a-delta[crop][mask])
                    temporal.append(dict(frame=i,scope=scope,method=name,pixels=count,coarse_sum=float(error[:,0].sum()),
                        fine_sum=float(error[:,2].sum()),full_sum=float(error[:,3].sum()),fine_ncc=float((normalized(a[:,2:3])*normalized(b[:,2:3])).mean())))
        previous=current
    result={}
    for scope in SCOPES:
        result[scope]={}
        for name in seq:
            rows=[r for r in temporal if r['scope']==scope and r['method']==name]
            sr=[r for r in spatial if r['scope']==scope and r['method']==name]
            n=sum(r['pixels'] for r in rows);sn=sum(r['pixels'] for r in sr);mse=sum(r['rgb_sse'] for r in sr)/(3*sn)
            result[scope][name]=dict(pairs=len(rows),excluded_pairs=149-len(rows),valid_pixels=n,
                coarse_mae=sum(r['coarse_sum'] for r in rows)/n if n else None,
                fine_mae=sum(r['fine_sum'] for r in rows)/n if n else None,
                full_mae=sum(r['full_sum'] for r in rows)/n if n else None,
                fine_ncc=sum(r['fine_ncc']*r['pixels'] for r in rows)/n if n else None,
                fine_amplitude=sum(r['fine_sum'] for r in sr)/sn,
                psnr_db=float(10*np.log10(255**2/mse)) if mse else None)
    return result,temporal,spatial,missing


def evaluate(run,case):
    root=run/case;out=root/'evaluation_valid_support';out.mkdir(exist_ok=False)
    rec=load_json(root/'post/receipt.json')
    for name,h in rec['outputs_sha256'].items():assert file_sha(root/'post'/(name+'.npy'))==h
    seq=dict(source=np.load(root/'source.npy',mmap_mode='r'),raw=np.load(root/'base_receive/decoded_rgb.npy',mmap_mode='r'),
             Q9_R25=np.load(root/'post/rgb.npy',mmap_mode='r'))
    scores,temporal,spatial,missing=score_valid_support(seq);dynamic,rows=motion_change_diagnostic(seq)
    save_csv(out/'temporal.csv',temporal);save_csv(out/'spatial.csv',spatial);save_csv(out/'dynamic.csv',rows)
    save_json(out/'missing_support.json',missing)
    send=load_json(root/'guide/send.json');base_bytes=(root/'send/base.glcsnd').stat().st_size
    labels=dict(source='Source',raw='Original GLC',Q9_R25='Fixed Q9_R25');previews={}
    for name,names in [('source_raw_Q9.mp4',list(seq)),('raw_vs_Q9.mp4',['raw','Q9_R25'])]:
        # Reuse bounded-memory writer, with a 150-frame sampling wrapper.
        from run_p0 import preview
        from verify_video1_persistent_structure import inspect_preview
        panels=[seq[n] for n in names]
        preview(FFMPEG,out/name,panels,15,0,[labels[n] for n in names])
        previews[name]=inspect_preview(FFMPEG,out/name,panels)
    sample_sheet(out/'contact.png',case,seq,dict(source='原片',raw='原GLC',Q9_R25='固定Q9_R25'),(20,64,80,128,149))
    value=dict(complete=True,case=case,profile='Q9_R25',scores=scores,dynamic=dynamic,low_support=missing,
        base_bytes=base_bytes,base_kbps=base_bytes*8/10000,guide_bytes=send['bytes'],guide_kbps=send['kbps'],total_kbps=send['total_kbps'],
        per_channel_budget_pass=base_bytes<=50000 and send['bytes']<=50000,source_scope='Archived source, first10s; not independent heldout',
        evaluation_dir='evaluation_valid_support',evaluation_note='Original evaluator stopped at low support. Same 64-pixel threshold and same metric; missing scope/frame entries excluded for ALL methods and disclosed.',
        subjective_review='pending',previews=previews,evaluation_code_sha256=file_sha(__file__))
    save_json(out/'summary.json',value);return value


def main(parent):
    run=parent/'archive_transfer';reg=load_json(run/'registration.json')
    assert all(file_sha(STAGE/name)==h for name,h in reg['code_sha256'].items())
    resume=run/'evaluation_resume';resume.mkdir(exist_ok=False)
    shutil.copy2(__file__,resume/Path(__file__).name)
    save_json(resume/'plan.json',dict(reason='Original evaluator raises on top_left frame51 shared support <64.',
        inference_unchanged=True,first_case_inference_reused=True,second_case_same_preregistered_Q9_R25=True,
        low_support_policy='Keep threshold 64; missing only for matching frame and scope across all methods. Never replace with zero.',
        original_registration_sha256=file_sha(run/'registration.json'),code_sha256=file_sha(__file__)))
    save_json(run/'reef_spotted_fish/evaluation/failure.json',dict(stage='evaluation',error='ValueError: Insufficient shared texture: top_left frame 51',
        inference_completed=True,original_code_sha256=reg['code_sha256']['run_q9_archive_transfer.py'],
        original_output_preserved=True,recovery='evaluation_valid_support with explicit shared-support exclusions'))
    results={}
    for case,spec in reg['cases'].items():
        root=run/case
        if not (root/'post/receipt.json').exists():
            assert not (root/'send').exists() and not (root/'base_receive').exists() and not (root/'post').exists()
            for role,folder in [('sender','send'),('receiver','base_receive')]:
                logs=root/(role+'_logs');logs.mkdir();service=Service(role,logs)
                try:
                    request=dict(output=str(root/folder),source=spec['source'],source_sha256=file_sha(spec['source'])) if role=='sender' else dict(output=str(root/folder),base=str(root/'send/base.glcsnd'))
                    service.request(request)
                finally:service.close()
            encode_packet(np.load(spec['source'],mmap_mode='r'),root/'send/base.glcsnd',root/'guide',9)
            with (root/'worker.log').open('x',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-B','-u','-X','utf8',str(STAGE/'run_q9_archive_transfer.py'),'worker','--run',str(parent),'--case',case],
                    stdout=log,stderr=subprocess.STDOUT,check=True,cwd=REPO)
        results[case]=evaluate(run,case)
        print('ARCHIVE COMPLETE',case,'rate',results[case]['total_kbps'],'coarse_mae',
            {n:v['coarse_mae'] for n,v in results[case]['scores']['full'].items()},'excluded',len(results[case]['low_support']),flush=True)
    save_json(run/'summary.json',dict(complete=True,cases=results,no_per_clip_tuning=True,subjective_review='pending',
        evaluation_resume_plan_sha256=file_sha(resume/'plan.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);main(args.run)
