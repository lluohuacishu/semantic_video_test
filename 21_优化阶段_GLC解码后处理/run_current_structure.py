"""Fixed CS50/CS100 watchability experiment, not a ghost-threshold search."""
import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import cv2
import numpy as np
from runtime_speed_common import STAGE,REPO,DATA,file_sha,load_json,save_json,check_inputs
from current_structure_preservation import PROFILES,preserve_current

CASES=('video1','video2','video3','reef_spotted_fish')
REGIONS={
 'video1':{'fish_and_water':[20,20,300,288]},
 'video2':{'main_fish':[130,110,430,288],'tail':[140,145,285,288]},
 'video3':{'foreground_fish':[0,160,270,288]},
 'reef_spotted_fish':{'main_fish':[190,30,330,245],'small_fish':[310,90,512,225]},
}
CONFIG_SHA='b5eeb844d0adcdfe97648351a0654a2575a98942e1627c98896ba302555f7b04'
CODE=('current_structure_preservation.py','run_current_structure.py','run_q9_ghost_exit.py',
      'runtime_speed_common.py','resume_q9_archive_transfer.py','evaluate_structure_guide.py',
      'finish_q9_ghost_exit.py','receiver_structure.py')

def freeze(run,parent):
    assert not (run/'execution_plan.json').exists()
    reg=load_json(run/'case_registration.json')
    assert tuple(reg['cases'])==CASES
    assert load_json(parent/'verification.json')['complete']
    check_inputs(reg['input_sha256'])
    assert file_sha(STAGE/'当前采用配置.json')==CONFIG_SHA
    inputs={}
    for case in CASES:
        rec=load_json(parent/case/'receive/receipt.json')
        for n,h in rec['baseline_sha256'].items():
            p=parent/case/'receive/Q9_R25'/(n+'.npy');assert file_sha(p)==h
            inputs[str(p)]=h
        inputs[str(parent/case/'receive/receipt.json')]=file_sha(parent/case/'receive/receipt.json')
        p=parent/case/'receive/ML3/rgb.npy'
        inputs[str(p)]=file_sha(p)
    plan=dict(date='2026-11-12',frozen_before_outputs=True,parent=str(parent),
       feedback_sha256=file_sha(run/'user_feedback_20261112.json'),
       registration_sha256=file_sha(run/'case_registration.json'),
       profiles=PROFILES,cases=list(CASES),new_outputs=8,no_per_clip_tuning=True,
       mechanism='Q9 + strength * clamp(Gaussian_sigma1(raw-Q9), min(raw-Q9,0), max(raw-Q9,0)); one final round. CS100 full current-structure diagnostic control, CS50 half-strength candidate. No source/semantic ROI/time gate.',
       rationale='Prioritize visible current fish and overall watchability; stop optimizing old empty positions. Keep Q9 fine detail while restoring smoothed current raw differences.',
       reference='Q9_R25 and previous ML3 (existing files, not regenerated)',
       review_regions=REGIONS,review_frames=[30,61,90,120],review_interval=[0,150],
       review_note='Broad fixed visual regions include background, not fish masks or fish counts. No prior empty-position ROI in current decision.',
       no_source_or_roi_in_receiver=True,new_flow_model_inference=False,extra_transmission_bytes=0,
       long_video_not_run=True,video4_deferred=True,seagrass_excluded=True,
       limitations=['Current raw may already have lost small fish; this cannot reconstruct absent evidence.',
                    'Current coarse change may reintroduce flicker and remaining Q9 high-frequency ghosts.',
                    'Per-channel bound makes this a constrained frequency correction, not an exact frequency partition.',
                    'Known development clips only; not generalization or real-time acceptance.'],
       input_sha256=inputs,code_sha256={n:file_sha(STAGE/n) for n in CODE},current_config_sha256=CONFIG_SHA,
       subjective_review='pending',automatic_selection=None)
    save_json(run/'execution_plan.json',plan)
    snap=run/'source_snapshot';snap.mkdir()
    for n in CODE:shutil.copy2(STAGE/n,snap/n)
    statepath=REPO/'05_资料与索引/会话收束_20261110/接续状态.json'
    shutil.copy2(statepath,run/'state_before_run.json')
    s=load_json(statepath);s.update(status='Q9_current_structure_running',active_experiment_plan=str(run/'execution_plan.json'),
       current_session_work='Prioritize video1/2/3 watchability plus reef fish presence; CS50/CS100 on four clips. Video4 dense coral deferred.')
    s['experiment_execution_gate']=dict(status='running',currently_running=True,new_run_started=True,date='2026-11-12')
    save_json(statepath,s)
    print('FROZEN',list(PROFILES),'four cases / 8 outputs',flush=True)

def worker(run,case):
    plan=load_json(run/'execution_plan.json');reg=load_json(run/'case_registration.json');s=reg['cases'][case]
    for n,h in plan['code_sha256'].items():assert file_sha(STAGE/n)==h
    parent=Path(plan['parent']);baseline=parent/case/'receive/Q9_R25'
    rec=load_json(parent/case/'receive/receipt.json')
    out=run/case/'receive';out.mkdir(parents=True,exist_ok=False)
    allowed={Path(s[k]).resolve() for k in ('raw','base','packet','base_receipt')}
    allowed.update((baseline/(n+'.npy')).resolve() for n in ('rgb','alpha','shared'));reads=set()
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p in allowed:reads.add(str(p));return
        if p.is_relative_to(out):return
        if p.suffix.lower() not in {'.py','.pyc','.pyd','.dll'} and (p.is_relative_to(DATA) or p.suffix.lower() in {'.npy','.npz','.mp4','.glcguide','.glcsnd'}):
            raise RuntimeError('Forbidden current-structure receiver input: '+str(p))
    sys.addaudithook(audit);blocked=[]
    for p in [Path(s['source']),Path(s['send_receipt']),run/'user_feedback_20261112.json',
              parent/'user_event_registration.json',parent/case/'receive/ML3/rgb.npy']:
        try:p.read_bytes()
        except RuntimeError:blocked.append(str(p))
        else:raise AssertionError('read guard failed')
    br=load_json(s['base_receipt'])
    assert br['reference_verified'] and br['source_media_reads_forbidden']
    assert file_sha(s['raw'])==br['decoded_rgb_sha256']
    assert file_sha(s['base'])==rec['base_sha256'] and file_sha(s['packet'])==rec['packet_sha256']
    (out/'Q9_R25').mkdir()
    for n,h in rec['baseline_sha256'].items():
        assert file_sha(baseline/(n+'.npy'))==h
        os.link(baseline/(n+'.npy'),out/'Q9_R25'/(n+'.npy'))
    raw=np.load(s['raw'],mmap_mode='r');q=np.load(baseline/'rgb.npy',mmap_mode='r')
    records={}
    for n in PROFILES:
        dest=out/n;dest.mkdir()
        rgb=np.lib.format.open_memmap(dest/'rgb.npy',mode='w+',dtype=np.uint8,shape=raw.shape)
        tick=time.perf_counter();changed=0
        for i in range(150):
            rgb[i]=preserve_current(raw[i],q[i],n)
            assert np.all(rgb[i]>=np.minimum(raw[i],q[i])) and np.all(rgb[i]<=np.maximum(raw[i],q[i]))
            assert np.array_equal(rgb[i][raw[i]==q[i]],raw[i][raw[i]==q[i]])
            changed+=int(np.any(rgb[i]!=q[i],axis=2).sum())
        rgb.flush();seconds=time.perf_counter()-tick;del rgb
        records[n]=dict(rgb_sha256=file_sha(dest/'rgb.npy'),changed_pixels_fraction=changed/(150*288*512),
                        component_seconds=seconds,includes_base_or_flow=False)
    save_json(out/'receipt.json',dict(complete=True,case=case,profiles=PROFILES,variants=records,
       baseline_sha256=rec['baseline_sha256'],base_sha256=file_sha(s['base']),packet_sha256=file_sha(s['packet']),
       reads=sorted(reads),allowed=sorted(map(str,allowed)),blocked=blocked,source_reads_blocked=True,
       code_sha256=plan['code_sha256'],no_new_packet=True,automatic_adoption=False))
    print('RECEIVED',case,flush=True)

def evaluate(run,case):
    from resume_q9_archive_transfer import score_valid_support,SCOPES
    from evaluate_structure_guide import motion_change_diagnostic
    from compare_deflicker import save_csv
    reg=load_json(run/'case_registration.json');plan=load_json(run/'execution_plan.json');s=reg['cases'][case]
    out=run/case/'evaluation';out.mkdir(exist_ok=False)
    seq={n:np.load(s[n],mmap_mode='r') for n in ('source','raw')}
    seq['Q9_R25']=np.load(run/case/'receive/Q9_R25/rgb.npy',mmap_mode='r')
    seq['ML3']=np.load(Path(plan['parent'])/case/'receive/ML3/rgb.npy',mmap_mode='r')
    for n in PROFILES:seq[n]=np.load(run/case/'receive'/n/'rgb.npy',mmap_mode='r')
    SCOPES.pop('reef',None)
    if s.get('reef'):SCOPES['reef']=tuple(s['reef'])
    scores,temporal,spatial,missing=score_valid_support(seq);dynamic,dr=motion_change_diagnostic(seq)
    save_csv(out/'temporal.csv',temporal);save_csv(out/'spatial.csv',spatial);save_csv(out/'dynamic.csv',dr)
    crops={};rows=[]
    for label,(x0,y0,x1,y1) in plan['review_regions'][case].items():
        crops[label]={}
        for n,arr in seq.items():
            vals=[];lows=[]
            for i in range(150):
                a=arr[i,y0:y1,x0:x1].astype(np.float32);b=seq['source'][i,y0:y1,x0:x1].astype(np.float32)
                v=float(np.abs(a-b).mean());lv=float(np.abs(cv2.GaussianBlur(a,(0,0),1)-cv2.GaussianBlur(b,(0,0),1)).mean())
                vals.append(v);lows.append(lv);rows.append(dict(region=label,frame=i,method=n,rgb_mae=v,sigma1_mae=lv))
            crops[label][n]=dict(mean_rgb_mae=float(np.mean(vals)),mean_sigma1_mae=float(np.mean(lows)),
                review_frames={str(i):dict(rgb_mae=vals[i],sigma1_mae=lows[i]) for i in plan['review_frames']})
    save_csv(out/'current_fish_regions.csv',rows)
    rate=load_json(s['send_receipt'])
    summary=dict(complete=True,case=case,scores=scores,dynamic=dynamic,low_support=missing,
                 current_regions=crops,total_kbps=rate['total_kbps'],guide_kbps=rate['kbps'],guide_bytes=rate['bytes'],
                 scope='Four known 10s clips; watchability/current presence, not empty-footprint optimization',
                 subjective_review='pending',automatic_selection=None)
    save_json(out/'summary.json',summary)
    print('EVALUATED',case,{n:round(v['coarse_mae'],5) for n,v in scores['full'].items()},flush=True)

def run_all(run):
    plan=load_json(run/'execution_plan.json')
    check_inputs(plan['input_sha256'])
    for case in plan['cases']:
        subprocess.run([sys.executable,'-B','-u','-X','utf8',str(Path(__file__).resolve()),'worker','--run',str(run),'--case',case],check=True,cwd=REPO)
        evaluate(run,case)
    save_json(run/'summary.json',dict(complete=True,cases={c:load_json(run/c/'evaluation/summary.json') for c in plan['cases']},
        profiles=PROFILES,new_outputs=8,automatic_selection=None,subjective_review='pending',
        long_video_not_run=True,video4_deferred=True))
    print('CURRENT STRUCTURE ROUND COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','worker','run'));p.add_argument('--run',type=Path,required=True)
    p.add_argument('--parent',type=Path);p.add_argument('--case')
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    try:
        if a.action=='freeze':freeze(a.run,a.parent)
        elif a.action=='worker':worker(a.run,a.case)
        else:run_all(a.run)
    except BaseException as e:
        out=a.run/a.case/'receive' if a.action=='worker' else a.run
        if out.exists():save_json(out/('failure_'+a.action+'.json'),dict(error=str(e),traceback=traceback.format_exc()))
        raise

