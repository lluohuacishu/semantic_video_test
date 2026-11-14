"""Three frozen received-guide fusion controls on video1/2/3."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import cv2
import numpy as np
from runtime_speed_common import STAGE,REPO,DATA,FFMPEG,file_sha,load_json,save_json,check_inputs
from received_structure_fusion import PROFILES,fuse_received_structure
import structure_guide_stream as guide_stream

CASES=('video1','video2','video3')
CONFIG_SHA='b5eeb844d0adcdfe97648351a0654a2575a98942e1627c98896ba302555f7b04'
CODE=('received_structure_fusion.py','run_received_structure.py','structure_guide_stream.py',
      'runtime_speed_common.py','resume_q9_archive_transfer.py','evaluate_structure_guide.py',
      'finish_q9_ghost_exit.py','receiver_structure.py')

def freeze(run,parent):
    assert not (run/'execution_plan.json').exists()
    reg=load_json(run/'case_registration.json');assert tuple(reg['cases'])==CASES
    assert load_json(parent/'verification.json')['complete']
    assert load_json(parent/'motion_extension/verification.json')['complete']
    check_inputs(reg['input_sha256'])
    assert file_sha(STAGE/'当前采用配置.json')==CONFIG_SHA
    inputs={}
    for case in CASES:
        rec=load_json(parent/case/'receive/receipt.json')
        for n,h in rec['baseline_sha256'].items():
            p=parent/case/'receive/Q9_R25'/(n+'.npy');assert file_sha(p)==h;inputs[str(p)]=h
        p=parent/'motion_extension'/case/'receive/CSM50/rgb.npy';inputs[str(p)]=file_sha(p)
        p=parent/case/'receive/receipt.json';inputs[str(p)]=file_sha(p)
    old=load_json(parent/'execution_plan.json')
    plan=dict(date='2026-11-13',frozen_before_outputs=True,adaptive_extension=False,parent=str(parent),
        authorization_sha256=file_sha(run/'authorization.json'),registration_sha256=file_sha(run/'case_registration.json'),
        diagnosis_sha256=file_sha(run/'diagnosis/current_structure_errors.json'),
        profiles=PROFILES,cases=list(CASES),new_outputs=9,no_per_clip_tuning=True,
        research_change='Candidate output may contain decoded current guide pixels. This explicitly differs from historical raw-GLC-only pixel reuse. Source images remain forbidden at receiver; old Q9/deployment is unchanged.',
        mechanism='GB25: Q9+.25*(guide-Q9). GS25/GS50: Q9+(.25 or .50)*Gaussian_sigma1(guide-Q9); clip and round once. Fixed current guide, no semantic masks, no source-derived weights.',
        design='GB25 controls direct blend versus same-strength frequency residual GS25; GS50 checks stronger current structure. All registered together.',
        reference='Q9_R25, old CSM50 and received guide itself (not new candidate outputs)',
        review_regions={c:old['review_regions'][c] for c in CASES},review_frames=[30,61,90,120],
        review_interval=[0,150],evaluation_note='Fixed regions include background; not fish masks or recovery counts. All baseline evaluation uses identical support.',
        extra_transmission_bytes=0,new_flow_model_inference=False,no_reencoding=True,
        formal_profile_unchanged=True,long_video_not_run=True,reef_not_run=True,video4_deferred=True,
        limitations=['Direct blending may blur texture. Frequency residual may retain false high-frequency texture or ring at boundaries.',
                     'Guide compression blocks and color errors can enter output; clipping is measured.',
                     'Guide may still lack small fish; no guarantee of recovered objects.',
                     'No new scene-cut handling or end-to-end speed claim. Known clips, not independent holdout.'],
        input_sha256=inputs,code_sha256={n:file_sha(STAGE/n) for n in CODE},current_config_sha256=CONFIG_SHA,
        subjective_review='pending',automatic_selection=None)
    save_json(run/'execution_plan.json',plan)
    snap=run/'source_snapshot';snap.mkdir()
    for n in CODE:shutil.copy2(STAGE/n,snap/n)
    statepath=REPO/'05_资料与索引/会话收束_20261110/接续状态.json'
    shutil.copy2(statepath,run/'state_before_run.json')
    s=load_json(statepath);s.update(status='Q9_received_structure_running',active_experiment_plan=str(run/'execution_plan.json'),
        current_session_work='Testing three registered current decoded-guide fusion controls on video1/2/3; video4 pending separately.')
    s['experiment_execution_gate']=dict(status='running',currently_running=True,new_run_started=True,date='2026-11-13')
    save_json(statepath,s)
    print('FROZEN',list(PROFILES),'three clips / 9 outputs; guide pixels explicitly enabled in candidates',flush=True)

def worker(run,case):
    plan=load_json(run/'execution_plan.json');reg=load_json(run/'case_registration.json');s=reg['cases'][case]
    for n,h in plan['code_sha256'].items():assert file_sha(STAGE/n)==h
    parent=Path(plan['parent']);baseline=parent/case/'receive/Q9_R25';rec=load_json(parent/case/'receive/receipt.json')
    out=run/case/'receive';out.mkdir(parents=True,exist_ok=False)
    cache=Path(s['cache'])/'guide_rgb.npy'
    allowed={Path(s[k]).resolve() for k in ('raw','base','packet','base_receipt')}
    allowed.update((baseline/(n+'.npy')).resolve() for n in ('rgb','alpha','shared'));allowed.add(cache.resolve());reads=set()
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p in allowed:reads.add(str(p));return
        if p.is_relative_to(out):return
        if p.suffix.lower() not in {'.py','.pyc','.pyd','.dll'} and (p.is_relative_to(DATA) or p.suffix.lower() in {'.npy','.npz','.mp4','.glcguide','.glcsnd'}):
            raise RuntimeError('Forbidden guide-fusion input: '+str(p))
    sys.addaudithook(audit);blocked=[]
    for p in [Path(s['source']),Path(s['send_receipt']),run/'diagnosis/current_structure_errors.json',
              parent/'user_feedback_20261112.json',parent/'motion_extension'/case/'receive/CSM50/rgb.npy']:
        try:p.read_bytes()
        except RuntimeError:blocked.append(str(p))
        else:raise AssertionError('read guard failed')
    br=load_json(s['base_receipt'])
    assert br['reference_verified'] and br['source_media_reads_forbidden']
    assert file_sha(s['raw'])==br['decoded_rgb_sha256']
    assert file_sha(s['base'])==rec['base_sha256'] and file_sha(s['packet'])==rec['packet_sha256']
    assert file_sha(cache)==s['cache_sha256']['guide_rgb']
    header,_,images=guide_stream.decode(Path(s['packet']).read_bytes(),file_sha(s['base']),FFMPEG)
    assert np.array_equal(images,np.load(cache,mmap_mode='r'))
    np.save(out/'received_guide.npy',images)
    (out/'Q9_R25').mkdir()
    for n,h in rec['baseline_sha256'].items():
        assert file_sha(baseline/(n+'.npy'))==h;os.link(baseline/(n+'.npy'),out/'Q9_R25'/(n+'.npy'))
    q=np.load(baseline/'rgb.npy',mmap_mode='r');records={}
    for n in PROFILES:
        dest=out/n;dest.mkdir();rgb=np.lib.format.open_memmap(dest/'rgb.npy',mode='w+',dtype=np.uint8,shape=q.shape)
        stats=[];tick=time.perf_counter()
        for i in range(150):rgb[i],v=fuse_received_structure(q[i],images[i],n);stats.append(v)
        rgb.flush();seconds=time.perf_counter()-tick;del rgb
        records[n]=dict(rgb_sha256=file_sha(dest/'rgb.npy'),component_seconds=seconds,
            clipping_channel_fraction=float(np.mean([v['clipped_channel_fraction'] for v in stats])),
            mean_abs_change=float(np.mean([v['mean_abs_change'] for v in stats])),
            guide_pixel_contribution=True)
    save_json(out/'receipt.json',dict(complete=True,case=case,variants=records,
        baseline_sha256=rec['baseline_sha256'],base_sha256=rec['base_sha256'],packet_sha256=rec['packet_sha256'],
        received_guide_sha256=file_sha(out/'received_guide.npy'),guide_redecode_bit_exact=True,
        reads=sorted(reads),allowed=sorted(map(str,allowed)),blocked=blocked,source_reads_blocked=True,
        code_sha256=plan['code_sha256'],extra_transmission_bytes=0,
        pixel_provenance='Q9 raw-GLC-derived detail plus already transmitted current guide RGB; not the old raw-only candidate contract.',
        automatic_adoption=False))
    print('RECEIVED',case,flush=True)

def evaluate(run,case):
    from resume_q9_archive_transfer import score_valid_support,SCOPES
    from evaluate_structure_guide import motion_change_diagnostic
    from compare_deflicker import save_csv
    plan=load_json(run/'execution_plan.json');s=load_json(run/'case_registration.json')['cases'][case]
    out=run/case/'evaluation';out.mkdir(exist_ok=False)
    seq={n:np.load(s[n],mmap_mode='r') for n in ('source','raw')}
    seq['Q9_R25']=np.load(run/case/'receive/Q9_R25/rgb.npy',mmap_mode='r')
    seq['CSM50']=np.load(Path(plan['parent'])/'motion_extension'/case/'receive/CSM50/rgb.npy',mmap_mode='r')
    seq['guide']=np.load(run/case/'receive/received_guide.npy',mmap_mode='r')
    for n in PROFILES:seq[n]=np.load(run/case/'receive'/n/'rgb.npy',mmap_mode='r')
    SCOPES.pop('reef',None)
    if s.get('reef'):SCOPES['reef']=tuple(s['reef'])
    scores,tr,sr,missing=score_valid_support(seq);dynamic,dr=motion_change_diagnostic(seq)
    save_csv(out/'temporal.csv',tr);save_csv(out/'spatial.csv',sr);save_csv(out/'dynamic.csv',dr)
    regions={};rows=[]
    for label,(x0,y0,x1,y1) in plan['review_regions'][case].items():
        regions[label]={}
        for n,a in seq.items():
            vals=[];low=[]
            for i in range(150):
                x=a[i,y0:y1,x0:x1].astype(np.float32);y=seq['source'][i,y0:y1,x0:x1].astype(np.float32)
                v=float(np.abs(x-y).mean());l=float(np.abs(cv2.GaussianBlur(x,(0,0),1)-cv2.GaussianBlur(y,(0,0),1)).mean())
                vals.append(v);low.append(l);rows.append(dict(region=label,method=n,frame=i,rgb_mae=v,sigma1_mae=l))
            regions[label][n]=dict(mean_rgb_mae=float(np.mean(vals)),mean_sigma1_mae=float(np.mean(low)),
                                  review_frames={str(i):dict(rgb_mae=vals[i],sigma1_mae=low[i]) for i in plan['review_frames']})
    save_csv(out/'current_fish_regions.csv',rows)
    rate=load_json(s['send_receipt'])
    summary=dict(complete=True,case=case,scores=scores,dynamic=dynamic,low_support=missing,current_regions=regions,
         total_kbps=rate['total_kbps'],guide_kbps=rate['kbps'],guide_bytes=rate['bytes'],
         guide_is_already_transmitted=True,subjective_review='pending',automatic_selection=None)
    save_json(out/'summary.json',summary)
    print('EVALUATED',case,{n:round(v['coarse_mae'],5) for n,v in scores['full'].items()},flush=True)

def run_all(run):
    plan=load_json(run/'execution_plan.json');check_inputs(plan['input_sha256'])
    for case in plan['cases']:
        subprocess.run([sys.executable,'-B','-u','-X','utf8',str(Path(__file__).resolve()),'worker','--run',str(run),'--case',case],check=True,cwd=REPO)
        evaluate(run,case)
    save_json(run/'summary.json',dict(complete=True,cases={c:load_json(run/c/'evaluation/summary.json') for c in plan['cases']},
          profiles=PROFILES,new_outputs=9,automatic_selection=None,subjective_review='pending',
          long_video_not_run=True,video4_deferred=True,reef_not_run=True))
    print('RECEIVED STRUCTURE ROUND COMPLETE',flush=True)

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

