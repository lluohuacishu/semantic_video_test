"""Adaptive registered motion-mask extension to the current-structure round."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import cv2
import numpy as np
from runtime_speed_common import STAGE,REPO,DATA,file_sha,load_json,save_json
from moving_structure_preservation import SETTINGS,preserve_moving_structure

CODE=('moving_structure_preservation.py','run_moving_structure.py','current_structure_preservation.py',
      'run_current_structure.py','mild_reliability_gate.py','resume_q9_archive_transfer.py',
      'evaluate_structure_guide.py')

def freeze(parent):
    assert load_json(parent/'summary.json')['complete']
    out=parent/'motion_extension';out.mkdir(exist_ok=False)
    shutil.copy2(parent/'case_registration.json',out/'case_registration.json')
    old=load_json(parent/'execution_plan.json')
    plan=dict(old,adaptive_extension=True,frozen_before_outputs=True,profiles={'CSM50':SETTINGS},new_outputs=4,
        first_phase_run=str(parent),first_phase_summary_sha256=file_sha(parent/'summary.json'),
        first_phase_plan_sha256=file_sha(parent/'execution_plan.json'),
        code_sha256={n:file_sha(STAGE/n) for n in CODE},
        mechanism='CS50 correction weighted by local guide flow departure from robust dominant affine frame motion. Deterministic three trimmed least-squares fits on alpha>0.5 grid; fixed .75..3px risk and sigma2 spatial feather.',
        motivation='CS50 reduces current-region source errors but video3/reef temporal regressions remain. Test one spatial motion-support constraint instead of global structure replacement.',
        limitations=['Local residual is not fish segmentation; parallax or faulty flow can trigger.',
                     'Low-rate guide can miss small fish, so this cannot restore missing motion evidence.',
                     'No temporal gate or scene-cut handling; masks may change at anchors.',
                     'A dominant object can bias affine background estimation.'])
    save_json(out/'execution_plan.json',plan)
    snap=out/'source_snapshot';snap.mkdir()
    for n in CODE:shutil.copy2(STAGE/n,snap/n)
    print('FROZEN ADAPTIVE CSM50 four outputs',flush=True)

def worker(parent,case):
    root=parent/'motion_extension';plan=load_json(root/'execution_plan.json')
    reg=load_json(root/'case_registration.json');s=reg['cases'][case]
    for n,h in plan['code_sha256'].items():assert file_sha(STAGE/n)==h
    rec=load_json(parent/case/'receive/receipt.json');baseline=parent/case/'receive/Q9_R25'
    out=root/case/'receive';out.mkdir(parents=True,exist_ok=False)
    flowpath=Path(s['cache'])/'to_anchor.npy'
    allowed={Path(s[k]).resolve() for k in ('raw','base','packet','base_receipt')}
    allowed.update((baseline/(n+'.npy')).resolve() for n in ('rgb','alpha','shared'));allowed.add(flowpath.resolve());reads=set()
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p in allowed:reads.add(str(p));return
        if p.is_relative_to(out):return
        if p.suffix.lower() not in {'.py','.pyc','.pyd','.dll'} and (p.is_relative_to(DATA) or p.suffix.lower() in {'.npy','.npz','.mp4','.glcguide','.glcsnd'}):
            raise RuntimeError('Forbidden moving-structure receiver input: '+str(p))
    sys.addaudithook(audit);blocked=[]
    for p in [Path(s['source']),Path(s['send_receipt']),parent/'user_feedback_20261112.json',Path(plan['parent'])/'user_event_registration.json',
              Path(plan['parent'])/case/'receive/ML3/rgb.npy']:
        try:p.read_bytes()
        except RuntimeError:blocked.append(str(p))
        else:raise AssertionError('guard failed')
    br=load_json(s['base_receipt'])
    assert br['reference_verified'] and file_sha(s['raw'])==br['decoded_rgb_sha256']
    assert file_sha(s['base'])==rec['base_sha256'] and file_sha(s['packet'])==rec['packet_sha256']
    assert file_sha(flowpath)==s['cache_sha256']['to_anchor']
    (out/'Q9_R25').mkdir()
    for n,h in rec['baseline_sha256'].items():
        assert file_sha(baseline/(n+'.npy'))==h;os.link(baseline/(n+'.npy'),out/'Q9_R25'/(n+'.npy'))
    raw=np.load(s['raw'],mmap_mode='r');q=np.load(baseline/'rgb.npy',mmap_mode='r');alpha=np.load(baseline/'alpha.npy',mmap_mode='r')
    flow=np.load(flowpath,mmap_mode='r')
    dest=out/'CSM50';dest.mkdir()
    rgb=np.lib.format.open_memmap(dest/'rgb.npy',mode='w+',dtype=np.uint8,shape=raw.shape)
    weight=np.lib.format.open_memmap(dest/'motion_weight.npy',mode='w+',dtype=np.float32,shape=raw.shape[:3])
    fits=[]
    for i in range(150):
        rgb[i],weight[i],fit=preserve_moving_structure(raw[i],q[i],flow[i],alpha[i]);fits.append(dict(frame=i,**fit))
        assert np.all(rgb[i]>=np.minimum(raw[i],q[i])) and np.all(rgb[i]<=np.maximum(raw[i],q[i]))
    rgb.flush();weight.flush();del rgb,weight
    save_json(out/'motion_fits.json',fits)
    save_json(out/'receipt.json',dict(complete=True,case=case,baseline_sha256=rec['baseline_sha256'],
       base_sha256=rec['base_sha256'],packet_sha256=rec['packet_sha256'],flow_sha256=file_sha(flowpath),
       output_sha256={n:file_sha(dest/(n+'.npy')) for n in ('rgb','motion_weight')},
       source_reads_blocked=True,blocked=blocked,reads=sorted(reads),allowed=sorted(map(str,allowed)),
       code_sha256=plan['code_sha256'],extra_transmission_bytes=0,automatic_adoption=False))
    print('RECEIVED MOTION',case,flush=True)

def run_all(parent):
    root=parent/'motion_extension';plan=load_json(root/'execution_plan.json')
    import run_current_structure as evaluation
    evaluation.PROFILES={'CSM50':SETTINGS}
    for case in plan['cases']:
        subprocess.run([sys.executable,'-B','-u','-X','utf8',str(Path(__file__).resolve()),'worker','--run',str(parent),'--case',case],check=True,cwd=REPO)
        evaluation.evaluate(root,case)
    save_json(root/'summary.json',dict(complete=True,cases={c:load_json(root/c/'evaluation/summary.json') for c in plan['cases']},
             profiles={'CSM50':SETTINGS},new_outputs=4,adaptive_extension=True,automatic_selection=None,
             subjective_review='pending',long_video_not_run=True,video4_deferred=True))
    print('MOTION EXTENSION COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','worker','run'));p.add_argument('--run',type=Path,required=True);p.add_argument('--case')
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    try:
        if a.action=='freeze':freeze(a.run)
        elif a.action=='worker':worker(a.run,a.case)
        else:run_all(a.run)
    except BaseException as e:
        out=a.run/'motion_extension'/a.case/'receive' if a.action=='worker' else a.run/'motion_extension'
        if out.exists():save_json(out/('failure_'+a.action+'.json'),dict(error=str(e),traceback=traceback.format_exc()))
        raise

