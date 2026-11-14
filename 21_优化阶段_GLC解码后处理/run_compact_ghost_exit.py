"""One explicitly adaptive follow-up: restrict EL12 to compact strong conflicts."""
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
from raw_conflict_exit import exit_frame

SETTINGS=dict(parent='EL12',support_min=.10,seed_min=.80,area_min=8,area_max=1024,connectivity=8)


def compact(raw,q9):
    _,risk=exit_frame(raw,q9,'EL12')
    n,labels,stats,_=cv2.connectedComponentsWithStats((risk>=SETTINGS['support_min']).astype(np.uint8),SETTINGS['connectivity'])
    maxima=np.zeros(n,np.float32);np.maximum.at(maxima,labels.ravel(),risk.ravel())
    keep=(stats[:,cv2.CC_STAT_AREA]>=SETTINGS['area_min'])&(stats[:,cv2.CC_STAT_AREA]<=SETTINGS['area_max'])&(maxima>=SETTINGS['seed_min'])
    keep[0]=False;weight=risk*keep[labels]
    rgb=np.rint(np.clip(q9.astype(np.float32)*(1-weight[...,None])+raw.astype(np.float32)*weight[...,None],0,255)).astype(np.uint8)
    return rgb,weight


def freeze(parent):
    assert load_json(parent/'summary.json')['complete']
    out=parent/'compact_extension';out.mkdir(exist_ok=False)
    shutil.copy2(parent/'case_registration.json',out/'case_registration.json')
    old=load_json(parent/'execution_plan.json');code={n:file_sha(STAGE/n) for n in ('run_compact_ghost_exit.py','raw_conflict_exit.py','run_q9_ghost_exit.py','resume_q9_archive_transfer.py','evaluate_structure_guide.py')}
    save_json(out/'execution_plan.json',dict(date='2026-11-12',adaptive_extension=True,frozen_before_extension_outputs=True,
        motivation='Phase1 lowered user-region source errors but broadly degraded temporal stability. Test one compact-support constraint, without altering phase1 outputs.',
        parent_summary_sha256=file_sha(parent/'summary.json'),parent_plan_sha256=file_sha(parent/'execution_plan.json'),profiles={'C32':SETTINGS},
        review_boxes=old['review_boxes'],review_timing=old['review_timing'],code_sha256=code,
        same_parameters_all_five_cases=True,not_semantic_segmentation=True,no_source_or_user_ROI=True,no_new_bytes=True,
        mechanism='Keep only connected raw-conflict supports of 8..1024 pixels with peak>=0.8. Max area is 32x32 pixels at fixed 512x288; masks recomputed per frame; no temporal hysteresis.',
        limitations='May miss larger artifacts and may switch as components merge/split; must retain motion/stability regression checks. No automatic adoption.'))
    snap=out/'source_snapshot';snap.mkdir()
    for name in code:shutil.copy2(STAGE/name,snap/name)
    print('COMPACT FROZEN C32',SETTINGS,flush=True)


def worker(parent,case):
    root=parent/'compact_extension';s=load_json(root/'case_registration.json')['cases'][case]
    plan=load_json(root/'execution_plan.json');rec=load_json(parent/case/'receive/receipt.json')
    assert rec['baseline_replay_bit_exact'] and rec['source_reads_blocked']
    for n,h in plan['code_sha256'].items():assert file_sha(STAGE/n)==h
    out=root/case/'receive';out.mkdir(parents=True,exist_ok=False)
    baseline=parent/case/'receive/Q9_R25'
    allowed={Path(s[k]).resolve() for k in ('raw','base','base_receipt','packet')}
    allowed.update((baseline/(n+'.npy')).resolve() for n in ('rgb','alpha','shared'))
    allowed.add((parent/case/'receive/receipt.json').resolve());reads=set()
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p in allowed:reads.add(str(p));return
        if p.is_relative_to(out):return
        if p.suffix.lower() not in {'.py','.pyc','.pyd','.dll'} and (p.is_relative_to(DATA) or p.suffix.lower() in {'.npy','.npz','.mp4','.glcsnd','.glcguide'}):
            raise RuntimeError('Forbidden compact exit input: '+str(p))
    sys.addaudithook(audit);blocked=[]
    for p in [Path(s['source']),parent/'diagnosis/events.json',parent/'user_event_registration.json',Path(s['send_receipt'])]:
        try:p.read_bytes()
        except RuntimeError:blocked.append(str(p))
        else:raise AssertionError('guard failed')
    assert file_sha(s['raw'])==load_json(s['base_receipt'])['decoded_rgb_sha256']
    assert file_sha(s['base'])==rec['base_sha256'] and file_sha(s['packet'])==rec['packet_sha256']
    (out/'Q9_R25').mkdir()
    for name,h in rec['baseline_sha256'].items():
        assert file_sha(baseline/(name+'.npy'))==h
        os.link(baseline/(name+'.npy'),out/'Q9_R25'/(name+'.npy'))
    raw=np.load(s['raw'],mmap_mode='r');q=np.load(baseline/'rgb.npy',mmap_mode='r');alpha=np.load(baseline/'alpha.npy',mmap_mode='r')
    rgb=np.empty_like(raw);weights=np.empty(raw.shape[:3],np.float32)
    for i in range(150):rgb[i],weights[i]=compact(raw[i],q[i])
    assert np.isfinite(weights).all() and np.all((weights>=0)&(weights<=1))
    assert np.array_equal(rgb[weights==0],q[weights==0]) and np.array_equal(rgb[weights==1],raw[weights==1])
    assert np.array_equal(rgb[alpha==0],raw[alpha==0])
    assert np.all(np.abs(rgb.astype(np.int16)-raw.astype(np.int16))<=np.abs(q.astype(np.int16)-raw.astype(np.int16)))
    dest=out/'C32';dest.mkdir();np.save(dest/'rgb.npy',rgb);np.save(dest/'exit_weight.npy',weights)
    save_json(out/'receipt.json',dict(complete=True,case=case,profile='C32',settings=SETTINGS,source_reads_blocked=True,
        blocked=blocked,reads=sorted(reads),allowed=sorted(map(str,allowed)),baseline_replay_reused_from=str(baseline),
        baseline_sha256=rec['baseline_sha256'],base_sha256=file_sha(s['base']),packet_sha256=file_sha(s['packet']),
        outputs_sha256={n:file_sha(dest/(n+'.npy')) for n in ('rgb','exit_weight')},
        changed_pixels_fraction=float(np.any(rgb!=q,axis=3).mean()),active_exit_fraction=float((weights>0).mean()),
        full_exit_fraction=float((weights==1).mean()),code_sha256=plan['code_sha256']))
    print('COMPACT RECEIVED',case,round(np.any(rgb!=q,axis=3).mean()*100,3),flush=True)


def run_all(parent):
    root=parent/'compact_extension';reg=load_json(root/'case_registration.json')
    import run_q9_ghost_exit as evaluation
    evaluation.PROFILES={'C32':SETTINGS}
    for case in reg['cases']:
        subprocess.run([sys.executable,'-B','-u','-X','utf8',str(Path(__file__).resolve()),'worker','--run',str(parent),'--case',case],check=True,cwd=REPO)
        evaluation.evaluate(root,case)
    save_json(root/'summary.json',dict(complete=True,adaptive_extension=True,cases={c:load_json(root/c/'evaluation/summary.json') for c in reg['cases']},
        automatic_selection=None,subjective_review='pending',long_video_not_run=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('freeze','run','worker'));p.add_argument('--run',type=Path,required=True);p.add_argument('--case')
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    try:
        if a.action=='freeze':freeze(a.run)
        elif a.action=='worker':worker(a.run,a.case)
        else:run_all(a.run)
    except BaseException as e:
        out=a.run/'compact_extension'/a.case/'receive' if a.action=='worker' else a.run/'compact_extension'
        save_json(out/('failure_'+a.action+'.json'),dict(error=str(e),traceback=traceback.format_exc()));raise
