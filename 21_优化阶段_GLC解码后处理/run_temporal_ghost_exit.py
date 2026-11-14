"""Run a frozen three-profile temporal-exit ablation on the five Q9 clips."""
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
from temporal_conflict_exit import TemporalExit,PROFILES,SETTINGS

CONFIG_SHA='b5eeb844d0adcdfe97648351a0654a2575a98942e1627c98896ba302555f7b04'
CODE=['temporal_conflict_exit.py','run_temporal_ghost_exit.py','run_q9_ghost_exit.py',
      'raw_conflict_exit.py','mild_reliability_gate.py','receiver_structure.py','cached_t3.py',
      'runtime_speed_common.py','resume_q9_archive_transfer.py','evaluate_structure_guide.py',
      'finish_q9_ghost_exit.py']

def freeze(run,parent):
    assert not run.exists()
    assert load_json(parent/'verification.json')['complete']
    reg=load_json(parent/'case_registration.json')
    check_inputs(reg['input_sha256'])
    assert file_sha(STAGE/'当前采用配置.json')==CONFIG_SHA
    run.mkdir()
    shutil.copy2(parent/'case_registration.json',run/'case_registration.json')
    shutil.copy2(parent/'user_event_registration.json',run/'user_event_registration.json')
    previous=load_json(parent/'execution_plan.json')
    inputs={}
    for case,s in reg['cases'].items():
        rec=load_json(parent/case/'receive/receipt.json')
        assert rec['baseline_replay_bit_exact']
        for n,h in rec['baseline_sha256'].items():
            p=parent/case/'receive/Q9_R25'/(n+'.npy')
            assert file_sha(p)==h
            inputs[str(p)]=h
        inputs[str(parent/case/'receive/receipt.json')]=file_sha(parent/case/'receive/receipt.json')
    plan=dict(date='2026-11-12',frozen_before_candidate_inference=True,
              parent=str(parent),parent_result_sha256=file_sha(parent/'round_result.json'),
              registration_sha256=file_sha(run/'case_registration.json'),
              user_event_sha256=file_sha(run/'user_event_registration.json'),
              profiles=PROFILES,settings=SETTINGS,cases=list(reg['cases']),
              new_candidate_outputs=15,adaptive_extension=False,no_per_clip_tuning=True,
              reference='Prior independently reproduced, bit-exact Q9_R25 receiver baseline',
              baseline_input_sha256=inputs,current_config_sha256=CONFIG_SHA,
              code_sha256={n:file_sha(STAGE/n) for n in CODE},
              review_boxes=previous['review_boxes'],review_timing=previous['review_timing'],
              mechanism='Guide-flow composition aligns the last two received raw/Q9 residuals. Three-frame signed conflict confirmation, bounded rise/fall. M3 gates by latched excess raw change over Q9 change. ML3 applies only the averaged sigma1 residual, channelwise bounded toward current raw.',
              ablation='P3 temporal persistence -> M3 adds motion proxy -> ML3 changes correction bandwidth; all three frozen together.',
              memory='Two historical blurred frames plus one gate/motion memory state; causal, reset at original anchors. No scene-cut detection.',
              limitations=['Motion proxy can respond to decoder instability; not semantic motion.',
                           'Guide-flow misalignment, confirmation delay and anchor resets can miss moving boundaries.',
                           'ML3 preserves high-frequency old texture and may leave fish contours.',
                           'Known historical clips, not independent holdout; no real-time claim.'],
              extra_transmission_bytes=0,new_flow_model_inference=False,no_source_or_roi_in_receiver=True,
              no_automatic_adoption=True,subjective_review='pending',long_video_not_run=True)
    save_json(run/'execution_plan.json',plan)
    snap=run/'source_snapshot';snap.mkdir()
    for n in CODE:shutil.copy2(STAGE/n,snap/n)
    statepath=REPO/'05_资料与索引/会话收束_20261110/接续状态.json'
    shutil.copy2(statepath,run/'state_before_run.json')
    state=load_json(statepath)
    state.update(status='Q9_temporal_exit_experiment_running',active_experiment_plan=str(run/'execution_plan.json'),
                 current_session_work='Three fixed causal temporal-exit profiles across five known Q9 clips.')
    state['experiment_execution_gate']=dict(status='running',currently_running=True,new_run_started=True,date='2026-11-12')
    save_json(statepath,state)
    print('FROZEN',list(PROFILES),'five clips / 15 outputs',flush=True)

def worker(run,case):
    plan=load_json(run/'execution_plan.json')
    s=load_json(run/'case_registration.json')['cases'][case]
    for n,h in plan['code_sha256'].items():assert file_sha(STAGE/n)==h
    parent=Path(plan['parent']);baseline=parent/case/'receive/Q9_R25'
    rec=load_json(parent/case/'receive/receipt.json')
    out=run/case/'receive';out.mkdir(parents=True,exist_ok=False)
    cache=Path(s['cache'])
    allowed={Path(s[k]).resolve() for k in ('raw','base','base_receipt','packet')}
    allowed.update((baseline/(n+'.npy')).resolve() for n in ('rgb','alpha','shared'))
    allowed.update((cache/(n+'.npy')).resolve() for n in ('to_anchor','from_anchor'))
    reads=set()
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p in allowed:reads.add(str(p));return
        if p.is_relative_to(out):return
        if p.suffix.lower() not in {'.py','.pyc','.pyd','.dll'} and (p.is_relative_to(DATA) or p.suffix.lower() in {'.npy','.npz','.mp4','.glcguide','.glcsnd'}):
            raise RuntimeError('Forbidden temporal receiver input: '+str(p))
    sys.addaudithook(audit)
    blocked=[]
    for p in [Path(s['source']),Path(s['send_receipt']),run/'user_event_registration.json',
              parent/'diagnosis/user_review_boxes.json',parent/'diagnosis/attribution.json']:
        try:p.read_bytes()
        except RuntimeError:blocked.append(str(p))
        else:raise AssertionError('read guard failed')
    base_rec=load_json(s['base_receipt'])
    assert base_rec['reference_verified'] and base_rec['source_media_reads_forbidden']
    assert file_sha(s['raw'])==base_rec['decoded_rgb_sha256']
    assert file_sha(s['base'])==rec['base_sha256'] and file_sha(s['packet'])==rec['packet_sha256']
    (out/'Q9_R25').mkdir()
    for n,h in rec['baseline_sha256'].items():
        assert file_sha(baseline/(n+'.npy'))==h
        os.link(baseline/(n+'.npy'),out/'Q9_R25'/(n+'.npy'))
    for n in ('to_anchor','from_anchor'):
        assert file_sha(cache/(n+'.npy'))==s['cache_sha256'][n]
    raw=np.load(s['raw'],mmap_mode='r')
    q=np.load(baseline/'rgb.npy',mmap_mode='r');alpha=np.load(baseline/'alpha.npy',mmap_mode='r')
    back=np.load(cache/'to_anchor.npy',mmap_mode='r');forward=np.load(cache/'from_anchor.npy',mmap_mode='r')
    outputs={}
    for n in PROFILES:
        dest=out/n;dest.mkdir()
        rgb=np.lib.format.open_memmap(dest/'rgb.npy',mode='w+',dtype=np.uint8,shape=raw.shape)
        weight=np.lib.format.open_memmap(dest/'weight.npy',mode='w+',dtype=np.float32,shape=raw.shape[:3])
        outputs[n]=(rgb,weight)
    proc=TemporalExit(*raw.shape[1:3]);rows=[];tick=time.perf_counter()
    for i in range(len(raw)):
        values,stats=proc.step(i,raw[i],q[i],alpha[i],back[i],forward[i])
        for n,(image,weight) in values.items():
            assert np.isfinite(weight).all() and np.all((weight>=0)&(weight<=1))
            assert np.array_equal(image[weight==0],q[i][weight==0])
            assert np.array_equal(image[alpha[i]==0],raw[i][alpha[i]==0])
            assert np.all(np.abs(image.astype(np.int16)-raw[i].astype(np.int16))<=np.abs(q[i].astype(np.int16)-raw[i].astype(np.int16)))
            outputs[n][0][i]=image;outputs[n][1][i]=weight
        assert np.all(values['M3'][1]<=values['P3'][1]+1e-6)
        assert np.array_equal(values['M3'][1],values['ML3'][1])
        rows.append(stats)
    seconds=time.perf_counter()-tick
    for rgb,weight in outputs.values():rgb.flush();weight.flush()
    del outputs
    with (out/'temporal_state.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    records={n:{k:file_sha(out/n/(k+'.npy')) for k in ('rgb','weight')} for n in PROFILES}
    save_json(out/'receipt.json',dict(complete=True,case=case,profiles=PROFILES,settings=SETTINGS,
              base_sha256=file_sha(s['base']),packet_sha256=file_sha(s['packet']),
              baseline_sha256=rec['baseline_sha256'],baseline_reuse_verified=True,
              source_reads_blocked=True,blocked=blocked,reads=sorted(reads),allowed=sorted(map(str,allowed)),
              output_sha256=records,code_sha256=plan['code_sha256'],
              three_profile_component_seconds=seconds,includes_base_decode=False,includes_flow_inference=False,
              output_memory_mapped=True,causal=True,automatic_adoption=False))
    print('RECEIVED',case,round(seconds,2),'seconds',flush=True)

def run_all(run):
    plan=load_json(run/'execution_plan.json')
    check_inputs(load_json(run/'case_registration.json')['input_sha256'])
    check_inputs(plan['baseline_input_sha256'])
    import run_q9_ghost_exit as evaluation
    evaluation.PROFILES=PROFILES
    for case in plan['cases']:
        subprocess.run([sys.executable,'-B','-u','-X','utf8',str(Path(__file__).resolve()),'worker','--run',str(run),'--case',case],check=True,cwd=REPO)
        evaluation.evaluate(run,case)
    save_json(run/'summary.json',dict(complete=True,cases={c:load_json(run/c/'evaluation/summary.json') for c in plan['cases']},
              profiles=PROFILES,new_outputs=15,no_per_clip_tuning=True,automatic_selection=None,
              subjective_review='pending',long_video_not_run=True))
    print('TEMPORAL ROUND COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('freeze','worker','run'))
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--parent',type=Path)
    p.add_argument('--case')
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    try:
        if a.action=='freeze':freeze(a.run,a.parent)
        elif a.action=='worker':worker(a.run,a.case)
        else:run_all(a.run)
    except BaseException as e:
        folder=a.run/a.case/'receive' if a.action=='worker' else a.run
        if folder.exists():save_json(folder/('failure_'+a.action+'.json'),dict(error=str(e),traceback=traceback.format_exc()))
        raise

