"""Frozen two-strength sharpness probe; all comparisons use accepted SQ3."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import cv2
import numpy as np
from runtime_speed_common import STAGE,REPO,DATA,WORKSPACE,file_sha,load_json,save_json,check_inputs
from sq3_detail_filter import SETTINGS,enhance
from preserve_sq3_accepted import SHORT,LONG,DEST,CONFIG

RUN=DATA/'sq3_detail_20261113_02'
PROFILES=tuple(SETTINGS['profiles'])
CODE=('run_sq3_detail.py','sq3_detail_filter.py','evaluate_video4_diagnosis.py',
      'run_video4_jitter.py','audit_deflicker_texture.py','probe_reef.py','runtime_speed_common.py')

def freeze():
    RUN.mkdir(exist_ok=False)
    check_inputs({str(STAGE/'当前采用配置.json'):CONFIG})
    archive=load_json(DEST/'manifest.json')
    for name,r in archive['files'].items():assert file_sha(DEST/name)==r['sha256']
    cases=load_json(SHORT/'case_registration.json')['cases']
    lp=load_json(LONG/'execution_plan.json')
    cases['long']=dict(source=lp['source'],raw=lp['raw'],frames=1239,fps=15,
        send_receipt=str(Path(lp['source']).parent/'source_metadata.json'))
    inputs={str(DEST/'manifest.json'):file_sha(DEST/'manifest.json')}
    for case,s in cases.items():
        s['sq3']=str(DEST/'outputs'/('long_SQ3.npy' if case=='long' else f'{case}_SQ3.npy'))
        s['sq3_sha256']=file_sha(s['sq3']);inputs[s['sq3']]=s['sq3_sha256']
        for k in ('source','raw'):inputs[s[k]]=file_sha(s[k])
        if case!='long':
            e=load_json(SHORT/case/'evaluation/summary.json');s['scopes']=e['scopes']
        else:s['scopes']=dict(full=[0,0,512,288],top_left=[0,0,256,144],top_right=[256,0,512,144],bottom_left=[0,144,256,288],bottom_right=[256,144,512,288])
    save_json(RUN/'case_registration.json',dict(cases=cases))
    plan=dict(date='2026-11-13',frozen_before_outputs=True,baseline='SQ3 user accepted',
        hypothesis='A small bounded luminance-only high-pass boost may recover perceived edge contrast without materially increasing SQ3 temporal error. It cannot restore missing structure.',
        profiles=SETTINGS,short_cases=['video4','video1','video2','video3'],
        purpose='Two fixed strengths form one controlled sharpness/stability tradeoff experiment. No per-video settings.',
        primary_candidate='SQ3_U15',secondary_candidate='SQ3_U30',
        long_extension_rule='Only the first candidate in fixed order U15 then U30 that passes every short-case/ROI numeric screen is extended to the full long video. This selects a viewing candidate, not adoption.',
        screen=dict(max_coarse_temporal_increase_percent=1,max_full_temporal_increase_percent=3,
            max_fine_temporal_increase_percent=5,max_full_acceleration_increase_percent=5,
            max_RGB_error_increase_percent=1,max_fine_spatial_error_increase_percent=1,
            min_PSNR_delta_db=-.1,min_full_fine_amplitude_gain_percent=2),
        threshold_note='Engineering screening tolerances frozen for this round, not perceptual just-noticeable-difference thresholds.',
        metrics='Reuse unchanged source-aligned metrics and regions. Check all frames, fine spatial error, dynamics and source-aligned acceleration; source used only in evaluator.',
        automatic_adoption=False,preferred_fallback='SQ3',independent_holdout=False,
        no_retraining=True,new_flow_inference=False,extra_transmission_bytes=0,lookahead=0,
        current_config_sha256=CONFIG,input_sha256=inputs,
        code_sha256={n:file_sha(STAGE/n) for n in CODE})
    save_json(RUN/'execution_plan.json',plan)
    for n in CODE:
        d=RUN/'source_snapshot'/n;d.parent.mkdir(exist_ok=True);shutil.copy2(STAGE/n,d)
    sp=REPO/'05_资料与索引/会话收束_20261110/接续状态.json';shutil.copy2(sp,RUN/'state_before_run.json')
    state=load_json(sp);state.update(status='SQ3_detail_running',active_experiment_plan=str(RUN/'execution_plan.json'),
        experiment_execution_gate=dict(status='running',currently_running=True,automatic_next_run=False),
        current_session_work='SQ3 preserved. Two bounded luminance-sharpening strengths; all four clips, then a fixed-order passing long candidate.')
    save_json(sp,state)
    print('FROZEN SQ3_U15/SQ3_U30; SQ3 remains preferred fallback',flush=True)

def receive(case,names):
    plan=load_json(RUN/'execution_plan.json');s=load_json(RUN/'case_registration.json')['cases'][case]
    for n,h in plan['code_sha256'].items():assert file_sha(STAGE/n)==h
    out=RUN/case/'receive';out.mkdir(parents=True,exist_ok=False)
    src=Path(s['sq3']).resolve();read_files=set()
    forbidden={Path(s['source']).resolve(),Path(s.get('send_receipt',RUN/'user_feedback.json')).resolve()}
    def guard(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p==src:read_files.add(str(p));return
        if p.is_relative_to(out):return
        if p in forbidden or p.is_relative_to(DATA) or (p.is_relative_to(DEST) and p!=src) or p.suffix.lower() in {'.npy','.npz','.mp4','.glcsnd','.glcg2','.glcguide'}:
            raise RuntimeError('SQ3 detail receiver forbidden read: '+str(p))
    sys.addaudithook(guard)
    blocked=[]
    for p in forbidden:
        try:p.read_bytes()
        except RuntimeError:blocked.append(str(p))
        else:raise AssertionError('Source read was not blocked')
    assert file_sha(src)==s['sq3_sha256']
    sq=np.load(src,mmap_mode='r',allow_pickle=False);assert sq.shape==(s['frames'],288,512,3) and sq.dtype==np.uint8
    tick=time.perf_counter();results={}
    for name in names:
        gain=SETTINGS['profiles'][name];p=out/name/'rgb.npy';p.parent.mkdir()
        target=np.lib.format.open_memmap(p,mode='w+',dtype=np.uint8,shape=sq.shape)
        maxdiff=0;changed=0
        for i,frame in enumerate(sq):
            result=enhance(frame,gain);d=result.astype(np.int16)-frame
            assert np.all(d==d[...,:1]),'RGB differences changed'
            maxdiff=max(maxdiff,int(np.max(np.abs(d))));assert maxdiff<=3
            changed+=int(np.any(d!=0,axis=2).sum());target[i]=result
        target.flush();results[name]=dict(sha256=file_sha(p),maximum_RGB_change=maxdiff,
            changed_pixel_fraction=changed/(len(sq)*288*512),gain=gain)
        del target
    save_json(out/'receipt.json',dict(complete=True,case=case,profiles=results,source_reads_blocked=True,
        blocked=blocked,actual_input_reads=sorted(read_files),input_sha256=s['sq3_sha256'],
        seconds=time.perf_counter()-tick,extra_transmission_bytes=0,no_added_temporal_state=True))
    print('RECEIVED',case,results,flush=True)

def evaluate(case,names):
    from evaluate_video4_diagnosis import score,dynamic
    from run_video4_jitter import jitter_diagnostic
    from compare_deflicker import save_csv
    s=load_json(RUN/'case_registration.json')['cases'][case];out=RUN/case/'evaluation';out.mkdir()
    seq=dict(source=np.load(s['source'],mmap_mode='r'),SQ3=np.load(s['sq3'],mmap_mode='r'))
    for n in names:seq[n]=np.load(RUN/case/'receive'/n/'rgb.npy',mmap_mode='r')
    scores,tr,sr,missing=score(seq,s['scopes']);print('SPATIAL/TEMPORAL',case,flush=True)
    dyn,dr=dynamic(seq);jitter,jr=jitter_diagnostic(seq,s['scopes'])
    for name,rows in [('temporal',tr),('spatial',sr),('dynamic',dr),('jitter',jr)]:save_csv(out/(name+'.csv'),rows)
    baseline_exact=None
    if case!='long':
        old=load_json(SHORT/case/'evaluation/summary.json')
        for scope in s['scopes']:
            assert scores[scope]['SQ3']==old['scores'][scope]['SQ3'],('metrics',case,scope)
            assert jitter[scope]['SQ3']==old['jitter'][scope]['SQ3'],('jitter',case,scope)
        assert dyn['SQ3']==old['dynamic']['SQ3'];baseline_exact=True
    result=dict(complete=True,case=case,scores=scores,dynamic=dyn,jitter=jitter,missing=missing,
        baseline_metrics_exact=baseline_exact,source_only_at_evaluation=True,subjective_review='pending',automatic_adoption=False)
    save_json(out/'summary.json',result)
    print('EVALUATED',case,{n:dict(fine=scores['full'][n]['fine_mae'],amplitude=scores['full'][n]['fine_amplitude'],psnr=scores['full'][n]['psnr_db'],accel=jitter['full'][n]['full_accel_mae']) for n in seq},flush=True)

def screen(cases,names):
    t=load_json(RUN/'execution_plan.json')['screen'];result={}
    for name in names:
        failures=[];changes={}
        for case in cases:
            e=load_json(RUN/case/'evaluation/summary.json');changes[case]={}
            for scope,values in e['scores'].items():
                a,b=values['SQ3'],values[name]
                change={k:100*(b[k]/a[k]-1) for k in ('coarse_mae','full_mae','fine_mae','rgb_mae','fine_spatial_mae','fine_amplitude')}
                change['psnr_db']=b['psnr_db']-a['psnr_db']
                ja,jb=e['jitter'][scope]['SQ3'],e['jitter'][scope][name]
                change['full_accel_mae']=100*(jb['full_accel_mae']/ja['full_accel_mae']-1)
                changes[case][scope]=change
                caps={'coarse_mae':t['max_coarse_temporal_increase_percent'],'full_mae':t['max_full_temporal_increase_percent'],
                    'fine_mae':t['max_fine_temporal_increase_percent'],'rgb_mae':t['max_RGB_error_increase_percent'],
                    'fine_spatial_mae':t['max_fine_spatial_error_increase_percent'],'full_accel_mae':t['max_full_acceleration_increase_percent']}
                for key,cap in caps.items():
                    if change[key]>cap:failures.append(dict(case=case,scope=scope,metric=key,change=change[key],limit=cap))
                if change['psnr_db']<t['min_PSNR_delta_db']:failures.append(dict(case=case,scope=scope,metric='psnr_db',change=change['psnr_db'],limit=t['min_PSNR_delta_db']))
                if scope=='full' and change['fine_amplitude']<t['min_full_fine_amplitude_gain_percent']:
                    failures.append(dict(case=case,scope=scope,metric='fine_amplitude',change=change['fine_amplitude'],limit=t['min_full_fine_amplitude_gain_percent']))
        result[name]=dict(pass_all=not failures,failures=failures,changes=changes)
    return result

def verify(cases,names):
    reg=load_json(RUN/'case_registration.json')['cases'];checks={}
    for case in cases:
        s=reg[case];sq=np.load(s['sq3'],mmap_mode='r');checks[case]={}
        for name in names:
            p=RUN/case/'receive'/name/'rgb.npy'
            if not p.exists():continue
            a=np.load(p,mmap_mode='r');gain=SETTINGS['profiles'][name]
            assert a.shape==sq.shape and a.dtype==np.uint8
            for i in range(len(a)):assert np.array_equal(a[i],enhance(sq[i],gain))
            for i in (0,len(a)//2,len(a)-1):assert np.array_equal(enhance(sq[i],0),sq[i])
            checks[case][name]=dict(all_frames_replayed_exact=True,zero_gain_identity=True,frames=len(a),sha256=file_sha(p),maximum_change=3)
    check_inputs(load_json(RUN/'execution_plan.json')['input_sha256'])
    assert file_sha(STAGE/'当前采用配置.json')==CONFIG
    save_json(RUN/'verification.json',dict(complete=True,cases=checks,input_hashes_unchanged=True,formal_config_unchanged=True,
        receiver_receipts=[str(RUN/c/'receive/receipt.json') for c in cases],no_new_temporal_state=True,chunk_boundaries_invariant_by_frame_independence=True))

def run():
    plan=load_json(RUN/'execution_plan.json');cases=plan['short_cases'];check_inputs(plan['input_sha256'])
    for case in cases:
        subprocess.run([sys.executable,'-B','-u','-X','utf8',__file__,'receive','--case',case,'--names',*PROFILES],check=True)
        evaluate(case,PROFILES)
    gates=screen(cases,PROFILES);selected=next((n for n in PROFILES if gates[n]['pass_all']),None)
    save_json(RUN/'short_screen.json',dict(complete=True,screen=gates,long_viewing_candidate=selected,automatic_adoption=False))
    print('SHORT SCREEN',{n:dict(pass_all=r['pass_all'],failures=r['failures']) for n,r in gates.items()},flush=True)
    allcases=list(cases)
    if selected:
        subprocess.run([sys.executable,'-B','-u','-X','utf8',__file__,'receive','--case','long','--names',selected],check=True)
        evaluate('long',[selected]);allcases.append('long')
        save_json(RUN/'long_screen.json',dict(complete=True,screen=screen(['long'],[selected]),not_perceptual_acceptance=True))
    verify(allcases,PROFILES)
    save_json(RUN/'summary.json',dict(complete=True,baseline='SQ3',profiles=PROFILES,cases=allcases,
        long_viewing_candidate=selected,automatic_adoption=False,preferred_fallback='SQ3'))
    print('DETAIL EXPERIMENT COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','receive','run']);p.add_argument('--case');p.add_argument('--names',nargs='+')
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    if a.action=='freeze':freeze()
    elif a.action=='receive':receive(a.case,a.names)
    else:run()
