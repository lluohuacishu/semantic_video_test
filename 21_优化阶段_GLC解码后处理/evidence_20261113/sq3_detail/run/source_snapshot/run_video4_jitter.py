"""Fixed, receiver-only short-history experiment after video4 jitter feedback."""
import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time
import cv2
import numpy as np
from runtime_speed_common import STAGE,REPO,DATA,WEIGHT,FFMPEG,file_sha,load_json,save_json,check_inputs
from short_history_filter import SETTINGS,blend_frame,guide_correction,quantize,maps_confidence

P4=DATA/'q9_video4_diagnosis_20261113_02'
P3=DATA/'q9_received_structure_20261113_01'
CONFIG='b5eeb844d0adcdfe97648351a0654a2575a98942e1627c98896ba302555f7b04'
NAMES=('RG25','SR3','SQ3')
CODE=('run_video4_jitter.py','short_history_filter.py','evaluate_video4_diagnosis.py',
      'raft_inference_runtime.py','runtime_speed_common.py','runtime_chain_worker.py',
      'q9_chunk_protocol.py','structure_guide_stream.py','received_structure_fusion.py',
      'audit_received_structure_blur.py','audit_deflicker_texture.py','probe_reef.py')


def freeze(run):
    assert not (run/'execution_plan.json').exists()
    assert file_sha(STAGE/'当前采用配置.json')==CONFIG
    old=load_json(P3/'case_registration.json')['cases']
    cases={}
    for case,s in old.items():
        s=dict(s)
        s.update(guide=str(P3/case/'receive/received_guide.npy'),
            q9=str(P3/case/'receive/Q9_R25/rgb.npy'),
            gs25=str(P3/case/'receive/GS25/rgb.npy'),
            alpha=str(Path(s['baseline'])/'alpha.npy'))
        cases[case]=s
    cases['video4']=dict(source=str(P4/'source.npy'),raw=str(P4/'video4/base_receive/decoded_rgb.npy'),
        base=str(P4/'video4/send/base.glcsnd'),base_receipt=str(P4/'video4/base_receive/receipt.json'),
        packet=str(P4/'guide/guide.glcg2'),send_receipt=str(P4/'guide/send.json'),
        guide=str(P4/'receive/guide_rgb.npy'),q9=str(P4/'receive/Q9_R25/rgb.npy'),
        gs25=str(P4/'receive/GS25/rgb.npy'),alpha=str(P4/'receive/Q9_R25/alpha.npy'),
        cache=str(P4/'receive'),frames=104,fps=15,reef=[0,128,512,288])
    inputs={str(WEIGHT):file_sha(WEIGHT)}
    for case,s in cases.items():
        for k in ('source','raw','base','base_receipt','packet','guide','q9','gs25','alpha','send_receipt'):
            inputs[s[k]]=file_sha(s[k])
    save_json(run/'case_registration.json',dict(cases=cases))
    plan=dict(date='2026-11-13',status='frozen_before_inference',cases=['video4','video1','video2','video3'],
        user_feedback=load_json(run/'user_feedback.json'),
        hypothesis='Long-anchor replacement and/or rapidly changing correspondence/support may add jitter. Test bypass and finite recent-frame averaging; association diagnostics do not alone establish causation.',
        profiles=dict(RG25='No historical Q9 replacement: current raw plus unchanged GS25 low-frequency guide correction. Ablation control.',
            SR3='Guide-aligned finite 3-frame raw blend, then unchanged 25% current guide correction.',
            SQ3='Guide-aligned finite 3-frame blend of existing GS25. No recursive filtered-output feedback.'),
        settings=SETTINGS,new_candidate_outputs=8,new_ablation_outputs=4,
        blur_control='For each clip match spatially blurred GS25 to SQ3 mean fine amplitude; diagnostic only, no source used for matching.',
        metrics=['unchanged source-aligned coarse/fine/full temporal MAE','PSNR/RGB and texture spatial error',
          'pixel dynamic amplitude','new source-aligned error second difference and large-area change fraction'],
        jitter_metric_note='Diagnostic proxies, not measured camera shake or a perceptual acceptance test.',
        guardrail='Do not accept based on scalar gains alone; check fish/turtle presence, trails, sharpness and natural motion. Preserve Q9 video123 feedback.',
        no_per_clip_parameter_tuning=True,no_source_at_receiver=True,extra_transmission_bytes=0,
        reencoding=False,new_flow_inference='Received-guide direct current->previous1/previous2, bidirectional RAFT12 FP32',
        lookahead_frames=0,added_temporal_history_frames=2,existing_Q9_history_remains_in_SQ3=True,
        long_video_not_run=True,independent_holdout=False,automatic_selection=None,
        current_config_sha256=CONFIG,input_sha256=inputs,code_sha256={n:file_sha(STAGE/n) for n in CODE})
    save_json(run/'execution_plan.json',plan)
    snap=run/'source_snapshot';snap.mkdir()
    for name in CODE:shutil.copy2(STAGE/name,snap/name)
    statepath=REPO/'05_资料与索引/会话收束_20261110/接续状态.json'
    shutil.copy2(statepath,run/'state_before_run.json')
    state=load_json(statepath)
    state.update(status='Q9_video4_jitter_running',active_experiment_plan=str(run/'execution_plan.json'),
        current_session_work='User rejects video4 frame-wide jitter. Test finite three-frame guide-aligned blend and no-anchor ablation; regress video123.')
    state['video4_jitter_feedback']=plan['user_feedback']
    state['experiment_execution_gate']=dict(status='running',currently_running=True,new_run_started=True,date='2026-11-13',automatic_next_run=False)
    save_json(statepath,state)
    print('FROZEN: RG25 / SR3 / SQ3, four clips; no automatic adoption',flush=True)


def receiver(run,case):
    import torch
    from runtime_chain_worker import ReadGuard
    from raft_inference_runtime import load_model,FlowRunner
    import q9_chunk_protocol
    import structure_guide_stream
    from audit_received_structure_blur import amplitude,match
    plan=load_json(run/'execution_plan.json');s=load_json(run/'case_registration.json')['cases'][case]
    for name,h in plan['code_sha256'].items():assert file_sha(STAGE/name)==h
    out=run/case/'receive';out.mkdir(parents=True,exist_ok=False)
    allowed=[Path(s[k]) for k in ('raw','base','base_receipt','packet','guide','q9','gs25','alpha')]+[WEIGHT]
    guard=ReadGuard('jitter-post');guard.authorize(out,allowed)
    blocked=[]
    for p in (Path(s['source']),Path(s['send_receipt']),run/'user_feedback.json',P4/'summary.json'):
        try:p.read_bytes()
        except RuntimeError:blocked.append(str(p))
        else:raise AssertionError('Receiver source guard failed')
    for p in allowed:assert file_sha(p)==plan['input_sha256'][str(p)]
    br=load_json(s['base_receipt'])
    assert br['reference_verified'] and br['source_media_reads_forbidden']
    assert file_sha(s['raw'])==br['decoded_rgb_sha256']
    packet=Path(s['packet']).read_bytes();base_sha=file_sha(s['base'])
    if case=='video4':_,guide=q9_chunk_protocol.decode(packet,base_sha,0,s['frames'],FFMPEG)
    else:_,_,guide=structure_guide_stream.decode(packet,base_sha,FFMPEG)
    assert np.array_equal(guide,np.load(s['guide'],mmap_mode='r'))
    np.save(out/'guide.npy',guide)
    raw=np.load(s['raw'],mmap_mode='r');gs=np.load(s['gs25'],mmap_mode='r')
    assert raw.shape==gs.shape==guide.shape==(s['frames'],288,512,3)
    n=len(raw);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model=load_model();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
    flow=np.lib.format.open_memmap(out/'flows.npy',mode='w+',dtype=np.float32,shape=(n,2,2,288,512,2));flow[:]=0
    pairs=0
    with torch.inference_mode():
        runner=FlowRunner(model,guide,'feature_cache');runner.prepare()
        for i in range(1,n):
            for lag in (1,2):
                if i<lag:continue
                value=runner.pair(i,i-lag).permute(0,2,3,1).contiguous().cpu().numpy()
                assert value.shape==(2,288,512,2) and np.isfinite(value).all()
                flow[i,lag-1]=value;pairs+=1
            if i%30==0:print('FLOW',case,i,'/',n,flush=True)
    torch.cuda.synchronize();seconds=time.perf_counter()-tick
    info=dict(seconds=seconds,pairs=pairs,peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2)
    flow.flush();del runner,model;torch.cuda.empty_cache()
    outputs={}
    for name in NAMES:
        d=out/name;d.mkdir();outputs[name]=np.lib.format.open_memmap(d/'rgb.npy',mode='w+',dtype=np.uint8,shape=raw.shape)
    stats=[];tick=time.perf_counter()
    for i in range(n):
        outputs['RG25'][i]=quantize(guide_correction(raw[i],guide[i]))
        sr,diag=blend_frame(raw,guide,flow,i)
        outputs['SR3'][i]=quantize(guide_correction(sr,guide[i]))
        sq,_=blend_frame(gs,guide,flow,i)
        outputs['SQ3'][i]=quantize(sq);stats.append(diag)
    for v in outputs.values():v.flush()
    render_seconds=time.perf_counter()-tick
    target=amplitude(outputs['SQ3'])
    (error,sigma,control),search=match(gs,target)
    (out/'BLUR_SQ3').mkdir();np.save(out/'BLUR_SQ3/rgb.npy',control)
    save_json(out/'blur_match.json',dict(sigma=sigma,target=target,relative_error=error/target,search=search,
        diagnostic_only=True,source_used=False,not_per_clip_deployment_tuning=True))
    hashes={name:file_sha(out/name/'rgb.npy') for name in (*NAMES,'BLUR_SQ3')}
    save_json(out/'receipt.json',dict(complete=True,case=case,source_reads_blocked=True,blocked=blocked,
        allowed=list(map(str,allowed)),guide_redecoded_exact=True,output_sha256=hashes,
        flow_sha256=file_sha(out/'flows.npy'),flow=info,render_seconds=render_seconds,
        new_transmission_bytes=0,frame_stats=stats,settings=SETTINGS,automatic_selection=None))
    print('RECEIVED',case,info,'blur sigma',sigma,flush=True)


def load_sequences(run,case):
    s=load_json(run/'case_registration.json')['cases'][case]
    seq={k:np.load(s[p],mmap_mode='r') for k,p in [('source','source'),('raw','raw'),('guide','guide'),('Q9_R25','q9'),('GS25','gs25')]}
    for name in (*NAMES,'BLUR_SQ3'):seq[name]=np.load(run/case/'receive'/name/'rgb.npy',mmap_mode='r')
    return seq


def jitter_diagnostic(seq,scopes):
    from audit_deflicker_texture import source_match
    from probe_reef import bands
    from temporal_filter import warp
    rows=[];previous={k:bands(v[0]) for k,v in seq.items()};last_error=None;last_mask=None
    for i in range(1,len(seq['source'])):
        cur={k:bands(v[i]) for k,v in seq.items()}
        mx,my,mask=source_match(seq['source'][i-1],seq['source'][i],previous['source'],cur['source'],np.s_[:,:])
        sd=cur['source']-warp(previous['source'],mx,my)
        errors={k:cur[k]-warp(previous[k],mx,my)-sd for k in seq}
        stable=None if last_mask is None else mask&(warp(last_mask.astype(np.float32),mx,my)>.999)
        for label,(x0,y0,x1,y1) in scopes.items():
            sl=np.s_[y0:y1,x0:x1];m=mask[sl];count=int(m.sum())
            if count<64:continue
            am=None if stable is None else stable[sl];an=0 if am is None else int(am.sum())
            for name,error in errors.items():
                e=np.abs(error[sl][m]);r=dict(frame=i,scope=label,method=name,pixels=count,
                    large_change_pixels=int((e[:,3]>16).sum()),coarse_sum=float(e[:,0].sum()),
                    fine_sum=float(e[:,2].sum()),full_sum=float(e[:,3].sum()),accel_pixels=an)
                for j,band in ((0,'coarse'),(2,'fine'),(3,'full')):
                    r[band+'_accel_sum']=0. if an<64 else float(np.abs(error-warp(last_error[name],mx,my))[sl][am,j].sum())
                if an<64:r['accel_pixels']=0
                rows.append(r)
        previous=cur;last_error=errors;last_mask=mask
    summary={}
    for scope in scopes:
        summary[scope]={}
        for name in seq:
            rr=[r for r in rows if r['scope']==scope and r['method']==name]
            total=sum(r['pixels'] for r in rr);atotal=sum(r['accel_pixels'] for r in rr)
            summary[scope][name]=dict(valid_pairs=len(rr),valid_acceleration_pairs=sum(r['accel_pixels']>0 for r in rr),
                large_change_fraction=sum(r['large_change_pixels'] for r in rr)/total if total else None,
                frame_full_mae_p95=float(np.percentile([r['full_sum']/r['pixels'] for r in rr],95)) if rr else None,
                **{band+'_accel_mae':sum(r[band+'_accel_sum'] for r in rr)/atotal if atotal else None for band in ('coarse','fine','full')})
    return summary,rows


def gate_diagnostic(run,case):
    s=load_json(run/'case_registration.json')['cases'][case]
    guide=np.load(s['guide'],mmap_mode='r');a=np.load(s['alpha'],mmap_mode='r')
    flow=np.load(run/case/'receive/flows.npy',mmap_mode='r')
    anchor=np.load(Path(s['cache'])/'to_anchor.npy',mmap_mode='r')
    q=np.load(s['q9'],mmap_mode='r');raw=np.load(s['raw'],mmap_mode='r')
    from receiver_structure import KEYS
    rows=[]
    for i in range(1,len(q)):
        mx,my,conf=maps_confidence(guide[i],guide[i-1],*flow[i,0])
        old=cv2.remap(a[i-1],mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101)
        m=conf>.5
        for scope,box in [('full',[0,0,512,288]),('reef',s['reef'])]:
            x0,y0,x1,y1=box;sl=np.s_[y0:y1,x0:x1];valid=m[sl];n=int(valid.sum())
            if n<64:continue
            flips=((a[i][sl]>.5)!=(old[sl]>.5))&valid
            residual=np.abs(q[i].astype(np.float32)-raw[i]).mean(axis=2)[sl]
            key=KEYS[np.searchsorted(KEYS,i,side='right')-1]
            prevkey=KEYS[np.searchsorted(KEYS,i-1,side='right')-1]
            disagreement=None
            if key==prevkey:
                predicted=flow[i,0,0]+cv2.remap(anchor[i-1],mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101)
                disagreement=float(np.linalg.norm(anchor[i]-predicted,axis=2)[sl][valid].mean())
            rows.append(dict(frame=i,scope=scope,pixels=n,flipped_pixels=int(flips.sum()),
                alpha_abs_sum=float(np.abs(a[i][sl]-old[sl])[valid].sum()),
                q9_raw_abs_on_flip_sum=float(residual[flips].sum()),
                anchor_flow_composition_disagreement_px=disagreement))
    return rows


def evaluate(run,case):
    from evaluate_video4_diagnosis import score,dynamic
    from compare_deflicker import save_csv
    s=load_json(run/'case_registration.json')['cases'][case]
    out=run/case/'evaluation';out.mkdir(exist_ok=False)
    seq=load_sequences(run,case)
    scopes=dict(full=[0,0,512,288],reef=s['reef'])
    if case=='video4':scopes.update(coral_lower=[0,128,512,288],turtle_wide=[50,150,410,288])
    else:
        scopes.update(load_json(P3/'execution_plan.json')['review_regions'][case])
    scores,tr,sr,missing=score(seq,scopes);dyn,dr=dynamic(seq)
    jitter,jr=jitter_diagnostic(seq,scopes);gates=gate_diagnostic(run,case)
    for name,rows in [('temporal',tr),('spatial',sr),('dynamic',dr),('jitter',jr),('gate',gates)]:save_csv(out/(name+'.csv'),rows)
    result=dict(complete=True,case=case,scopes=scopes,scores=scores,dynamic=dyn,jitter=jitter,
        missing=missing,automatic_selection=None,subjective_review='pending',
        new_metric_definition='Source-aligned error second temporal difference; large-change fraction uses abs full-gray temporal error >16 on common support. Not camera shake magnitude.')
    save_json(out/'summary.json',result)
    print('EVALUATED',case,{name:dict(coarse=round(v['coarse_mae'],3),fine=round(v['fine_mae'],3),psnr=round(v['psnr_db'],3),accel=round(jitter['full'][name]['full_accel_mae'],3)) for name,v in scores['full'].items() if name!='source'},flush=True)


def run_all(run):
    plan=load_json(run/'execution_plan.json');check_inputs(plan['input_sha256'])
    for case in plan['cases']:
        subprocess.run([sys.executable,'-B','-u','-X','utf8',str(Path(__file__).resolve()),'receive','--run',str(run),'--case',case],check=True,cwd=REPO)
        evaluate(run,case)
    check_inputs(plan['input_sha256'])
    assert file_sha(STAGE/'当前采用配置.json')==CONFIG
    save_json(run/'summary.json',dict(complete=True,cases={c:load_json(run/c/'evaluation/summary.json') for c in plan['cases']},
        profiles=NAMES,input_hashes_unchanged=True,current_config_unchanged=True,automatic_selection=None))
    print('JITTER EXPERIMENT COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','receive','evaluate','run']);p.add_argument('--run',type=Path,required=True);p.add_argument('--case')
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    if a.action=='freeze':freeze(a.run)
    elif a.action=='receive':receiver(a.run,a.case)
    elif a.action=='evaluate':evaluate(a.run,a.case)
    else:run_all(a.run)
