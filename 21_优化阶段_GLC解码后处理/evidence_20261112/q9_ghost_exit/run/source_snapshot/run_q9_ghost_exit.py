"""Two frozen local-history exit tests on five unchanged Q9 packets and flows."""
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
from raw_conflict_exit import PROFILES,exit_frame
from receiver_structure import render,EVENTS
from mild_reliability_gate import mild_gate
import structure_guide_stream as guide

CODE=['raw_conflict_exit.py','run_q9_ghost_exit.py','mild_reliability_gate.py','receiver_structure.py',
      'cached_t3.py','receive_guide_texture.py','runtime_speed_common.py','structure_guide_stream.py',
      'resume_q9_archive_transfer.py','evaluate_structure_guide.py']


def freeze(run):
    reg=load_json(run/'case_registration.json');check_inputs(reg['input_sha256'])
    assert not (run/'execution_plan.json').exists()
    save_json(run/'execution_plan.json',dict(date='2026-11-12',frozen_before_candidate_inference=True,
        registration_sha256=file_sha(run/'case_registration.json'),user_events_sha256=file_sha(run/'user_event_registration.json'),
        profiles=PROFILES,reference='Unmodified Q9_R25 replay from same received packet and local RAFT cache.',
        operation='Exit weight from final Q9-versus-current-raw sigma3/sigma1 RGB disagreement, applied as one added rounded convex blend toward current raw.',
        motivation='E32 uses prior D25 12..32 sigma3 thresholds but permits full exit. EL12 uses finer localized conflict and both scales, one pixel dilation.',
        parameter_status='Research choices frozen after baseline diagnosis, before candidates; identical across five clips. No threshold search or per-clip settings.',
        no_source_or_roi_in_receiver=True,no_new_flow_or_packet=True,extra_transmission_bytes=0,
        no_automatic_adoption=True,subjective_review='pending',long_video_not_run=True,
        review_boxes={'reef_old_fish_A':[416,112,452,144],'reef_old_fish_B':[488,112,512,144],
                      'video2_tail':[140,145,285,288]},
        review_timing={'reef_departure':[53,75],'user_exact_reef_frame':61,'video2_tail':[45,75]},
        evaluation='Source errors at preregistered user-event crops plus full/quadrants, old reef ROIs, dynamic amplitude and valid-support temporal metrics. No pixel count is called semantic ghost count.',
        negative_control='Also evaluate originally accepted video1/video3 and seagrass; local gain cannot automatically overrule regression elsewhere.',
        code_sha256={n:file_sha(STAGE/n) for n in CODE}))
    snap=run/'source_snapshot';snap.mkdir()
    for n in CODE:shutil.copy2(STAGE/n,snap/n)
    print('FROZEN',list(PROFILES),'five cases; unchanged transmitted bytes',flush=True)


def receive(run,case):
    reg=load_json(run/'case_registration.json');plan=load_json(run/'execution_plan.json');s=reg['cases'][case]
    for n,h in plan['code_sha256'].items():assert file_sha(STAGE/n)==h
    out=run/case/'receive';out.mkdir(parents=True,exist_ok=False);cache=Path(s['cache'])
    allowed={Path(s[k]).resolve() for k in ('base','raw','base_receipt','packet','cache_receipt')}
    allowed.update((cache/(n+'.npy')).resolve() for n in s['cache_sha256']);reads=set()
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p in allowed:reads.add(str(p));return
        if p.is_relative_to(out):return
        runtime=p.suffix.lower() in {'.py','.pyc','.pyd','.dll'}
        if not runtime and (p.is_relative_to(DATA) or p.suffix.lower() in {'.npy','.npz','.mp4','.glcguide','.glcsnd'}):
            raise RuntimeError('Forbidden ghost-exit receiver input: '+str(p))
    sys.addaudithook(audit);blocked=[]
    for p in [Path(s['source']),Path(s['baseline'])/'rgb.npy',Path(s['send_receipt']),run/'diagnosis/events.json',run/'user_event_registration.json']:
        try:p.read_bytes()
        except RuntimeError:blocked.append(str(p))
        else:raise AssertionError('Read guard failed')
    for p in allowed:assert file_sha(p)==reg['input_sha256'][str(p)]
    base_rec=load_json(s['base_receipt']);assert base_rec['reference_verified'] and base_rec['source_media_reads_forbidden']
    assert file_sha(s['base'])==base_rec['stream_sha256'] and file_sha(s['raw'])==base_rec['decoded_rgb_sha256']
    raw=np.load(s['raw'],mmap_mode='r');header,_,images=guide.decode(Path(s['packet']).read_bytes(),file_sha(s['base']),FFMPEG)
    assert np.array_equal(images,np.load(cache/'guide_rgb.npy',mmap_mode='r'))
    flows=[np.load(cache/(n+'.npy'),mmap_mode='r') for n in ('to_anchor','from_anchor','transition_flow')]
    tick=time.perf_counter();baseline=render(raw,images,*flows,gate_fn=mild_gate(images,raw,'R25'))
    folder=out/'Q9_R25';folder.mkdir()
    for n,a in baseline.items():
        np.save(folder/(n+'.npy'),a);assert file_sha(folder/(n+'.npy'))==s['baseline_sha256'][n]
    baseline_seconds=time.perf_counter()-tick;records={}
    for name in PROFILES:
        tick=time.perf_counter();folder=out/name;folder.mkdir()
        rgb=np.empty_like(raw);weights=np.empty(raw.shape[:3],np.float32)
        for i in range(150):rgb[i],weights[i]=exit_frame(raw[i],baseline['rgb'][i],name)
        assert np.isfinite(weights).all() and np.all((weights>=0)&(weights<=1))
        np.save(folder/'rgb.npy',rgb);np.save(folder/'exit_weight.npy',weights)
        assert np.array_equal(rgb[weights==0],baseline['rgb'][weights==0])
        assert np.array_equal(rgb[weights==1],raw[weights==1])
        assert np.array_equal(rgb[baseline['alpha']==0],raw[baseline['alpha']==0])
        assert np.all(np.abs(rgb.astype(np.int16)-raw.astype(np.int16))<=np.abs(baseline['rgb'].astype(np.int16)-raw.astype(np.int16)))
        records[name]=dict(seconds=time.perf_counter()-tick,active_exit_fraction=float((weights>0).mean()),
            full_exit_fraction=float((weights==1).mean()),mean_exit=float(weights.mean()),
            changed_pixels_fraction=float(np.any(rgb!=baseline['rgb'],axis=3).mean()),
            outputs_sha256={n:file_sha(folder/(n+'.npy')) for n in ('rgb','exit_weight')})
    save_json(out/'receipt.json',dict(complete=True,case=case,base_sha256=file_sha(s['base']),packet_sha256=file_sha(s['packet']),
        guide_decode_exact=True,baseline_replay_bit_exact=True,baseline_seconds=baseline_seconds,baseline_sha256=s['baseline_sha256'],
        variants=records,source_reads_blocked=True,blocked=blocked,reads=sorted(reads),allowed=sorted(map(str,allowed)),
        no_additional_packets=True,no_new_flow=True,no_source_no_roi=True,code_sha256=plan['code_sha256']))
    print('RECEIVED',case,{n:round(v['changed_pixels_fraction']*100,2) for n,v in records.items()},flush=True)


def evaluate(run,case):
    from resume_q9_archive_transfer import score_valid_support,SCOPES
    from evaluate_structure_guide import motion_change_diagnostic
    from compare_deflicker import save_csv
    reg=load_json(run/'case_registration.json');s=reg['cases'][case];out=run/case/'evaluation';out.mkdir(exist_ok=False)
    seq=dict(source=np.load(s['source'],mmap_mode='r'),raw=np.load(s['raw'],mmap_mode='r'))
    for n in ('Q9_R25',*PROFILES):seq[n]=np.load(run/case/'receive'/n/'rgb.npy',mmap_mode='r')
    SCOPES.pop('reef',None)
    if s.get('reef'):SCOPES['reef']=tuple(s['reef'])
    scores,temporal,spatial,missing=score_valid_support(seq);dynamic,dr=motion_change_diagnostic(seq)
    save_csv(out/'temporal.csv',temporal);save_csv(out/'spatial.csv',spatial);save_csv(out/'dynamic.csv',dr)
    plan=load_json(run/'execution_plan.json');crops={};rows=[]
    if case=='reef_spotted_fish':
        boxes={k:v for k,v in plan['review_boxes'].items() if k.startswith('reef')};ids=range(*plan['review_timing']['reef_departure'])
    elif case=='video2':boxes={'tail':plan['review_boxes']['video2_tail']};ids=range(*plan['review_timing']['video2_tail'])
    else:boxes={};ids=[]
    for label,(x0,y0,x1,y1) in boxes.items():
        crops[label]={}
        for name,arr in seq.items():
            vals=[];low=[]
            for i in ids:
                a=arr[i,y0:y1,x0:x1].astype(np.float32);b=seq['source'][i,y0:y1,x0:x1].astype(np.float32)
                mae=float(np.abs(a-b).mean());lm=float(np.abs(cv2.GaussianBlur(a,(0,0),1)-cv2.GaussianBlur(b,(0,0),1)).mean())
                vals.append(mae);low.append(lm);rows.append(dict(crop=label,frame=i,method=name,rgb_mae=mae,sigma1_mae=lm))
            exact=[r for r in rows if r['crop']==label and r['frame']==61 and r['method']==name]
            crops[label][name]=dict(mean_rgb_mae=float(np.mean(vals)),mean_sigma1_mae=float(np.mean(low)),exact_frame61=exact[0] if exact else None)
    if rows:save_csv(out/'user_event_crops.csv',rows)
    rates=load_json(s['send_receipt'])
    value=dict(complete=True,case=case,scores=scores,dynamic=dynamic,low_support=missing,user_event_crops=crops,
        total_kbps=rates['total_kbps'],guide_kbps=rates['kbps'],guide_bytes=rates['bytes'],
        scope='Five historical same-shot 10s cases; no new long-video optimization.',subjective_review='pending',automatic_selection=None)
    save_json(out/'summary.json',value)
    print('EVALUATED',case,{n:round(v['coarse_mae'],5) for n,v in scores['full'].items()},flush=True)


def run_all(run):
    reg=load_json(run/'case_registration.json');check_inputs(reg['input_sha256'])
    for case in reg['cases']:
        subprocess.run([sys.executable,'-B','-u','-X','utf8',str(Path(__file__).resolve()),'worker','--run',str(run),'--case',case],check=True,cwd=REPO)
        evaluate(run,case)
    save_json(run/'summary.json',dict(complete=True,cases={c:load_json(run/c/'evaluation/summary.json') for c in reg['cases']},
        no_per_clip_tuning=True,automatic_selection=None,subjective_review='pending',long_video_not_run=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('freeze','run','worker'));p.add_argument('--run',type=Path,required=True);p.add_argument('--case')
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    try:
        if a.action=='freeze':freeze(a.run)
        elif a.action=='worker':receive(a.run,a.case)
        else:run_all(a.run)
    except BaseException as e:
        # A worker read guard only permits its own output directory.
        folder=a.run/a.case/'receive' if a.action=='worker' else a.run
        save_json(folder/('failure_'+a.action+'.json'),dict(error=str(e),traceback=traceback.format_exc()));raise
