"""Verify and render the fixed P3/M3/ML3 temporal-exit round."""
import argparse
from pathlib import Path
import csv
import shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from runtime_speed_common import STAGE, FFMPEG, file_sha, load_json, save_json, check_inputs
from receiver_structure import KEYS
from temporal_conflict_exit import PROFILES, TemporalExit
from finish_q9_ghost_exit import scores_from_csv, inspect
from run_p0 import preview

METHODS=('source','raw','Q9_R25','P3','M3','ML3')

def sequences(run,case,s):
    seq={n:np.load(s[n],mmap_mode='r') for n in ('source','raw')}
    for n in METHODS[2:]:seq[n]=np.load(run/case/'receive'/n/'rgb.npy',mmap_mode='r')
    return seq

def synthetic_checks():
    shape=(40,40,3);zeros=np.zeros(shape,np.uint8);flow=np.zeros((40,40,2),np.float32)
    alpha=np.ones((40,40),np.float32);proc=TemporalExit(40,40)
    entered=False;motion_entered=False
    for i in range(35):
        raw=np.full(shape,80 if 3<=i<12 else 0,np.uint8)
        results,_=proc.step(i,raw,zeros,alpha,flow,flow)
        if i in KEYS:
            assert all(not w.any() for _,w in results.values())
        if 6<=i<12:
            entered|=bool(results['P3'][1][20,20]>.5)
            motion_entered|=bool(results['M3'][1][20,20]>.5)
        if i==34:
            assert all(not w.any() for _,w in results.values())
    assert entered and motion_entered
    proc=TemporalExit(40,40)
    for i in range(36):
        raw=np.full(shape,i%256,np.uint8)
        result,_=proc.step(i,raw,raw,alpha,flow,flow)
        assert all(np.array_equal(x,raw) and not w.any() for x,w in result.values())
    return dict(unchanged_baseline_stays_exact=True,persistent_departure_triggers=True,
                motion_proxy_triggers=True,conflict_release_returns_to_baseline=True,
                anchor_resets_gate=True)

def verify(run):
    reg=load_json(run/'case_registration.json');plan=load_json(run/'execution_plan.json')
    assert load_json(run/'summary.json')['complete']
    check_inputs(reg['input_sha256']);check_inputs(plan['baseline_input_sha256'])
    assert file_sha(run/'case_registration.json')==plan['registration_sha256']
    assert file_sha(run/'user_event_registration.json')==plan['user_event_sha256']
    for n,h in plan['code_sha256'].items():
        assert file_sha(STAGE/n)==h==file_sha(run/'source_snapshot'/n)
    assert file_sha(STAGE/'当前采用配置.json')==plan['current_config_sha256']
    synthetic=synthetic_checks();checks={}
    for case,s in reg['cases'].items():
        out=run/case/'receive';rec=load_json(out/'receipt.json')
        assert rec['complete'] and rec['source_reads_blocked'] and len(rec['blocked'])==5
        assert set(rec['reads'])<=set(rec['allowed']) and not set(rec['reads'])&set(rec['blocked'])
        assert file_sha(s['packet'])==rec['packet_sha256'] and file_sha(s['base'])==rec['base_sha256']
        for n,h in rec['baseline_sha256'].items():assert file_sha(out/'Q9_R25'/(n+'.npy'))==h==s['baseline_sha256'][n]
        seq=sequences(run,case,s)
        alpha=np.load(out/'Q9_R25/alpha.npy',mmap_mode='r')
        weights={n:np.load(out/n/'weight.npy',mmap_mode='r') for n in PROFILES}
        stats={}
        for n in PROFILES:
            assert seq[n].shape==(150,288,512,3) and seq[n].dtype==np.uint8
            assert weights[n].shape==(150,288,512) and weights[n].dtype==np.float32
            assert np.isfinite(weights[n]).all() and np.all((weights[n]>=0)&(weights[n]<=1))
            for kind,h in rec['output_sha256'][n].items():assert file_sha(out/n/(kind+'.npy'))==h
            changed=0
            for i in range(150):
                raw=seq['raw'][i];q=seq['Q9_R25'][i];current=seq[n][i];w=weights[n][i]
                assert np.array_equal(current[w==0],q[w==0])
                assert np.array_equal(current[alpha[i]==0],raw[alpha[i]==0])
                assert np.all(current>=np.minimum(raw,q)) and np.all(current<=np.maximum(raw,q))
                if n!='ML3':
                    expected=np.rint(q.astype(np.float32)+w[...,None]*(raw.astype(np.float32)-q.astype(np.float32))).astype(np.uint8)
                    assert np.array_equal(current,expected)
                changed+=int(np.any(current!=q,axis=2).sum())
                if i in KEYS:assert not w.any() and np.array_equal(current,q)
            stats[n]=dict(changed_pixels_fraction=changed/(150*288*512),mean_weight=float(weights[n].mean()),
                          active_fraction=float((weights[n]>0).mean()))
        assert np.all(weights['M3']<=weights['P3']+1e-6) and np.array_equal(weights['M3'],weights['ML3'])
        for i in range(150):
            q=seq['Q9_R25'][i].astype(np.int16)
            assert np.all(np.abs(seq['ML3'][i].astype(np.int16)-q)<=np.abs(seq['M3'][i].astype(np.int16)-q))
        # Reconstruct a user-event prefix from received inputs, checking actual output arrays.
        if case in ('video2','reef_spotted_fish'):
            proc=TemporalExit(288,512)
            back=np.load(Path(s['cache'])/'to_anchor.npy',mmap_mode='r')
            forward=np.load(Path(s['cache'])/'from_anchor.npy',mmap_mode='r')
            for i in range(76):
                results,_=proc.step(i,seq['raw'][i],seq['Q9_R25'][i],alpha[i],back[i],forward[i])
                for n,(rgb,w) in results.items():
                    assert np.array_equal(rgb,seq[n][i]) and np.array_equal(w,weights[n][i])
        summary=load_json(run/case/'evaluation/summary.json');scores_from_csv(run/case/'evaluation',summary)
        old=load_json(Path(plan['parent'])/case/'evaluation/summary.json')
        for scope in summary['scores']:
            for n in ('source','raw','Q9_R25'):assert summary['scores'][scope][n]==old['scores'][scope][n]
        if case in ('video2','reef_spotted_fish'):
            frames=range(*plan['review_timing']['reef_departure' if case=='reef_spotted_fish' else 'video2_tail'])
            for label,methods in summary['user_event_crops'].items():
                box=plan['review_boxes'][label if label.startswith('reef') else 'video2_tail']
                x0,y0,x1,y1=box
                for n,value in methods.items():
                    vals=[float(np.abs(seq[n][i,y0:y1,x0:x1].astype(np.float32)-seq['source'][i,y0:y1,x0:x1].astype(np.float32)).mean()) for i in frames]
                    assert abs(np.mean(vals)-value['mean_rgb_mae'])<1e-10
                    val=float(np.abs(seq[n][61,y0:y1,x0:x1].astype(np.float32)-seq['source'][61,y0:y1,x0:x1].astype(np.float32)).mean())
                    assert abs(val-value['exact_frame61']['rgb_mae'])<1e-10
        rates=load_json(s['send_receipt'])
        assert rates['bytes']==Path(s['packet']).stat().st_size
        assert abs(rates['total_kbps']-(Path(s['base']).stat().st_size+Path(s['packet']).stat().st_size)*8/10000)<1e-12
        checks[case]=dict(complete=True,baseline_exact=True,input_boundary_checked=True,output_properties=stats,
                          csv_reaggregation_exact=True,user_crops_exact=True,bytes_unchanged=True,
                          causal_prefix_replay_exact=case in ('video2','reef_spotted_fish'))
        print('VERIFIED',case,flush=True)
    snap=run/'verification_source';snap.mkdir(exist_ok=False);shutil.copy2(__file__,snap/Path(__file__).name)
    save_json(run/'verification.json',dict(complete=True,cases=checks,new_outputs=15,
              all_frozen_inputs_and_codes_unchanged=True,current_config_unchanged=True,
              synthetic_checks=synthetic,automatic_adoption=False,verifier_sha256=file_sha(__file__)))

def media(run):
    assert load_json(run/'verification.json')['complete']
    reg=load_json(run/'case_registration.json')
    out=run/'previews';out.mkdir(exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15);checks={}
    for case,s in reg['cases'].items():
        seq=sequences(run,case,s);panels=[seq[n] for n in METHODS];p=out/f'{case}_all_six.mp4'
        preview(FFMPEG,p,panels,15,0,list(METHODS));checks[p.name]=inspect(p,panels)
        if case in ('video2','reef_spotted_fish'):
            p=out/f'{case}_Q9_vs_ML3.mp4';panels=[seq[n] for n in ('Q9_R25','ML3')]
            preview(FFMPEG,p,panels,15,0,['Original Q9_R25','ML3 diagnostic / NOT adopted']);checks[p.name]=inspect(p,panels)
            box=np.s_[100:225,310:512] if case=='reef_spotted_fish' else np.s_[145:288,140:285]
            p=out/f'{case}_user_event_crop_all.mp4';panels=[seq[n][45:75] for n in METHODS]
            preview(FFMPEG,p,panels,15,45,list(METHODS),crop=box);checks[p.name]=inspect(p,panels,box)
            ids=[53,61,69] if case=='reef_spotted_fish' else [50,61,70]
            w=280;h=round((box[0].stop-box[0].start)*w/(box[1].stop-box[1].start))
            canvas=Image.new('RGB',(w*6,(h+25)*3),(24,35,48));d=ImageDraw.Draw(canvas)
            for row,i in enumerate(ids):
                for col,n in enumerate(METHODS):
                    d.text((col*w+3,row*(h+25)+3),f'{n} f{i} {i/15:.2f}s',font=font,fill='white')
                    canvas.paste(Image.fromarray(seq[n][i][box]).resize((w,h)),(col*w,row*(h+25)+25))
            canvas.save(out/f'{case}_user_event_contact.png')
        print('MEDIA VERIFIED',case,flush=True)
    seq=sequences(run,'reef_spotted_fish',reg['cases']['reef_spotted_fish'])
    canvas=Image.new('RGB',(280*6,570),(24,35,48));d=ImageDraw.Draw(canvas)
    for row,(x0,y0,x1,y1) in enumerate(([416,112,452,144],[488,112,512,144])):
        for col,n in enumerate(METHODS):
            d.text((col*280+3,row*285+3),f'old fish {row+1} {n} f61',font=font,fill='white')
            canvas.paste(Image.fromarray(seq[n][61,y0:y1,x0:x1]).resize((280,256)),(col*280,row*285+25))
    canvas.save(out/'reef_exact4p07_footprints.png')
    save_json(out/'verification.json',dict(complete=True,videos=checks,video_count=len(checks),fps=15,
              note='Row-major: source/raw/Q9_R25/P3/M3/ML3. Paired ML3 views are diagnostic, not adopted.',
              user_subjective_review='pending',verifier_sha256=file_sha(__file__)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('verify','media'));p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    (verify if a.action=='verify' else media)(a.run)

