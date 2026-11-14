"""Verify received-guide pixel provenance and inspect watchability exports."""
import argparse
import csv
from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from runtime_speed_common import STAGE,FFMPEG,file_sha,load_json,save_json,check_inputs
from finish_q9_ghost_exit import scores_from_csv,inspect
from received_structure_fusion import PROFILES,fuse_received_structure
import structure_guide_stream as guide_stream
from run_p0 import preview

METHODS=('source','raw','guide','Q9_R25','CSM50','GB25','GS25','GS50')

def sequences(run,case,s,plan):
    seq={n:np.load(s[n],mmap_mode='r') for n in ('source','raw')}
    seq['Q9_R25']=np.load(run/case/'receive/Q9_R25/rgb.npy',mmap_mode='r')
    seq['guide']=np.load(run/case/'receive/received_guide.npy',mmap_mode='r')
    seq['CSM50']=np.load(Path(plan['parent'])/'motion_extension'/case/'receive/CSM50/rgb.npy',mmap_mode='r')
    for n in PROFILES:seq[n]=np.load(run/case/'receive'/n/'rgb.npy',mmap_mode='r')
    return seq

def verify(run):
    plan=load_json(run/'execution_plan.json');reg=load_json(run/'case_registration.json')
    assert load_json(run/'summary.json')['complete']
    check_inputs(reg['input_sha256']);check_inputs(plan['input_sha256'])
    assert file_sha(run/'case_registration.json')==plan['registration_sha256']
    assert file_sha(run/'authorization.json')==plan['authorization_sha256']
    assert file_sha(run/'diagnosis/current_structure_errors.json')==plan['diagnosis_sha256']
    for n,h in plan['code_sha256'].items():assert file_sha(STAGE/n)==h==file_sha(run/'source_snapshot'/n)
    assert file_sha(STAGE/'当前采用配置.json')==plan['current_config_sha256']
    test=np.arange(24*24*3,dtype=np.int32).reshape(24,24,3)%256
    test=test.astype(np.uint8)
    for n,strength in [('GB25',.25),('GS25',.25),('GS50',.5)]:
        assert np.array_equal(fuse_received_structure(test,test,n)[0],test)
        a=np.full((24,24,3),80,np.uint8);b=np.full_like(a,120)
        assert np.all(fuse_received_structure(a,b,n)[0]==round(80+40*strength))
    checks={}
    for case,s in reg['cases'].items():
        out=run/case/'receive';rec=load_json(out/'receipt.json')
        assert rec['complete'] and rec['source_reads_blocked'] and len(rec['blocked'])==5
        assert set(rec['reads'])<=set(rec['allowed']) and not set(rec['reads'])&set(rec['blocked'])
        assert file_sha(s['packet'])==rec['packet_sha256'] and file_sha(s['base'])==rec['base_sha256']
        _,_,guide=guide_stream.decode(Path(s['packet']).read_bytes(),rec['base_sha256'],FFMPEG)
        seq=sequences(run,case,s,plan)
        assert np.array_equal(guide,seq['guide']) and np.array_equal(guide,np.load(Path(s['cache'])/'guide_rgb.npy',mmap_mode='r'))
        assert file_sha(out/'received_guide.npy')==rec['received_guide_sha256']
        for n,h in rec['baseline_sha256'].items():assert file_sha(out/'Q9_R25'/(n+'.npy'))==h==s['baseline_sha256'][n]
        for n in PROFILES:
            assert seq[n].shape==(150,288,512,3) and seq[n].dtype==np.uint8
            assert file_sha(out/n/'rgb.npy')==rec['variants'][n]['rgb_sha256']
            clipping=[];change=[]
            for i in range(150):
                q=seq['Q9_R25'][i].astype(np.float32);g=guide[i].astype(np.float32)
                correction=g-q
                if n!='GB25':correction=cv2.GaussianBlur(correction,(0,0),1.)
                x=q+(.5 if n=='GS50' else .25)*correction
                expected=np.rint(np.clip(x,0,255)).astype(np.uint8)
                assert np.array_equal(expected,seq[n][i])
                if n=='GB25':
                    assert np.all(expected>=np.minimum(q,g)) and np.all(expected<=np.maximum(q,g))
                clipping.append(float(((x<0)|(x>255)).mean()))
                change.append(float(np.abs(expected.astype(np.float32)-q).mean()))
            assert abs(np.mean(clipping)-rec['variants'][n]['clipping_channel_fraction'])<1e-12
            assert abs(np.mean(change)-rec['variants'][n]['mean_abs_change'])<1e-12
        summary=load_json(run/case/'evaluation/summary.json')
        scores_from_csv(run/case/'evaluation',summary)
        old=load_json(Path(plan['parent'])/'motion_extension'/case/'evaluation/summary.json')
        for scope in summary['scores']:
            for n in ('source','raw','Q9_R25','CSM50'):assert summary['scores'][scope][n]==old['scores'][scope][n]
        with (run/case/'evaluation/current_fish_regions.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        for label,methods in summary['current_regions'].items():
            x0,y0,x1,y1=plan['review_regions'][case][label]
            for n,value in methods.items():
                rr=[r for r in rows if r['region']==label and r['method']==n]
                assert len(rr)==150 and [int(r['frame']) for r in rr]==list(range(150))
                for metric in ('rgb_mae','sigma1_mae'):
                    assert abs(np.mean([float(r[metric]) for r in rr])-value['mean_'+metric])<1e-10
                direct=[]
                for i in range(150):
                    a=seq[n][i,y0:y1,x0:x1].astype(np.float32);b=seq['source'][i,y0:y1,x0:x1].astype(np.float32)
                    direct.append(float(np.abs(a-b).mean()))
                assert abs(np.mean(direct)-value['mean_rgb_mae'])<1e-10
        rate=load_json(s['send_receipt'])
        assert rate['bytes']==Path(s['packet']).stat().st_size
        assert abs(rate['total_kbps']-(Path(s['packet']).stat().st_size+Path(s['base']).stat().st_size)*8/10000)<1e-12
        checks[case]=dict(complete=True,baseline_exact=True,guide_packet_redecode_exact=True,
          source_read_guard_verified=True,all_candidate_frames_independent_formula_exact=True,
          original_baseline_scores_unchanged=True,region_errors_directly_recomputed=True,
          csv_reaggregation_exact=True,transmitted_bytes_unchanged=True,clipping_recorded=True)
        print('VERIFIED',case,flush=True)
    snap=run/'verification_source';snap.mkdir(exist_ok=False);shutil.copy2(__file__,snap/Path(__file__).name)
    save_json(run/'verification.json',dict(complete=True,cases=checks,new_outputs=9,
        current_config_unchanged=True,guide_contributes_pixels_explicitly=True,
        historical_raw_only_bounds_and_zero_alpha_invariance_not_applicable_to_new_fusion=True,
        identity_and_constant_offset_checks=True,automatic_adoption=False,verifier_sha256=file_sha(__file__)))

def media(run):
    assert load_json(run/'verification.json')['complete']
    reg=load_json(run/'case_registration.json');plan=load_json(run/'execution_plan.json')
    out=run/'previews';out.mkdir(exist_ok=False);checks={}
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    for case,s in reg['cases'].items():
        seq=sequences(run,case,s,plan)
        groups=[('all_eight',METHODS),('Q9_vs_GS25',('Q9_R25','GS25')),('Q9_vs_GS50',('Q9_R25','GS50'))]
        for suffix,names in groups:
            p=out/f'{case}_{suffix}.mp4';panels=[seq[n] for n in names]
            labels=list(names) if len(names)>2 else ['Original Q9_R25',names[1]+' received guide / NOT adopted']
            preview(FFMPEG,p,panels,15,0,labels);checks[p.name]=inspect(p,panels)
        w,h=320,180;ids=plan['review_frames']
        canvas=Image.new('RGB',(8*w,(h+25)*len(ids)),(24,35,48));d=ImageDraw.Draw(canvas)
        for row,i in enumerate(ids):
            for col,n in enumerate(METHODS):
                d.text((col*w+3,row*(h+25)+3),f'{n} f{i} {i/15:.2f}s',font=font,fill='white')
                canvas.paste(Image.fromarray(seq[n][i]).resize((w,h)),(col*w,row*(h+25)+25))
        canvas.save(out/f'{case}_whole_scene_contact.png')
        if case in ('video2','video3'):
            region='main_fish' if case=='video2' else 'foreground_fish'
            x0,y0,x1,y1=plan['review_regions'][case][region];box=np.s_[y0:y1,x0:x1]
            names=('source','guide','Q9_R25','GS25','GS50','GB25');panels=[seq[n] for n in names]
            p=out/f'{case}_current_fish_crop.mp4'
            preview(FFMPEG,p,panels,15,0,list(names),crop=box);checks[p.name]=inspect(p,panels,box)
        print('MEDIA VERIFIED',case,flush=True)
    save_json(out/'verification.json',dict(complete=True,videos=checks,video_count=len(checks),fps=15,
         note='Eight panels row-major: source/raw/guide/Q9_R25/CSM50/GB25/GS25/GS50. Pairs Q9 vs GS25 or GS50. Fish crops source/guide/Q9/GS25/GS50/GB25. All original-speed 10s.',
         user_original_speed_review='pending',verifier_sha256=file_sha(__file__)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('verify','media'));p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);(verify if a.action=='verify' else media)(a.run)

