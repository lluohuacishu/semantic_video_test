"""Check current-structure outputs and export whole-scene watchability comparisons."""
import argparse
import csv
from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from runtime_speed_common import STAGE,FFMPEG,file_sha,load_json,save_json,check_inputs
from finish_q9_ghost_exit import scores_from_csv,inspect
from current_structure_preservation import PROFILES,preserve_current
from run_p0 import preview

METHODS=('source','raw','Q9_R25','CS50','CS100','CSM50')
CROP={'video1':(20,20,300,288),'video2':(130,110,430,288),
      'video3':(0,160,270,288),'reef_spotted_fish':(190,30,330,245)}

def sequences(run,case,s,plan):
    seq={n:np.load(s[n],mmap_mode='r') for n in ('source','raw')}
    seq['Q9_R25']=np.load(run/case/'receive/Q9_R25/rgb.npy',mmap_mode='r')
    seq['ML3']=np.load(Path(plan['parent'])/case/'receive/ML3/rgb.npy',mmap_mode='r')
    for n in PROFILES:seq[n]=np.load(run/case/'receive'/n/'rgb.npy',mmap_mode='r')
    if (run/'motion_extension'/case/'receive/CSM50/rgb.npy').exists():
        seq['CSM50']=np.load(run/'motion_extension'/case/'receive/CSM50/rgb.npy',mmap_mode='r')
    return seq

def verify(run):
    plan=load_json(run/'execution_plan.json');reg=load_json(run/'case_registration.json')
    assert load_json(run/'summary.json')['complete']
    check_inputs(reg['input_sha256']);check_inputs(plan['input_sha256'])
    assert file_sha(run/'case_registration.json')==plan['registration_sha256']
    assert file_sha(run/'user_feedback_20261112.json')==plan['feedback_sha256']
    for n,h in plan['code_sha256'].items():assert file_sha(STAGE/n)==h==file_sha(run/'source_snapshot'/n)
    assert file_sha(STAGE/'当前采用配置.json')==plan['current_config_sha256']
    # Independent constant-offset oracle, sign reversal and identity checks.
    ramp=np.arange(64*64*3,dtype=np.int32).reshape(64,64,3)%156+50
    raw=ramp.astype(np.uint8);q=(ramp-30).astype(np.uint8)
    for n,strength in [('CS50',.5),('CS100',1.)]:
        assert np.array_equal(preserve_current(raw,raw,n),raw)
        const=np.full((40,40,3),120,np.uint8);old=np.full_like(const,80)
        assert np.all(preserve_current(const,old,n)==round(80+40*strength))
        assert np.all(preserve_current(old,const,n)==round(120-40*strength))
    cases={}
    for case,s in reg['cases'].items():
        out=run/case/'receive';rec=load_json(out/'receipt.json')
        assert rec['complete'] and rec['source_reads_blocked'] and len(rec['blocked'])==5
        assert set(rec['reads'])<=set(rec['allowed']) and not set(rec['reads'])&set(rec['blocked'])
        assert file_sha(s['packet'])==rec['packet_sha256'] and file_sha(s['base'])==rec['base_sha256']
        for n,h in rec['baseline_sha256'].items():assert file_sha(out/'Q9_R25'/(n+'.npy'))==h==s['baseline_sha256'][n]
        seq=sequences(run,case,s,plan);alpha=np.load(out/'Q9_R25/alpha.npy',mmap_mode='r')
        for n in PROFILES:
            assert seq[n].shape==(150,288,512,3) and seq[n].dtype==np.uint8
            assert file_sha(out/n/'rgb.npy')==rec['variants'][n]['rgb_sha256']
        changed={n:0 for n in PROFILES}
        for i in range(150):
            raw=seq['raw'][i];q=seq['Q9_R25'][i]
            residual=raw.astype(np.float32)-q.astype(np.float32)
            low=cv2.GaussianBlur(residual,(0,0),1.)
            bounded=np.where(residual>=0,np.minimum(np.maximum(low,0),residual),
                             np.maximum(np.minimum(low,0),residual))
            for n,strength in [('CS50',.5),('CS100',1.)]:
                expected=np.rint(q.astype(np.float32)+strength*bounded).astype(np.uint8)
                current=seq[n][i]
                assert np.array_equal(current,expected)
                assert np.all(current>=np.minimum(raw,q)) and np.all(current<=np.maximum(raw,q))
                assert np.array_equal(current[alpha[i]==0],raw[alpha[i]==0])
                changed[n]+=int(np.any(current!=q,axis=2).sum())
            assert np.all(np.abs(seq['CS50'][i].astype(np.int16)-q.astype(np.int16))<=
                          np.abs(seq['CS100'][i].astype(np.int16)-q.astype(np.int16)))
        summary=load_json(run/case/'evaluation/summary.json')
        scores_from_csv(run/case/'evaluation',summary)
        prior=load_json(Path(plan['parent'])/case/'evaluation/summary.json')
        for scope in summary['scores']:
            for n in ('source','raw','Q9_R25','ML3'):assert summary['scores'][scope][n]==prior['scores'][scope][n]
        with (run/case/'evaluation/current_fish_regions.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        for label,values in summary['current_regions'].items():
            x0,y0,x1,y1=plan['review_regions'][case][label]
            for n,v in values.items():
                rr=[r for r in rows if r['region']==label and r['method']==n]
                assert len(rr)==150 and [int(r['frame']) for r in rr]==list(range(150))
                assert abs(np.mean([float(r['rgb_mae']) for r in rr])-v['mean_rgb_mae'])<1e-10
                assert abs(np.mean([float(r['sigma1_mae']) for r in rr])-v['mean_sigma1_mae'])<1e-10
                direct=[]
                for i in range(150):
                    a=seq[n][i,y0:y1,x0:x1].astype(np.float32);b=seq['source'][i,y0:y1,x0:x1].astype(np.float32)
                    direct.append(float(np.abs(a-b).mean()))
                assert abs(np.mean(direct)-v['mean_rgb_mae'])<1e-10
        rates=load_json(s['send_receipt'])
        assert rates['bytes']==Path(s['packet']).stat().st_size
        assert abs(summary['total_kbps']-(Path(s['base']).stat().st_size+Path(s['packet']).stat().st_size)*8/10000)<1e-12
        cases[case]=dict(complete=True,baseline_and_ML3_unchanged=True,
             all_candidate_frames_match_independent_formula=True,source_read_guard_verified=True,
             zero_alpha_preserved=True,channelwise_current_raw_Q9_bounds=True,CS50_change_no_larger_than_CS100=True,
             csv_reaggregation_exact=True,current_region_errors_exact=True,packet_bytes_unchanged=True,
             changed_fraction={n:changed[n]/(150*288*512) for n in PROFILES})
        print('VERIFIED',case,flush=True)
    snap=run/'verification_source';snap.mkdir(exist_ok=False);shutil.copy2(__file__,snap/Path(__file__).name)
    save_json(run/'verification.json',dict(complete=True,cases=cases,new_outputs=8,
              frozen_inputs_and_code_unchanged=True,current_config_unchanged=True,
              identity_and_constant_offset_checks=True,automatic_adoption=False,verifier_sha256=file_sha(__file__)))
    verify_motion(run)

def verify_motion(run):
    from moving_structure_preservation import local_motion_weight,preserve_moving_structure
    ext=run/'motion_extension';plan=load_json(ext/'execution_plan.json');reg=load_json(ext/'case_registration.json')
    assert load_json(ext/'summary.json')['complete']
    assert file_sha(run/'summary.json')==plan['first_phase_summary_sha256']
    assert file_sha(run/'execution_plan.json')==plan['first_phase_plan_sha256']
    for n,h in plan['code_sha256'].items():assert file_sha(STAGE/n)==h==file_sha(ext/'source_snapshot'/n)
    # Dominant camera motion should not count as independent local motion.
    yy,xx=np.mgrid[:80,:80].astype(np.float32);alpha=np.ones((80,80),np.float32)
    field=np.stack((.02*xx+.01*yy+2,-.01*xx+.03*yy-1),axis=2)
    w,_=local_motion_weight(field,alpha);assert not w.any()
    field[28:44,28:44,0]+=6
    w,_=local_motion_weight(field,alpha);assert w[33:39,33:39].mean()>.8 and w[:10,:10].max()<.01
    checks={}
    for case,s in reg['cases'].items():
        rec=load_json(ext/case/'receive/receipt.json')
        assert rec['complete'] and rec['source_reads_blocked'] and len(rec['blocked'])==5
        assert set(rec['reads'])<=set(rec['allowed']) and not set(rec['reads'])&set(rec['blocked'])
        for n,h in rec['output_sha256'].items():assert file_sha(ext/case/'receive/CSM50'/(n+'.npy'))==h
        seq=sequences(run,case,s,plan);a=np.load(run/case/'receive/Q9_R25/alpha.npy',mmap_mode='r')
        flow=np.load(Path(s['cache'])/'to_anchor.npy',mmap_mode='r')
        weights=np.load(ext/case/'receive/CSM50/motion_weight.npy',mmap_mode='r')
        assert seq['CSM50'].shape==(150,288,512,3) and seq['CSM50'].dtype==np.uint8
        assert weights.shape==(150,288,512) and weights.dtype==np.float32
        assert np.isfinite(weights).all() and np.all((weights>=0)&(weights<=1))
        for i in range(150):
            expected,w,_=preserve_moving_structure(seq['raw'][i],seq['Q9_R25'][i],flow[i],a[i])
            current=seq['CSM50'][i];q=seq['Q9_R25'][i];raw=seq['raw'][i]
            assert np.array_equal(current,expected) and np.array_equal(w,weights[i])
            assert np.all(current>=np.minimum(raw,q)) and np.all(current<=np.maximum(raw,q))
            assert np.array_equal(current[a[i]==0],raw[a[i]==0])
            assert np.array_equal(current[w==0],q[w==0])
            assert np.all(np.abs(current.astype(np.int16)-q.astype(np.int16))<=np.abs(seq['CS50'][i].astype(np.int16)-q.astype(np.int16)))
        summary=load_json(ext/case/'evaluation/summary.json');scores_from_csv(ext/case/'evaluation',summary)
        base=load_json(run/case/'evaluation/summary.json')
        for scope in summary['scores']:
            for n in ('source','raw','Q9_R25','ML3'):assert summary['scores'][scope][n]==base['scores'][scope][n]
        for label,values in summary['current_regions'].items():
            x0,y0,x1,y1=plan['review_regions'][case][label]
            for n,v in values.items():
                direct=[float(np.abs(seq[n][i,y0:y1,x0:x1].astype(np.float32)-seq['source'][i,y0:y1,x0:x1].astype(np.float32)).mean()) for i in range(150)]
                assert abs(np.mean(direct)-v['mean_rgb_mae'])<1e-10
        assert file_sha(s['packet'])==rec['packet_sha256'] and file_sha(s['base'])==rec['base_sha256']
        checks[case]=dict(complete=True,read_guard_checked=True,full_replay_exact=True,source_region_errors_exact=True,
             baseline_scores_unchanged=True,bounded_change_no_larger_than_CS50=True,bytes_unchanged=True,
             mean_motion_weight=float(weights.mean()),active_fraction=float((weights>0).mean()))
        print('VERIFIED MOTION',case,flush=True)
    save_json(ext/'verification.json',dict(complete=True,cases=checks,new_outputs=4,adaptive_extension=True,
        synthetic_affine_camera_rejected=True,synthetic_local_motion_retained=True,
        verifier_sha256=file_sha(__file__),automatic_adoption=False))

def media(run):
    assert load_json(run/'verification.json')['complete']
    assert load_json(run/'motion_extension/verification.json')['complete']
    plan=load_json(run/'execution_plan.json');reg=load_json(run/'case_registration.json')
    out=run/'previews';out.mkdir(exist_ok=False);checks={}
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    for case,s in reg['cases'].items():
        seq=sequences(run,case,s,plan)
        for suffix,names in [('all_six',METHODS),('Q9_vs_CSM50',('Q9_R25','CSM50'))]:
            p=out/f'{case}_{suffix}.mp4';panels=[seq[n] for n in names]
            labels=list(names) if len(names)>2 else ['Original Q9_R25','CSM50 motion structure / NOT adopted']
            preview(FFMPEG,p,panels,15,0,labels);checks[p.name]=inspect(p,panels)
        # Full-scene contacts are primary; fish crops are a secondary diagnostic.
        w,h=384,216;ids=plan['review_frames']
        canvas=Image.new('RGB',(w*6,(h+25)*len(ids)),(24,35,48));draw=ImageDraw.Draw(canvas)
        for row,i in enumerate(ids):
            for col,n in enumerate(METHODS):
                draw.text((col*w+3,row*(h+25)+3),f'{n} f{i} {i/15:.2f}s',font=font,fill='white')
                canvas.paste(Image.fromarray(seq[n][i]).resize((w,h)),(col*w,row*(h+25)+25))
        canvas.save(out/f'{case}_whole_scene_contact.png')
        if case in ('video2','reef_spotted_fish'):
            x0,y0,x1,y1=CROP[case];box=np.s_[y0:y1,x0:x1]
            p=out/f'{case}_current_fish_crop.mp4';names=('source','raw','Q9_R25','CSM50');panels=[seq[n] for n in names]
            preview(FFMPEG,p,panels,15,0,list(names),crop=box);checks[p.name]=inspect(p,panels,box)
        print('MEDIA VERIFIED',case,flush=True)
    save_json(out/'verification.json',dict(complete=True,videos=checks,video_count=len(checks),fps=15,
              note='Six row-major: source/raw/Q9_R25/CS50/CS100/CSM50. Pairs: Q9/CSM50. Fish crops: source/raw/Q9/CSM50, all 10 seconds, original speed.',
              original_speed_user_review='pending',verifier_sha256=file_sha(__file__)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('verify','media'));p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    (verify if a.action=='verify' else media)(a.run)

