"""Verify all ghost-exit diagnostics and export faithful original-speed views."""
import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from runtime_speed_common import STAGE,FFMPEG,file_sha,load_json,save_json,check_inputs
from run_p0 import preview
import structure_guide_stream as guide

METHODS=('source','raw','Q9_R25','E32','EL12','C32')


def sequences(run,case,s):
    values=dict(source=np.load(s['source'],mmap_mode='r'),raw=np.load(s['raw'],mmap_mode='r'))
    for n in METHODS[2:]:
        root=run/'compact_extension' if n=='C32' else run
        values[n]=np.load(root/case/'receive'/n/'rgb.npy',mmap_mode='r')
    return values


def scores_from_csv(folder,summary):
    def rows(name):
        with (folder/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    temporal=rows('temporal.csv');spatial=rows('spatial.csv');dynamic=rows('dynamic.csv')
    for scope,methods in summary['scores'].items():
        supports={}
        for name,value in methods.items():
            tr=[r for r in temporal if r['scope']==scope and r['method']==name]
            n=sum(int(r['pixels']) for r in tr);supports[name]=[(int(r['frame']),int(r['pixels'])) for r in tr]
            assert len(tr)==value['pairs'] and n==value['valid_pixels']
            assert len(tr)+sum(r['scope']==scope for r in summary['low_support'])==149
            assert abs(sum(float(r['coarse_sum']) for r in tr)/n-value['coarse_mae'])<1e-12
            sr=[r for r in spatial if r['scope']==scope and r['method']==name]
            mse=sum(float(r['rgb_sse']) for r in sr)/(3*sum(int(r['pixels']) for r in sr))
            if mse:assert abs(10*np.log10(255**2/mse)-value['psnr_db'])<1e-10
            else:assert value['psnr_db'] is None
        assert all(v==supports['raw'] for v in supports.values())
    for n,v in summary['dynamic'].items():
        rr=[r for r in dynamic if r['method']==n]
        assert abs(sum(float(r['delta_error_sum']) for r in rr)/(3*sum(int(r['pixels']) for r in rr))-v['RGB_delta_error'])<1e-10
        assert abs(sum(float(r['change_sum']) for r in rr)/sum(float(r['source_change_sum']) for r in rr)-v['change_amplitude_ratio'])<1e-10


def verify(run):
    reg=load_json(run/'case_registration.json');check_inputs(reg['input_sha256'])
    first=load_json(run/'execution_plan.json');extension=load_json(run/'compact_extension/execution_plan.json')
    assert file_sha(run/'summary.json')==extension['parent_summary_sha256']
    assert file_sha(run/'execution_plan.json')==extension['parent_plan_sha256']
    assert file_sha(run/'case_registration.json')==first['registration_sha256']
    assert file_sha(run/'user_event_registration.json')==first['user_events_sha256']
    for root,plan in [(run,first),(run/'compact_extension',extension)]:
        for n,h in plan['code_sha256'].items():assert file_sha(STAGE/n)==h==file_sha(root/'source_snapshot'/n)
    results={}
    for case,s in reg['cases'].items():
        rec=load_json(run/case/'receive/receipt.json');ext=load_json(run/'compact_extension'/case/'receive/receipt.json')
        for item,blocked in [(rec,5),(ext,4)]:
            assert item['complete'] and item['source_reads_blocked'] and len(item['blocked'])==blocked
            assert set(item['reads'])<=set(item['allowed']) and not set(item['reads'])&set(item['blocked'])
            assert item['base_sha256']==file_sha(s['base']) and item['packet_sha256']==file_sha(s['packet'])
        assert rec['baseline_replay_bit_exact']
        for n,h in s['baseline_sha256'].items():
            assert file_sha(run/case/'receive/Q9_R25'/(n+'.npy'))==h==file_sha(Path(s['baseline'])/(n+'.npy'))
            assert file_sha(run/'compact_extension'/case/'receive/Q9_R25'/(n+'.npy'))==h
        _,_,images=guide.decode(Path(s['packet']).read_bytes(),file_sha(s['base']),FFMPEG)
        assert np.array_equal(images,np.load(Path(s['cache'])/'guide_rgb.npy',mmap_mode='r'))
        seq=sequences(run,case,s);alpha=np.load(run/case/'receive/Q9_R25/alpha.npy',mmap_mode='r')
        weights={};stat={}
        for name in ('E32','EL12','C32'):
            root=run/'compact_extension' if name=='C32' else run;folder=root/case/'receive'/name
            expected=ext['outputs_sha256'] if name=='C32' else rec['variants'][name]['outputs_sha256']
            for n,h in expected.items():assert file_sha(folder/(n+'.npy'))==h
            w=np.load(folder/'exit_weight.npy',mmap_mode='r');weights[name]=w
            assert w.shape==(150,288,512) and w.dtype==np.float32 and np.isfinite(w).all()
            assert seq[name].shape==(150,288,512,3) and seq[name].dtype==np.uint8
            assert np.all((w>=0)&(w<=1))
            changed=0
            for t in range(150):
                a=seq[name][t];raw=seq['raw'][t];q=seq['Q9_R25'][t]
                assert np.array_equal(a[w[t]==0],q[w[t]==0]) and np.array_equal(a[w[t]==1],raw[w[t]==1])
                assert np.array_equal(a[alpha[t]==0],raw[alpha[t]==0])
                assert np.all(np.minimum(q,raw)<=a) and np.all(a<=np.maximum(q,raw))
                changed+=int(np.any(a!=q,axis=2).sum())
            stat[name]=dict(changed_fraction=changed/(150*288*512),full_exit_fraction=float((w==1).mean()),unchanged_outside_exit=True,
                            zero_original_alpha_unchanged=True,convex_bounds_verified=True)
        assert np.all(weights['C32']<=weights['EL12'])
        summaries=[]
        for root in (run,run/'compact_extension'):
            summary=load_json(root/case/'evaluation/summary.json');scores_from_csv(root/case/'evaluation',summary);summaries.append(summary)
            for label,crop_scores in summary['user_event_crops'].items():
                box=first['review_boxes']['video2_tail' if label=='tail' else label];x0,y0,x1,y1=box
                ids=range(*(first['review_timing']['video2_tail' if label=='tail' else 'reef_departure']))
                for name,values in crop_scores.items():
                    actual=float(np.mean([np.abs(seq[name][i,y0:y1,x0:x1].astype(np.float32)-seq['source'][i,y0:y1,x0:x1]).mean() for i in ids],dtype=np.float64))
                    assert abs(actual-values['mean_rgb_mae'])<1e-10
                    if values['exact_frame61']:
                        actual=float(np.abs(seq[name][61,y0:y1,x0:x1].astype(np.float32)-seq['source'][61,y0:y1,x0:x1]).mean())
                        assert abs(actual-values['exact_frame61']['rgb_mae'])<1e-10
        for scope in summaries[0]['scores']:
            for n in ('source','raw','Q9_R25'):assert summaries[0]['scores'][scope][n]==summaries[1]['scores'][scope][n]
        rate=load_json(s['send_receipt']);assert rate['bytes']==Path(s['packet']).stat().st_size
        assert abs(rate['total_kbps']-(Path(s['packet']).stat().st_size+Path(s['base']).stat().st_size)*8/10000)<1e-12
        results[case]=dict(complete=True,baseline_exact=True,guide_redecode_exact=True,read_guard_verified=True,
            candidate_properties=stat,csv_reaggregation_exact=True,user_crop_values_exact=True,transmitted_bytes_unchanged=True)
        print('VERIFIED',case,flush=True)
    save_json(run/'verification.json',dict(complete=True,cases=results,all_frozen_inputs_and_codes_unchanged=True,
        first_phase_new_outputs=10,adaptive_extension_new_outputs=5,automatic_adoption=False,verifier_sha256=file_sha(__file__)))


def inspect(path,panels,crop=None):
    probe=load_json_from_command([str(FFMPEG.with_name('ffprobe.exe')),'-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=width,height,avg_frame_rate,nb_read_frames,duration','-of','json',str(path)])['streams'][0]
    n=len(panels[0]);assert int(probe['nb_read_frames'])==n and Fraction(probe['avg_frame_rate'])==15
    h,w=panels[0].shape[1:3] if crop is None else (2*(crop[0].stop-crop[0].start),2*(crop[1].stop-crop[1].start))
    assert (probe['width'],probe['height'])==(w*2,(h+26+h%2)*((len(panels)+1)//2))
    assert abs(float(probe['duration'])-n/15)<.001
    ids=sorted(set([0,n//4,n//2,3*n//4,n-1]));sel='+'.join(f'eq(n\\,{i})' for i in ids)
    data=subprocess.check_output([str(FFMPEG),'-v','error','-threads','1','-i',str(path),'-vf','select='+sel,'-fps_mode','vfr','-pix_fmt','rgb24','-threads','1','-f','rawvideo','pipe:1'])
    arr=np.frombuffer(data,np.uint8).reshape(len(ids),probe['height'],probe['width'],3);errors=[]
    for j,a in enumerate(panels):
        x,y=(j%2)*w,(j//2)*(h+26+h%2)+26;values=[]
        for k,i in enumerate(ids):
            target=a[i] if crop is None else cv2.resize(a[i][crop],(w,h),interpolation=cv2.INTER_NEAREST)
            values.append(float(np.abs(arr[k,y:y+h,x:x+w].astype(np.float32)-target).mean()))
        assert max(values)<6;errors.append(values)
    return dict(probe=probe,frame_indices=ids,panel_rgb_mae=errors,sha256=file_sha(path),sample_only_decode=True)


def load_json_from_command(cmd):return json.loads(subprocess.check_output(cmd))


def media(run):
    assert load_json(run/'verification.json')['complete'];reg=load_json(run/'case_registration.json')
    out=run/'previews';out.mkdir(exist_ok=False);checks={};font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    for case,s in reg['cases'].items():
        seq=sequences(run,case,s);path=out/f'{case}_all_six.mp4';panels=[seq[n] for n in METHODS]
        preview(FFMPEG,path,panels,15,0,list(METHODS));checks[path.name]=inspect(path,panels)
        if case in ('reef_spotted_fish','video2'):
            path=out/f'{case}_Q9_vs_EL12.mp4';panels=[seq[n] for n in ('Q9_R25','EL12')]
            preview(FFMPEG,path,panels,15,0,['Original Q9_R25','EL12 diagnostic / NOT adopted']);checks[path.name]=inspect(path,panels)
            box=(np.s_[100:225,310:512] if case=='reef_spotted_fish' else np.s_[145:288,140:285])
            path=out/f'{case}_user_event_crop_all.mp4';panels=[seq[n][45:75] for n in METHODS]
            preview(FFMPEG,path,panels,15,45,list(METHODS),crop=box);checks[path.name]=inspect(path,panels,box)
            ids=[53,61,69] if case=='reef_spotted_fish' else [50,61,70]
            ch=box[0].stop-box[0].start;cw=box[1].stop-box[1].start;w=280;h=round(ch*w/cw)
            canvas=Image.new('RGB',(w*len(METHODS),(h+25)*len(ids)),(24,35,48));d=ImageDraw.Draw(canvas)
            for row,t in enumerate(ids):
                for col,n in enumerate(METHODS):
                    d.text((col*w+3,row*(h+25)+3),f'{n} f{t} {t/15:.2f}s',font=font,fill='white')
                    canvas.paste(Image.fromarray(seq[n][t][box]).resize((w,h)),(col*w,row*(h+25)+25))
            canvas.save(out/f'{case}_user_event_contact.png')
        print('MEDIA VERIFIED',case,flush=True)
    # Large old-fish footprints at the exact user time for direct attribution.
    s=reg['cases']['reef_spotted_fish'];seq=sequences(run,'reef_spotted_fish',s)
    canvas=Image.new('RGB',(280*len(METHODS),2*285),(24,35,48));d=ImageDraw.Draw(canvas)
    for row,box in enumerate(([416,112,452,144],[488,112,512,144])):
        x0,y0,x1,y1=box
        for col,n in enumerate(METHODS):
            d.text((col*280+3,row*285+3),f'old fish {row+1} {n} f61',font=font,fill='white')
            canvas.paste(Image.fromarray(seq[n][61,y0:y1,x0:x1]).resize((280,256)),(col*280,row*285+25))
    canvas.save(out/'reef_exact4p07_footprints.png')
    save_json(out/'verification.json',dict(complete=True,videos=checks,video_count=len(checks),fps=15,
        note='Six panels row-major: source/raw/Q9/E32/EL12/C32. EL12 pairs are diagnostic, not adopted; crops cover original3.0-5.0s.',
        user_subjective_review='pending',verifier_sha256=file_sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('verify','media'));p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);(verify if a.action=='verify' else media)(a.run)
