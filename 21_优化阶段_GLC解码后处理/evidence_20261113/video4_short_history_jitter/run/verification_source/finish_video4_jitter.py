"""Verify finite-history experiment and produce original-speed comparisons."""
import argparse
import csv
from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from runtime_speed_common import STAGE,FFMPEG,file_sha,load_json,save_json,check_inputs
from run_video4_jitter import load_sequences,P3,P4,CONFIG,NAMES
from short_history_filter import blend_frame,guide_correction,quantize
from audit_received_structure_blur import blur,amplitude
from finish_q9_ghost_exit import inspect
from run_p0 import preview


def read_csv(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def close(a,b):
    assert abs(a-b)<1e-8,(a,b)


def checks(run):
    plan=load_json(run/'execution_plan.json');check_inputs(plan['input_sha256'])
    assert file_sha(STAGE/'当前采用配置.json')==CONFIG
    for name,h in plan['code_sha256'].items():assert file_sha(STAGE/name)==h==file_sha(run/'source_snapshot'/name)
    verified={}
    for case in plan['cases']:
        spec=load_json(run/'case_registration.json')['cases'][case]
        rec=load_json(run/case/'receive/receipt.json')
        assert rec['complete'] and rec['source_reads_blocked'] and len(rec['blocked'])==4
        assert rec['new_transmission_bytes']==0 and rec['guide_redecoded_exact']
        seq=load_sequences(run,case);n=len(seq['source'])
        flows=np.load(run/case/'receive/flows.npy',mmap_mode='r')
        assert flows.shape==(n,2,2,288,512,2)
        assert file_sha(run/case/'receive/flows.npy')==rec['flow_sha256']
        assert rec['flow']['pairs']==2*n-3
        assert np.array_equal(seq['SQ3'][0],seq['GS25'][0])
        for name in (*NAMES,'BLUR_SQ3'):
            assert seq[name].shape==seq['source'].shape and seq[name].dtype==np.uint8
            assert file_sha(run/case/'receive'/name/'rgb.npy')==rec['output_sha256'][name]
        for i in range(n):
            assert np.isfinite(flows[i]).all()
            direct=quantize(seq['raw'][i].astype(np.float32)+.25*cv2.GaussianBlur(seq['guide'][i].astype(np.float32)-seq['raw'][i],(0,0),1.))
            assert np.array_equal(direct,seq['RG25'][i])
            for name,base in [('SR3','raw'),('SQ3','GS25')]:
                f,stats=blend_frame(seq[base],seq['guide'],flows,i)
                if name=='SR3':f=guide_correction(f,seq['guide'][i])
                assert np.array_equal(quantize(f),seq[name][i]),(case,name,i)
                assert stats['history_weight_max']<=.5+1e-6
        # Perturb frames outside the two-frame window: output must be unchanged.
        test_index=n//2;test=np.array(seq['GS25']);original,_=blend_frame(test,seq['guide'],flows,test_index)
        test[:test_index-2]=255-test[:test_index-2];test[test_index+1:]=255-test[test_index+1:]
        changed,_=blend_frame(test,seq['guide'],flows,test_index)
        assert np.array_equal(original,changed)
        bm=load_json(run/case/'receive/blur_match.json')
        assert bm['relative_error']<.005
        assert np.array_equal(blur(seq['GS25'],bm['sigma']),seq['BLUR_SQ3'])
        close(amplitude(seq['SQ3']),bm['target'])
        folder=run/case/'evaluation';summary=load_json(folder/'summary.json')
        temporal=read_csv(folder/'temporal.csv');spatial=read_csv(folder/'spatial.csv')
        jitter=read_csv(folder/'jitter.csv');dynamic=read_csv(folder/'dynamic.csv')
        for scope,methods in summary['scores'].items():
            support=None
            for name,sc in methods.items():
                rr=[r for r in temporal if r['scope']==scope and r['method']==name]
                layout=[(r['frame'],r['pixels']) for r in rr]
                if support is None:support=layout
                else:assert layout==support
                pixels=sum(int(r['pixels']) for r in rr)
                for band in ('coarse','fine','full'):
                    if pixels:close(sum(float(r[band+'_sum']) for r in rr)/pixels,sc[band+'_mae'])
                ss=[r for r in spatial if r['scope']==scope and r['method']==name]
                assert len(ss)==n
                sn=sum(int(r['pixels']) for r in ss)
                close(sum(float(r['rgb_abs_sum']) for r in ss)/(3*sn),sc['rgb_mae'])
                mse=sum(float(r['rgb_sse']) for r in ss)/(3*sn)
                if mse:close(10*np.log10(255**2/mse),sc['psnr_db'])
                jj=[r for r in jitter if r['scope']==scope and r['method']==name]
                jp=sum(int(r['pixels']) for r in jj);ap=sum(int(r['accel_pixels']) for r in jj)
                js=summary['jitter'][scope][name]
                if jp:close(sum(int(r['large_change_pixels']) for r in jj)/jp,js['large_change_fraction'])
                if ap:
                    for band in ('coarse','fine','full'):close(sum(float(r[band+'_accel_sum']) for r in jj)/ap,js[band+'_accel_mae'])
            assert summary['jitter'][scope]['source']['full_accel_mae']==0
        for name,d in summary['dynamic'].items():
            rr=[r for r in dynamic if r['method']==name]
            close(sum(float(r['change_sum']) for r in rr)/sum(float(r['source_change_sum']) for r in rr),d['change_amplitude_ratio'])
        prior=load_json(P4/'summary.json') if case=='video4' else load_json(P3/case/'evaluation/summary.json')
        for name in ('raw','guide','Q9_R25','GS25'):
            for metric in ('coarse_mae','fine_mae','full_mae','psnr_db'):
                close(summary['scores']['full'][name][metric],prior['scores']['full'][name][metric])
        verified[case]=dict(all_frames_replayed_exact=True,extra_history_bounded_to_two_frames=True,
            no_output_feedback=True,current_frame_weight_at_least_half=True,common_support_equal=True,
            original_baseline_metrics_exact=True,metrics_resummed=True,source_guard_negative_probes_passed=True,
            matched_blur_verified=True,frames=n)
        print('VERIFIED',case,flush=True)
    save_json(run/'verification.json',dict(complete=True,cases=verified,current_config_sha256=CONFIG,
        input_hashes_unchanged=True,verifier_sha256=file_sha(__file__),subjective_review='pending'))
    snap=run/'verification_source';snap.mkdir();shutil.copy2(__file__,snap/Path(__file__).name)


def media(run):
    assert load_json(run/'verification.json')['complete']
    out=run/'previews';out.mkdir();records={}
    for case in ('video4','video1','video2','video3'):
        seq=load_sequences(run,case)
        groups=[(case+'_GS25_vs_SQ3',('GS25','SQ3'),None)]
        if case=='video4':
            groups.extend([
              ('video4_source_raw_GS25_SQ3',('source','raw','GS25','SQ3'),None),
              ('video4_raw_GS25_SR3_SQ3',('raw','GS25','SR3','SQ3'),None),
              ('video4_Q9_RG25_SR3_SQ3',('Q9_R25','RG25','SR3','SQ3'),None),
              ('video4_coral',('source','raw','GS25','SQ3'),np.s_[128:288,0:512]),
              ('video4_turtle',('source','raw','GS25','SQ3'),np.s_[150:288,50:410]),
              ('video4_blur_vs_SQ3',('BLUR_SQ3','SQ3'),None)])
        else:groups.append((case+'_source_Q9_SR3_SQ3',('source','Q9_R25','SR3','SQ3'),None))
        for filename,names,crop in groups:
            path=out/(filename+'.mp4');panels=[seq[n] for n in names]
            labels=[name+(' / candidate' if name in ('SR3','SQ3') else '') for name in names]
            preview(FFMPEG,path,panels,15,0,labels,crop=crop)
            records[path.name]=inspect(path,panels,crop)
            print('MEDIA',filename,flush=True)
        if case=='video4':
            font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
            names=('source','raw','Q9_R25','GS25','SR3','SQ3','BLUR_SQ3')
            for suffix,ids in [('contact',[0,26,52,78,103]),('event64',[62,63,64,65,66])]:
                canvas=Image.new('RGB',(320*len(names),205*len(ids)),(24,35,48));draw=ImageDraw.Draw(canvas)
                for row,i in enumerate(ids):
                    for col,name in enumerate(names):
                        draw.text((col*320+3,row*205+3),f'{name}  f{i}',font=font,fill='white')
                        canvas.paste(Image.fromarray(seq[name][i]).resize((320,180)),(col*320,row*205+25))
                canvas.save(out/('video4_'+suffix+'.png'))
            # Space-time strip, same source-registered coral row for every method.
            # Displays a temporal trace without claiming a video was watched.
            names=('source','raw','Q9_R25','GS25','SR3','SQ3','BLUR_SQ3')
            canvas=Image.new('RGB',(512*len(names),104*3+28),(24,35,48));draw=ImageDraw.Draw(canvas)
            for col,name in enumerate(names):
                draw.text((col*512+3,3),name+' y=220; time down',font=font,fill='white')
                strip=np.rint(np.asarray(seq[name][:,219:222]).mean(axis=1)).astype(np.uint8)
                canvas.paste(Image.fromarray(strip).resize((512,312),Image.Resampling.NEAREST),(col*512,28))
            canvas.save(out/'video4_time_strip.png')
    save_json(out/'verification.json',dict(complete=True,videos=records,user_original_speed_review='pending'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);checks(a.run);media(a.run)
