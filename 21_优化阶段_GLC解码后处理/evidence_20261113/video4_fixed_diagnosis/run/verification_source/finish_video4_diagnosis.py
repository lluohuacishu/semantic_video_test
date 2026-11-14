"""Verify the complete short-clip chain and export original-speed video4 comparisons."""
import argparse
import csv
from pathlib import Path
import shutil
import subprocess
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from runtime_speed_common import STAGE,REPO,FFMPEG,file_sha,load_json,save_json,check_inputs
from receiver_structure import render,EVENTS
from mild_reliability_gate import mild_gate
from evaluate_video4_diagnosis import sequences
from finish_q9_ghost_exit import inspect
from run_p0 import preview
import q9_chunk_protocol as protocol

def rows(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def verify(run):
    p=load_json(run/'execution_plan.json');r=load_json(run/'case_registration.json')
    s=load_json(run/'summary.json');rec=load_json(run/'receive/receipt.json')
    assert s['complete'] and p['frames']==r['frames']==s['frames']==104
    check_inputs(p['input_sha256'])
    for name,h in p['code_sha256'].items():assert file_sha(STAGE/name)==h==file_sha(run/'source_snapshot'/name)
    for name,h in p['external_code_sha256'].items():assert file_sha(REPO/name)==h==file_sha(run/'source_snapshot/repository'/name)
    ep=load_json(run/'evaluation_plan.json')
    for name,h in ep['code_sha256'].items():assert file_sha(STAGE/name)==h==file_sha(run/'evaluation_source'/name)
    assert file_sha(STAGE/'当前采用配置.json')==p['current_config_sha256']
    assert file_sha(run/'case_registration.json')==p['registration_sha256']
    assert file_sha(run/'source.npy')==r['source_npy_sha256']
    seq=sequences(run);n=104
    assert all(a.shape==(n,288,512,3) and a.dtype==np.uint8 for a in seq.values())
    extracted=subprocess.check_output([str(FFMPEG),'-v','error','-i',r['source_video'],'-map','0:v:0','-an',
        '-vf','fps=15,scale=512:288:flags=area','-pix_fmt','rgb24','-f','rawvideo','pipe:1'])
    assert np.array_equal(np.frombuffer(extracted,np.uint8).reshape(n,288,512,3),seq['source'])
    assert int(np.any(seq['source'][1:]!=seq['source'][:-1],axis=(1,2,3)).sum())==103
    base=run/'video4/send/base.glcsnd';br=load_json(run/'video4/base_receive/receipt.json')
    sr=load_json(run/'video4/send/receipt.json')
    assert br['complete'] and br['reference_verified'] and br['source_media_reads_forbidden'] and br['source_reads_blocked']
    assert br['frames']==sr['header']['frames']==n
    assert sr['input_sha256']==r['source_npy_sha256']
    assert str(run/'source.npy') in br['blocked']
    assert file_sha(base)==rec['base_sha256']==br['stream_sha256']==sr['stream_sha256']
    assert file_sha(run/'video4/base_receive/decoded_rgb.npy')==br['decoded_rgb_sha256']==rec['raw_sha256']
    packet=(run/'guide/guide.glcg2').read_bytes()
    assert file_sha(run/'guide/guide.glcg2')==rec['guide_sha256']
    h,images=protocol.decode(packet,file_sha(base),0,n,FFMPEG)
    assert np.array_equal(images,seq['guide']) and h==rec['guide_header']
    bad=bytearray(packet);bad[-5]^=1;rejected=[]
    for name,args in [('CRC',(bytes(bad),file_sha(base),0,n)),('truncated',(packet[:-1],file_sha(base),0,n)),
                      ('base',(packet,'0'*64,0,n)),('start',(packet,file_sha(base),150,n)),
                      ('total',(packet,file_sha(base),0,n+1))]:
        try:protocol.unpack(*args)
        except (ValueError,TypeError):rejected.append(name)
        else:raise AssertionError('Invalid guide accepted')
    assert rec['source_reads_blocked'] and len(rec['blocked'])==5
    assert set(rec['reads'])<=set(rec['allowed']) and not set(rec['reads'])&set(rec['blocked'])
    for name,hashvalue in rec['cache_sha256'].items():
        assert file_sha(run/'receive'/(name+'.npy'))==hashvalue
    flow=[np.load(run/'receive'/(name+'.npy'),mmap_mode='r') for name in ('to_anchor','from_anchor','transition_flow')]
    assert flow[0].shape==flow[1].shape==(104,288,512,2)
    assert flow[2].shape==(18,2,288,512,2) and all(np.isfinite(a).all() for a in flow)
    expected=render(seq['raw'],images,*flow,gate_fn=mild_gate(images,seq['raw'],'R25'))
    for name,a in expected.items():
        saved=np.load(run/'receive/Q9_R25'/(name+'.npy'),mmap_mode='r')
        assert np.array_equal(a[:len(saved)],saved)
        assert file_sha(run/'receive/Q9_R25'/(name+'.npy'))==rec['Q9_sha256'][name]
    alpha=expected['alpha']
    assert np.all((alpha>=0)&(alpha<=1))
    assert np.array_equal(seq['Q9_R25'][alpha==0],seq['raw'][alpha==0])
    for j,t in enumerate([i for i in EVENTS if i<n]):
        assert np.all(expected['shared'][j]<=alpha[t])
    clips=[];changes=[]
    for i in range(n):
        q=seq['Q9_R25'][i].astype(np.float32);g=images[i].astype(np.float32)
        unbounded=q+.25*cv2.GaussianBlur(g-q,(0,0),1.)
        gs=np.rint(np.clip(unbounded,0,255)).astype(np.uint8)
        assert np.array_equal(gs,seq['GS25'][i])
        clips.append(float(((unbounded<0)|(unbounded>255)).mean()))
        changes.append(float(np.abs(gs.astype(np.float32)-q).mean()))
    assert float(np.mean(clips))==rec['clipped_channel_fraction']
    assert float(np.mean(changes))==rec['mean_abs_change']
    assert file_sha(run/'receive/GS25/rgb.npy')==rec['GS25_sha256']
    rate=load_json(run/'guide/send.json');seconds=n/15
    assert rate['base_bytes']==base.stat().st_size and rate['guide_bytes']==len(packet) and rate['index_bytes']==0
    assert rate['total_kbps']==(base.stat().st_size+len(packet))*8/seconds/1000
    assert rate==s['rates']
    for name in ('base150_verification.json','prefix_regression.json','evaluation150_verification.json'):
        assert load_json(run/'regression'/name)['complete']
    tr=rows(run/'evaluation/temporal.csv');sp=rows(run/'evaluation/spatial.csv');dr=rows(run/'evaluation/dynamic.csv')
    for scope,methods in s['scores'].items():
        supports={}
        for name,v in methods.items():
            t=[x for x in tr if x['scope']==scope and x['method']==name]
            a=[x for x in sp if x['scope']==scope and x['method']==name]
            total=sum(int(x['pixels']) for x in t);pixels=sum(int(x['pixels']) for x in a)
            supports[name]=[(int(x['frame']),int(x['pixels'])) for x in t]
            assert len(a)==n and [int(x['frame']) for x in a]==list(range(n))
            assert len(t)==v['pairs'] and n-1-len(t)==v['excluded_pairs']
            assert len(t)+sum(x['scope']==scope for x in s['low_support'])==n-1
            for b in ('coarse','fine','full'):
                value=sum(float(x[b+'_sum']) for x in t)/total if total else None
                assert value==v[b+'_mae']
            if total:
                assert sum(float(x['fine_ncc'])*int(x['pixels']) for x in t)/total==v['fine_ncc']
            else:assert v['fine_ncc'] is None
            assert sum(float(x['fine_sum']) for x in a)/pixels==v['fine_amplitude']
            assert sum(float(x['rgb_abs_sum']) for x in a)/(3*pixels)==v['rgb_mae']
            for b in ('coarse','fine'):
                assert sum(float(x[b+'_spatial_sum']) for x in a)/pixels==v[b+'_spatial_mae']
            mse=sum(float(x['rgb_sse']) for x in a)/(3*pixels)
            assert (float(10*np.log10(255**2/mse)) if mse else None)==v['psnr_db']
            x0,y0,x1,y1=ep['scopes'][scope]
            direct=sum(float(np.abs(seq[name][i,y0:y1,x0:x1].astype(np.float32)-seq['source'][i,y0:y1,x0:x1]).sum()) for i in range(n))/(n*(x1-x0)*(y1-y0)*3)
            assert direct==v['rgb_mae']
        assert all(x==supports['source'] for x in supports.values())
    for name,v in s['dynamic'].items():
        rr=[x for x in dr if x['method']==name]
        assert len(rr)==n-1
        assert sum(float(x['delta_error_sum']) for x in rr)/(3*sum(int(x['pixels']) for x in rr))==v['RGB_delta_error']
        assert sum(float(x['change_sum']) for x in rr)/sum(float(x['source_change_sum']) for x in rr)==v['change_amplitude_ratio']
    snap=run/'verification_source';snap.mkdir();shutil.copy2(__file__,snap/Path(__file__).name)
    save_json(run/'verification.json',dict(complete=True,actual_frames=104,source_reextract_exact=True,
        no_repeated_padding=True,base_reference_verified=True,guide_redecode_exact=True,
        malformed_packets_rejected=rejected,receiver_read_boundary_verified=True,Q9_legacy_replay_exact=True,
        GS25_independent_formula_exact=True,all_scoped_metrics_reaggregated=True,
        scoped_rgb_mae_directly_recomputed=True,full_rate_accounting=True,
        old150_base_and_metric_regressions_exact=True,old_three_prefix_render_regressions_exact=True,
        old_video1_prefix_flow_regression_exact=True,
        current_config_sha256=file_sha(STAGE/'当前采用配置.json'),verifier_sha256=file_sha(__file__)))
    print('VIDEO4 CHAIN AND METRICS VERIFIED',flush=True)

def media(run):
    assert load_json(run/'verification.json')['complete']
    p=load_json(run/'execution_plan.json');seq=sequences(run)
    out=run/'previews';out.mkdir();checks={}
    groups=[('video4_source_raw_Q9_GS25',('source','raw','Q9_R25','GS25'),None),
            ('video4_Q9_vs_GS25',('Q9_R25','GS25'),None),
            ('video4_source_vs_guide',('source','guide'),None),
            ('video4_coral_crop',('source','raw','Q9_R25','GS25'),np.s_[128:288,0:512]),
            ('video4_turtle_crop',('source','raw','Q9_R25','GS25'),np.s_[150:288,50:410])]
    for suffix,names,crop in groups:
        path=out/(suffix+'.mp4');panels=[seq[n] for n in names]
        labels=[n+(' / NOT adopted' if n=='GS25' else '') for n in names]
        preview(FFMPEG,path,panels,15,0,labels,crop=crop)
        checks[path.name]=inspect(path,panels,crop)
        print('VIDEO4 MEDIA',suffix,flush=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    ids=p['review_frames'];names=tuple(seq)
    canvas=Image.new('RGB',(320*5,205*len(ids)),(24,35,48));draw=ImageDraw.Draw(canvas)
    for row,i in enumerate(ids):
        for col,name in enumerate(names):
            draw.text((col*320+3,row*205+3),f'{name} f{i} {i/15:.2f}s',font=font,fill='white')
            canvas.paste(Image.fromarray(seq[name][i]).resize((320,180)),(col*320,row*205+25))
    canvas.save(out/'video4_whole_scene_contact.png')
    for label,box in [('coral',(0,128,512,288)),('turtle',(50,150,410,288))]:
        x0,y0,x1,y1=box;w=x1-x0;h=y1-y0
        ids=[26,52,78];names=('source','raw','guide','Q9_R25','GS25')
        canvas=Image.new('RGB',(w*5,(h+25)*len(ids)),(24,35,48));draw=ImageDraw.Draw(canvas)
        for row,i in enumerate(ids):
            for col,name in enumerate(names):
                draw.text((col*w+3,row*(h+25)+3),f'{name} f{i}',font=font,fill='white')
                canvas.paste(Image.fromarray(seq[name][i,y0:y1,x0:x1]),(col*w,row*(h+25)+25))
        canvas.save(out/('video4_'+label+'_contact.png'))
    save_json(out/'verification.json',dict(complete=True,videos=checks,video_count=5,
        actual_frames=104,fps=15,playback_seconds=104/15,original_video_stream_seconds=6.9,
        four_panel_order=['source','raw','Q9_R25','GS25'],source_guide_pair=['source','guide'],
        user_original_speed_review='pending'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);verify(a.run);media(a.run)

