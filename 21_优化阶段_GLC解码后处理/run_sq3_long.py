"""Fixed SQ3 on the verified 82.6-second sequence; byte-matched H.265 control."""
import argparse
from fractions import Fraction
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import cv2
import numpy as np
from runtime_speed_common import STAGE,REPO,DATA,WORKSPACE,WEIGHT,FFMPEG,file_sha,load_json,save_json,check_inputs
from short_history_filter import SETTINGS,blend_frame,quantize
from received_structure_fusion import fuse_received_structure
from cache_hevc_baseline import X265_PARAMS
import q9_chunk_protocol as protocol

OLD=DATA/'q9_long_transfer_20261112_03'
SHORT=DATA/'q9_video4_jitter_20261113_03'
CONFIG='b5eeb844d0adcdfe97648351a0654a2575a98942e1627c98896ba302555f7b04'
CODE=('run_sq3_long.py','short_history_filter.py','received_structure_fusion.py',
      'raft_inference_runtime.py','runtime_speed_common.py','runtime_chain_worker.py',
      'q9_chunk_protocol.py','cache_hevc_baseline.py')


def prepare(run):
    assert not (run/'execution_plan.json').exists()
    assert file_sha(STAGE/'当前采用配置.json')==CONFIG
    old=load_json(OLD/'execution_plan.json');rec=load_json(OLD/'receive/receipt.json')
    send=load_json(OLD/'sender_receipt.json');manifest=load_json(OLD/'guide/manifest.json')
    assert load_json(OLD/'verification.json')['complete'] and rec['source_reads_blocked']
    assert rec['frames']==1239 and rec['first_window_legacy_exact']
    q9=OLD/'receive/continuous/rgb.npy'
    assert file_sha(q9)==rec['outputs_sha256']['continuous']['rgb']
    assert file_sha(old['raw'])==rec['raw_sha256']
    assert file_sha(old['base'])==rec['base_sha256']==manifest['base_sha256']
    assert file_sha(OLD/'guide/manifest.json')==rec['manifest_sha256']==send['manifest_sha256']
    shortplan=load_json(SHORT/'execution_plan.json')
    for name in ('short_history_filter.py','raft_inference_runtime.py','received_structure_fusion.py'):
        assert file_sha(STAGE/name)==shortplan['code_sha256'][name]
    inputs={str(Path(p)):file_sha(p) for p in [old['source'],old['raw'],old['base'],old['raw_receipt'],
        OLD/'execution_plan.json',OLD/'receive/receipt.json',OLD/'guide/manifest.json',OLD/'verification.json',
        OLD/'sender_receipt.json',q9,WEIGHT,SHORT/'execution_plan.json',SHORT/'verification.json']}
    for row in manifest['packets']:
        p=OLD/'guide'/row['file'];assert file_sha(p)==row['sha256'];inputs[str(p)]=file_sha(p)
    assert sum(r['bytes'] for r in manifest['packets'])==send['packet_bytes']
    assert (OLD/'guide/manifest.json').stat().st_size==send['index_bytes']
    regress={}
    specs=load_json(SHORT/'case_registration.json')['cases']
    for case in ('video4','video1'):
        paths=dict(guide=specs[case]['guide'],gs25=specs[case]['gs25'],source=specs[case]['source'],
            flow=str(SHORT/case/'receive/flows.npy'),sq3=str(SHORT/case/'receive/SQ3/rgb.npy'))
        regress[case]=paths
        for p in paths.values():inputs[p]=file_sha(p)
    budget=send['base_bytes']+send['auxiliary_bytes']
    plan=dict(date='2026-11-13',frozen_before_inference=True,request=load_json(run/'user_request.json'),
        source=old['source'],source_metadata=old['source_metadata'],raw=old['raw'],base=old['base'],raw_receipt=old['raw_receipt'],
        q9=str(q9),q9_receipt=str(OLD/'receive/receipt.json'),manifest=str(OLD/'guide/manifest.json'),guide_root=str(OLD/'guide'),
        frames=1239,fps=15,seconds=82.6,base_bytes=send['base_bytes'],guide_packet_bytes=send['packet_bytes'],
        guide_index_bytes=send['index_bytes'],auxiliary_bytes=send['auxiliary_bytes'],total_bytes=budget,total_kbps=budget*8/82.6/1000,
        input_note='Reuse verified continuous historical full-length GLC/Q9; base encoder TRT FP16. No new base/guide encoding.',
        sq3=dict(settings=SETTINGS,input='GS25 from continuous Q9 plus unchanged 25% current guide correction',
            window_frames=150,history_overlap=2,reset_at_processing_boundaries=False,scene_cut_detector=False,
            no_new_lookahead=True,per_scene_tuning=False,extra_transmission_bytes=0),
        regression=dict(cases=regress,window_frames=32,require_rgb_exact=True,require_flow_exact=True),
        h265=dict(preset='medium',x265_params=X265_PARAMS,color='bt709 limited yuv420p',
            budget_bytes=budget,billing='Complete MP4 file including mux/headers, no audio/padding',
            search='CRF [28,51] at 1/16 step, byte-only bisection; target <=0.5% shortfall, require <=2%',
            comparison='Equal actual total communication bytes to GLC+guide+manifest; raw GLC alone has fewer bits'),
        deliverable=dict(layout=['source','H.265','raw_GLC','SQ3'],geometry=[1024,688],fps=15,frames=1239,
            labels='Chinese titles and actual rates; all panels same source timestamps, 512x288, video only'),
        evaluation='Output replay and input hashes, all frame coverage, original-speed four-way, first/middle/last plus every 150-frame boundary. Basic per-frame spatial/dynamic diagnostics, no extensive parameter search.',
        no_automatic_adoption=True,current_config_sha256=CONFIG,input_sha256=inputs,code_sha256={n:file_sha(STAGE/n) for n in CODE})
    save_json(run/'execution_plan.json',plan);snap=run/'source_snapshot';snap.mkdir()
    for name in CODE:shutil.copy2(STAGE/name,snap/name)
    statepath=REPO/'05_资料与索引/会话收束_20261110/接续状态.json';shutil.copy2(statepath,run/'state_before_run.json')
    state=load_json(statepath);state.update(status='SQ3_long_fourway_running',active_experiment_plan=str(run/'execution_plan.json'),
        current_session_work='Run frozen SQ3 on full 82.6-second historical source; make source/H265/rawGLC/SQ3 four-way at matched total bitrate.')
    state['SQ3_short_subjective_feedback']=plan['request']
    state['experiment_execution_gate']=dict(status='running',currently_running=True,new_run_started=True,automatic_next_run=False,date='2026-11-13')
    save_json(statepath,state)
    print('FROZEN',plan['frames'],'frames; budget',budget,'bytes /',plan['total_kbps'],'kbps',flush=True)


def regress(run):
    import torch
    from raft_inference_runtime import load_model,FlowRunner
    from runtime_chain_worker import ReadGuard
    plan=load_json(run/'execution_plan.json');out=run/'regression';out.mkdir()
    allowed=[Path(v) for s in plan['regression']['cases'].values() for k,v in s.items() if k!='source']+[WEIGHT]
    guard=ReadGuard('sq3-window-regression');guard.authorize(out,allowed)
    blocked=[]
    for s in plan['regression']['cases'].values():
        try:Path(s['source']).read_bytes()
        except RuntimeError:blocked.append(s['source'])
        else:raise AssertionError('Regression source guard failed')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);model=load_model();results={}
    for case,s in plan['regression']['cases'].items():
        guide=np.load(s['guide'],mmap_mode='r');gs=np.load(s['gs25'],mmap_mode='r')
        expected=np.load(s['sq3'],mmap_mode='r');oldflow=np.load(s['flow'],mmap_mode='r');n=len(gs);pairs=0
        for start in range(0,n,32):
            stop=min(start+32,n);begin=max(0,start-2);images=np.array(guide[begin:stop])
            flow=np.zeros((len(images),2,2,288,512,2),np.float32)
            with torch.inference_mode():
                runner=FlowRunner(model,images,'feature_cache');runner.prepare()
                for t in range(start,stop):
                    i=t-begin
                    for lag in (1,2):
                        if t<lag:continue
                        pred=runner.pair(i,i-lag).permute(0,2,3,1).contiguous().cpu().numpy()
                        assert np.array_equal(pred,oldflow[t,lag-1]),(case,t,lag,float(np.abs(pred-oldflow[t,lag-1]).max()))
                        flow[i,lag-1]=pred;pairs+=1
                    rgb,_=blend_frame(gs[begin:stop],images,flow,i)
                    assert np.array_equal(quantize(rgb),expected[t]),(case,t)
                del runner
            del images,flow;gc.collect();torch.cuda.empty_cache()
        results[case]=dict(frames=n,pairs=pairs,all_flow_exact=True,all_RGB_exact=True,window_frames=32,overlap=2)
        print('REGRESSION',case,results[case],flush=True)
    save_json(out/'verification.json',dict(complete=True,cases=results,source_reads_blocked=True,blocked=blocked,reads=sorted(guard.reads)))


def receive(run):
    import torch
    from raft_inference_runtime import load_model,FlowRunner
    from runtime_chain_worker import ReadGuard,process_memory
    plan=load_json(run/'execution_plan.json');assert load_json(run/'regression/verification.json')['complete']
    for n,h in plan['code_sha256'].items():assert file_sha(STAGE/n)==h
    manifest=load_json(plan['manifest']);prior=load_json(plan['q9_receipt']);out=run/'receive';out.mkdir()
    allowed=[Path(plan[k]) for k in ('raw','base','raw_receipt','q9','q9_receipt','manifest')]+[WEIGHT]
    allowed += [Path(plan['guide_root'])/r['file'] for r in manifest['packets']]
    guard=ReadGuard('sq3-long');guard.authorize(out,allowed);blocked=[]
    for p in (Path(plan['source']),Path(plan['source_metadata']),OLD/'sender_receipt.json',run/'user_request.json'):
        try:p.read_bytes()
        except RuntimeError:blocked.append(str(p))
        else:raise AssertionError('Long source guard failed')
    for p in allowed:assert file_sha(p)==plan['input_sha256'][str(p)]
    assert prior['complete'] and prior['source_reads_blocked'] and prior['first_window_legacy_exact']
    assert file_sha(plan['q9'])==prior['outputs_sha256']['continuous']['rgb']
    raw_receipt=load_json(plan['raw_receipt']);base_sha=file_sha(plan['base'])
    assert raw_receipt['reference_verified'] and raw_receipt['source_media_reads_forbidden']
    assert base_sha==raw_receipt['stream_sha256']==manifest['base_sha256']
    q9=np.load(plan['q9'],mmap_mode='r');n=plan['frames'];shape=(n,288,512,3)
    assert q9.shape==shape
    rgb={name:np.lib.format.open_memmap(out/(name+'.npy'),mode='w+',dtype=np.uint8,shape=shape) for name in ('guide','GS25','SQ3')}
    torch.set_num_threads(1);torch.set_num_interop_threads(1);model=load_model();tick=time.perf_counter()
    records=[];cursor=0
    for packet in manifest['packets']:
        start=packet['start'];count=packet['frames'];stop=start+count;begin=max(0,start-2)
        assert start==cursor;cursor=stop
        path=Path(plan['guide_root'])/packet['file'];assert file_sha(path)==packet['sha256']
        header,images=protocol.decode(path.read_bytes(),base_sha,start,n,FFMPEG)
        assert len(images)==count
        rgb['guide'][start:stop]=images
        for t in range(start,stop):rgb['GS25'][t]=fuse_received_structure(q9[t],images[t-start],'GS25')[0]
        for a in rgb.values():a.flush()
        guide=np.array(rgb['guide'][begin:stop]);gs=np.array(rgb['GS25'][begin:stop])
        flowpath=out/f'flows_{start:06d}.npy'
        flow=np.lib.format.open_memmap(flowpath,mode='w+',dtype=np.float32,shape=(len(guide),2,2,288,512,2));flow[:]=0
        pairs=0;weights=[];began=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            runner=FlowRunner(model,guide,'feature_cache');runner.prepare()
            for t in range(start,stop):
                i=t-begin
                for lag in (1,2):
                    if t<lag:continue
                    pred=runner.pair(i,i-lag).permute(0,2,3,1).contiguous().cpu().numpy()
                    assert np.isfinite(pred).all();flow[i,lag-1]=pred;pairs+=1
                frame,info=blend_frame(gs,guide,flow,i);rgb['SQ3'][t]=quantize(frame);weights.append(info)
            del runner
        flow.flush();flow._mmap.close();del flow
        for a in rgb.values():a.flush()
        torch.cuda.synchronize();seconds=time.perf_counter()-began;peak=torch.cuda.max_memory_allocated()/1024**2
        del images,guide,gs;gc.collect();torch.cuda.empty_cache()
        record=dict(start=start,frames=count,history_begin=begin,pairs=pairs,processing_seconds=seconds,
            cached_frames=stop-begin,flow_file=flowpath.name,flow_sha256=file_sha(flowpath),
            peak_gpu_mib=peak,process_memory=process_memory(),frame_stats=weights)
        records.append(record)
        save_json(out/'progress.json',dict(complete_frames=cursor,total_frames=n,windows=[{k:v for k,v in r.items() if k!='frame_stats'} for r in records]))
        print('SQ3 LONG',start,stop,'/',n,'seconds',round(seconds,2),flush=True)
    assert cursor==n and sum(r['pairs'] for r in records)==2*n-3
    for a in rgb.values():a.flush();a._mmap.close()
    save_json(out/'receipt.json',dict(complete=True,frames=n,source_reads_blocked=True,blocked=blocked,
        reads=sorted(guard.reads),allowed=sorted(map(str,allowed)),input_Q9_sha256=file_sha(plan['q9']),
        output_sha256={name:file_sha(out/(name+'.npy')) for name in rgb},records=records,
        seconds=time.perf_counter()-tick,extra_transmission_bytes=0,
        no_boundary_history_reset=True,no_new_scene_cut_rule=True,settings=SETTINGS,
        timing_note='Additional SQ3 stage including guide decode, flow, render, hashes and IO; excludes reused base/Q9 and is not end-to-end.') )
    print('SQ3 LONG COMPLETE',n,flush=True)


def hevc(run):
    plan=load_json(run/'execution_plan.json');out=run/'h265';out.mkdir();budget=plan['total_bytes']
    assert file_sha(plan['source'])==plan['input_sha256'][plan['source']]
    source=np.load(plan['source'],mmap_mode='r');n=len(source);common=[str(FFMPEG),'-hide_banner','-loglevel','error','-nostdin'];cache={}
    def encode(tick):
        if tick in cache:return cache[tick][0]
        path=out/f'crf_{tick}.mp4'
        command=common+['-f','rawvideo','-pixel_format','rgb24','-video_size','512x288','-framerate','15','-i','pipe:0','-an',
            '-vf','scale=out_color_matrix=bt709:out_range=tv,format=yuv420p','-color_range','tv','-colorspace','bt709',
            '-color_primaries','bt709','-color_trc','bt709','-c:v','libx265','-preset','medium','-crf',str(tick/16),
            '-x265-params',X265_PARAMS,'-tag:v','hvc1','-movflags','+faststart','-n',str(path)]
        with (out/f'crf_{tick}.log').open('wb') as log:
            p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
            try:
                for start in range(0,n,32):p.stdin.write(source[start:start+32].tobytes())
                p.stdin.close();assert p.wait(timeout=60)==0
            finally:
                if p.poll() is None:p.kill();p.wait()
        cache[tick]=(path.stat().st_size,path)
        save_json(out/'search_progress.json',{str(k/16):v[0] for k,v in sorted(cache.items())})
        print('H265 CRF',tick/16,'bytes',cache[tick][0],'budget',budget,flush=True)
        return cache[tick][0]
    lo,hi=28*16,51*16
    assert encode(lo)>budget and encode(hi)<=budget
    while hi-lo>1:
        mid=(lo+hi)//2
        if encode(mid)<=budget:hi=mid
        else:lo=mid
        best=max(v[0] for v in cache.values() if v[0]<=budget)
        if (budget-best)/budget<=.005:break
    best=max(v[0] for v in cache.values() if v[0]<=budget)
    if (budget-best)/budget>.005:
        for tick in range(max(28*16,hi-4),min(51*16,hi+4)+1):
            encode(tick);best=max(v[0] for v in cache.values() if v[0]<=budget)
            if (budget-best)/budget<=.005:break
    tick,(size,path)=max(((k,v) for k,v in cache.items() if v[0]<=budget),key=lambda z:z[1][0])
    assert (budget-size)/budget<=.02
    chosen=out/'h265_matched.mp4';shutil.copy2(path,chosen)
    rgb=np.lib.format.open_memmap(out/'rgb.npy',mode='w+',dtype=np.uint8,shape=source.shape)
    with (out/'decode.log').open('wb') as log:
        p=subprocess.Popen(common+['-i',str(chosen),'-map','0:v:0','-vf','scale=in_color_matrix=bt709:in_range=tv,format=rgb24','-f','rawvideo','pipe:1'],stdout=subprocess.PIPE,stderr=log)
        try:
            for i in range(n):
                frame=p.stdout.read(288*512*3);assert len(frame)==288*512*3
                rgb[i]=np.frombuffer(frame,np.uint8).reshape(288,512,3)
            assert not p.stdout.read(1) and p.wait(timeout=30)==0
        finally:
            if p.poll() is None:p.kill();p.wait()
    rgb.flush();rgb._mmap.close()
    probe=json.loads(subprocess.check_output([str(FFMPEG.with_name('ffprobe.exe')),'-v','error','-select_streams','v:0','-count_frames',
        '-show_entries','stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration,pix_fmt','-of','json',str(chosen)]))['streams'][0]
    assert probe['codec_name']=='hevc' and int(probe['nb_read_frames'])==n and Fraction(probe['avg_frame_rate'])==15
    assert (probe['width'],probe['height'])==(512,288) and abs(float(probe['duration'])-plan['seconds'])<.001
    save_json(out/'receipt.json',dict(complete=True,bytes=size,budget_bytes=budget,kbps=size*8/plan['seconds']/1000,
        relative_shortfall=(budget-size)/budget,crf=tick/16,preset='medium',x265_params=X265_PARAMS,
        complete_file_billing=True,zero_padding=True,source_sha256=file_sha(plan['source']),stream=probe,
        stream_sha256=file_sha(chosen),rgb_sha256=file_sha(out/'rgb.npy'),candidates={str(k/16):v[0] for k,v in sorted(cache.items())}))
    print('H265 COMPLETE',size,'bytes',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','regress','receive','hevc']);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    dict(prepare=prepare,regress=regress,receive=receive,hevc=hevc)[a.action](a.run)
