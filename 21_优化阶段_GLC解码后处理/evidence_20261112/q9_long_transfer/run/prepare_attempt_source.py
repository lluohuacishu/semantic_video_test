"""Frozen Q9_R25 full-length test with bounded windows and boundary control."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np

from runtime_speed_common import STAGE,REPO,WORKSPACE,FFMPEG,WEIGHT,file_sha,load_json,save_json,check_inputs
from mild_reliability_gate import mild_gate
from receiver_structure import RAMP
import q9_chunk_protocol as protocol

ORIGIN = WORKSPACE/'02_实验数据/优化_编解码执行复核/full_video_15fps_20261105_01/width512'
CODE = ('run_q9_long_transfer.py','q9_chunk_protocol.py','mild_reliability_gate.py','cached_t3.py',
        'receive_guide_texture.py','receiver_structure.py','raft_inference_runtime.py','runtime_speed_common.py')
MODES = ('continuous','reset')


def event(run,stage,**data):
    save_json(run/'progress.json',dict(stage=stage,**data))
    print(json.dumps(dict(stage=stage,**data),ensure_ascii=False),flush=True)


def prepare(run):
    if (run/'execution_plan.json').exists():
        raise FileExistsError('Plan already frozen')
    source_meta=ORIGIN/'prepare/result.json'
    raw_meta=ORIGIN/'receiver/result.json'
    raw_path=ORIGIN/'receiver/decoded_rgb.npy'
    base=ORIGIN/'sender/stream.glcsnd'
    source=load_json(source_meta);receiver=load_json(raw_meta)
    assert source['frames']==1239 and source['fps']==15
    assert receiver['reference_verified'] and receiver['source_media_reads_forbidden']
    assert file_sha(raw_path)==receiver['rgb_sha256'] and file_sha(base)==receiver['stream_sha256']
    inputs={str(p):file_sha(p) for p in [source_meta,raw_meta,raw_path,base,WEIGHT,STAGE/'当前采用配置.json']}
    total=0
    for chunk in source['chunks']:
        assert chunk['start']==total and file_sha(chunk['path'])==chunk['sha256']
        inputs[chunk['path']]=chunk['sha256'];total+=chunk['frames']
    assert total==1239
    source_npy=run/'source.npy'
    data=np.lib.format.open_memmap(source_npy,mode='w+',dtype=np.uint8,shape=(total,288,512,3))
    for chunk in source['chunks']:
        a=np.load(chunk['path'],mmap_mode='r');data[chunk['start']:chunk['start']+chunk['frames']]=a
    data.flush();del data
    inputs[str(source_npy)]=file_sha(source_npy)
    plan=dict(date='2026-11-12',frozen_before_inference=True,authorization='那继续推进吧',
        reference='Q9_R25; user positively reviewed video1/2/3',source=str(source_npy),
        source_metadata=str(source_meta),raw=str(raw_path),raw_receipt=str(raw_meta),base=str(base),
        frames=total,fps=15,duration_seconds=total/15,base_bytes=base.stat().st_size,
        base_note='Reuse actual continuous full-length stream and independently verified RGB; historical input encoder TRT FP16, not a new short-clip FP32 encoding.',
        guide=dict(internal_kbps=9,size=[256,144],preset='slower',two_pass=True,window_frames=150,
                   tail='Actual 39 frames, no duplicate padding',container='GLCGDE02 with timeline, full-base SHA and CRC',
                   index='Receiver manifest bytes included in auxiliary total'),
        continuous=dict(anchors='0, then 15+30*k until EOF; do not reset at chunk boundaries',
                        history_frames=30,maximum_frames_in_model_window=180,transition_frames=6,
                        receiver='Original R25 gate, 2px feather, two-stage rounding, raw reference only'),
        reset_control='Same received guides and base; restart six-anchor schedule in each 150-frame chunk. Isolates reference-reset effects.',
        fixed=dict(raft_iterations=12,precision='FP32',TF32=False,no_per_scene_thresholds=True,
                   scene_cut_detector=False,source_in_receiver=False),
        preplanned_views=dict(full='source/raw/continuous/reset',boundary_windows='All 8 boundaries, from 15 frames before to 30 after, chronological'),
        evaluation='All 1238 source-aligned pairs, full/four quadrants, PSNR, dynamic error; boundary and anchor subsets. Low support flagged, never silently zeroed.',
        scope='One complete historical marine source; old first10s plus later timeline, not independent heldout dataset.',
        timing='Postprocess only; guide two-pass/window lookahead and full-base SHA binding make this offline, not streaming realtime.',
        input_sha256=inputs,code_sha256={name:file_sha(STAGE/name) for name in CODE})
    save_json(run/'execution_plan.json',plan)
    snap=run/'source_snapshot';snap.mkdir()
    for name in CODE:shutil.copy2(STAGE/name,snap/name)
    # The first full window uses the same encoder parameters as the 10s Q9 test.
    event(run,'prepared',frames=total,windows=9,tail_frames=39,base_kbps=base.stat().st_size*8/(total/15)/1000)


def encode(run):
    plan=load_json(run/'execution_plan.json');check_inputs(plan['input_sha256'])
    source=np.load(plan['source'],mmap_mode='r');total=len(source);base_sha=file_sha(plan['base'])
    out=run/'guide';out.mkdir(exist_ok=False);records=[]
    for start in range(0,total,150):
        frames=min(150,total-start);folder=out/f'{start:06d}';folder.mkdir()
        low=np.stack([cv2.resize(x,(256,144),interpolation=cv2.INTER_AREA) for x in source[start:start+frames]])
        common=[str(FFMPEG),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','256x144','-r','15','-i','pipe:0',
                '-an','-c:v','libx265','-preset','slower','-b:v','9k','-pix_fmt','yuv420p','-threads','2']
        params='log-level=error:pools=2:frame-threads=1:keyint=150:min-keyint=150:scenecut=0:repeat-headers=1:info=0:stats=guide_pass.log'
        tick=time.perf_counter()
        for n in (1,2):
            result=subprocess.run([*common,'-x265-params',params+f':pass={n}','-f','hevc','pipe:1'],
                input=low.tobytes(),stdout=subprocess.PIPE if n==2 else subprocess.DEVNULL,
                stderr=subprocess.PIPE,check=True,cwd=folder)
            (folder/f'pass{n}.log').write_bytes(result.stderr)
        packet,header=protocol.pack(result.stdout,base_sha,start,frames,total)
        path=folder/'guide.glcg2';path.write_bytes(packet)
        records.append(dict(start=start,frames=frames,file=path.relative_to(out).as_posix(),
                            bytes=len(packet),sha256=file_sha(path),auxiliary_kbps=len(packet)*8/(frames/15)/1000,
                            encode_seconds=time.perf_counter()-tick))
        event(run,'guide_encoded',start=start,frames=frames,kbps=records[-1]['auxiliary_kbps'])
    manifest=dict(schema='Q9-window-manifest-v1',base_sha256=base_sha,total_frames=total,fps=15,
                  profile='Q9_R25',anchor_schedule='0 then 15+30*k',history_frames=30,packets=records)
    save_json(out/'manifest.json',manifest)
    packet_bytes=sum(x['bytes'] for x in records);index_bytes=(out/'manifest.json').stat().st_size
    assert (packet_bytes+index_bytes)*8/(total/15)<40000
    save_json(run/'sender_receipt.json',dict(complete=True,packet_bytes=packet_bytes,index_bytes=index_bytes,
        auxiliary_bytes=packet_bytes+index_bytes,base_bytes=plan['base_bytes'],
        auxiliary_kbps=(packet_bytes+index_bytes)*8/(total/15)/1000,
        total_kbps=(packet_bytes+index_bytes+plan['base_bytes'])*8/(total/15)/1000,
        manifest_sha256=file_sha(out/'manifest.json'),packets=records,source_sha256=file_sha(plan['source'])))


def receive(run):
    plan=load_json(run/'execution_plan.json');manifest_path=run/'guide/manifest.json'
    manifest=load_json(manifest_path);raw_path=Path(plan['raw']);base_path=Path(plan['base'])
    out=run/'receive'
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    import torch
    from raft_inference_runtime import FlowRunner,load_model,estimate
    from runtime_chain_worker import process_memory
    allowed={Path(p).resolve() for p in [base_path,raw_path,plan['raw_receipt'],WEIGHT,manifest_path]}
    allowed.update((run/'guide'/r['file']).resolve() for r in manifest['packets'])
    reads=set()
    def audit(name,args):
        if name!='open' or not args or not isinstance(args[0],(str,bytes)):return
        path=Path(os.fsdecode(args[0])).resolve()
        if path in allowed:reads.add(str(path));return
        if path.is_relative_to(out):return
        runtime=path.suffix.lower() in ('.py','.pyc','.pyd','.dll')
        if not runtime and (path.is_relative_to(WORKSPACE/'02_实验数据') or path.suffix.lower() in ('.npy','.npz','.mp4','.glcsnd','.glcg2','.glcguide')):
            raise RuntimeError('Forbidden long receiver input: '+str(path))
    sys.addaudithook(audit)
    blocked=[]
    for path in [Path(plan['source']),Path(plan['source_metadata']),run/'sender_receipt.json',
                 WORKSPACE/'02_实验数据/优化_GLC后处理/generalization_reliability_20261112_02/archive82_first10s/Q9/receive/R25/rgb.npy']:
        try:path.read_bytes()
        except RuntimeError:blocked.append(str(path))
        else:raise ValueError('Source guard failure')
    base_sha=file_sha(base_path);raw_receipt=load_json(plan['raw_receipt'])
    assert manifest['base_sha256']==base_sha==raw_receipt['stream_sha256']
    assert file_sha(raw_path)==raw_receipt['rgb_sha256'] and raw_receipt['reference_verified']
    raw=np.load(raw_path,mmap_mode='r');total=len(raw)
    assert raw.shape==(1239,288,512,3) and manifest['total_frames']==total
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    started=time.perf_counter();model=load_model();load_seconds=time.perf_counter()-started
    outputs={}
    for mode in MODES:
        folder=out/mode;folder.mkdir()
        outputs[mode]={name:np.lib.format.open_memmap(folder/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)
                       for name,dtype,shape in [('rgb',np.uint8,raw.shape),('alpha',np.float32,raw.shape[:3]),('shared',np.float32,raw.shape[:3])]}
    history=None;records=[];expected_start=0;total_pairs=0
    def anchor(t):return 0 if t<15 else 15+30*((t-15)//30)
    for block in manifest['packets']:
        start=block['start'];n=block['frames']
        assert start==expected_start and n==min(150,total-start);expected_start+=n
        tick=time.perf_counter();packet_path=run/'guide'/block['file']
        assert file_sha(packet_path)==block['sha256'] and packet_path.stat().st_size==block['bytes']
        header,new_images=protocol.decode(packet_path.read_bytes(),base_sha,start,total,FFMPEG)
        assert header['frames']==n
        begin=max(0,start-30)
        images=new_images if history is None else np.concatenate((history,new_images))
        assert images.shape==(start-begin+n,288,512,3) and len(images)<=180
        raw_window=raw[begin:start+n]
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            runner=FlowRunner(model,images,'feature_cache');runner.prepare()
            gate=mild_gate(images,raw_window,'R25')
            pairs=0;render_seconds=0.;flow_seconds=0.
            for t in range(start,start+n):
                i=t-begin;pair_cache={};render_cache={}
                def pair(key):
                    nonlocal pairs,flow_seconds
                    local=key-begin
                    assert 0<=local<len(images)
                    if key not in pair_cache:
                        if key==t:pair_cache[key]=np.zeros((2,288,512,2),np.float32)
                        else:
                            t0=time.perf_counter()
                            pred=runner.pair(i,local)
                            value=pred.permute(0,2,3,1).contiguous().cpu().numpy()
                            assert np.isfinite(value).all()
                            pair_cache[key]=value;pairs+=1;flow_seconds+=time.perf_counter()-t0
                    return pair_cache[key]
                def compose(key,old_key,age,transition):
                    nonlocal render_seconds
                    ident=(key,old_key,age,transition)
                    if ident in render_cache:return render_cache[ident]
                    back,forward=pair(key)
                    t0=time.perf_counter()
                    a,mx,my=gate(images,i,key-begin,back,forward)
                    new=cv2.remap(raw[key].astype(np.float32),mx,my,cv2.INTER_CUBIC)
                    rgb=np.rint(np.clip(raw[t]*(1-a[...,None])+new*a[...,None],0,255))
                    render_seconds+=time.perf_counter()-t0
                    shared=np.zeros((288,512),np.float32)
                    if transition:
                        back,forward=pair(old_key)
                        t0=time.perf_counter()
                        old_a,ox,oy=gate(images,i,old_key-begin,back,forward)
                        shared=np.minimum(a,old_a)
                        old=cv2.remap(raw[old_key].astype(np.float32),ox,oy,cv2.INTER_CUBIC)
                        weight=shared*RAMP[age]
                        rgb=np.rint(np.clip(rgb+weight[...,None]*(old-new),0,255))
                        render_seconds+=time.perf_counter()-t0
                    value=(rgb.astype(np.uint8),a,shared)
                    render_cache[ident]=value;return value
                key=anchor(t);old_key=0 if key==15 else max(0,key-30);age=t-key
                continuous=compose(key,old_key,age,key>0 and age<6)
                local_key=anchor(t-start);reset_key=start+local_key
                reset_old=start+(0 if local_key==15 else max(0,local_key-30))
                reset=compose(reset_key,reset_old,t-reset_key,local_key>0 and t-reset_key<6)
                for mode,value in zip(MODES,(continuous,reset)):
                    for name,array in zip(('rgb','alpha','shared'),value):outputs[mode][name][t]=array
            del pair_cache,render_cache,gate,runner
        peak=torch.cuda.max_memory_allocated()/1024**2
        history=np.array(images[-30:],copy=True)
        for values in outputs.values():
            for array in values.values():array.flush()
        # First window must exactly match the original 150-frame implementation.
        legacy_exact=None
        if start==0:
            from receiver_structure import render,EVENTS
            flows,_=estimate(model,images,'feature_cache')
            control=render(raw_window,images,*flows,gate_fn=mild_gate(images,raw_window,'R25'))
            assert np.array_equal(control['rgb'],outputs['continuous']['rgb'][:150])
            assert np.array_equal(control['alpha'],outputs['continuous']['alpha'][:150])
            assert np.array_equal(control['shared'],outputs['continuous']['shared'][list(EVENTS)])
            assert np.array_equal(outputs['continuous']['rgb'][:150],outputs['reset']['rgb'][:150])
            legacy_exact=True;del flows,control
        del images,new_images,raw_window
        gc.collect();torch.cuda.empty_cache()
        mem=process_memory();total_pairs+=pairs
        records.append(dict(start=start,frames=n,cached_frames=start-begin+n,pairs=pairs,
            processing_seconds=time.perf_counter()-tick,flow_seconds=flow_seconds,render_seconds=render_seconds,
            peak_gpu_allocated_mib=peak,process_memory_after_release=mem,first_window_legacy_exact=legacy_exact))
        save_json(out/'progress.json',dict(complete_frames=expected_start,total_frames=total,records=records))
        print('LONG WINDOW',start,n,'pairs',pairs,'seconds',round(records[-1]['processing_seconds'],2),flush=True)
    assert expected_start==total
    hashes={mode:{name:file_sha(out/mode/(name+'.npy')) for name in values} for mode,values in outputs.items()}
    save_json(out/'receipt.json',dict(complete=True,frames=total,base_sha256=base_sha,raw_sha256=file_sha(raw_path),
        manifest_sha256=file_sha(manifest_path),source_reads_blocked=True,blocked=blocked,reads=sorted(reads),
        allowed=sorted(map(str,allowed)),outputs_sha256=hashes,records=records,total_pairs=total_pairs,
        max_window_frames=max(r['cached_frames'] for r in records),model_load_seconds=load_seconds,
        postprocess_wall_seconds=time.perf_counter()-started,first_window_legacy_exact=True,
        continuous_anchor_rule='0,15,45,...',reference_chain='original base RGB only',
        no_source_pixels=True,no_source_masks=True,not_realtime_or_streaming=True,
        code_sha256=file_sha(__file__),protocol_sha256=file_sha(STAGE/'q9_chunk_protocol.py')))
    print('LONG RECEIVE COMPLETE',total,flush=True)


def evaluate(run):
    from probe_reef import bands
    from audit_deflicker_texture import source_match,normalized
    from temporal_filter import warp
    from compare_deflicker import save_csv
    plan=load_json(run/'execution_plan.json');receipt=load_json(run/'receive/receipt.json')
    check_inputs(plan['input_sha256'])
    seq=dict(source=np.load(plan['source'],mmap_mode='r'),raw=np.load(plan['raw'],mmap_mode='r'))
    for mode in MODES:
        for name,digest in receipt['outputs_sha256'][mode].items():
            assert file_sha(run/'receive'/mode/(name+'.npy'))==digest
        seq[mode]=np.load(run/'receive'/mode/'rgb.npy',mmap_mode='r')
        a=np.load(run/'receive'/mode/'alpha.npy',mmap_mode='r')
        shared=np.load(run/'receive'/mode/'shared.npy',mmap_mode='r')
        for start in range(0,plan['frames'],150):
            sl=slice(start,start+150);assert np.isfinite(a[sl]).all() and np.isfinite(shared[sl]).all()
            assert np.all((a[sl]>=0)&(a[sl]<=1)) and np.all((shared[sl]>=0)&(shared[sl]<=a[sl]+1e-7))
            zero=a[sl]==0;assert np.array_equal(seq[mode][sl][zero],seq['raw'][sl][zero])
    out=run/'evaluation';out.mkdir(exist_ok=False)
    scopes={'full':(0,0,512,288),'top_left':(0,0,256,144),'top_right':(256,0,512,144),
            'bottom_left':(0,144,256,288),'bottom_right':(256,144,512,288)}
    temporal=[];spatial=[];dynamic=[];low_support=[];previous=None;previous_rgb=None
    for i in range(plan['frames']):
        rgb={k:v[i] for k,v in seq.items()};current={k:bands(v) for k,v in rgb.items()}
        if i:
            mx,my,common=source_match(previous_rgb['source'],rgb['source'],previous['source'],current['source'],np.s_[:,:])
            aligned={k:warp(v,mx,my) for k,v in previous.items()}
            delta=current['source']-aligned['source']
            source_delta=rgb['source'].astype(np.float32)-previous_rgb['source']
            magnitude=np.max(np.abs(source_delta),axis=2)
            mask=magnitude>=max(12,float(np.quantile(magnitude,.9)));count=int(mask.sum())
            for name,value in rgb.items():
                change=value.astype(np.float32)-previous_rgb[name]
                dynamic.append(dict(frame=i,method=name,pixels=count,delta_error_sum=float(np.abs(change-source_delta)[mask].sum()),
                    change_sum=float(np.abs(change)[mask].sum()),source_change_sum=float(np.abs(source_delta)[mask].sum())))
        for scope,(x0,y0,x1,y1) in scopes.items():
            crop=np.s_[y0:y1,x0:x1]
            for name,value in rgb.items():
                error=value[crop].astype(np.float32)-rgb['source'][crop]
                spatial.append(dict(frame=i,scope=scope,method=name,pixels=(x1-x0)*(y1-y0),rgb_sse=float(np.square(error).sum())))
            if i:
                mask=common[crop];count=int(mask.sum())
                if count<64:
                    low_support.append(dict(frame=i,scope=scope,pixels=count))
                    continue
                for name in rgb:
                    a=aligned[name][crop][mask];b=current[name][crop][mask];error=np.abs(b-a-delta[crop][mask])
                    temporal.append(dict(frame=i,scope=scope,method=name,pixels=count,coarse_sum=float(error[:,0].sum()),
                        fine_sum=float(error[:,2].sum()),full_sum=float(error[:,3].sum()),
                        fine_ncc=float((normalized(a[:,2:3])*normalized(b[:,2:3])).mean())))
        previous,previous_rgb=current,rgb
        if i%150==0:print('LONG EVAL',i,flush=True)
    save_csv(out/'temporal.csv',temporal);save_csv(out/'spatial.csv',spatial);save_csv(out/'dynamic.csv',dynamic)
    def aggregate(frame_test=lambda i:True,scope='full'):
        result={}
        for method in seq:
            rows=[r for r in temporal if r['scope']==scope and r['method']==method and frame_test(r['frame'])]
            sr=[r for r in spatial if r['scope']==scope and r['method']==method and frame_test(r['frame'])]
            dr=[r for r in dynamic if r['method']==method and frame_test(r['frame'])]
            n=sum(x['pixels'] for x in rows);sn=sum(x['pixels'] for x in sr);dn=sum(x['pixels'] for x in dr)
            mse=sum(x['rgb_sse'] for x in sr)/(3*sn) if sn else None
            amp=sum(x['source_change_sum'] for x in dr)
            result[method]=dict(valid_pairs=len(rows),valid_pixels=n,
                coarse_mae=sum(x['coarse_sum'] for x in rows)/n if n else None,
                fine_mae=sum(x['fine_sum'] for x in rows)/n if n else None,
                full_mae=sum(x['full_sum'] for x in rows)/n if n else None,
                fine_ncc=sum(x['fine_ncc']*x['pixels'] for x in rows)/n if n else None,
                psnr_db=float(10*np.log10(255**2/mse)) if mse else None,
                p95=float(np.percentile([x['coarse_sum']/x['pixels'] for x in rows],95)) if rows else None,
                dynamic_error=sum(x['delta_error_sum'] for x in dr)/(3*dn) if dn else None,
                change_amplitude_ratio=sum(x['change_sum'] for x in dr)/amp if amp else None)
        return result
    boundaries=list(range(150,plan['frames'],150))
    windows={str(b):aggregate(lambda i,b=b:b-5<=i<=b+20) for b in boundaries}
    segments={str(b):aggregate(lambda i,b=b:b<=i<min(b+150,plan['frames'])) for b in range(0,plan['frames'],150)}
    worst={}
    for name in seq:
        rows=[dict(frame=r['frame'],mae=r['coarse_sum']/r['pixels']) for r in temporal if r['scope']=='full' and r['method']==name]
        worst[name]=sorted(rows,key=lambda r:r['mae'],reverse=True)[:12]
    full=aggregate();save_json(run/'summary.json',dict(complete=True,frames=plan['frames'],duration_seconds=plan['duration_seconds'],
        scores={scope:aggregate(scope=scope) for scope in scopes},segments=segments,boundary_windows=windows,
        boundary_exact=aggregate(lambda i:i in boundaries),boundary_next20=aggregate(lambda i:any(b<=i<=b+20 for b in boundaries)),
        nonboundary=aggregate(lambda i:not any(b<=i<=b+20 for b in boundaries)),
        worst_frames=worst,low_support=low_support,rates=load_json(run/'sender_receipt.json'),
        scope=plan['scope'],subjective_review='pending',automatic_adoption=False))
    event(run,'evaluated',full_mae={n:v['coarse_mae'] for n,v in full.items()},low_support_pairs=len(low_support))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','encode','receive','evaluate'))
    p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    try:globals()[a.action](a.run.resolve())
    except BaseException as error:
        save_json(a.run/f'failure_{a.action}.json',dict(error=str(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
