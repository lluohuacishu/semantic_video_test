"""Repeated complete local file chains with independent cold/warm services."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np

from runtime_speed_common import *
import structure_guide_stream as guide


class Service:
    def __init__(self,role,logs):
        self.role=role;self.events=[]
        self.log=(logs/(role+'.stderr.log')).open('x',encoding='utf-8')
        tick=time.perf_counter()
        self.process=subprocess.Popen([sys.executable,'-B','-u','-X','utf8',str(STAGE/'runtime_chain_worker.py'),'--role',role],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf-8',cwd=REPO)
        self.ready=self.read('ready');self.startup_seconds=time.perf_counter()-tick

    def read(self,target):
        while True:
            line=self.process.stdout.readline()
            if not line:raise RuntimeError(self.role+' worker exited; see stderr')
            try:value=json.loads(line)
            except json.JSONDecodeError:continue
            if value.get('event')=='error':raise RuntimeError(self.role+': '+value['message'])
            if value.get('event')=='first_frame':
                self.events.append(dict(kind=value['kind'],received_at=time.perf_counter()))
            if value.get('event')==target:return value

    def request(self,payload):
        self.events=[];tick=time.perf_counter()
        self.process.stdin.write(json.dumps(payload,ensure_ascii=False)+'\n');self.process.stdin.flush()
        response=self.read('result')['result']
        response['request_wall_seconds']=time.perf_counter()-tick
        response['frame_events']=list(self.events)
        return response

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"stop":true}\n');self.process.stdin.flush()
            self.process.wait(timeout=30)
        self.log.close()


def extract(out):
    out.mkdir(exist_ok=False);tick=time.perf_counter()
    command=[str(FFMPEG),'-hide_banner','-loglevel','error','-nostdin','-ss','0.0','-i',str(SOURCE_VIDEO),
        '-map','0:v:0','-an','-vf','fps=15,scale=512:288:flags=area,format=rgb24',
        '-frames:v','150','-f','rawvideo','pipe:1']
    pixels=subprocess.check_output(command)
    array=np.frombuffer(pixels,np.uint8).reshape(150,288,512,3)
    path=out/'source.npy';np.save(path,array,allow_pickle=False)
    digest=file_sha(path)
    if digest!='97e44a828004587c5d362a45017b8d6cc41ddd8258bc0c7098e9d84bae15a751':
        raise ValueError('Raw source extraction changed')
    return array,path,dict(seconds=time.perf_counter()-tick,source_npy_sha256=digest,command=command)


def encode_guide(source,base,out):
    out.mkdir(exist_ok=False);tick=time.perf_counter()
    low=np.stack([cv2.resize(v,(256,144),interpolation=cv2.INTER_AREA) for v in source])
    common=[str(FFMPEG),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','256x144','-r','15','-i','pipe:0',
        '-an','-c:v','libx265','-preset','slower','-b:v','6k','-pix_fmt','yuv420p','-threads','2']
    params='log-level=error:pools=2:frame-threads=1:keyint=150:min-keyint=150:scenecut=0:repeat-headers=1:info=0:stats=guide_pass.log'
    passes=[]
    for number in (1,2):
        begin=time.perf_counter()
        p=subprocess.run([*common,'-x265-params',params+f':pass={number}','-f','hevc','pipe:1'],
            input=low.tobytes(),stdout=subprocess.PIPE if number==2 else subprocess.DEVNULL,
            stderr=subprocess.PIPE,cwd=out,check=True)
        (out/f'pass{number}.log').write_bytes(p.stderr);passes.append(time.perf_counter()-begin)
    packet,_=guide.pack(p.stdout,file_sha(base),(256,144));path=out/'guide.glcguide';path.write_bytes(packet)
    digest=file_sha(path)
    if digest!='2883c3b13c98bd384b0a896e3f4563d5ae46b8b5a702b1e75998e60435dc987f':
        raise ValueError('Re-encoded S8I bytes differ from frozen accepted packet')
    return dict(seconds=time.perf_counter()-tick,pass_seconds=passes,bytes=len(packet),sha256=digest)


def chain(folder,index,mode,temperature,raft_mode,services=None):
    out=folder/f'{temperature}_{index:02}_{mode}';out.mkdir()
    owned=[];records={};first={};started=time.perf_counter()
    before=gpu_snapshot()
    def stage_service(role):
        if services is not None:return services[role]
        log=out/(role+'_logs');log.mkdir();obj=Service(role,log);owned.append(obj)
        return obj
    def progress(name):
        print('CHAIN',temperature,index,mode,name,round(time.perf_counter()-started,2),flush=True)
        save_json(out/'progress.json',dict(stage=name,elapsed=time.perf_counter()-started,records=records))
    try:
        source,path,records['source']=extract(out/'input');progress('source')
        sender=stage_service('sender')
        records['sender']=sender.request(dict(output=str(out/'send'),source=str(path),source_sha256=records['source']['source_npy_sha256']))
        records['sender']['startup_seconds']=sender.startup_seconds if services is None else 0.
        if records['sender']['stream_sha256']!='cf9f519f9dee6918c1dd49bce4c2572c1020425e1bb3789abcf7b06854d60fd5':
            raise ValueError('New GLC stream differs from frozen base')
        if services is None:sender.close();owned.remove(sender)
        progress('base_encoded')
        records['guide_sender']=encode_guide(source,out/'send/base.glcsnd',out/'guide');del source
        progress('guide_encoded')
        receiver=stage_service('receiver')
        records['receiver']=receiver.request(dict(output=str(out/'receive'),base=str(out/'send/base.glcsnd')))
        records['receiver']['startup_seconds']=receiver.startup_seconds if services is None else 0.
        if records['receiver']['decoded_rgb_sha256']!='66892444a3dac9ab1addbbb6b6239b5990d8cb0981db899295dbeab285fea281':
            raise ValueError('Independent GLC RGB changed')
        if services is None:receiver.close();owned.remove(receiver)
        progress('base_decoded')
        post=stage_service('post')
        records['post']=post.request(dict(output=str(out/'post'),base=str(out/'send/base.glcsnd'),
            raw=str(out/'receive/decoded_rgb.npy'),base_receipt=str(out/'receive/receipt.json'),
            guide=str(out/'guide/guide.glcguide'),raft_mode='original' if mode=='original' else raft_mode,
            cached_t3=mode!='original'))
        records['post']['startup_seconds']=post.startup_seconds if services is None else 0.
        elapsed=time.perf_counter()-started
        for role in ('sender','receiver','post'):
            for event in records[role]['frame_events']:
                first[event['kind']]=event['received_at']-started
        cfg=load_json(STAGE/'当前采用配置.json')
        exact={name:digest==cfg['artifacts'][name]['sha256'] for name,digest in records['post']['outputs_sha256'].items()}
        if not all(exact.values()):raise ValueError('Complete chain final RGB/alpha/shared mismatch')
        result=dict(complete=True,index=index,mode=mode,temperature=temperature,
            external_task_wall_seconds=elapsed,complete_task_fps=150/elapsed,
            first_frame_seconds=first,validated_viewing_file_ready_seconds=elapsed,
            stages=records,outputs_bit_exact=exact,source_free_receivers=True,
            base_bytes=31963,guide_bytes=8865,total_kbps=32.6624,
            gpu_before=before,gpu_after=gpu_snapshot(),
            timing_scope='Source video sampling -> real GLC/S8I encode -> independent base decode -> postprocess -> RGB/alpha/shared and viewing MP4. Includes child startup for cold. Excludes initial benchmark-driver imports, final external regression checks and last worker shutdown. No network.',
            first_frame_note='First computed packet/RGB notifications; final deliverable is released only after complete integrity verification. Offline postprocess waits for all flow pairs.')
        save_json(out/'result.json',result);progress('complete_exact')
        return result
    except BaseException as error:
        save_json(out/'failure.json',dict(error=str(error),traceback=traceback.format_exc(),elapsed=time.perf_counter()-started,records=records))
        raise
    finally:
        for obj in owned:obj.close()


def run(run_folder):
    out=run_folder/'complete_chain';out.mkdir(exist_ok=False)
    frozen=load_json(run_folder/'execution_plan.json')['fixed_input_sha256'];check_inputs(frozen)
    raft=load_json(run_folder/'raft/summary.json')['selected_exact_candidate']
    order=['original','optimized','optimized','original','original','optimized']
    names=('run_complete_runtime_benchmark.py','runtime_chain_worker.py','runtime_speed_common.py',
        'raft_inference_runtime.py','cached_t3.py','structure_guide_stream.py','receiver_structure.py','receive_guide_texture.py')
    code={name:file_sha(STAGE/name) for name in names}
    save_json(out/'plan.json',dict(frozen_before_trials=True,raft_candidate=raft,order_per_temperature=order,
        cold='New sender/base-receiver/post processes for every trial; startup included, no artificial warmup',
        warm='Same three independent resident services; one complete warmup excluded, then balanced six trials',
        sender='Eager FP32, same frozen source/GLCSND header, fill-on as original source encoding',
        receiver='Eager FP32 base decoder, fill-off; no hidden TensorRT change',
        source_sha256=file_sha(SOURCE_VIDEO),code_sha256=code,
        outputs='All newly encoded stream/guide bytes and final RGB/alpha/shared must match accepted artifacts',
        local_flow_storage='RAM in both modes; no local .npy flow diagnostic dumps in full-chain timing',
        no_generalization_this_round=True))
    snapshot=out/'source';snapshot.mkdir()
    for name in names:shutil.copy2(STAGE/name,snapshot/name)
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    rows=[]
    for index,mode in enumerate(order):
        rows.append(chain(out,index,mode,'cold',raft));save_json(out/'trials.json',rows)
    services={};logs=out/'resident_services';logs.mkdir()
    try:
        for role in ('sender','receiver','post'):
            target=logs/role;target.mkdir();services[role]=Service(role,target)
        warmup=chain(out,0,'original','warmup',raft,services)
        save_json(out/'warmup.json',warmup)
        for index,mode in enumerate(order):
            rows.append(chain(out,index,mode,'warm',raft,services));save_json(out/'trials.json',rows)
    finally:
        for service in services.values():service.close()
    summary={}
    for temp in ('cold','warm'):
        summary[temp]={mode:stats([r['external_task_wall_seconds'] for r in rows if r['temperature']==temp and r['mode']==mode])
                       for mode in ('original','optimized')}
        summary[temp]['median_time_reduction_percent']=(1-summary[temp]['optimized']['median']/summary[temp]['original']['median'])*100
    check_inputs(frozen)
    save_json(out/'summary.json',dict(complete=True,timing=summary,selected_raft=raft,
        trial_count=len(rows),all_transmission_and_display_artifacts_exact=True,
        input_hashes_unchanged=True,scope='Complete local offline video-file chain; not network or realtime',
        memory='Per-worker peak torch allocation and Windows process-lifetime peak working set in each trial; not summed nonconcurrent peaks',
        raw_trials='trials.json',warmup_excluded=True))
    print('COMPLETE CHAIN DONE',summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();run(a.run.resolve())
