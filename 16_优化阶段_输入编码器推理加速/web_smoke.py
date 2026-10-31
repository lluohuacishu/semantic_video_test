"""Explicit real-HTTP acceptance of UI-started modes, GPU cancel and ordinary upload."""
from __future__ import annotations
import argparse
import json
import time
from urllib.request import Request,build_opener,ProxyHandler
from experiment_common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--four-only',action='store_true',help='Check only the four-mode job, without starting extra GPU jobs')
    args=p.parse_args();args.output.mkdir(exist_ok=True);base='http://127.0.0.1:8773/inference/'
    opener=build_opener(ProxyHandler({}))
    def call(path,value=None,method=None,headers=None):
        data=json.dumps(value).encode() if value is not None else None
        with opener.open(Request(base+path,data=data,method=method or ('POST' if data is not None else 'GET'),
            headers=headers or ({'Content-Type':'application/json'} if data is not None else {})),timeout=30) as r:return json.load(r)
    def wait(identifier,predicate,timeout=120):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            j=call('api/jobs/'+identifier)
            if predicate(j):return j
            if j['status'] in ['error','cancelled','interrupted']:raise RuntimeError(str(j))
            time.sleep(.2)
        raise TimeoutError(identifier)
    j=wait(args.job,lambda x:x['status']=='done');save_json(args.output/'four_modes.json',j)
    if len(j['results'])!=4 or not all(r['reference_verified'] for r in j['results']):raise ValueError('Four-mode UI test incomplete')
    trial=WORKSPACE/'02_实验数据/优化_输入编码器推理/inference_20261030_01'
    jobs=load_settings()['data_dir'].parent/'C5_loopback/sender/inference_lab'
    for r in j['results']:
        actual=load_json(jobs/j['id']/(r['mode']+'_sender/result.json'))
        historical=load_json(trial/'full/long108_q0_640'/r['mode']/'sender1/result.json')
        if any(a['latent_sha256']!=b['latent_sha256'] for a,b in zip(actual['records'],historical['records'])):
            raise ValueError('Live encoder differs from prior correct reference chain')
        request=Request('http://127.0.0.1:8773'+r['video'],headers={'Range':'bytes=0-31'})
        with opener.open(request) as response:
            if response.status!=206 or len(response.read())!=32:raise ValueError('Live video range failed')
    print('Four live modes match prior reference chains; four video ranges passed',flush=True)
    if args.four_only:return
    params=dict(sample='long108_q0_640',frames=66,size=640,q=0,modes=['eager'])
    cancel=call('api/jobs',params);call('api/jobs/'+cancel['id']+'/start',{})
    active=wait(cancel['id'],lambda x:x.get('progress',{}).get('phase')=='编码中')
    call('api/jobs/'+cancel['id']+'/cancel',{})
    deadline=time.monotonic()+75
    while time.monotonic()<deadline:
        cancelled=call('api/jobs/'+cancel['id'])
        if cancelled['status']=='cancelled':break
        time.sleep(.2)
    else:raise TimeoutError('Cancellation did not finish')
    if cancelled['results']:raise ValueError('Cancelled partial mode published as complete')
    process=load_json(jobs/cancel['id']/'eager_sender_process/process.json')
    if process['returncode']!=2 or process['state']!='cancelled':raise ValueError('GPU worker not confirmed exited')
    save_json(args.output/'cancelled.json',dict(job=cancelled,process=process,requested_while=active.get('progress')))
    print('Running GPU cancellation confirmed; no partial result published',flush=True)
    video=trial/'playback/compact_trt_fp16_stream/video.mp4';payload=video.read_bytes()
    upload=call('api/jobs',dict(filename='web-upload-test.mp4',bytes=len(payload),frames=4,size=640,q=12,modes=['trt_fp16_stream']))
    with opener.open(Request(base+'api/jobs/'+upload['id']+'/upload',data=payload,method='PUT',headers={'Content-Type':'application/octet-stream'}),timeout=30) as response:
        if json.load(response)['status']!='staged':raise ValueError('Upload not staged')
    call('api/jobs/'+upload['id']+'/start',{})
    done=wait(upload['id'],lambda x:x['status']=='done')
    if done['source']['frames']!=4 or not done['results'][0]['reference_verified']:raise ValueError('Upload/recovery decode failed')
    save_json(args.output/'uploaded_video.json',done)
    save_json(args.output/'web_acceptance.json',dict(passed=True,four_mode_job=j['id'],four_mode_reference_matches=True,
        live_video_ranges=4,cancelled_running_gpu_job=cancel['id'],gpu_process_exit_confirmed=True,
        uploaded_job=done['id'],upload_and_gpu_recovery_verified=True,
        scope='HTTP/GPU acceptance; CPU checks run separately',training_steps=0,old_benchmark_matrix_rerun=False))
    print('Ordinary video upload and GPU recovery passed',flush=True)


if __name__=='__main__':main()
