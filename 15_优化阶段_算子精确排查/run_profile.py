"""Freeze current role codec, run clean timing and native CUPTI/NVTX campaigns."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(REPO/'14_优化阶段_部署精简'))
sys.path.insert(0,str(REPO/'13_优化阶段_发送端免RGB重建'))
from sender_common import file_sha,load_json,load_settings,save_json

HERE=Path(__file__).resolve().parent


def compare(reference,result,role):
    if len(reference['records'])!=len(result['records']):raise ValueError('Frame count changed')
    keys=['index','kind','latent_sha256','payload_sha256']+(['rgb_sha256'] if role=='receiver' else [])
    for expected,actual in zip(reference['records'],result['records']):
        for key in keys:
            if expected[key]!=actual[key]:raise ValueError(f'Frame {actual["index"]} differs: {key}')


def gpu_snapshot():
    command=['nvidia-smi','--query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,clocks.current.sm,power.draw,temperature.gpu','--format=csv,noheader']
    p=subprocess.run(command,capture_output=True,text=True,timeout=10)
    return dict(returncode=p.returncode,value=p.stdout.strip(),error=p.stderr.strip())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    runtime=load_settings()['runtime'];python=runtime['python']
    if (output/'plan.json').exists():plan=load_json(output/'plan.json')
    else:
        output.mkdir(parents=True,exist_ok=False)
        previous=load_json(REPO/'14_优化阶段_部署精简/evidence_20261029/plan.json')
        paths=[*HERE.glob('*.py'),REPO/'14_优化阶段_部署精简/role_codec.py',REPO/'14_优化阶段_部署精简/deploy_codec.py',
               REPO/'08_新阶段C_编解码加速与网页展示/c1_io.py',REPO/'08_新阶段C_编解码加速与网页展示/c2_native.py',
               REPO/'04_外部依赖快照/semantic-video-training/scripts/glc_video_codec.py']
        plan=dict(schema='glc-native-cupti-profile-v1',date='2026-10-30',clip=previous['clips'][0],assets=previous['assets'],
            assets_manifest_sha256=previous['assets_manifest_sha256'],samples=[0,1,2,3,9,33,64,65],warmup_frames=16,
            baseline_repeats=3,nvtx_repeats=2,code_sha256={p.relative_to(REPO).as_posix():file_sha(p) for p in paths},
            measurement='FP32, native rANS, real CUDA 12.4 CUPTI Activity API 22; no model edits or precision changes',
            gpu_at_start=gpu_snapshot())
        save_json(output/'plan.json',plan);save_json(output/'plan_identity.json',dict(sha256=file_sha(output/'plan.json')))
    if file_sha(output/'plan.json')!=load_json(output/'plan_identity.json')['sha256']:raise ValueError('Plan changed')
    for name,digest in plan['code_sha256'].items():
        if file_sha(REPO/name)!=digest:raise ValueError('Frozen code changed: '+name)
    teacher=load_json(plan['clip']['teacher_result']);results={}
    for role in ['sender','receiver']:
        results[role]={}
        for mode,count in [('baseline',3),('activity',1),('nvtx',2)]:
            values=[]
            for repeat in range(1,count+1):
                folder=output/role/f'{mode}_{repeat}';path=folder/'result.json'
                if path.exists():value=load_json(path)
                else:
                    folder.parent.mkdir(parents=True,exist_ok=True)
                    before=gpu_snapshot();start=time.perf_counter()
                    command=[str(python),'-X','utf8',str(HERE/'profile_worker.py'),'--plan',str(output/'plan.json'),
                             '--role',role,'--mode',mode,'--output',str(folder)]
                    with folder.with_suffix('.log').open('w',encoding='utf-8') as log:
                        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                    value=load_json(path);value.update(process_wall_seconds=time.perf_counter()-start,gpu_before=before,gpu_after=gpu_snapshot())
                    save_json(path,value)
                compare(teacher,value,role)
                if role=='sender' and value['stream_sha256']!=plan['clip']['stream_sha256']:raise ValueError('Full sender file changed')
                values.append(value)
                print(f'{role} {mode} {repeat}: core {value["core_seconds"]:.3f}s, capture exact',flush=True)
            results[role][mode]=values
            save_json(output/'progress_summary.json',dict(results=results,complete=False))
        folder=output/'pipeline'/role
        if not (folder/'result.json').exists():
            folder.parent.mkdir(parents=True,exist_ok=True)
            command=[str(python),'-X','utf8',str(HERE/'pipeline_timing.py'),'--plan',str(output/'plan.json'),'--role',role,'--output',str(folder)]
            start=time.perf_counter()
            with folder.with_suffix('.log').open('w',encoding='utf-8') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            save_json(folder/'process_wall.json',dict(seconds=time.perf_counter()-start))
        pipeline=load_json(folder/'result.json')
        if role=='sender' and pipeline['stream_sha256']!=plan['clip']['stream_sha256']:raise ValueError('Real-file sender differs')
        results[role]['pipeline']=pipeline
    save_json(output/'summary.json',dict(complete=True,results=results,plan_sha256=file_sha(output/'plan.json'),
        exact_frame_hashes=True,cupti_record_drops=0,production_changed=False,Orin_tested=False,training_steps=0))
    print('Completed actual timing, CUPTI-only controls, native ATen/NVTX profiling and file pipeline',flush=True)


if __name__=='__main__':main()
