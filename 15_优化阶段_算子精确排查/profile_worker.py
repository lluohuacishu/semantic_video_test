"""One source-blind receiver or sender; baseline and real-CUPTI modes separate."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from contextlib import nullcontext
import os
from pathlib import Path
import sys
import time

if '--mode' in sys.argv and sys.argv[sys.argv.index('--mode')+1]=='nvtx':
    # NVIDIA's documented embedded-CUPTI injection, scoped to this child process.
    os.environ['NVTX_INJECTION64_PATH']=str(Path(sys.prefix)/'Lib/site-packages/torch/lib/cupti64_2024.1.0.dll')

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(REPO/'14_优化阶段_部署精简'))
from role_codec import RoleCodec, ref
from sender_common import file_sha, load_json, save_json
from sender_container import ReferenceChecksums, SenderWriter, read_sender_stream
from trial_worker import payload_fingerprint
import numpy as np
import torch


def save_profile(folder, profile):
    folder.mkdir(parents=True,exist_ok=False)
    save_json(folder/'profile.json',profile)
    rows=[]
    for row in profile['operators']:
        rows.append(dict(operator=row['name'],module=row['module'],phase=row['phase'],calls=row['calls'],
            cpu_wall_ms=row['cpu_wall_ns']/1e6,thread_cpu_ms=row['thread_cpu_ns']/1e6 if row['thread_cpu_ns'] is not None else None,
            gpu_kernel_ms=row['kernel_sum_ns']/1e6,kernel_count=row['kernel_count'],
            actual_memcpy_ms=row['memcpy_sum_ns']/1e6,memcpy_bytes=row['memcpy_bytes'],
            runtime_sync_cpu_ms=row['runtime_sync_wall_ns']/1e6,inputs=str(row['inputs'])))
    with (folder/'operators.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]) if rows else ['operator','module','gpu_kernel_ms'])
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--role',choices=['sender','receiver'],required=True)
    parser.add_argument('--mode',choices=['baseline','trace','activity','nvtx'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    plan=load_json(args.plan);clip=plan['clip']
    if args.role=='receiver':
        forbidden={str(Path(clip['source']).resolve()).casefold()}
        def audit(event,values):
            if event=='open' and values and isinstance(values[0],(str,bytes)):
                name=str(values[0]);normalized=name.replace('\\','/').lower()
                if str(Path(name).resolve()).casefold() in forbidden or normalized.endswith(('.npy','.png','.jpg','.jpeg','.mkv','.mp4','.avi')):
                    raise RuntimeError('Independent profiler receiver tried to read source media')
        sys.addaudithook(audit)
    start=time.perf_counter()
    codec=RoleCodec(plan['assets'],args.role)
    torch.cuda.synchronize()
    init_seconds=time.perf_counter()-start
    header,packets,integrity=read_sender_stream(clip['stream'])
    if header['arithmetic']!=codec.profile or header['table_sha256']!=codec.entropy.table_sha256:
        raise ValueError('Model arithmetic/entropy differs')
    if args.role=='sender':
        if file_sha(clip['source'])!=clip['source_sha256']:
            raise ValueError('Source array changed')
        source=np.load(clip['source'],mmap_mode='r')
    def input_tensor(index):
        value=torch.from_numpy(np.array(source[index])).to(codec.device).permute(2,0,1)[None].float().div(255)
        return ref.F.pad(value,(0,(-header['width'])%64,0,(-header['height'])%64),mode='replicate').mul(2).sub(1)
    def frame(index,x):
        if args.role=='sender':return codec.frame(header,index,x=x,render_rgb=False)
        return codec.frame(header,index,packet=packets[index])
    warmup_frames=min(plan.get('warmup_frames',16),header['frames'])
    for index in range(warmup_frames):frame(index,input_tensor(index) if args.role=='sender' else None)
    torch.cuda.synchronize()
    codec.dpb=dict(ref_latent=None,ref_y=None)
    tracer=observer=None
    if args.mode!='baseline':
        from cupti_trace import CuptiTrace
        from operator_observer import Observer,DispatchObserver
        tracer=CuptiTrace(kinds=CuptiTrace.KINDS+(12,13) if args.mode=='nvtx' else None);tracer.start()
        # Prime CUPTI/cudnn tracing after ordinary model warmup, then discard it.
        for index in range(warmup_frames):frame(index,input_tensor(index) if args.role=='sender' else None)
        torch.cuda.synchronize();tracer.take()
        codec.dpb=dict(ref_latent=None,ref_y=None)
        observer=Observer(tracer,native_nvtx=args.mode=='nvtx')
        if args.mode in ('nvtx','trace'):observer.install(codec,ref)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    checks=ReferenceChecksums();rows=[];profiles=[]
    writer=SenderWriter(args.output/'stream.glcsnd',header) if args.role=='sender' else None
    loop_started=time.perf_counter()
    try:
        for index in range(header['frames']):
            sampled=args.mode!='baseline' and index in plan['samples']
            if tracer:tracer.take()
            if sampled:
                observer.reset()
                context=torch.autograd.profiler.emit_nvtx(record_shapes=True) if args.mode=='nvtx' else nullcontext()
                with context:
                    tracer.take();observer.active=True;begin=tracer.timestamp()
                    wall_start=time.perf_counter_ns()
                    with DispatchObserver(observer) if args.mode=='trace' else nullcontext():
                        with observer.scope('input_tensor'):
                            tick=time.perf_counter_ns()
                            x=input_tensor(index) if args.role=='sender' else None
                            torch.cuda.synchronize()
                            input_ns=time.perf_counter_ns()-tick
                        with observer.scope('codec.frame'):
                            tick=time.perf_counter_ns();rgb,packet=frame(index,x)
                            with observer.scope('tail_sync'):torch.cuda.synchronize()
                            core_ns=time.perf_counter_ns()-tick
                    elapsed_ns=time.perf_counter_ns()-wall_start
                    end=tracer.timestamp();observer.active=False
                    records=tracer.take()
                if args.mode=='nvtx':
                    from nvtx_binding import native_binding
                    profile=native_binding(observer,records,begin,end)
                else:profile=observer.summarize(records,begin,end)
                profile.update(index=index,role=args.role,frame_kind='I' if packet['kind']==0 else 'P',
                    observed_wall_ns=elapsed_ns,input_ns=input_ns,core_ns=core_ns,
                    format='Real CUPTI kernel/memcpy/API activities; '+args.mode+' attribution')
                save_profile(args.output/'frames'/f'{index:03d}',profile)
                if profile['unattributed_kernel_count'] and args.mode!='activity':
                    raise ValueError('Some GPU kernels could not be assigned to ATen operators')
                profiles.append({key:profile[key] for key in ['index','frame_kind','kernel_count','kernel_sum_ns',
                    'gpu_activity_union_ns','window_ns','core_ns','runtime_sync_wall_ns','memcpy_bytes','memcpy_sum_ns',
                    'unattributed_kernel_count','capture_dropped_records']})
            else:
                tick=time.perf_counter_ns()
                x=input_tensor(index) if args.role=='sender' else None
                torch.cuda.synchronize();input_ns=time.perf_counter_ns()-tick
                tick=time.perf_counter_ns();rgb,packet=frame(index,x);torch.cuda.synchronize()
                core_ns=time.perf_counter_ns()-tick
            if writer:writer.add(packet)
            checks.add(index,packet)
            row=dict(index=index,kind=packet['kind'],seconds=core_ns/1e9,input_seconds=input_ns/1e9,
                latent_sha256=packet['latent_sha256'],payload_sha256=payload_fingerprint(packet),
                payload_bytes=len(packet['z'])+sum(map(len,packet['y'])),sampled=sampled)
            if 'rgb_sha256' in packet:row['rgb_sha256']=packet['rgb_sha256']
            rows.append(row)
            if sampled or (index+1)%16==0:
                save_json(args.output/'progress.json',dict(role=args.role,mode=args.mode,frame=index+1,total=header['frames']))
                print(f'{args.role} {args.mode} {index+1}/{header["frames"]} {core_ns/1e6:.2f} ms',flush=True)
        checks.verify(integrity['expected_reference_checksums'])
        if writer:writer.finish()
    finally:
        if writer:writer.close()
        if tracer:tracer.close()
    result=dict(role=args.role,mode=args.mode,header=header,records=rows,profiles=profiles,
        init_seconds=init_seconds,core_seconds=sum(row['seconds'] for row in rows),
        input_seconds=sum(row['input_seconds'] for row in rows),loop_seconds_including_exports=time.perf_counter()-loop_started,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
        source_media_forbidden=args.role=='receiver',reference_verified=True,
        stream_sha256=file_sha(args.output/'stream.glcsnd') if writer else file_sha(clip['stream']),
        cupti_api_version=tracer.version if tracer else None,capture_dropped_records=tracer.dropped_records if tracer else None)
    save_json(args.output/'result.json',result)
    print(f'completed {args.role} {args.mode}: core={result["core_seconds"]:.3f}s',flush=True)


if __name__=='__main__':main()
