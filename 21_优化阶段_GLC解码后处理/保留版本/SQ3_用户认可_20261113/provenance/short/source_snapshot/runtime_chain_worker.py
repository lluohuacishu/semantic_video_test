"""Independent resident sender/base-receiver/postprocessor for timing only.

JSON-line requests allow a real sequential warm file chain with resident models.
Only the sender can read source pixels. Receiver guards start before model load.
"""
import argparse
import ctypes
from ctypes import wintypes
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np
import torch

from runtime_speed_common import (REPO, DATA, BASE, PACKET, WEIGHT, ASSETS, SOURCE_VIDEO,
    UPSTREAM, FFMPEG, file_sha, load_json, save_json, render)


def emit(event,**values):
    print(json.dumps(dict(event=event,**values),ensure_ascii=False,allow_nan=False),flush=True)


def process_memory():
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
            (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize',
            'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    counters=Counters();counters.cb=ctypes.sizeof(counters)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi=ctypes.WinDLL('psapi',use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(working_set_mib=counters.WorkingSetSize/1024**2,
        process_lifetime_peak_working_set_mib=counters.PeakWorkingSetSize/1024**2,
        scope='This Python worker, not FFmpeg children or a simultaneous whole-system peak')


class ReadGuard:
    def __init__(self,role):
        self.allowed={WEIGHT.resolve()}
        self.outputs=[];self.reads=set()
        def audit(event,args):
            if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
            p=Path(os.fsdecode(args[0])).resolve()
            if p in self.allowed:self.reads.add(str(p));return
            if any(p.is_relative_to(out) for out in self.outputs):return
            runtime=p.suffix.lower() in {'.py','.pyc','.pyd','.dll'}
            asset=p.is_relative_to(ASSETS) and p.suffix.lower() in {'.json','.pth','.pt'}
            if not runtime and not asset and (p.is_relative_to(DATA) or p.suffix.lower() in
                {'.npy','.npz','.glcsnd','.glcguide','.glccsp2','.glccsp3','.mp4','.hevc'}):
                raise RuntimeError('Forbidden '+role+' source/input read: '+str(p))
        sys.addaudithook(audit)
        self.blocked=[]
        for p in (BASE/'source.npy',BASE/'base_cut.json',BASE/'flow/to_anchor.npy',
            BASE/'flow/from_anchor.npy',PACKET.parent/'send.json',PACKET.parent/'sender_expected_guide.npy',SOURCE_VIDEO):
            try:p.read_bytes()
            except RuntimeError:self.blocked.append(str(p))
            else:raise ValueError('Receiver guard failed')

    def authorize(self,out,inputs):
        self.outputs.append(out.resolve())
        self.allowed.update(Path(p).resolve() for p in inputs)


class Worker:
    def __init__(self,role):
        self.role=role
        self.guard=None if role=='sender' else ReadGuard(role)
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
        started=time.perf_counter()
        if role=='post':
            from raft_inference_runtime import load_model
            self.model=load_model()
        else:
            sys.path.insert(0,str(REPO/'16_优化阶段_输入编码器推理加速'))
            from experiment_common import input_tensor
            from role_codec import RoleCodec,ref
            from sender_container import SenderWriter,ReferenceChecksums,read_sender_stream,PROFILE
            self.input_tensor=input_tensor;self.ref=ref;self.Writer=SenderWriter
            self.Checks=ReferenceChecksums;self.read_stream=read_sender_stream;self.profile=PROFILE
            self.model=RoleCodec(ASSETS,'sender' if role=='sender' else 'receiver')
            # Same frozen sender fill-on, independent receiver fill-off.
            torch.utils.deterministic.fill_uninitialized_memory=role=='sender'
        torch.cuda.synchronize()
        self.load_seconds=time.perf_counter()-started

    @torch.inference_mode()
    def job(self,request):
        started=time.perf_counter();out=Path(request['output']).resolve()
        inputs=[request[k] for k in ('base','raw','base_receipt','guide') if k in request]
        if self.guard:self.guard.authorize(out,inputs)
        out.mkdir(parents=True,exist_ok=False)
        torch.cuda.reset_peak_memory_stats()
        if self.role=='sender':result=self.send(request,out)
        elif self.role=='receiver':result=self.receive(request,out)
        else:result=self.post(request,out)
        torch.cuda.synchronize()
        result.update(role=self.role,job_seconds=time.perf_counter()-started,
            peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            peak_gpu_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
            cpu_memory=process_memory(),model_load_seconds=self.load_seconds)
        if self.guard:result.update(source_reads_blocked=True,blocked=self.guard.blocked,reads=sorted(self.guard.reads))
        save_json(out/'receipt.json',result)
        gc.collect();torch.cuda.empty_cache()
        return result

    def send(self,r,out):
        source_path=Path(r['source']);source=np.load(source_path,mmap_mode='r',allow_pickle=False)
        if source.shape!=(150,288,512,3) or file_sha(source_path)!=r['source_sha256']:
            raise ValueError('Sender source binding')
        codec=self.model;codec.dpb=dict(ref_latent=None,ref_y=None)
        header=dict(width=512,height=288,frames=150,fps=15,q=0,intra_period=64,reset_interval=32,
            revision=self.ref.REVISION,entropy=self.ref.ENTROPY_ID,table_sha256=codec.entropy.table_sha256,
            arithmetic=codec.profile,weights={k:v['sha256'] for k,v in self.ref.ASSETS.items()},
            container_version=3,integrity_profile=self.profile,sender_rgb_reconstructed=False)
        path=out/'base.glcsnd';writer=self.Writer(path,header);times=[]
        try:
            for i,rgb in enumerate(source):
                torch.cuda.synchronize();tick=time.perf_counter()
                _,packet=codec.frame(header,i,x=self.input_tensor(rgb),render_rgb=False)
                writer.add(packet);torch.cuda.synchronize();times.append(time.perf_counter()-tick)
                if i==0:emit('first_frame',kind='first_base_packet')
            writer.finish()
        finally:writer.close()
        codec.dpb=dict(ref_latent=None,ref_y=None)
        return dict(complete=True,stream_sha256=file_sha(path),stream_bytes=path.stat().st_size,
            core_with_input_and_write_seconds=sum(times),frame_seconds=times,header=header,
            input_sha256=r['source_sha256'],warmup_frames=0)

    def receive(self,r,out):
        tick=time.perf_counter();header,packets,integrity=self.read_stream(r['base'])
        codec=self.model
        if header['arithmetic']!=codec.profile or header['table_sha256']!=codec.entropy.table_sha256:
            raise ValueError('Decoder arithmetic identity')
        parse_seconds=time.perf_counter()-tick
        codec.dpb=dict(ref_latent=None,ref_y=None);checks=self.Checks();times=[]
        raw=np.empty((150,288,512,3),np.uint8)
        for i,packet in enumerate(packets):
            torch.cuda.synchronize();tick=time.perf_counter()
            rgb,value=codec.frame(header,i,packet=packet)
            checks.add(i,value);raw[i]=rgb
            torch.cuda.synchronize();times.append(time.perf_counter()-tick)
            if i==0:emit('first_frame',kind='first_base_rgb')
        checks.verify(integrity['expected_reference_checksums'])
        tick=time.perf_counter();path=out/'decoded_rgb.npy';np.save(path,raw,allow_pickle=False)
        save_seconds=time.perf_counter()-tick;codec.dpb=dict(ref_latent=None,ref_y=None)
        return dict(complete=True,stream_sha256=file_sha(r['base']),stream_bytes=Path(r['base']).stat().st_size,
            decoded_rgb_sha256=file_sha(path),reference_verified=True,source_media_reads_forbidden=True,
            parse_seconds=parse_seconds,core_seconds=sum(times),frame_seconds=times,
            output_save_seconds=save_seconds,backend='eager',warmup_frames=0)

    def post(self,r,out):
        import structure_guide_stream as guide
        from raft_inference_runtime import estimate
        tick=time.perf_counter();receipt=load_json(r['base_receipt'])
        digest=file_sha(r['base']);raw_path=Path(r['raw'])
        if digest!=receipt['stream_sha256'] or file_sha(raw_path)!=receipt['decoded_rgb_sha256'] or not receipt['reference_verified']:
            raise ValueError('Independent base decode binding')
        raw=np.load(raw_path,mmap_mode='r',allow_pickle=False)
        validation_seconds=time.perf_counter()-tick
        tick=time.perf_counter();packet=Path(r['guide']).read_bytes()
        _,_,images=guide.decode(packet,digest,FFMPEG);guide_seconds=time.perf_counter()-tick
        flows,flow_info=estimate(self.model,images,r['raft_mode'])
        tick=time.perf_counter()
        values=render(raw,images,*flows,cached=r['cached_t3'],on_first=lambda:emit('first_frame',kind='first_post_rgb'))
        render_seconds=time.perf_counter()-tick
        tick=time.perf_counter()
        for name,value in values.items():np.save(out/(name+'.npy'),value,allow_pickle=False)
        save_seconds=time.perf_counter()-tick
        del flows
        tick=time.perf_counter()
        command=[str(FFMPEG),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','512x288','-r','15','-i','pipe:0',
            '-an','-c:v','libx264','-preset','veryfast','-crf','12','-pix_fmt','yuv420p','-threads','2',
            '-movflags','+faststart','-n',str(out/'video.mp4')]
        subprocess.run(command,input=values['rgb'].tobytes(),stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True)
        video_seconds=time.perf_counter()-tick
        return dict(complete=True,raft_mode=r['raft_mode'],cached_t3=r['cached_t3'],
            validation_seconds=validation_seconds,guide_decode_seconds=guide_seconds,
            flow=flow_info,render_seconds=render_seconds,output_save_seconds=save_seconds,
            viewing_encode_seconds=video_seconds,base_sha256=digest,auxiliary_sha256=file_sha(r['guide']),
            outputs_sha256={name:file_sha(out/(name+'.npy')) for name in values},
            zero_weight_unchanged=True,scope='Local complete-file receiver; no network, display loop or streaming claim')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--role',choices=('sender','receiver','post'),required=True)
    a=p.parse_args()
    try:
        worker=Worker(a.role);emit('ready',role=a.role,load_seconds=worker.load_seconds,pid=os.getpid())
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('stop'):break
            result=worker.job(request);emit('result',result=result)
    except BaseException as error:
        traceback.print_exc(file=sys.stderr);emit('error',message=str(error));raise


if __name__=='__main__':main()
