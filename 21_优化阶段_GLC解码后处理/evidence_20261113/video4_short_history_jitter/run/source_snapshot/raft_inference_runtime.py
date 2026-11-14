"""Inference-only RAFT candidates; original checkpoint and recurrent arithmetic."""
import argparse
from contextlib import contextmanager, nullcontext
import sys
import time

import numpy as np
import torch

from runtime_speed_common import UPSTREAM, WEIGHT, KEYS, EVENTS
from receiver_structure import WEIGHT_SHA, file_sha

sys.path.insert(0, str(UPSTREAM))
from src.models.stage_1.core.raft import RAFT
from src.models.stage_1.core.corr import CorrBlock


def load_model():
    if file_sha(WEIGHT) != WEIGHT_SHA:
        raise ValueError('RAFT checkpoint changed')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # Match the original stand-alone CURRENT_I12 worker, not GLC global setup.
    if torch.are_deterministic_algorithms_enabled():
        raise ValueError('RAFT worker inherited a different deterministic setting')
    model=RAFT(argparse.Namespace(small=False,mixed_precision=False,alternate_corr=False))
    state=torch.load(WEIGHT,map_location='cpu',weights_only=True)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    return model.cuda().eval()


class Events:
    def __init__(self):self.items=[]

    @contextmanager
    def section(self,name):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record();self.items.append((name,start,end))

    def result(self):
        torch.cuda.synchronize();result={}
        for name,start,end in self.items:
            result[name]=result.get(name,0.)+start.elapsed_time(end)
        return result


class FlowRunner:
    def __init__(self,model,images,mode):
        if mode not in ('original','instrumented_original','final_only','feature_cache'):
            raise ValueError(mode)
        self.model,self.mode=model,mode
        self.frames=torch.from_numpy(np.array(images)).permute(0,3,1,2).to(device='cuda',dtype=torch.float32)
        self.fmap=self.net=self.inp=None
        self.prepare_seconds=0.

    def prepare(self):
        if self.mode!='feature_cache':return
        tick=time.perf_counter();features=[];nets=[];inputs=[]
        n=len(self.frames)
        # Keep original fnet batch=4 and cnet batch=2 kernel shapes. Actual
        # per-image numerical equivalence is tested, never assumed.
        for start in range(0,n,4):
            ids=[min(start+j,n-1) for j in range(4)]
            images=(2*(self.frames[ids]/255.0)-1.0).contiguous()
            features.append(self.model.fnet(images)[:min(4,n-start)].float())
        for start in range(0,n,2):
            ids=[min(start+j,n-1) for j in range(2)]
            images=(2*(self.frames[ids]/255.0)-1.0).contiguous()
            net,inp=torch.split(self.model.cnet(images),[128,128],dim=1)
            nets.append(torch.tanh(net)[:min(2,n-start)])
            inputs.append(torch.relu(inp)[:min(2,n-start)])
        self.fmap=torch.cat(features);self.net=torch.cat(nets);self.inp=torch.cat(inputs)
        torch.cuda.synchronize();self.prepare_seconds=time.perf_counter()-tick

    def pair(self,frame,key,events=None):
        batch=self.frames[[frame,key]]
        if self.mode=='original':return self.model(batch,batch.flip(0),iters=12,test_mode=True)[1]
        section=events.section if events is not None else lambda name:nullcontext()
        with section('normalization'):
            image1=(2*(batch/255.0)-1.0).contiguous()
            image2=(2*(batch.flip(0)/255.0)-1.0).contiguous()
        with section('feature_network'):
            if self.mode=='feature_cache':
                fmap1=self.fmap[[frame,key]];fmap2=self.fmap[[key,frame]]
            else:
                fmap1,fmap2=self.model.fnet([image1,image2])
                fmap1=fmap1.float();fmap2=fmap2.float()
        with section('correlation_volume'):
            corr_fn=CorrBlock(fmap1,fmap2,radius=self.model.args.corr_radius)
        with section('context_network'):
            if self.mode=='feature_cache':net=self.net[[frame,key]];inp=self.inp[[frame,key]]
            else:
                context=self.model.cnet(image1)
                net,inp=torch.split(context,[128,128],dim=1)
                net=torch.tanh(net);inp=torch.relu(inp)
        with section('initialize_coordinates'):
            coords0,coords1=self.model.initialize_flow(image1)
        for itr in range(12):
            coords1=coords1.detach()
            with section('correlation_lookup'):
                corr=corr_fn(coords1)
            with section('recurrent_update'):
                flow=coords1-coords0
                net,up_mask,delta_flow=self.model.update_block(net,inp,corr,flow)
                coords1=coords1+delta_flow
            if self.mode=='instrumented_original' or itr==11:
                with section('upsample'):
                    flow_up=self.model.upsample_flow(coords1-coords0,up_mask)
        return flow_up


@torch.inference_mode()
def estimate(model,images,mode,on_pair=None):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    begin=time.perf_counter()
    runner=FlowRunner(model,images,mode);runner.prepare()
    back=np.zeros((150,288,512,2),np.float32)
    forward=np.zeros_like(back)
    transition=np.empty((len(EVENTS),2,288,512,2),np.float32)
    transfer_seconds=0.;pair_count=0
    def pair(frame,key):
        nonlocal transfer_seconds,pair_count
        prediction=runner.pair(frame,key)
        tick=time.perf_counter()
        value=prediction.permute(0,2,3,1).contiguous().cpu().numpy()
        transfer_seconds+=time.perf_counter()-tick
        if value.shape!=(2,288,512,2) or not np.isfinite(value).all():raise ValueError('Invalid flow')
        pair_count+=1
        if on_pair is not None:on_pair(pair_count)
        return value
    for key,stop in zip(KEYS,(*KEYS[1:],150),strict=True):
        for frame in range(key+1,stop):back[frame],forward[frame]=pair(frame,key)
    for index,frame in enumerate(EVENTS):transition[index]=pair(frame,KEYS[index//6])
    torch.cuda.synchronize();elapsed=time.perf_counter()-begin
    info=dict(seconds=elapsed,feature_precompute_seconds=runner.prepare_seconds,pairs=pair_count,
        peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        peak_gpu_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
        cpu_transfer_call_wall_seconds=transfer_seconds,
        transfer_note='Includes synchronization/backpressure; not exclusive memcpy time',
        flow_storage='Receiver-local RAM; no transmitted sidecar and no flow file writes in this benchmark')
    del runner
    return (back,forward,transition),info
