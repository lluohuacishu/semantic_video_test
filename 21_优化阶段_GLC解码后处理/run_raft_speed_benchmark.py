"""Balanced fixed-I12 RAFT trials and a separately instrumented diagnostic."""
import argparse
import gc
import shutil
import time

import cv2
import numpy as np
import torch

from runtime_speed_common import *
from raft_inference_runtime import load_model, estimate, FlowRunner, Events


def compare(left,right):
    exact=bool(np.array_equal(left,right))
    if exact:return dict(exact=True,max_abs=0.,different=0)
    # Bound temporary memory while comparing all frame/pair elements.
    maximum=0.;different=0
    for a,b in zip(left,right,strict=True):
        maximum=max(maximum,float(np.abs(a-b).max()));different+=int(np.count_nonzero(a!=b))
    return dict(exact=False,max_abs=maximum,different=different)


@torch.inference_mode()
def run(folder):
    out=folder/'raft';out.mkdir(exist_ok=False)
    frozen=load_json(folder/'execution_plan.json')['fixed_input_sha256'];check_inputs(frozen)
    order=['original','final_only','feature_cache','final_only','feature_cache','original','feature_cache','original','final_only']
    code={name:file_sha(STAGE/name) for name in ('run_raft_speed_benchmark.py','raft_inference_runtime.py','runtime_speed_common.py','cached_t3.py')}
    for name in ('raft.py','corr.py','extractor.py','update.py'):
        path=UPSTREAM/'src/models/stage_1/core'/name;code[str(path)]=file_sha(path)
    save_json(out/'plan.json',dict(frozen_before_trials=True,order=order,iterations=12,keys=list(KEYS),
        original='Unmodified upstream forward',final_only='Same recurrent loop; only final upsampling executed',
        feature_cache='final_only plus per-image fnet/context reuse, original batch4/batch2 shapes; all precomputation included',
        precision='FP32, TF32 off',exact_requirement='All 174 bidirectional pairs compared with frozen CURRENT_I12 flows',
        benchmark='Same resident model after two warmup pairs. Three balanced rounds. Checks excluded from time.',
        code_sha256=code,gpu_before=gpu_snapshot()))
    snap=out/'source';snap.mkdir()
    for name in ('run_raft_speed_benchmark.py','raft_inference_runtime.py','runtime_speed_common.py','cached_t3.py'):
        shutil.copy2(STAGE/name,snap/name)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    images=np.load(ACCEPTED/'receive/guide_rgb.npy',mmap_mode='r')
    expected=[np.load(ACCEPTED/'flow'/name,mmap_mode='r') for name in ('to_anchor.npy','from_anchor.npy','transition_flow.npy')]
    model=load_model();warm=FlowRunner(model,images,'original')
    for frame,key in ((1,0),(15,0)):warm.pair(frame,key)
    torch.cuda.synchronize();del warm
    rows=[];render_verified=set()
    for index,mode in enumerate(order):
        before=gpu_snapshot();flows,info=estimate(model,images,mode)
        comparisons={name:compare(a,b) for name,a,b in zip(('back','forward','transition'),flows,expected,strict=True)}
        exact=all(v['exact'] for v in comparisons.values())
        row=dict(index=index,mode=mode,**info,comparison=comparisons,all_flow_bit_exact=exact,gpu_before=before,gpu_after=gpu_snapshot())
        if mode not in render_verified:
            raw=np.load(BASE/'base_receive/decoded_rgb.npy',mmap_mode='r')
            values=render(raw,images,*flows,cached=True)
            row['render_exact']={name:bool(np.array_equal(value,np.load(ACCEPTED/'receive'/(name+'.npy'),mmap_mode='r')))
                                 for name,value in values.items()}
            if exact and not all(row['render_exact'].values()):raise ValueError('Renderer regression with equal flow')
            if mode!='original':
                target=out/mode;target.mkdir()
                for name,value in values.items():np.save(target/(name+'.npy'),value,allow_pickle=False)
            render_verified.add(mode);del values
        rows.append(row);save_json(out/'trials.json',rows)
        print('RAFT',index,mode,round(info['seconds'],4),'flow_exact',exact,'peakMiB',round(info['peak_gpu_allocated_mib'],1),flush=True)
        if mode=='original' and not exact:raise ValueError('Original RAFT does not replay accepted flow')
        del flows;gc.collect();torch.cuda.empty_cache()
    times={mode:stats([r['seconds'] for r in rows if r['mode']==mode]) for mode in ('original','final_only','feature_cache')}
    valid=[mode for mode in times if all(r['all_flow_bit_exact'] for r in rows if r['mode']==mode)]
    selected=min(valid,key=lambda mode:times[mode]['median'])
    # Separate event-instrumented diagnostic; never mix these times into trials.
    profiler=FlowRunner(model,images,'instrumented_original');original=FlowRunner(model,images,'original');diagnostic=[]
    for frame,key in ((1,0),(14,0),(16,15),(64,45),(105,75),(128,105),(149,135)):
        events=Events();actual=profiler.pair(frame,key,events);reference=original.pair(frame,key)
        if not torch.equal(actual,reference):raise ValueError('Instrumented original changed output')
        diagnostic.append(dict(frame=frame,key=key,cuda_ms=events.result()))
    save_json(out/'diagnostic.json',dict(complete=True,rows=diagnostic,
        scope='Seven representative bidirectional calls, CUDA event diagnostic only; event recording perturbs execution',
        includes_intermediate_upsampling=True))
    check_inputs(frozen)
    save_json(out/'summary.json',dict(complete=True,times=times,selected_exact_candidate=selected,
        exact_candidates=valid,median_time_reduction_percent=(1-times[selected]['median']/times['original']['median'])*100,
        all_flow_pairs=174,all_directions=348,precision_or_iterations_changed=False,
        input_hashes_unchanged=True,diagnostic='diagnostic.json',
        selection_scope='Only this fixed video1; full local chain must still verify selected candidate'))
    print('RAFT COMPLETE selected',selected,times,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();run(a.run.resolve())
