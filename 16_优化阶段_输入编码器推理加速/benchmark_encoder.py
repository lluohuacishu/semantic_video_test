"""Single encoder: actual frames, paired orders, GPU events and synchronized wall time."""
from __future__ import annotations
import argparse
import importlib.util
import platform
import statistics
import time
from experiment_common import *
from encoder_variants import EncoderVariant
from role_codec import RoleCodec
import torch


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',required=True);p.add_argument('--clip',default='long108_q0_640')
    p.add_argument('--model',choices=['image','video'],default='video')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    plan=load_json(args.plan);frozen(plan);clip=next(c for c in plan['clips'] if c['id']==args.clip)
    codec=RoleCodec(plan['assets'],'sender');model=getattr(codec,args.model).vqgan.encoder
    source=np.load(clip['source'],mmap_mode='r');indexes=plan['encoder_samples']
    inputs=[input_tensor(source[i]) for i in indexes]
    refs=[model(x).cpu().numpy() for x in inputs]
    # All candidates use the same fill-off baseline; fill-on is a separate ablation.
    torch.utils.deterministic.fill_uninitialized_memory=False
    base=model
    geometry=f'{inputs[0].shape[-1]}x{inputs[0].shape[-2]}'
    candidate=EncoderVariant(model,inputs[0],args.variant,args.plan.parent/'engines'/geometry/args.model)
    changes=[difference(a,candidate(x).cpu().numpy()) for a,x in zip(refs,inputs)]
    for x in inputs:
        for _ in range(plan['encoder_warmup']):candidate(x)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    resident=torch.cuda.memory_allocated()/1024**2
    rows=[]
    # A candidate may transform weight layout. Eager control uses those same values/layout;
    # the original-layout standalone baseline is also measured in its own eager job.
    for rep in range(plan['encoder_repeats']):
        for name in (['baseline','candidate'] if rep%2==0 else ['candidate','baseline']):
            function=base if name=='baseline' else candidate
            for i,x in zip(indexes,inputs):
                for _ in range(plan['encoder_iterations']):
                    torch.cuda.synchronize();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    tick=time.perf_counter_ns();a.record();y=function(x);b.record();b.synchronize()
                    wall=(time.perf_counter_ns()-tick)/1e6
                    rows.append(dict(repeat=rep,mode=name,index=i,event_ms=a.elapsed_time(b),wall_ms=wall))
    ratios=[]
    for rep in range(plan['encoder_repeats']):
        a=statistics.median(r['wall_ms'] for r in rows if r['repeat']==rep and r['mode']=='baseline')
        b=statistics.median(r['wall_ms'] for r in rows if r['repeat']==rep and r['mode']=='candidate')
        ratios.append(a/b)
    result=dict(variant=args.variant,clip=clip['id'],model=args.model,input_shape=list(inputs[0].shape),
        parameters=sum(p.numel() for p in model.parameters()),setup_seconds=candidate.setup_seconds,
        changes=changes,records=rows,paired_speedup_median=statistics.median(ratios),
        candidate_wall_median_ms=statistics.median(r['wall_ms'] for r in rows if r['mode']=='candidate'),
        baseline_wall_median_ms=statistics.median(r['wall_ms'] for r in rows if r['mode']=='baseline'),
        candidate_event_median_ms=statistics.median(r['event_ms'] for r in rows if r['mode']=='candidate'),
        resident_mib=resident,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        includes_input_layout_conversion=True,includes_graph_or_engine_output_copy=True,fill_uninitialized_memory=False,
        tf32_outside_encoder=False,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        source_sha256=clip['source_sha256'],plan_sha256=file_sha(args.plan),
        baseline_note='channels_last transforms shared weight layout; compare its standalone eager control, not this paired ratio')
    save_json(args.output/'result.json',result)
    print(f'{args.model} {args.variant}: {result["candidate_wall_median_ms"]:.3f}ms, exact={all(x["bit_exact"] for x in changes)}',flush=True)


if __name__=='__main__':main()
