"""Run optional backends sequentially; preserve failures as experimental evidence."""
from __future__ import annotations
import argparse
import statistics
import traceback
from experiment_common import *
from run_experiment import run,compare
from run_profile import gpu_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['encoder','full'],required=True)
    p.add_argument('--variants',nargs='+',required=True)
    p.add_argument('--clips',nargs='+',default=['long108_q0_640','sf007_q12_512','sf025_q32_512','uv096_q63_512_255'])
    args=p.parse_args();out=args.output.resolve();plan=load_json(out/'plan.json');frozen(plan)
    rows=[];failures=[]
    def job(folder,script,extra):
        try:
            cached=(folder/'result.json').exists()
            names={script,'experiment_common.py','encoder_variants.py','extra_variants.py',
                   'benchmark_encoder.py','benchmark_encoder_isolated.py','codec_worker.py','candidate_worker.py'}
            hashes={name:file_sha(HERE/name) for name in sorted(names)}
            before=gpu_snapshot() if not cached else None
            value=run(folder,script,extra,out/'plan.json')
            if not cached:
                if hashes!={name:file_sha(HERE/name) for name in sorted(names)}:raise ValueError('Job implementation changed while running')
                save_json(folder.with_suffix('.code.json'),dict(code_sha256=hashes,gpu_before=before,gpu_after=gpu_snapshot()))
            return value
        except Exception as exc:
            failures.append(dict(folder=str(folder),type=type(exc).__name__,error=str(exc)))
            save_json(out/(args.phase+'_failures.json'),failures)
            print('FAILED',folder,repr(exc),flush=True)
            return None
    if args.phase=='encoder':
        for model in ['video','image']:
            for variant in args.variants:
                folder=out/'encoder'/model/variant
                result=job(folder,'benchmark_encoder_isolated.py',['--model',model,'--variant',variant])
                if result is not None:
                    rows.append(result)
                    print(f'{model} {variant}: {result["candidate_wall_median_ms"]:.3f}ms, exact={all(x["bit_exact"] for x in result["changes"])}',flush=True)
                save_json(out/'encoder_summary.json',dict(complete=False,reports=rows,failures=failures))
        save_json(out/'encoder_summary.json',dict(complete=True,reports=rows,failures=failures))
    else:
        for cid in args.clips:
            clip=next(c for c in plan['clips'] if c['id']==cid)
            variants=['eager',*args.variants]
            senders_by_variant={v:[] for v in variants}
            for repeat in range(1,3):
                for variant in (variants if repeat==1 else list(reversed(variants))):
                    folder=out/'full'/cid/variant/f'sender{repeat}'
                    result=job(folder,'candidate_worker.py',['--clip',cid,'--role','sender','--variant',variant,'--fill','off'])
                    if result is None:continue
                    senders=senders_by_variant[variant]
                    if senders:
                        compare(senders[0],result)
                        if senders[0]['stream_sha256']!=result['stream_sha256']:raise ValueError('Candidate not repeatable')
                    senders.append(result)
                    print(f'full sender {cid} {variant} repeat={repeat}: {result["core_seconds"]:.3f}s',flush=True)
            for variant in variants:
                if len(senders_by_variant['eager'])!=2:raise ValueError('Full baseline incomplete')
                senders=senders_by_variant[variant]
                if len(senders)!=2:continue
                source=out/'full'/cid/variant/'sender1/stream.glcsnd'
                receiver=job(out/'full'/cid/variant/'receiver','codec_worker.py',
                    ['--clip',cid,'--role','receiver','--variant','eager','--fill','off','--stream',str(source),'--save-rgb'])
                if receiver is None:continue
                compare(senders[0],receiver)
                rows.append(dict(clip=cid,variant=variant,sender_repeats=senders,receiver=receiver,
                    sender_core_median_seconds=statistics.median(x['core_seconds'] for x in senders),
                    paired_speedup_vs_eager=[a['core_seconds']/b['core_seconds'] for a,b in zip(senders_by_variant['eager'],senders)],
                    order_policy='repeat1 forward variants; repeat2 reverse variants; decode after all sender timings',
                    sender_stream_repeatable=True,independent_reference_and_symbols_verified=True))
                print(f'full {cid} {variant}: encode {rows[-1]["sender_core_median_seconds"]:.3f}s; independent decode verified',flush=True)
                save_json(out/'full_summary.json',dict(complete=False,reports=rows,failures=failures))
        save_json(out/'full_summary.json',dict(complete=True,reports=rows,failures=failures))


if __name__=='__main__':main()
