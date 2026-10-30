"""Sequential GPU campaign with resumable, hash-bound results."""
from __future__ import annotations
import argparse
import statistics
import subprocess
import time
from experiment_common import *


def run(folder,script,args,plan):
    path=folder/'result.json'
    if not path.exists():
        folder.parent.mkdir(parents=True,exist_ok=True)
        command=[str(load_settings()['runtime']['python']),'-X','utf8',str(HERE/script),'--plan',str(plan),
                 '--output',str(folder),*args]
        with folder.with_suffix('.log').open('w',encoding='utf-8') as log:
            p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError(f'{script} failed ({p.returncode}): {folder.with_suffix(".log")}')
    value=load_json(path)
    if value['plan_sha256']!=file_sha(plan):raise ValueError('Cached result from different plan')
    return value


def compare(a,b,rgb=False):
    if len(a['records'])!=len(b['records']):raise ValueError('Frame count changed')
    for x,y in zip(a['records'],b['records']):
        for k in ['index','kind','latent_sha256','payload_sha256','entropy']+(['rgb_sha256'] if rgb else []):
            if x[k]!=y[k]:raise ValueError(f'Frame {x["index"]} differs: {k}')


def fills(out,plan):
    reports=[]
    for c in plan['clips']:
        for role in ['sender','receiver']:
            pairs=[]
            stream=Path(c['stream']) if c.get('stream') else out/'fill'/c['id']/'sender/pair1_on/stream.glcsnd'
            reference=None
            for pair in range(1,plan['fill_pairs']+1):
                row={}
                for fill in (['on','off'] if pair%2 else ['off','on']):
                    folder=out/'fill'/c['id']/role/f'pair{pair}_{fill}'
                    args=['--clip',c['id'],'--role',role,'--variant','eager','--fill',fill]
                    if role=='receiver':args+=['--stream',str(stream)]
                    value=run(folder,'codec_worker.py',args,out/'plan.json')
                    if c.get('teacher_result'):
                        old=load_json(c['teacher_result'])
                        if len(old['records'])!=len(value['records']):raise ValueError('Historical frame count differs')
                        for expected,actual in zip(old['records'],value['records']):
                            for key in ['index','kind','latent_sha256','payload_sha256']+(['rgb_sha256'] if role=='receiver' else []):
                                if expected[key]!=actual[key]:raise ValueError('Historical output differs: '+key)
                    if reference is None:reference=value
                    compare(reference,value,rgb=role=='receiver')
                    if role=='sender':
                        expected=c.get('stream_sha256',reference['stream_sha256'])
                        if value['stream_sha256']!=expected:raise ValueError('Fill changed complete stream')
                    row[fill]=value
                    print(f'{c["id"]} {role} fill={fill} pair={pair}: {value["core_seconds"]:.3f}s exact',flush=True)
                pairs.append(row)
            report=dict(clip=c['id'],role=role,frames=c['frames'],q=c['q'],pairs=pairs,
                on_median_seconds=statistics.median(r['on']['core_seconds'] for r in pairs),
                off_median_seconds=statistics.median(r['off']['core_seconds'] for r in pairs),
                paired_speedup_median=statistics.median(r['on']['core_seconds']/r['off']['core_seconds'] for r in pairs))
            reports.append(report);save_json(out/'fill_summary.json',dict(complete=False,reports=reports))
    save_json(out/'fill_summary.json',dict(complete=True,reports=reports,default_changed=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['fill','encoder'],required=True)
    p.add_argument('--variants',nargs='+',default=['eager','channels_last','trace','graph','tf32','amp_fp16'])
    args=p.parse_args();out=args.output.resolve();plan=load_json(out/'plan.json');frozen(plan)
    if file_sha(out/'plan.json')!=load_json(out/'plan_identity.json')['sha256']:raise ValueError('Plan changed')
    if args.phase=='fill':fills(out,plan)
    else:
        rows=[]
        for model in ['video','image']:
            for variant in args.variants:
                folder=out/'encoder'/model/variant
                result=run(folder,'benchmark_encoder.py',['--model',model,'--variant',variant],out/'plan.json')
                rows.append(result)
                print(f'{model} {variant}: {result["candidate_wall_median_ms"]:.3f} ms exact={all(x["bit_exact"] for x in result["changes"])}',flush=True)
                save_json(out/'encoder_summary.json',dict(complete=False,reports=rows))
        save_json(out/'encoder_summary.json',dict(complete=True,reports=rows))


if __name__=='__main__':main()
