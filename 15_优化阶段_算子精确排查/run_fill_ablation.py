"""Diagnostic-only allocation-fill ablation; defaults and algorithms unchanged."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'14_优化阶段_部署精简'))
from role_codec import RoleCodec
from sender_common import load_json,save_json,file_sha,load_settings
import torch


def worker(args,remaining):
    import profile_worker
    original=profile_worker.RoleCodec
    def factory(*pos,**kwargs):
        value=original(*pos,**kwargs)
        torch.utils.deterministic.fill_uninitialized_memory=args.fill=='on'
        if not torch.are_deterministic_algorithms_enabled():raise ValueError('Deterministic algorithms changed')
        return value
    profile_worker.RoleCodec=factory
    folder=Path(remaining[remaining.index('--output')+1])
    sys.argv=[str(HERE/'profile_worker.py'),*remaining]
    profile_worker.main()
    result=load_json(folder/'result.json')
    result.update(fill_uninitialized_memory=args.fill=='on',deterministic_algorithms=True,
                  arithmetic_changed=False,setting_changed_in_diagnostic_child_only=True)
    save_json(folder/'result.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fill',choices=['on','off'])
    args,remaining=parser.parse_known_args()
    if args.fill:
        worker(args,remaining);return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(remaining)
    from run_profile import compare
    import statistics
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    plan=load_json(args.campaign/'plan.json');teacher=load_json(plan['clip']['teacher_result'])
    results={}
    for role in ['sender','receiver']:
        pairs=[]
        for pair,order in enumerate([['on','off'],['off','on'],['on','off']],1):
            row={}
            for setting in order:
                folder=output/role/f'pair{pair}_{setting}';folder.parent.mkdir(parents=True,exist_ok=True)
                command=[str(load_settings()['runtime']['python']),'-X','utf8',str(__file__),'--fill',setting,
                    '--plan',str(args.campaign/'plan.json'),'--role',role,'--mode','baseline','--output',str(folder)]
                begin=time.perf_counter()
                with folder.with_suffix('.log').open('w',encoding='utf-8') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                value=load_json(folder/'result.json');value['process_wall_seconds']=time.perf_counter()-begin
                compare(teacher,value,role)
                if role=='sender' and value['stream_sha256']!=plan['clip']['stream_sha256']:raise ValueError('Allocation-fill ablation changed full stream')
                save_json(folder/'result.json',value);row[setting]=value
                print(f'{role} fill {setting}: {value["core_seconds"]:.4f}s, exact',flush=True)
            pairs.append(row)
        results[role]=dict(pairs=pairs,on_median_seconds=statistics.median(row['on']['core_seconds'] for row in pairs),
                          off_median_seconds=statistics.median(row['off']['core_seconds'] for row in pairs),
                          paired_speedup_median=statistics.median(row['on']['core_seconds']/row['off']['core_seconds'] for row in pairs))
    save_json(output/'summary.json',dict(results=results,complete=True,default_modified=False,
        deterministic_algorithms_preserved=True,reference_stream_and_pixels_exact=True,
        code_sha256={p.name:file_sha(p) for p in [HERE/'run_fill_ablation.py',HERE/'profile_worker.py']}))
    print('Completed allocation-fill ablation; production default remains unchanged')


if __name__=='__main__':main()
