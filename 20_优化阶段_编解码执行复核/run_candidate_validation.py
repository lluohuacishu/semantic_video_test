"""Four independent ABBA receiver processes on one immutable existing stream."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text('utf-8-sig'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',required=True)
    ap.add_argument('--stream',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cuda-graph',action='store_true')
    ap.add_argument('--worker',type=Path,default=HERE/'candidate_receiver.py')
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    runs=[]
    for name,mode,save_rgb in [('baseline_a','baseline',True),('candidate_b',args.mode,True),
                              ('candidate_b2',args.mode,False),('baseline_a2','baseline',False)]:
        folder=args.output/name
        cmd=[sys.executable,'-X','utf8','-B',str(args.worker),
             '--mode',mode,'--stream',str(args.stream),'--cache',str(args.cache),'--output',str(folder)]
        if save_rgb:
            cmd.extend(['--save-rgb','--video'])
        if args.cuda_graph and mode!='baseline':
            cmd.append('--cuda-graph')
        print('START '+name,flush=True)
        with (args.output/(name+'.log')).open('wb') as f:
            subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,
                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        result=load(folder/'result.json')
        runs.append(dict(name=name,result_path=str(folder/'result.json'),**result))
        print(f'DONE {name} core={result["core_seconds"]:.3f}s FPS={result["fps"]:.3f}',flush=True)
    first=runs[0]
    for run in runs:
        for key in ('frames','stream_sha256','reference_checksums','entropy_symbols_and_cdf_sha256'):
            if run[key]!=first[key]:
                raise ValueError('Reference/entropy/stream differs: '+key)
        if not run['reference_verified'] or not run['source_media_reads_forbidden']:
            raise ValueError('Independent reference validation missing')
        if [r['latent_sha256'] for r in run['records']]!=[r['latent_sha256'] for r in first['records']]:
            raise ValueError('Per-frame reference differs')
    for a,b in [(runs[0],runs[3]),(runs[1],runs[2])]:
        if [r['rgb_sha256'] for r in a['records']]!=[r['rgb_sha256'] for r in b['records']]:
            raise ValueError('Same backend did not repeat RGB')
    base=statistics.median(r['core_seconds'] for r in runs if r['mode']=='baseline')
    cand=statistics.median(r['core_seconds'] for r in runs if r['mode']!='baseline')
    summary=dict(complete=True,mode=args.mode,candidate_cuda_graph=args.cuda_graph,frames=first['frames'],stream_sha256=first['stream_sha256'],
        baseline_median_core_seconds=base,candidate_median_core_seconds=cand,
        baseline_fps=first['frames']/base,candidate_fps=first['frames']/cand,speedup=base/cand,
        time_saved_percent=(1-cand/base)*100,all_reference_states_equal=True,all_entropy_symbols_and_cdf_equal=True,
        per_backend_rgb_repeatable=True,runs=[{k:r[k] for k in ('name','result_path','mode','core_seconds','fps','worker_seconds')} for r in runs])
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PASS '+json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
