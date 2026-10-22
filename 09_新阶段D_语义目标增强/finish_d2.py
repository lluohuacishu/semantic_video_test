"""Wait for the resumed D2 matrix, then evaluate and create local viewing copies."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from prepare_d2 import read_json, save_json


def run(args):
    root=args.run.resolve(); attempt=root/args.attempt
    status=attempt/'postprocess_progress.json'
    if status.exists(): raise FileExistsError('Postprocessing already dispatched; inspect its status')
    save_json(status,{'state':'waiting_for_codec','pid':os.getpid()})
    print('Waiting for the existing resumed matrix; no GPU jobs are dispatched here.',flush=True)
    while True:
        state=read_json(attempt/'progress.json')['state']
        if state=='failed':
            save_json(status,{'state':'blocked_by_codec_failure'})
            raise RuntimeError('Codec stopped; preserve existing outputs')
        if state=='codec_complete' and read_json(root/'progress.json')['state']=='codec_complete': break
        time.sleep(2)
    stage=Path(__file__).resolve().parent
    for phase,script,extra in [
        ('evaluate','evaluate_d2.py',[]),
        ('report','report_d2.py',['--output',str(stage/'D2数值附表_20261022.md')]),
        ('visualize','visualize_d2.py',[])]:
        save_json(status,{'state':'running','phase':phase,'pid':os.getpid()})
        print('START '+phase,flush=True)
        with (attempt/(phase+'_postprocess.log')).open('w',encoding='utf-8') as log:
            code=subprocess.run([sys.executable,'-u',str(stage/script),'--run',str(root),*extra],
                                stdout=log,stderr=subprocess.STDOUT).returncode
        if code:
            save_json(status,{'state':'failed','phase':phase,'returncode':code})
            raise RuntimeError(phase+' failed; inspect its log')
        print('DONE '+phase,flush=True)
    save_json(status,{'state':'completed','date':'2026-10-22'})
    print('D2 evaluation and viewing completed.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--attempt',default='resume_20261022_r1')
    run(parser.parse_args())
