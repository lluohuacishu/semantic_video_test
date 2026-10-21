"""Run the registered D2 matrix with serial, separate sender/receiver processes."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from prepare_d2 import read_json, save_json

HERE=Path(__file__).resolve().parent


def run(args):
    root=args.run.resolve()
    if (root/'matrix.json').exists():
        raise FileExistsError('This run was already dispatched; inspect its progress instead of duplicating it')
    plan=read_json(root/'plan.json')
    inputs=read_json(root/'inputs.json')
    sys.path.insert(0,str(HERE.parent/'08_新阶段C_编解码加速与网页展示'))
    from c4_config import load_settings
    runtime={k:str(v) for k,v in load_settings()['runtime'].items()}
    tasks=[]
    for clip in inputs['clips']:
        for method in plan['groups']:
            for q in plan['codec']['q_candidates']:
                label=f'v{clip["video_id"]:03d}_{method}_q{q:02d}'
                tasks.append({'id':label,'video_id':clip['video_id'],'method':method,'q':q,
                              'width':clip['width'],'height':clip['height'],'fps':clip['fps'],
                              'frames':[f['paths'][method] for f in clip['records']],
                              'output':str(root/'points'/label/'encoder')})
    if len(tasks)!=24:
        raise ValueError('Matrix differs from the registered 24 points')
    anchors=[t for t in tasks if t['method']=='A' and t['q']==32]
    others=[t for t in tasks if t not in anchors]
    save_json(root/'matrix.json',{'tasks':tasks,'runtime':runtime,'phases':['encode_anchors','decode_anchors','encode_remaining','decode_remaining']})
    started=time.perf_counter()
    for phase,role,subset in [('encode_anchors','encode',anchors),('decode_anchors','decode',anchors),
                              ('encode_remaining','encode',others),('decode_remaining','decode',others)]:
        if role=='decode':
            subset=[{'id':t['id'],'input':str(Path(t['output'])/'stream.glcv'),
                     'output':str(Path(t['output']).parent/'decoder')} for t in subset]
        config=root/(phase+'_config.json')
        save_json(config,{'phase':phase,'role':role,'run_root':str(root),'runtime':runtime,'tasks':subset})
        save_json(root/'progress.json',{'state':'running','phase':phase,'elapsed_seconds':time.perf_counter()-started})
        print('START '+phase,flush=True)
        env=dict(os.environ,PYTHONUTF8='1')
        with (root/(phase+'.log')).open('w',encoding='utf-8') as log:
            child=subprocess.Popen([runtime['python'],'-u',str(HERE/'d2_codec_worker.py'),'--config',str(config)],
                                    stdout=log,stderr=subprocess.STDOUT,env=env)
            save_json(root/'progress.json',{'state':'running','phase':phase,'child_pid':child.pid,
                                          'elapsed_seconds':time.perf_counter()-started})
            code=child.wait()
        if code:
            save_json(root/'progress.json',{'state':'failed','phase':phase,'returncode':code,
                                          'elapsed_seconds':time.perf_counter()-started})
            raise RuntimeError(f'{phase} failed; see {phase}.log')
        print('DONE '+phase,flush=True)
    save_json(root/'progress.json',{'state':'codec_complete','points':24,'independent_decodes':24,
                                  'seconds':time.perf_counter()-started})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    run(parser.parse_args())
