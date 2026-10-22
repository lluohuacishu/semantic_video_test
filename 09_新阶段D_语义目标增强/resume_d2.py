"""Resume only unfinished points of the registered D2 matrix, retaining old evidence."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import time

from prepare_d2 import read_json, save_json

HERE=Path(__file__).resolve().parent


def run(args):
    root=args.run.resolve(); resumed=root/args.attempt
    matrix=read_json(root/'matrix.json'); tasks=matrix['tasks']
    encoded=[t for t in tasks if (Path(t['output'])/'encode.json').exists()]
    decoded=[t for t in tasks if (Path(t['output']).parent/'decoder/decode.json').exists()]
    missing_enc=[t for t in tasks if t not in encoded]
    missing_dec=[t for t in tasks if t not in decoded]
    if (len(tasks),len(encoded),len(decoded),len(missing_enc),len(missing_dec))!=(24,15,5,9,19):
        raise ValueError('Checkpoint differs from the 15-encode/5-decode user handoff; inspect before dispatch')
    for t in missing_enc:
        if Path(t['output']).exists():
            raise FileExistsError('Preserve the interrupted encoder folder before retrying: '+t['id'])
    for t in missing_dec:
        if (Path(t['output']).parent/'decoder').exists():
            raise FileExistsError('Incomplete receiver output needs deliberate preservation: '+t['id'])
    resumed.mkdir(exist_ok=False)
    prior=read_json(root/'progress.json')
    save_json(resumed/'prior_progress.json',prior)
    save_json(resumed/'resume_plan.json',{
        'schema':'d2-resume-v1','date':'2026-10-22','parent_plan':'../plan.json',
        'reason':'User explicitly continued after the previous time limit.',
        'existing_encodes':[t['id'] for t in encoded],
        'existing_decodes':[t['id'] for t in decoded],
        'remaining_encodes':[t['id'] for t in missing_enc],
        'remaining_decodes':[t['id'] for t in missing_dec],
        'original_STOP_preserved':True,'active_STOP':str(resumed/'STOP'),
        'interrupted_output_archive':'../interrupted_20261021/v002_B_q32_encoder',
        'dataset_or_weight_integrity_rescan':False,'new_q_candidates':False,
        'new_training_or_semantic_inference':False,'codec_arithmetic_unchanged':True})
    started=time.perf_counter()
    runtime=matrix['runtime']
    for phase,role,subset in [('encode_resume','encode',missing_enc),('decode_resume','decode',missing_dec)]:
        if role=='decode':
            subset=[{'id':t['id'],'input':str(Path(t['output'])/'stream.glcv'),
                     'output':str(Path(t['output']).parent/'decoder')} for t in subset]
        config=resumed/(phase+'_config.json')
        save_json(config,{'phase':phase,'role':role,'run_root':str(resumed),'runtime':runtime,'tasks':subset})
        print(f'START {phase}: {len(subset)} points',flush=True)
        with (resumed/(phase+'.log')).open('w',encoding='utf-8') as log:
            child=subprocess.Popen([runtime['python'],'-u',str(HERE/'d2_codec_worker.py'),'--config',str(config)],
                                   stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUTF8='1'))
            save_json(resumed/'progress.json',{'state':'running','phase':phase,'child_pid':child.pid,
                      'elapsed_seconds':time.perf_counter()-started})
            code=child.wait()
        if code:
            save_json(resumed/'progress.json',{'state':'failed','phase':phase,'returncode':code,
                      'elapsed_seconds':time.perf_counter()-started})
            raise RuntimeError(phase+' stopped; retain outputs and inspect its log')
        print('DONE '+phase,flush=True)
    # Completed points span multiple actual processes. Keep their identities
    # instead of fabricating the interrupted original phase's missing summary.
    anchors=read_json(root/'encode_anchors_summary.json')
    old_encoder_pid=read_json(root/'timeboxed_20261021/plan.json')['encoder_remaining_pid']
    index={}
    for t in tasks:
        enc=read_json(Path(t['output'])/'encode.json')
        dec=read_json(Path(t['output']).parent/'decoder/decode.json')
        if t in encoded:
            phase='encode_anchors' if t['method']=='A' and t['q']==32 else 'encode_remaining_interrupted'
            pid=anchors['pid'] if phase=='encode_anchors' else old_encoder_pid
            sender=f'20261021/{phase}'
        else:
            pid=enc['pid']; sender=args.attempt+'/'+enc['worker_phase']
        if t in decoded:
            receiver='20261021/'+('decode_anchors' if t['method']=='A' else 'decode_timeboxed_q32')
        else:
            receiver=args.attempt+'/'+dec['worker_phase']
        index[t['id']]={'sender_pid':pid,'sender_execution':sender,
                        'receiver_pid':dec['pid'],'receiver_execution':receiver,
                        'sender_receiver_separate_invocations':sender!=receiver}
    save_json(root/'execution_index.json',index)
    phases=[]
    for name in ('encode_anchors','decode_anchors'):
        phases.append(dict(phase='20261021/'+name,**read_json(root/(name+'_summary.json'))))
    old_progress=read_json(root/'encode_progress.json')
    phases.append({'phase':'20261021/encode_remaining_interrupted','state':'interrupted_by_user',
        'pid':old_encoder_pid,'tasks':[t['id'] for t in encoded if not (t['method']=='A' and t['q']==32)],
        'seconds':old_progress['elapsed_seconds'],'seconds_is_lower_bound':True,
        'model_load_seconds':read_json(root/'points/v001_A_q00/encoder/encode.json')['model_load_seconds']})
    phases.append(dict(phase='20261021/decode_timeboxed_q32',
                       **read_json(root/'timeboxed_20261021/decode_timeboxed_q32_summary.json')))
    for name in ('encode_resume','decode_resume'):
        phases.append(dict(phase=args.attempt+'/'+name,**read_json(resumed/(name+'_summary.json'))))
    save_json(root/'execution_summary.json',{'phases':phases,'pause_time_excluded':True,
        'interrupted_phase_seconds_are_last_frame_progress_lower_bound':True})
    result={'state':'codec_complete','points':24,'independent_decodes':24,
            'resume_seconds':time.perf_counter()-started,'resumed_at':args.attempt,
            'original_interruption_preserved':True,'completion_date':'2026-10-22'}
    save_json(resumed/'progress.json',result)
    save_json(root/'progress.json',result)
    print('CODEC_COMPLETE: 24 encoded points and 24 independent decodes',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--attempt',default='resume_20261022_r1')
    run(parser.parse_args())
