"""Verify saved outputs and the explicit fast receiver entry after timing ends."""
import argparse
import ast
from pathlib import Path
import shutil
import subprocess
import sys
import time

from runtime_speed_common import *


def verify(folder):
    out=folder/'verification';out.mkdir(exist_ok=False)
    summary=load_json(folder/'complete_chain/summary.json')
    if not summary['complete'] or summary['trial_count']!=12:raise ValueError('Timing matrix incomplete')
    cfg=load_json(STAGE/'当前采用配置.json')
    code_checks={}
    plans=[load_json(folder/'execution_plan.json')['t3_code_sha256'],
           load_json(folder/'raft/plan.json')['code_sha256'],
           load_json(folder/'complete_chain/plan.json')['code_sha256']]
    for hashes in plans:
        for name,digest in hashes.items():
            path=Path(name) if Path(name).is_absolute() else STAGE/name
            code_checks[str(path)]=file_sha(path)==digest
    for name,digest in load_json(BASE/'base_receive/result.json')['code_sha256'].items():
        code_checks[str(REPO/name)]=file_sha(REPO/name)==digest
    if not all(code_checks.values()):raise ValueError('Executed or historical frozen code changed')
    check_inputs(load_json(folder/'execution_plan.json')['fixed_input_sha256'])
    # One externally timed smoke run exercises the usable single-receive CLI.
    smoke=folder/'fast_entry_smoke'
    command=[sys.executable,'-B','-u','-X','utf8',str(STAGE/'receive_current_i12_fast.py'),
        '--base',str(BASE/'base.glcsnd'),'--base-receipt',str(BASE/'base_receive/result.json'),
        '--raw',str(BASE/'base_receive/decoded_rgb.npy'),'--guide',str(PACKET),'--output',str(smoke)]
    shutil.copy2(STAGE/'receive_current_i12_fast.py',out/'receive_current_i12_fast.py')
    begin=time.perf_counter()
    with (out/'entry_smoke.log').open('x',encoding='utf-8') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,text=True,encoding='utf-8',cwd=REPO,check=True)
    external=time.perf_counter()-begin
    targets=sorted((folder/'complete_chain').glob('*/result.json'))
    artifact_checks=[];media_checks=[]
    probe=FFMPEG.with_name('ffprobe.exe')
    for result_path in targets:
        trial=result_path.parent;record=load_json(result_path)
        if not record['complete']:raise ValueError('Incomplete saved trial')
        checks={
            'base':file_sha(trial/'send/base.glcsnd')==cfg['artifacts']['base']['sha256'],
            'guide':file_sha(trial/'guide/guide.glcguide')==cfg['artifacts']['guide']['sha256'],
            'base_RGB':file_sha(trial/'receive/decoded_rgb.npy')==load_json(BASE/'base_receive/result.json')['decoded_rgb_sha256']}
        checks.update({name:file_sha(trial/'post'/(name+'.npy'))==cfg['artifacts'][name]['sha256'] for name in ('rgb','alpha','shared')})
        if not all(checks.values()):raise ValueError('Saved trial bytes changed: '+trial.name)
        for role in ('receiver','post'):
            receipt=load_json(trial/('receive' if role=='receiver' else 'post')/'receipt.json')
            if not receipt['source_reads_blocked'] or len(receipt['blocked'])!=7:raise ValueError('Source read guard not exercised')
        artifact_checks.append(dict(trial=trial.name,checks=checks))
    smoke_checks={name:file_sha(smoke/(name+'.npy'))==cfg['artifacts'][name]['sha256'] for name in ('rgb','alpha','shared')}
    if not all(smoke_checks.values()):raise ValueError('Explicit receiver entry changed pixels/weights')
    for path in [*(p.parent/'post/video.mp4' for p in targets),smoke/'video.mp4']:
        r=subprocess.run([str(probe),'-v','error','-select_streams','v:0','-show_entries',
            'stream=width,height,nb_frames,r_frame_rate:format=duration','-of','json',str(path)],
            capture_output=True,text=True,check=True)
        import json
        info=json.loads(r.stdout);video=info['streams'][0]
        if (video['width'],video['height'],video['nb_frames'],video['r_frame_rate'])!=(512,288,'150','15/1') or float(info['format']['duration'])!=10:
            raise ValueError('Viewing video geometry/timebase changed')
        media_checks.append(dict(path=str(path),verified=True,info=info))
    syntax={}
    for name in ('cached_t3.py','runtime_speed_common.py','raft_inference_runtime.py','run_t3_cache_benchmark.py',
        'run_raft_speed_benchmark.py','runtime_chain_worker.py','run_complete_runtime_benchmark.py','receive_current_i12_fast.py'):
        ast.parse((STAGE/name).read_text(encoding='utf-8'));syntax[name]=True
    check_inputs(load_json(folder/'execution_plan.json')['fixed_input_sha256'])
    save_json(out/'summary.json',dict(complete=True,timed_trials=12,warmup_trials=1,
        artifact_checks=artifact_checks,all_artifacts_match=True,media_checks=media_checks,
        frozen_code_checks=code_checks,syntax_checks=syntax,original_inputs_unchanged=True,
        smoke=dict(complete=True,external_process_wall_seconds=external,checks=smoke_checks,
                   entry_sha256=file_sha(STAGE/'receive_current_i12_fast.py'),command=command),
        no_generalization_run=True))
    print('VERIFICATION COMPLETE: 12 timed chains, warmup, explicit entry, 14 videos; all exact.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();verify(a.run.resolve())
