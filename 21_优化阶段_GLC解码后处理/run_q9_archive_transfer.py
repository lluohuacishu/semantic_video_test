"""Two preregistered historical marine sources, fixed Q9_R25 without tuning."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from runtime_speed_common import STAGE,REPO,FFMPEG,WEIGHT,file_sha,load_json,save_json
from run_complete_runtime_benchmark import Service
from run_generalization_extra import encode_packet
from mild_reliability_gate import mild_gate
from receiver_structure import render
import structure_guide_stream as guide


def prepare(parent):
    run=parent/'archive_transfer';run.mkdir(exist_ok=False)
    inventory=load_json(parent/'archived_source_inventory.json')
    selected=[('reef_spotted_fish',inventory[2]),('seagrass_spotted_fish',inventory[3])]
    cases={}
    for name,item in selected:
        assert file_sha(item['path'])==item['sha256']
        cases[name]=dict(video=item['path'],video_sha256=item['sha256'],probe=item['probe'],
            selection='Chronological first10s, no output-based selection',fps_conversion='Timestamp fps=15 from 30000/1001; no artificial slow playback',
            source=str(run/name/'source.npy'),frames=150)
    code={name:file_sha(STAGE/name) for name in ['run_q9_archive_transfer.py','run_generalization_extra.py','run_complete_runtime_benchmark.py',
        'runtime_chain_worker.py','raft_inference_runtime.py','mild_reliability_gate.py','receiver_structure.py',
        'cached_t3.py','receive_guide_texture.py','structure_guide_stream.py','evaluate_fullframe_reference.py']}
    save_json(run/'registration.json',dict(date='2026-11-12',registered_before_extraction_and_inference=True,cases=cases,
        profile='Q9_R25',no_parameter_tuning=True,scope='Additional archived source videos, not a statistically independent heldout dataset',
        code_sha256=code,parent_plan_sha256=file_sha(parent/'execution_plan.json')))
    snap=run/'source_snapshot';snap.mkdir()
    for name in code:shutil.copy2(STAGE/name,snap/name)
    for name,spec in cases.items():
        out=run/name;out.mkdir()
        result=subprocess.run([str(FFMPEG),'-v','error','-i',spec['video'],'-map','0:v:0','-an',
            '-vf','fps=15,scale=512:288:flags=area','-frames:v','150','-pix_fmt','rgb24','-f','rawvideo','pipe:1'],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
        assert len(result.stdout)==150*288*512*3
        source=np.frombuffer(result.stdout,np.uint8).reshape(150,288,512,3)
        np.save(spec['source'],source,allow_pickle=False)
        save_json(out/'source_receipt.json',dict(source_sha256=file_sha(spec['source']),video_sha256=file_sha(spec['video']),
            frames=150,fps=15,duration_seconds=10,distinct_adjacent_frames=int(np.any(source[1:]!=source[:-1],axis=(1,2,3)).sum())))
    print('ARCHIVE REGISTERED',list(cases),flush=True)


def receive(run,case):
    spec=load_json(run/'registration.json')['cases'][case]
    root=run/case;out=root/'post';base=root/'send/base.glcsnd';raw_path=root/'base_receive/decoded_rgb.npy'
    raw_receipt=root/'base_receive/receipt.json';packet=root/'guide/guide.glcguide'
    import torch
    from runtime_chain_worker import ReadGuard
    from raft_inference_runtime import load_model,estimate
    guard=ReadGuard('archive-Q9');allowed=[base,raw_path,raw_receipt,packet,WEIGHT];guard.authorize(out,allowed)
    blocked=[]
    for path in [Path(spec['source']),Path(spec['video']),root/'source_receipt.json',root/'guide/send.json']:
        try:path.read_bytes()
        except RuntimeError:blocked.append(str(path))
        else:raise ValueError('Receiver read boundary failed')
    out.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    receipt=load_json(raw_receipt);base_sha=file_sha(base)
    assert receipt['stream_sha256']==base_sha and receipt['reference_verified'] and receipt['source_media_reads_forbidden']
    assert file_sha(raw_path)==receipt['decoded_rgb_sha256']
    raw=np.load(raw_path,mmap_mode='r')
    header,_,images=guide.decode(packet.read_bytes(),base_sha,FFMPEG)
    model=load_model();flows,flow_info=estimate(model,images,'feature_cache')
    del model;torch.cuda.empty_cache()
    for name,array in zip(('to_anchor','from_anchor','transition_flow'),flows):np.save(out/(name+'.npy'),array)
    np.save(out/'guide_rgb.npy',images)
    tick=time.perf_counter();values=render(raw,images,*flows,gate_fn=mild_gate(images,raw,'R25'));elapsed=time.perf_counter()-tick
    for name,array in values.items():np.save(out/(name+'.npy'),array)
    save_json(out/'receipt.json',dict(complete=True,profile='Q9_R25',base_sha256=base_sha,guide_sha256=file_sha(packet),
        guide_header=header,flow=flow_info,render_seconds=elapsed,source_reads_blocked=True,blocked=blocked,
        reads=sorted(guard.reads),allowed=[str(p.resolve()) for p in allowed],no_per_clip_parameters=True,
        outputs_sha256={name:file_sha(out/(name+'.npy')) for name in values},
        local_cache_sha256={name:file_sha(out/(name+'.npy')) for name in ('guide_rgb','to_anchor','from_anchor','transition_flow')},
        code_sha256=file_sha(__file__)))


def evaluate(run,case):
    import evaluate_fullframe_reference as scoring
    from evaluate_structure_guide import motion_change_diagnostic
    from compare_deflicker import save_csv
    from run_p0 import preview
    from verify_video1_persistent_structure import inspect_preview
    from finish_generalization_factorial import sample_sheet
    root=run/case;out=root/'evaluation';out.mkdir(exist_ok=False)
    rec=load_json(root/'post/receipt.json')
    for name,h in rec['outputs_sha256'].items():assert file_sha(root/'post'/(name+'.npy'))==h
    seq=dict(source=np.load(root/'source.npy',mmap_mode='r'),raw=np.load(root/'base_receive/decoded_rgb.npy',mmap_mode='r'),
             Q9_R25=np.load(root/'post/rgb.npy',mmap_mode='r'))
    a=np.load(root/'post/alpha.npy',mmap_mode='r')
    assert np.array_equal(seq['Q9_R25'][a==0],seq['raw'][a==0])
    scoring.SCOPES.pop('reef',None)
    scores,temporal,spatial=scoring.score(seq)
    dynamic,rows=motion_change_diagnostic(seq)
    save_csv(out/'temporal.csv',temporal);save_csv(out/'spatial.csv',spatial);save_csv(out/'dynamic.csv',rows)
    packet=load_json(root/'guide/send.json');base_bytes=(root/'send/base.glcsnd').stat().st_size
    panels=[seq['source'],seq['raw'],seq['Q9_R25']]
    # A blank fourth panel is explicit in the preview helper's unused canvas.
    path=out/'source_raw_Q9.mp4';preview(FFMPEG,path,panels,15,0,['Source','Original GLC',f'Q9_R25 / {packet["total_kbps"]:.2f} kbps'])
    verified=inspect_preview(FFMPEG,path,panels)
    path2=out/'raw_vs_Q9.mp4';preview(FFMPEG,path2,panels[1:],15,0,['Original GLC',f'Q9_R25 / {packet["total_kbps"]:.2f} kbps'])
    verified2=inspect_preview(FFMPEG,path2,panels[1:])
    sample_sheet(out/'contact.png',case,seq,dict(source='原片',raw='原GLC',Q9_R25='固定Q9_R25'),(20,64,80,128,149))
    save_json(out/'summary.json',dict(complete=True,case=case,profile='Q9_R25',scores=scores,dynamic=dynamic,
        base_bytes=base_bytes,base_kbps=base_bytes*8/10000,guide_bytes=packet['bytes'],guide_kbps=packet['kbps'],
        total_kbps=packet['total_kbps'],per_channel_budget_pass=base_bytes<=50000 and packet['bytes']<=50000,
        zero_alpha_unchanged=True,source_scope='Additional archived source, first10s; not independent heldout',
        subjective_review='pending',previews={path.name:verified,path2.name:verified2}))


def run_all(parent):
    run=parent/'archive_transfer'
    registration=load_json(run/'registration.json')
    assert all(file_sha(STAGE/name)==h for name,h in registration['code_sha256'].items())
    # Avoid overlapping GPU experiments with the long receiver.
    while not (parent/'receive/receipt.json').exists():
        if (parent/'failure_receive.json').exists():raise RuntimeError('Parent receiver failed; do not overlap a retry')
        time.sleep(1)
    results={}
    for case,spec in registration['cases'].items():
        root=run/case
        for role,folder in [('sender','send'),('receiver','base_receive')]:
            logs=root/(role+'_logs');logs.mkdir()
            service=Service(role,logs)
            try:
                request=dict(output=str(root/folder),source=spec['source'],source_sha256=file_sha(spec['source'])) if role=='sender' else dict(output=str(root/folder),base=str(root/'send/base.glcsnd'))
                service.request(request)
            finally:service.close()
        encode_packet(np.load(spec['source'],mmap_mode='r'),root/'send/base.glcsnd',root/'guide',9)
        with (root/'worker.log').open('x',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-B','-u','-X','utf8',str(Path(__file__).resolve()),'worker','--run',str(parent),'--case',case],
                stdout=log,stderr=subprocess.STDOUT,check=True,cwd=REPO)
        evaluate(run,case)
        results[case]=load_json(root/'evaluation/summary.json')
        print('ARCHIVE COMPLETE',case,'rate',results[case]['total_kbps'],'mae',
              {k:v['coarse_mae'] for k,v in results[case]['scores']['full'].items()},flush=True)
    save_json(run/'summary.json',dict(complete=True,cases=results,no_per_clip_tuning=True,subjective_review='pending'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run','worker'))
    p.add_argument('--run',type=Path,required=True);p.add_argument('--case');a=p.parse_args()
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    if a.action=='prepare':prepare(a.run)
    elif a.action=='worker':receive(a.run/'archive_transfer',a.case)
    else:run_all(a.run)
