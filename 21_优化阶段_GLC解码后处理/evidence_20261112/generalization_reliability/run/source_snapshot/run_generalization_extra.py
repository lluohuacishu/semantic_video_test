"""A pre-registered extra marine scene, same fixed factorial with no retuning.

The scene registry is injected into the unchanged receiver controller; all
arithmetic, thresholds and profiles are exactly those frozen before video1/2/3.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from runtime_speed_common import STAGE,REPO,FFMPEG,WEIGHT,load_json,save_json,file_sha
from run_complete_runtime_benchmark import Service
import run_generalization_factorial as factorial
import structure_guide_stream as guide

CASE='archive82_first10s'


def encode_packet(source,base,out,rate):
    out.mkdir(exist_ok=False);tick=time.perf_counter()
    low=np.stack([cv2.resize(v,(256,144),interpolation=cv2.INTER_AREA) for v in source])
    common=[str(FFMPEG),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','256x144','-r','15','-i','pipe:0',
        '-an','-c:v','libx265','-preset','slower','-b:v',str(rate)+'k','-pix_fmt','yuv420p','-threads','2']
    params='log-level=error:pools=2:frame-threads=1:keyint=150:min-keyint=150:scenecut=0:repeat-headers=1:info=0:stats=guide_pass.log'
    for n in (1,2):
        p=subprocess.run([*common,'-x265-params',params+f':pass={n}','-f','hevc','pipe:1'],input=low.tobytes(),
            stdout=subprocess.PIPE if n==2 else subprocess.DEVNULL,stderr=subprocess.PIPE,check=True,cwd=out)
        (out/f'pass{n}.log').write_bytes(p.stderr)
    packet,header=guide.pack(p.stdout,file_sha(base),(256,144));guide.unpack(packet,file_sha(base))
    path=out/'guide.glcguide';path.write_bytes(packet)
    save_json(out/'send.json',dict(complete=True,bytes=len(packet),kbps=len(packet)*8/10000,
        total_kbps=(len(packet)+base.stat().st_size)*8/10000,sha256=file_sha(path),header=header,
        encode_seconds=time.perf_counter()-tick,rate_selection='Fixed internal target; no byte/quality search'))


def run_extra(parent):
    registration=load_json(parent/'registered_extra_source/registration.json')
    source_path=parent/'registered_extra_source/source.npy';assert file_sha(source_path)==registration['source_npy_sha256']
    run=parent/'extra_scene';run.mkdir(exist_ok=False);root=run/CASE;root.mkdir()
    plan=load_json(parent/'execution_plan.json')
    assert file_sha(STAGE/'run_generalization_factorial.py')==plan['code_sha256']['run_generalization_factorial.py']
    save_json(run/'execution_plan.json',dict(date='2026-11-12',frozen_before_extra_inference=True,case=CASE,
        source_registration=registration,profiles=list(factorial.VARIANTS),parent_plan_sha256=file_sha(parent/'execution_plan.json'),
        code_sha256=file_sha(__file__),receiver_code_sha256=file_sha(STAGE/'run_generalization_factorial.py'),
        scope='Historical marine clip outside video1/2/3; extra fixed-rule stress test, not independent unseen data or full-length test',
        no_parameter_tuning=True,selected_before_quality_results=True))
    services=[]
    try:
        for role in ('sender','receiver'):
            logs=root/(role+'_logs');logs.mkdir();service=Service(role,logs);services.append(service)
            if role=='sender':request=dict(source=str(source_path),source_sha256=file_sha(source_path),output=str(root/'send'))
            else:request=dict(base=str(root/'send/base.glcsnd'),output=str(root/'receive'))
            record=service.request(request);service.close();services.remove(service)
            print('EXTRA',role,'complete',flush=True)
        raw=root/'receive/decoded_rgb.npy';base=root/'send/base.glcsnd'
        assert base.stat().st_size<=50000
        spec={CASE:dict(source=str(source_path),base=str(base),raw=str(raw),base_receipt=str(root/'receive/receipt.json'))}
        save_json(run/'receiver_input_registry.json',spec)
        source=np.load(source_path,mmap_mode='r',allow_pickle=False)
        for quality,rate in (('Q6',6),('Q12',12)):
            encode_packet(source,base,root/quality,rate)
            with (root/quality/'worker.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-B','-u','-X','utf8',str(Path(__file__).resolve()),'worker','--run',str(run),
                    '--quality',quality],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=REPO)
            print('EXTRA',quality,'three variants complete',flush=True)
    finally:
        for service in services:service.close()
    # Additional replay through the original standalone I12 code path.
    logs=root/'original_replay_logs';logs.mkdir();post=Service('post',logs)
    try:
        replay=post.request(dict(base=str(base),raw=str(raw),base_receipt=str(root/'receive/receipt.json'),
            guide=str(root/'Q6/guide.glcguide'),output=str(root/'original_i12'),raft_mode='original',cached_t3=False))
    finally:post.close()
    baseline=load_json(root/'Q6/receive/receipt.json')['variants']['T3']['outputs_sha256']
    assert replay['outputs_sha256']==baseline
    save_json(run/'chain_verification.json',dict(complete=True,original_I12_baseline_exact=True,
        source_sha256=file_sha(source_path),base_sha256=file_sha(base),base_bytes=base.stat().st_size,
        base_reference_verified=load_json(root/'receive/receipt.json')['reference_verified']))
    evaluate(run)


def evaluate(run):
    import evaluate_fullframe_reference as scoring
    from evaluate_structure_guide import motion_change_diagnostic
    from compare_deflicker import save_csv
    from run_p0 import preview
    from verify_video1_persistent_structure import inspect_preview
    r=load_json(run/'receiver_input_registry.json')[CASE];root=run/CASE;out=root/'evaluation';out.mkdir(exist_ok=False)
    seq=dict(source=np.load(r['source'],mmap_mode='r'),raw=np.load(r['raw'],mmap_mode='r'));rates={};verifications={}
    for quality in ('Q6','Q12'):
        rec=load_json(root/quality/'receive/receipt.json');send=load_json(root/quality/'send.json')
        assert rec['source_reads_blocked'] and len(rec['blocked'])==5 and rec['auxiliary_sha256']==send['sha256']
        baseline=np.load(root/quality/'receive/T3/alpha.npy',mmap_mode='r')
        for mode in ('T3','F8','C8'):
            name=quality+'_'+mode;folder=root/quality/'receive'/mode
            for field,h in rec['variants'][mode]['outputs_sha256'].items():assert file_sha(folder/(field+'.npy'))==h
            seq[name]=np.load(folder/'rgb.npy',mmap_mode='r');alpha=np.load(folder/'alpha.npy',mmap_mode='r')
            assert np.array_equal(seq[name][alpha==0],seq['raw'][alpha==0])
            assert np.all((alpha>=0)&(alpha<=baseline+1e-7))
            rates[name]=send['total_kbps'];verifications[name]=dict(hashes_valid=True,zero_alpha_unchanged=True,within_base_support=True)
    scoring.SCOPES.pop('reef',None);scores,temporal,spatial=scoring.score(seq)
    save_csv(out/'temporal.csv',temporal);save_csv(out/'spatial.csv',spatial)
    dynamic,rows=motion_change_diagnostic(seq);save_csv(out/'dynamic.csv',rows)
    parent=load_json(run.parent/'summary.json');chosen=parent['selected'] or parent['best_video2_diagnostic']
    names=('source','raw','Q6_T3',chosen);panels=[seq[n] for n in names]
    labels=['Archive source','Original GLC',f'Current / {rates["Q6_T3"]:.2f} kbps',f'{chosen} / {rates[chosen]:.2f} kbps']
    path=out/'archive82_source_raw_current_candidate.mp4';preview(FFMPEG,path,panels,15,0,labels)
    view=inspect_preview(FFMPEG,path,panels)
    path2=out/'archive82_current_vs_candidate.mp4';preview(FFMPEG,path2,panels[2:],15,0,labels[2:])
    view2=inspect_preview(FFMPEG,path2,panels[2:])
    save_json(run/'summary.json',dict(complete=True,case=CASE,scores=scores,rates=rates,dynamic=dynamic,
        verified_variants=verifications,viewing_candidate=chosen,previews={path.name:view,path2.name:view2},
        subjective_review='pending',scope='Additional historical scene, frozen before original matrix scores; not unseen holdout or long-video validation'))
    print('EXTRA COMPLETE',json.dumps({n:v['coarse_mae'] for n,v in scores['full'].items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('run','worker'));p.add_argument('--run',type=Path,required=True)
    p.add_argument('--quality',choices=('Q6','Q12'));a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    if a.action=='worker':
        registry=load_json(a.run/'receiver_input_registry.json')
        factorial.cases=lambda:registry
        factorial.receive(a.run.resolve(),CASE,a.quality)
    else:run_extra(a.run.resolve())
