"""Frozen full-sequence exact replay and alternating T3 cache timing."""
import argparse
import shutil
import time

import cv2
import numpy as np

from runtime_speed_common import *
from run_anchor_time_experiment import render_profile


def run(folder):
    folder.mkdir(parents=True, exist_ok=False)
    out = folder/'t3'
    out.mkdir()
    manifest = fixed_inputs()
    for name in ('to_anchor.npy','from_anchor.npy','transition_flow.npy'):
        path = ACCEPTED/'flow'/name
        manifest[str(path)] = file_sha(path)
    inputs = [BASE/'base_receive/decoded_rgb.npy', ACCEPTED/'receive/guide_rgb.npy']
    raw, images = [np.load(p, mmap_mode='r', allow_pickle=False) for p in inputs]
    flow = [np.load(ACCEPTED/'flow'/name, mmap_mode='r') for name in ('to_anchor.npy','from_anchor.npy','transition_flow.npy')]
    expected = {n:np.load(ACCEPTED/'receive'/(n+'.npy'), mmap_mode='r') for n in ('rgb','alpha','shared')}
    manifest[str(inputs[1])] = file_sha(inputs[1])
    order = ['original','cached','cached','original','original','cached']
    code = {name:file_sha(STAGE/name) for name in ('run_t3_cache_benchmark.py','runtime_speed_common.py','cached_t3.py',
            'run_anchor_time_experiment.py','receive_guide_texture.py','receiver_structure.py')}
    save_json(folder/'execution_plan.json', dict(date='2026-11-11', frozen_before_trials=True,
        scope='video1 S8I CURRENT_I12; no generalization or long-video experiments this round',
        phases=['exact T3 cache','isolated RAFT inference optimization','cold/warm complete local file chain'],
        user_authorization='Complete this timing/optimization round; generalization to other marine and long videos is the subsequent direction.',
        fixed_input_sha256=manifest, fixed_keys=list(KEYS), raft_iterations=12,
        t3_order=order, t3_code_sha256=code,
        exact_rule='All full-sequence RGB/alpha/shared arrays equal accepted outputs, not only metrics',
        memory_policy='Four-entry LRU RGB blur cache, not an all-frame RGB blur tensor',
        next_phase='Generalization, independent marine clips and long sequences, deferred'))
    snapshot=folder/'source_t3';snapshot.mkdir()
    for name in code:shutil.copy2(STAGE/name,snapshot/name)
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    rows=[]
    for index, mode in enumerate(order):
        tick=time.perf_counter()
        values=render_profile(raw,images,*flow,KEYS) if mode=='original' else render(raw,images,*flow,cached=True)
        elapsed=time.perf_counter()-tick
        exact={name:bool(np.array_equal(value,expected[name])) for name,value in values.items()}
        rows.append(dict(index=index,mode=mode,seconds=elapsed,exact=exact))
        save_json(out/'trials.json',rows)
        if not all(exact.values()):raise ValueError('T3 exact replay failed')
        if mode=='cached' and index==1:
            for name,value in values.items():np.save(out/(name+'.npy'),value,allow_pickle=False)
        del values
        print('T3',index,mode,round(elapsed,4),'all_exact',all(exact.values()),flush=True)
    times={mode:stats([r['seconds'] for r in rows if r['mode']==mode]) for mode in ('original','cached')}
    check_inputs(manifest)
    save_json(out/'summary.json',dict(complete=True,times=times,
        median_time_reduction_percent=(1-times['cached']['median']/times['original']['median'])*100,
        full_rgb_alpha_shared_bit_exact=True,original_and_inputs_unchanged=True,
        timing_scope='Entire render call, including texture std precomputation, lazy blur cache, blend, arrays and zero-weight check; excludes source file reads and output writes',
        extra_RGB_blur_cache_limit_frames=4,
        extra_RGB_blur_cache_payload_bytes=4*2*288*512*3*4,
        outputs_sha256={p.name:file_sha(p) for p in out.glob('*.npy')}))
    print('T3 COMPLETE',times,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();run(a.run.resolve())
