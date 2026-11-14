"""Fixed Q9/GS25 transfer on the entire short video4; no scene-specific tuning."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
import cv2
import numpy as np
from runtime_speed_common import STAGE,REPO,DATA,FFMPEG,WEIGHT,file_sha,load_json,save_json,check_inputs
from receiver_structure import KEYS,EVENTS,render
from mild_reliability_gate import mild_gate
from received_structure_fusion import fuse_received_structure
import q9_chunk_protocol as protocol

PRIOR=DATA/'q9_received_structure_20261113_01'
CONFIG='b5eeb844d0adcdfe97648351a0654a2575a98942e1627c98896ba302555f7b04'
CODE=('run_video4_diagnosis.py','variable_clip_base_worker.py','runtime_chain_worker.py',
      'runtime_speed_common.py','q9_chunk_protocol.py','raft_inference_runtime.py',
      'mild_reliability_gate.py','cached_t3.py','receive_guide_texture.py','receiver_structure.py',
      'received_structure_fusion.py')
EXTERNAL=('14_优化阶段_部署精简/role_codec.py','13_优化阶段_发送端免RGB重建/sender_container.py',
          '16_优化阶段_输入编码器推理加速/experiment_common.py')

def freeze(run):
    assert not (run/'execution_plan.json').exists()
    reg=load_json(run/'case_registration.json')
    assert reg['frames']==104
    assert file_sha(reg['source_video'])==reg['source_video_sha256']
    assert file_sha(reg['source_npy'])==reg['source_npy_sha256']
    assert file_sha(STAGE/'当前采用配置.json')==CONFIG
    previous=load_json(PRIOR/'case_registration.json')['cases']
    inputs={str(p):file_sha(p) for p in [Path(reg['source_video']),Path(reg['source_npy']),
            PRIOR/'execution_plan.json',PRIOR/'summary.json',PRIOR/'verification.json',WEIGHT]}
    for case,s in previous.items():
        for name in ('source','raw','base'):
            inputs[s[name]]=file_sha(s[name])
        for name in ('guide_rgb','to_anchor','from_anchor','transition_flow'):
            p=Path(s['cache'])/(name+'.npy');inputs[str(p)]=file_sha(p)
        for name in ('rgb','alpha','shared'):
            p=Path(s['baseline'])/(name+'.npy');inputs[str(p)]=file_sha(p)
        p=PRIOR/case/'receive/GS25/rgb.npy';inputs[str(p)]=file_sha(p)
    p=dict(date='2026-11-13',frozen_before_new_inference=True,case='video4',
        registration_sha256=file_sha(run/'case_registration.json'),
        authorization='好了，我把video4整过去了，你可以开始下一轮了',
        frames=104,fps=15,billing_seconds=104/15,source_video_seconds=6.9,
        selection='Whole user-supplied dense-coral clip; 30->15 FPS by timestamps, no duplicate padding',
        methods=['raw','guide','Q9_R25','GS25'],new_candidate_outputs=2,
        reference='Fixed Q9_R25; GS25 candidate from the completed three-clip round',
        base=dict(q=0,intra_period=64,reset_interval=32,arithmetic='Same independent eager FP32 sender/receiver worker; frame count generalized only'),
        guide=dict(internal_kbps=9,size=[256,144],fps=15,preset='slower',two_pass=True,
                   keyint=150,scenecut=0,container='Existing GLCGDE02 variable-length format; one self-describing packet, no separate transmitted index needed'),
        fixed=dict(anchors=[k for k in KEYS if k<104],transition_frames=6,raft_iterations=12,precision='FP32',TF32=False,
                   gate='unchanged R25',gs25_strength=.25,gs25_sigma=1,no_per_clip_tuning=True),
        regression=dict(base_video1_150='Re-encode and independently decode with length-only worker; compare frozen base/RGB hashes.',
            q9_video1_prefix104='Recompute receiver flow on guide prefix104 and compare original prefix flows; render exact prefix.',
            all_three_prefix104='Replay unchanged Q9 and GS25 on old cached prefixes; compare original 104 frames bit-for-bit.'),
        regions=dict(coral_lower=[0,128,512,288],water_upper=[0,0,512,96],turtle_wide=[50,150,410,288]),
        region_note='Source-only pre-inference visual registration. Wide rectangles include background; not masks or object recovery percentages.',
        review_frames=[0,26,52,78,103],review_intervals=[[0,104]],
        evaluation='Full, four quadrants and fixed regions; same source-aligned bands/support, N-1 pairs, dynamic amplitude, RGB error and PSNR. Anchors and GLC reset/I-frame events reported separately without causal claims.',
        causal_scope='New independent FP32 encoding of selected short clip; not identical to historical full-video TRT stream/timeline. Do not attribute old-long vs new-short differences to Q9 or GS25.',
        holds_out=False,development_failure_case=True,scene_cut_detector=False,
        automatic_selection=None,subjective_review='pending',adoption_unchanged=True,
        current_config_sha256=CONFIG,video123_feedback_preserved=True,
        input_sha256=inputs,code_sha256={n:file_sha(STAGE/n) for n in CODE},
        external_code_sha256={n:file_sha(REPO/n) for n in EXTERNAL})
    save_json(run/'execution_plan.json',p)
    snap=run/'source_snapshot';snap.mkdir()
    for n in CODE:shutil.copy2(STAGE/n,snap/n)
    for n in EXTERNAL:
        dst=snap/'repository'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/n,dst)
    statepath=REPO/'05_资料与索引/会话收束_20261110/接续状态.json'
    shutil.copy2(statepath,run/'state_before_run.json')
    state=load_json(statepath)
    state.update(status='Q9_video4_fixed_diagnosis_running',active_experiment_plan=str(run/'execution_plan.json'),
        current_session_work='Registered whole 6.9s video4; 104 frames at 15 FPS. Fixed Q9/GS25, length-adapter regressions; no scene tuning.')
    state['experiment_execution_gate']=dict(status='running',currently_running=True,new_run_started=True,date='2026-11-13')
    save_json(statepath,state)
    print('VIDEO4 PLAN FROZEN: whole 104 frames, Q9 and GS25',flush=True)

def base_job(root,role,source=None):
    root.mkdir(parents=True,exist_ok=True)
    request=dict(output=str(root/('send' if role=='sender' else 'base_receive')))
    if role=='sender':
        request.update(source=str(source),source_sha256=file_sha(source))
    else:
        request.update(base=str(root/'send/base.glcsnd'),forbidden_source=str(source))
    req=root/(role+'_request.json');save_json(req,request)
    with (root/(role+'.log')).open('x',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-B','-u','-X','utf8',str(STAGE/'variable_clip_base_worker.py'),
            '--role',role,'--request',str(req)],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=REPO)

def pair_schedule(n):
    assert 0<n<=150
    keys=[k for k in KEYS if k<n]
    pairs=[(i,k) for k,stop in zip(keys,keys[1:]+[n]) for i in range(k+1,stop)]
    events=[(j,t,KEYS[j//6]) for j,t in enumerate(EVENTS) if t<n]
    return pairs,events

def estimate_short(model,images):
    import torch
    from raft_inference_runtime import FlowRunner
    n=len(images);pairs,events=pair_schedule(n)
    back=np.zeros((n,288,512,2),np.float32);forward=np.zeros_like(back)
    transition=np.empty((len(events),2,288,512,2),np.float32)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
    with torch.inference_mode():
        runner=FlowRunner(model,images,'feature_cache');runner.prepare()
        def pair(i,k):
            v=runner.pair(i,k).permute(0,2,3,1).contiguous().cpu().numpy()
            assert v.shape==(2,288,512,2) and np.isfinite(v).all()
            return v
        for i,k in pairs:back[i],forward[i]=pair(i,k)
        for j,i,k in events:transition[j]=pair(i,k)
    torch.cuda.synchronize()
    info=dict(seconds=time.perf_counter()-tick,pairs=len(pairs)+len(events),frames=n,
        event_frames=[i for j,i,k in events],peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
    del runner
    return (back,forward,transition),info

def render_short(raw,images,flows):
    values=render(raw,images,*flows,gate_fn=mild_gate(images,raw,'R25'))
    values['shared']=values['shared'][:sum(t<len(raw) for t in EVENTS)]
    return values

def regression_render(run,infer=False):
    old=load_json(PRIOR/'case_registration.json')['cases']
    out=run/'regression';out.mkdir(exist_ok=True);results={}
    model=None
    if infer:
        import torch
        from raft_inference_runtime import load_model
        torch.set_num_threads(1);torch.set_num_interop_threads(1);model=load_model()
    for case,s in old.items():
        q=np.load(Path(s['baseline'])/'rgb.npy',mmap_mode='r')
        raw=np.load(s['raw'],mmap_mode='r')[:104]
        images=np.load(Path(s['cache'])/'guide_rgb.npy',mmap_mode='r')[:104]
        cached=[np.load(Path(s['cache'])/(n+'.npy'),mmap_mode='r') for n in ('to_anchor','from_anchor','transition_flow')]
        flows=[cached[0][:104],cached[1][:104],cached[2][:18]]
        info=None
        if infer and case=='video1':
            fresh,info=estimate_short(model,images)
            for a,b in zip(fresh,flows):assert np.array_equal(a,b),'Length-specific flow regression differs'
            flows=fresh
        values=render_short(raw,images,flows)
        for n,a in values.items():
            expected=np.load(Path(s['baseline'])/(n+'.npy'),mmap_mode='r')
            assert np.array_equal(a,expected[:len(a)]),(case,n)
        gs=np.load(PRIOR/case/'receive/GS25/rgb.npy',mmap_mode='r')
        for i in range(104):assert np.array_equal(fuse_received_structure(values['rgb'][i],images[i],'GS25')[0],gs[i])
        results[case]=dict(Q9_prefix_rgb_alpha_shared_exact=True,GS25_prefix_exact=True,
             first104_frames=True,fresh_flow_prefix_exact=(case=='video1' and infer),flow_info=info)
        print('PREFIX REGRESSION',case,flush=True)
    save_json(out/'prefix_regression.json',dict(complete=True,cases=results,
        source_pixels_not_used_for_render=True,old_files_not_modified=True))

def encode(run):
    source=np.load(run/'source.npy',mmap_mode='r');n=len(source);out=run/'guide';out.mkdir()
    base=run/'video4/send/base.glcsnd'
    low=np.stack([cv2.resize(v,(256,144),interpolation=cv2.INTER_AREA) for v in source])
    common=[str(FFMPEG),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','256x144','-r','15','-i','pipe:0',
        '-an','-c:v','libx265','-preset','slower','-b:v','9k','-pix_fmt','yuv420p','-threads','2']
    params='log-level=error:pools=2:frame-threads=1:keyint=150:min-keyint=150:scenecut=0:repeat-headers=1:info=0:stats=guide_pass.log'
    for number in (1,2):
        p=subprocess.run([*common,'-x265-params',params+f':pass={number}','-f','hevc','pipe:1'],
            input=low.tobytes(),stdout=subprocess.PIPE if number==2 else subprocess.DEVNULL,
            stderr=subprocess.PIPE,check=True,cwd=out)
        (out/f'pass{number}.log').write_bytes(p.stderr)
    packet,header=protocol.pack(p.stdout,file_sha(base),0,n,n)
    path=out/'guide.glcg2';path.write_bytes(packet)
    seconds=n/15;bb=base.stat().st_size
    save_json(out/'send.json',dict(complete=True,base_bytes=bb,guide_bytes=len(packet),index_bytes=0,
        base_kbps=bb*8/seconds/1000,guide_kbps=len(packet)*8/seconds/1000,
        total_kbps=(bb+len(packet))*8/seconds/1000,billing_seconds=seconds,header=header,
        sha256=file_sha(path),base_sha256=file_sha(base),source_sha256=file_sha(run/'source.npy'),
        per_channel_budget_pass=max(bb,len(packet))*8/seconds/1000<=40,
        no_search=True,notes='One self-describing GLCGDE02 packet; no separate network manifest. Full header and CRC counted.'))
    print('VIDEO4 GUIDE ENCODED',flush=True)

def receive(run):
    import torch
    from runtime_chain_worker import ReadGuard
    from raft_inference_runtime import load_model
    plan=load_json(run/'execution_plan.json');reg=load_json(run/'case_registration.json')
    for n,h in plan['code_sha256'].items():assert file_sha(STAGE/n)==h
    out=run/'receive';base=run/'video4/send/base.glcsnd';rawp=run/'video4/base_receive/decoded_rgb.npy'
    recp=run/'video4/base_receive/receipt.json';packet=run/'guide/guide.glcg2'
    guard=ReadGuard('video4-post')
    allowed=[base,rawp,recp,packet,WEIGHT]
    guard.authorize(out,allowed);blocked=[]
    for p in (run/'source.npy',Path(reg['source_video']),run/'guide/send.json',PRIOR/'summary.json',
              PRIOR/'video2/receive/GS25/rgb.npy'):
        try:p.read_bytes()
        except RuntimeError:blocked.append(str(p))
        else:raise AssertionError('Forbidden post input was readable')
    out.mkdir();br=load_json(recp)
    assert br['reference_verified'] and br['source_media_reads_forbidden'] and br['source_reads_blocked']
    assert file_sha(base)==br['stream_sha256'] and file_sha(rawp)==br['decoded_rgb_sha256']
    raw=np.load(rawp,mmap_mode='r')
    h,images=protocol.decode(packet.read_bytes(),file_sha(base),0,plan['frames'],FFMPEG)
    assert raw.shape==images.shape==(104,288,512,3)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);model=load_model()
    flows,info=estimate_short(model,images);del model;torch.cuda.empty_cache()
    for n,a in zip(('to_anchor','from_anchor','transition_flow'),flows):np.save(out/(n+'.npy'),a)
    np.save(out/'guide_rgb.npy',images)
    values=render_short(raw,images,flows);(out/'Q9_R25').mkdir()
    for n,a in values.items():np.save(out/'Q9_R25'/(n+'.npy'),a)
    (out/'GS25').mkdir()
    gs=np.empty_like(values['rgb']);stats=[]
    for i in range(len(raw)):gs[i],s=fuse_received_structure(values['rgb'][i],images[i],'GS25');stats.append(s)
    np.save(out/'GS25/rgb.npy',gs)
    save_json(out/'receipt.json',dict(complete=True,base_sha256=file_sha(base),guide_sha256=file_sha(packet),
        raw_sha256=file_sha(rawp),guide_header=h,frames=len(raw),flow=info,
        source_reads_blocked=True,blocked=blocked,reads=sorted(guard.reads),
        allowed=sorted(str(p.resolve()) for p in allowed),
        Q9_sha256={n:file_sha(out/'Q9_R25'/(n+'.npy')) for n in values},
        GS25_sha256=file_sha(out/'GS25/rgb.npy'),
        cache_sha256={n:file_sha(out/(n+'.npy')) for n in ('guide_rgb','to_anchor','from_anchor','transition_flow')},
        clipped_channel_fraction=float(np.mean([s['clipped_channel_fraction'] for s in stats])),
        mean_abs_change=float(np.mean([s['mean_abs_change'] for s in stats])),
        Q9_raw_only=True,GS25_received_guide_pixels=True,extra_GS25_transmission_bytes=0,
        automatic_adoption=False))
    print('VIDEO4 Q9 AND GS25 COMPLETE',flush=True)

def all_steps(run):
    plan=load_json(run/'execution_plan.json');check_inputs(plan['input_sha256'])
    prior=load_json(PRIOR/'case_registration.json')['cases']['video1']
    regroot=run/'regression/video1_base150'
    for role in ('sender','receiver'):
        base_job(regroot,role,Path(prior['source']))
        print('BASE150 REGRESSION',role,flush=True)
    assert file_sha(regroot/'send/base.glcsnd')==file_sha(prior['base'])
    assert file_sha(regroot/'base_receive/decoded_rgb.npy')==file_sha(prior['raw'])
    save_json(run/'regression/base150_verification.json',dict(complete=True,base_packet_exact=True,
        independent_raw_exact=True,source_sha256=file_sha(prior['source'])))
    subprocess.run([sys.executable,'-B','-u','-X','utf8',str(Path(__file__).resolve()),'regression','--run',str(run)],check=True,cwd=REPO)
    for role in ('sender','receiver'):
        base_job(run/'video4',role,run/'source.npy')
        print('VIDEO4 BASE',role,flush=True)
    encode(run)
    subprocess.run([sys.executable,'-B','-u','-X','utf8',str(Path(__file__).resolve()),'receive','--run',str(run)],check=True,cwd=REPO)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('freeze','run','regression','receive'));p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    {'freeze':freeze,'run':all_steps,'regression':lambda r:regression_render(r,True),'receive':receive}[a.action](a.run)

