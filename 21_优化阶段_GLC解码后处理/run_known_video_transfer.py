"""Frozen S8I/I12 transfer to the two previously used 10-second marine clips.

New experiment controller only: all receiver arithmetic is imported unchanged.
Source pixels are used by sender/evaluation; independent receivers use packets.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from runtime_speed_common import (STAGE, REPO, WORKSPACE, DATA, FFMPEG, WEIGHT,
    file_sha, load_json, save_json, check_inputs, gpu_snapshot)
from run_complete_runtime_benchmark import Service
from make_current_fourway_preview import match_hevc
from cache_hevc_baseline import X265_PARAMS
import structure_guide_stream as guide
import temporal_filter

SOURCES = WORKSPACE/'01_项目/GLC后处理/素材/20261109_三个片段'
HISTORY = DATA/'source_dense_flow_three_clips_20261109_01'
OLD_K = DATA/'support_refinement_20261111_01'
CASES = {
    'video2': dict(sha256='b7a05eaa1b52f5d45461a1948087505ae7729ecc98a5b39b7e61f33862d7fb5c', reef=[380,156,507,281]),
    'video3': dict(sha256='d1b29b1eb9ba8048e31fb2ad394e67268b5fede6008db6e58916f07dcb062a5d', reef=[145,190,510,286]),
}
CODE = ['run_known_video_transfer.py','runtime_chain_worker.py','runtime_speed_common.py',
    'run_complete_runtime_benchmark.py','raft_inference_runtime.py','cached_t3.py',
    'receiver_structure.py','receive_guide_texture.py','structure_guide_stream.py',
    'make_current_fourway_preview.py','cache_hevc_baseline.py','evaluate_fullframe_reference.py',
    'evaluate_structure_guide.py','audit_deflicker_texture.py','probe_reef.py']


def announce(root, stage, **details):
    print(json.dumps(dict(stage=stage,**details),ensure_ascii=False),flush=True)
    save_json(root/'progress.json',dict(stage=stage,**details))


def prepare(run):
    run.mkdir(parents=True,exist_ok=False)
    records={};frozen={}
    for case,profile in CASES.items():
        video=SOURCES/(case+'.mp4')
        if file_sha(video)!=profile['sha256']:raise ValueError('Historical source mismatch: '+case)
        probe=json.loads(subprocess.check_output([str(FFMPEG.with_name('ffprobe.exe')),'-v','error',
            '-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,duration,nb_frames',
            '-of','json',str(video)]))['streams'][0]
        paths=[video,HISTORY/case/'source.npy',HISTORY/case/'base.glcsnd',
            HISTORY/case/'base_receive/decoded_rgb.npy',HISTORY/case/'base_receive/result.json',
            OLD_K/case/'K/receive/rgb.npy',OLD_K/case/'K/receive/receipt.json',
            OLD_K/case/'K/support.glccsp2',OLD_K/case/'evaluation/summary.json']
        for p in paths:frozen[str(p)]=file_sha(p)
        records[case]=dict(source=str(video),source_sha256=profile['sha256'],source_probe=probe,
            start_seconds=0,frames=150,fps=15,shape=[288,512,3],historical_reef_box=profile['reef'])
    for p in [WEIGHT,STAGE/'当前采用配置.json',Path(temporal_filter.__file__)]:frozen[str(p)]=file_sha(p)
    code={name:file_sha(STAGE/name) for name in CODE}
    snapshot=run/'source_snapshot';snapshot.mkdir()
    for name in CODE:shutil.copy2(STAGE/name,snapshot/name)
    cfg=load_json(STAGE/'当前采用配置.json')
    assert cfg['profile_name']=='S8I + CURRENT_I12' and cfg['anchor_frames']==[0,15,45,75,105,135]
    assert cfg['raft']['iterations']==12
    plan=dict(date='2026-11-11',frozen_before_trials=True,cases=records,
        purpose='Initial transfer to previously used development videos, not unseen holdout or long-video validation',
        profile='S8I + CURRENT_I12 + exact feature/T3 caches',
        source_sampling='First 10 seconds, fps=15, scale=512:288:flags=area; must equal historical source.npy',
        anchor_frames=[0,15,45,75,105,135],raft=dict(iterations=12,precision='FP32',TF32=False,mode='feature_cache'),
        t3='Existing sigma3 std>=3, geometric/photometric confidence, feather=2px, transition=6 frames, two rounding stages',
        guide=dict(size=[256,144],fps=15,preset='slower',target_kbps=6,two_pass=True,pix_fmt='yuv420p',
            parameters='keyint=150:min-keyint=150:scenecut=0:repeat-headers=1:info=0:pools=2:frame-threads=1'),
        base=dict(q=0,intra_period=64,reset_interval=32,backend='eager FP32'),
        tests=['Actual source sampling, new GLC/S8I packets, independent base decoding, current cached postprocess',
            'Original uncached I12 postprocess replay, compare complete RGB/alpha/shared on each clip',
            'Original GLC, historical K, current I12, complete-file byte-matched H265; no parameter search for current algorithm'],
        evaluation=dict(scopes='Full frame, four quadrants, pre-existing reef boxes',
            temporal='Existing source-defined common-support alignment; coarse MAE, fine NCC, per-frame p95 and event windows',
            spatial='RGB PSNR against same sampled source',dynamic='Existing source-selected top-decile RGB change diagnostic',
            limitations='Metrics are diagnostics, not semantic tracking or subjective acceptance; no per-clip tuning'),
        h265=dict(preset='medium',x265_params=X265_PARAMS,search='CRF 28..51 in 1/16 steps, bytes only, <=2% below total bytes; no padding'),
        timing='One cold process chain per clip; descriptive single-run time, not benchmark medians; excludes evaluation and validation replay',
        source_free_receivers=True,input_sha256=frozen,code_sha256=code,
        protected_inference_code_unchanged=True,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        gpu_at_prepare=gpu_snapshot(),user_visual_acceptance='pending')
    save_json(run/'execution_plan.json',plan)
    announce(run,'prepared',cases=list(CASES),plan=str(run/'execution_plan.json'))


def encode_guide(source,base,out):
    out.mkdir(exist_ok=False);start=time.perf_counter()
    low=np.stack([cv2.resize(v,(256,144),interpolation=cv2.INTER_AREA) for v in source])
    common=[str(FFMPEG),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','256x144','-r','15','-i','pipe:0',
        '-an','-c:v','libx265','-preset','slower','-b:v','6k','-pix_fmt','yuv420p','-threads','2']
    params='log-level=error:pools=2:frame-threads=1:keyint=150:min-keyint=150:scenecut=0:repeat-headers=1:info=0:stats=guide_pass.log'
    times=[]
    for number in (1,2):
        tick=time.perf_counter()
        p=subprocess.run([*common,'-x265-params',params+f':pass={number}','-f','hevc','pipe:1'],
            input=low.tobytes(),stdout=subprocess.PIPE if number==2 else subprocess.DEVNULL,
            stderr=subprocess.PIPE,cwd=out,check=True)
        (out/f'pass{number}.log').write_bytes(p.stderr);times.append(time.perf_counter()-tick)
    packet,header=guide.pack(p.stdout,file_sha(base),(256,144))
    guide.unpack(packet,file_sha(base))
    path=out/'guide.glcguide';path.write_bytes(packet)
    result=dict(seconds=time.perf_counter()-start,pass_seconds=times,bytes=len(packet),
        kbps=len(packet)*8/10000,sha256=file_sha(path),header=header)
    save_json(out/'send.json',result);return result


def run_case(run,case):
    plan=load_json(run/'execution_plan.json');check_inputs(plan['input_sha256'])
    assert all(file_sha(STAGE/name)==digest for name,digest in plan['code_sha256'].items())
    out=run/case;out.mkdir(exist_ok=False)
    records={};start=time.perf_counter();active=[]
    def service(role):
        logs=out/(role+'_logs');logs.mkdir();obj=Service(role,logs);active.append(obj);return obj
    def step(name):announce(out,name,case=case,elapsed=time.perf_counter()-start)
    try:
        folder=out/'input';folder.mkdir();tick=time.perf_counter()
        command=[str(FFMPEG),'-hide_banner','-loglevel','error','-nostdin','-ss','0.0','-i',str(SOURCES/(case+'.mp4')),
            '-map','0:v:0','-an','-vf','fps=15,scale=512:288:flags=area,format=rgb24','-frames:v','150','-f','rawvideo','pipe:1']
        pixels=subprocess.check_output(command);source=np.frombuffer(pixels,np.uint8).reshape(150,288,512,3)
        path=folder/'source.npy';np.save(path,source,allow_pickle=False);digest=file_sha(path)
        assert digest==plan['input_sha256'][str(HISTORY/case/'source.npy')]
        records['source']=dict(seconds=time.perf_counter()-tick,sha256=digest,command=command);step('source_sampled')
        sender=service('sender');records['sender']=sender.request(dict(output=str(out/'send'),source=str(path),source_sha256=digest))
        records['sender']['startup_seconds']=sender.startup_seconds
        assert records['sender']['stream_sha256']==plan['input_sha256'][str(HISTORY/case/'base.glcsnd')]
        sender.close();active.remove(sender);step('base_encoded_exact')
        records['guide_sender']=encode_guide(source,out/'send/base.glcsnd',out/'guide');step('guide_encoded')
        receiver=service('receiver');records['receiver']=receiver.request(dict(output=str(out/'receive'),base=str(out/'send/base.glcsnd')))
        records['receiver']['startup_seconds']=receiver.startup_seconds
        assert records['receiver']['decoded_rgb_sha256']==plan['input_sha256'][str(HISTORY/case/'base_receive/decoded_rgb.npy')]
        receiver.close();active.remove(receiver);step('base_decoded_exact')
        post=service('post')
        request=dict(base=str(out/'send/base.glcsnd'),raw=str(out/'receive/decoded_rgb.npy'),
            base_receipt=str(out/'receive/receipt.json'),guide=str(out/'guide/guide.glcguide'))
        records['post']=post.request(dict(**request,output=str(out/'post'),raft_mode='feature_cache',cached_t3=True))
        records['post']['startup_seconds']=post.startup_seconds
        elapsed=time.perf_counter()-start;step('current_complete')
        # Separate replay excluded from the full-chain time; freezes the original arithmetic on this new source.
        replay=post.request(dict(**request,output=str(out/'uncached_replay'),raft_mode='original',cached_t3=False))
        exact={k:h==replay['outputs_sha256'][k] for k,h in records['post']['outputs_sha256'].items()}
        if not all(exact.values()):raise ValueError('Cached and original I12 disagree on '+case)
        post.close();active.remove(post)
        sizes=dict(base=Path(request['base']).stat().st_size,guide=Path(request['guide']).stat().st_size)
        assert all(v<=50000 for v in sizes.values())
        for role in ('receiver','post'):
            assert records[role]['source_reads_blocked']
            forbidden={str(SOURCES/(case+'.mp4')),str(path),str(HISTORY/case/'source.npy')}
            assert not forbidden.intersection(records[role]['reads'])
        result=dict(complete=True,case=case,stages=records,full_chain_cold_seconds=elapsed,
            full_chain_task_fps=150/elapsed,timing_repetitions=1,base_bytes=sizes['base'],guide_bytes=sizes['guide'],
            total_bytes=sum(sizes.values()),base_kbps=sizes['base']*8/10000,guide_kbps=sizes['guide']*8/10000,
            total_kbps=sum(sizes.values())*8/10000,base_packet_and_decode_match_history=True,
            exact_against_uncached_I12=exact,uncached_replay=replay,source_free_receivers=True)
        save_json(out/'chain_result.json',result);step('chain_and_replay_verified')
        return result
    except BaseException as error:
        save_json(out/'failure.json',dict(error=str(error),traceback=traceback.format_exc(),stages=records));raise
    finally:
        for obj in active:obj.close()


def make_quad(out,case,sequences,rates):
    width,height,bar=1024,688,56
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    labels=['原视频',f'H.265  {rates["h265"]:.2f} kbps',f'原始 GLC  {rates["base"]:.2f} kbps',f'当前 I12  {rates["current"]:.2f} kbps']
    notes=[case+' · 15 FPS · 前10秒','按完整文件匹配当前方案总码率','仅本体码流','固定参数 S8I + T3 + 六锚点']
    order=['source','H265','raw','current'];positions=[]
    canvas=Image.new('RGB',(width,height),(20,29,40));draw=ImageDraw.Draw(canvas)
    for j in range(4):
        x,y=j%2*512,j//2*344;positions.append((x,y+bar))
        draw.text((x+10,y+4),labels[j],font=font,fill=(239,246,252))
        draw.text((x+10,y+31),notes[j],font=small,fill=(169,198,218))
    path=out/(case+'_四宫格_原片_H265_原GLC_当前I12.mp4')
    cmd=[str(FFMPEG),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','15','-i','pipe:0','-an',
        '-vf','scale=out_color_matrix=bt709:out_range=tv,format=yuv420p','-color_range','tv','-colorspace','bt709',
        '-color_primaries','bt709','-color_trc','bt709','-c:v','libx264','-preset','fast','-crf','12','-threads','4','-movflags','+faststart','-n',str(path)]
    with (out/'quad.ffmpeg.log').open('wb') as log:
        process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        try:
            for i in range(150):
                for name,position in zip(order,positions):canvas.paste(Image.fromarray(sequences[name][i]),position)
                process.stdin.write(canvas.tobytes())
                if i in (20,64,80,128,149):canvas.save(out/f'{case}_frame{i:03d}.png')
            process.stdin.close()
            if process.wait(timeout=60):raise RuntimeError('Quad encode failed')
        finally:
            if process.poll() is None:process.kill();process.wait()
    probe=json.loads(subprocess.check_output([str(FFMPEG.with_name('ffprobe.exe')),'-v','error','-select_streams','v:0',
        '-count_frames','-show_entries','stream=width,height,avg_frame_rate,nb_read_frames,duration','-of','json',str(path)]))['streams'][0]
    assert int(probe['nb_read_frames'])==150 and Fraction(probe['avg_frame_rate'])==15 and float(probe['duration'])==10
    decoded=subprocess.check_output([str(FFMPEG),'-v','error','-i',str(path),'-vf','scale=in_color_matrix=bt709:in_range=tv,format=rgb24','-f','rawvideo','pipe:1'])
    frames=np.frombuffer(decoded,np.uint8).reshape(150,height,width,3);errors={}
    for name,(x,y) in zip(order,positions):
        errors[name]=[float(np.abs(frames[i,y:y+288,x:x+512].astype(np.float32)-sequences[name][i]).mean()) for i in (0,75,149)]
        assert max(errors[name])<6
    return dict(path=str(path),sha256=file_sha(path),probe=probe,order=order,labels=labels,panel_mae=errors)


def evaluate_case(run,case):
    import evaluate_fullframe_reference as scoring
    from evaluate_structure_guide import motion_change_diagnostic
    from compare_deflicker import save_csv
    from run_p0 import preview
    from verify_video1_persistent_structure import inspect_preview
    root=run/case;chain=load_json(root/'chain_result.json');out=root/'evaluation';out.mkdir(exist_ok=False)
    source=np.load(root/'input/source.npy',mmap_mode='r',allow_pickle=False)
    sequences=dict(source=source,raw=np.load(root/'receive/decoded_rgb.npy',mmap_mode='r',allow_pickle=False),
        K=np.load(OLD_K/case/'K/receive/rgb.npy',mmap_mode='r',allow_pickle=False),
        current=np.load(root/'post/rgb.npy',mmap_mode='r',allow_pickle=False))
    h265,record=match_hevc(FFMPEG,source,chain['total_bytes'],out/'matched_hevc');sequences['H265']=h265
    scoring.SCOPES['historical_reef']=tuple(CASES[case]['reef'])
    scores,temporal,spatial=scoring.score(sequences)
    archive=load_json(OLD_K/case/'evaluation/summary.json')['scores']
    for scope in archive:
        if scope not in scoring.SCOPES:continue
        for method in ('raw','K'):
            for field in ('coarse_mae','fine_ncc','psnr_db','valid_pixels'):
                if abs(scores[scope][method][field]-archive[scope][method][field])>1e-9:
                    raise ValueError('Historical metric regression: '+scope+'/'+method+'/'+field)
    save_csv(out/'temporal.csv',temporal);save_csv(out/'spatial.csv',spatial)
    dynamic,dynamic_rows=motion_change_diagnostic(sequences);save_csv(out/'source_change_diagnostic.csv',dynamic_rows)
    alpha=np.load(root/'post/alpha.npy',mmap_mode='r',allow_pickle=False)
    shared=np.load(root/'post/shared.npy',mmap_mode='r',allow_pickle=False)
    assert alpha.shape==(150,288,512) and shared.shape==(30,288,512)
    assert np.isfinite(alpha).all() and float(alpha.min())>=0 and float(alpha.max())<=1
    assert np.array_equal(sequences['current'][alpha==0],sequences['raw'][alpha==0])
    coverage={scope:dict(active_fraction=float((alpha[:,y0:y1,x0:x1]>0).mean()),mean_alpha=float(alpha[:,y0:y1,x0:x1].mean()))
        for scope,(x0,y0,x1,y1) in scoring.SCOPES.items()}
    spikes={}
    for scope in scoring.SCOPES:
        spikes[scope]={}
        for method in ('raw','K','current','H265'):
            rows=[dict(frame=r['frame'],coarse_mae=r['coarse_sum']/r['pixels'],pixels=r['pixels'])
                for r in temporal if r['scope']==scope and r['method']==method]
            values=[r['coarse_mae'] for r in rows]
            events={}
            for name,indices in [('anchor_transition',{k+i for k in (15,45,75,105,135) for i in range(6)}),
                ('GLC_intra_neighborhood',{k+i for k in (64,128) for i in (-1,0,1,2)}),('all',set(range(1,150)))]:
                subset=[r for r in rows if r['frame'] in indices]
                events[name]=dict(frames=[r['frame'] for r in subset],mean_coarse_mae=float(np.mean([r['coarse_mae'] for r in subset])))
            spikes[scope][method]=dict(frame_p95=float(np.percentile(values,95)),maximum=max(values),
                worst_frames=sorted(rows,key=lambda r:r['coarse_mae'],reverse=True)[:10],events=events)
    old_aux=(OLD_K/case/'K/support.glccsp2').stat().st_size
    rates=dict(base=chain['base_kbps'],guide=chain['guide_kbps'],current=chain['total_kbps'],
        K=(chain['base_bytes']+old_aux)*8/10000,h265=record['kbps'])
    quad=make_quad(out,case,sequences,rates)
    pairs={}
    for tag,names in [('原GLC_当前I12',('raw','current')),('旧K_当前I12',('K','current'))]:
        path=out/(case+'_'+tag+'.mp4');panels=[sequences[n] for n in names]
        labels=[f'{n} / {rates["base" if n=="raw" else n]:.2f} kbps' for n in names]
        preview(FFMPEG,path,panels,15,0,labels);pairs[tag]=inspect_preview(FFMPEG,path,panels)
    result=dict(complete=True,case=case,scores=scores,rates_kbps=rates,coverage=coverage,temporal_spikes=spikes,
        source_change_diagnostic=dynamic,matched_hevc=record,quad=quad,pairs=pairs,
        historical_raw_K_scores_exact=True,zero_alpha_pixels_unchanged=True,
        exact_against_uncached_I12=chain['exact_against_uncached_I12'],full_chain_cold_seconds=chain['full_chain_cold_seconds'],
        current_changed_fraction=float(np.any(sequences['current']!=sequences['raw'],axis=-1).mean()),
        interpretation='Initial fixed-rule transfer on known development clips; subjective review pending, no unseen/long-video claim')
    save_json(out/'summary.json',result)
    announce(root,'evaluated',case=case,rates=rates,full_scores=scores['full'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run','evaluate'))
    p.add_argument('--run',type=Path,required=True);p.add_argument('--case',choices=CASES)
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    run=a.run.resolve()
    if a.action=='prepare':prepare(run)
    elif a.action=='run':
        for case in ([a.case] if a.case else CASES):run_case(run,case)
    else:
        for case in ([a.case] if a.case else CASES):evaluate_case(run,case)
    if a.action!='prepare':check_inputs(load_json(run/'execution_plan.json')['input_sha256'])


if __name__=='__main__':main()
