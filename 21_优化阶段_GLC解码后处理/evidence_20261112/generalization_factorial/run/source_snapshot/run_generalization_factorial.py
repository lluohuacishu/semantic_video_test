"""Fixed 2x3 guide-quality / conservative-blend experiment on known clips.

No per-clip tuning. Source RGB is confined to encode/evaluate actions.
Receiver action only reads packets, independently decoded RGB, and local assets.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np

from runtime_speed_common import (STAGE,REPO,WORKSPACE,DATA,FFMPEG,WEIGHT,
    file_sha,load_json,save_json,check_inputs,render as original_render)
from receiver_structure import KEYS,RAMP,render
from cached_t3 import cached_texture_gate
import structure_guide_stream as guide

HISTORY=DATA/'source_dense_flow_three_clips_20261109_01'
TRANSFER=DATA/'known_video_transfer_20261111_02'
VARIANTS=('Q6_T3','Q6_F8','Q6_C8','Q12_T3','Q12_F8','Q12_C8')
CODE=('run_generalization_factorial.py','runtime_speed_common.py','raft_inference_runtime.py',
    'runtime_chain_worker.py','receiver_structure.py','receive_guide_texture.py','cached_t3.py',
    'structure_guide_stream.py','evaluate_fullframe_reference.py','evaluate_structure_guide.py')


def cases():
    cfg=load_json(STAGE/'当前采用配置.json')
    result={}
    for case in ('video1','video2','video3'):
        base=HISTORY/case
        result[case]=dict(source=str(base/'source.npy'),base=str(base/'base.glcsnd'),
            raw=str(base/'base_receive/decoded_rgb.npy'),base_receipt=str(base/'base_receive/result.json'),
            guide=str(Path(cfg['artifacts']['guide']['path']) if case=='video1' else TRANSFER/case/'guide/guide.glcguide'),
            current=str(Path(cfg['artifacts']['rgb']['path']).parent if case=='video1' else TRANSFER/case/'post'),
            reef={'video1':[342,10,502,105],'video2':[380,156,507,281],'video3':[145,190,510,286]}[case])
    return result


def progress(out,stage,**values):
    value=dict(stage=stage,**values);save_json(out/'progress.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def prepare(run):
    run.mkdir(parents=True,exist_ok=False);records=cases();frozen={}
    for case,r in records.items():
        for name in ('source','base','raw','base_receipt','guide'):frozen[r[name]]=file_sha(r[name])
        for name in ('rgb','alpha','shared'):
            p=Path(r['current'])/(name+'.npy');frozen[str(p)]=file_sha(p)
        old=DATA/'support_refinement_20261111_01'/case/'K/receive/rgb.npy';frozen[str(old)]=file_sha(old)
    frozen[str(WEIGHT)]=file_sha(WEIGHT);frozen[str(STAGE/'当前采用配置.json')]=file_sha(STAGE/'当前采用配置.json')
    plan=dict(date='2026-11-12',frozen_before_trials=True,
        authorization='User resumed experiments: 好的，那开始进一步实验吧，整一整泛用性',
        cases=records,variants=VARIANTS,guide_factor=dict(Q6='Existing transmitted S8I, internal 6 kbps',
            Q12='New fixed 12 kbps internal two-pass x265 slower, same 256x144/15fps; actual packet billed'),
        receiver_factor=dict(T3='Unchanged accepted T3, 2px feather',
            F8='Same valid support; feather ramps over 8px, never exceeds original alpha',
            C8='F8 multiplied by smooth confidence in local guide-flow deformation: 1 at gradient<=0.10, 0 at>=0.50, cubic smoothstep; 2px Gaussian smoothing of gradient'),
        hypothesis=['Guide information loss may produce wrong correspondence/support on dense spotted moving fish',
            'Narrow alpha boundaries and nonrigid flow may cause visible boundaries/jumps; conservative blending may help but may sacrifice stabilization'],
        fixed=dict(anchors=list(KEYS),iterations=12,precision='FP32',TF32=False,transition=list(RAMP),
            all_references='Historical unfiltered GLC RGB',new_per_clip_rules=False,new_source_masks=False),
        trial_count=18,baseline_exact_required=['RGB','alpha','shared'],
        numeric_screen=dict(description='Prioritization only; never subjective acceptance',
            video2_full_mae_ratio_max=.90,video2_boundary_residual_ratio_max=.95,video2_dynamic_error_ratio_max=1.01,
            video2_psnr_drop_max_db=.05,other_full_and_reef_mae_ratio_max=1.05,other_psnr_drop_max_db=.10,
            other_dynamic_error_ratio_max=1.03),
        evaluation='Same source-defined alignment, full/four quadrants/old reef boxes, RGB PSNR, source-high-change error, frame p95/worst; fixed baseline support-border residual gradient (diagnostic only)',
        scope='Known development clips. No holdout/long-video/realtime claim. If no candidate passes, report negative result and do not auto-adopt.',
        next_test='A selected frozen candidate may be tested on a separately registered existing marine clip; no further threshold tuning in this round.',
        input_sha256=frozen,code_sha256={n:file_sha(STAGE/n) for n in CODE},
        base_encode_rerun=False,base_note='Reuse hash-verified actual stream and independently decoded RGB; no complete-chain timing claim')
    save_json(run/'execution_plan.json',plan);snap=run/'source_snapshot';snap.mkdir()
    for name in CODE:shutil.copy2(STAGE/name,snap/name)
    state_path=REPO/'05_资料与索引/会话收束_20261110/接续状态.json'
    shutil.copy2(state_path,run/'state_before_resume.json');state=load_json(state_path)
    state['experiment_execution_gate']=dict(status='resumed_by_user',date='2026-11-12',authorization=plan['authorization'])
    state['status']='generalization_factorial_running_fixed_2x3'
    state['user_latest_decision']='好的，那开始进一步实验吧，整一整泛用性'
    state['current_session_work']='Running fixed guide-quality by conservative-blend factorial on video1/2/3; no auto-adoption.'
    state['active_experiment_plan']=str(run/'execution_plan.json');save_json(state_path,state)
    progress(run,'prepared',variants=list(VARIANTS),cases=list(records))


def encode(run):
    plan=load_json(run/'execution_plan.json');check_inputs(plan['input_sha256'])
    for case,r in plan['cases'].items():
        root=run/case;root.mkdir(exist_ok=False)
        for quality,rate in (('Q6',6),('Q12',12)):
            out=root/quality;out.mkdir();packet=out/'guide.glcguide'
            if quality=='Q6':
                shutil.copy2(r['guide'],packet);record=dict(reused=True,source_packet=r['guide'],encode_seconds=None)
            else:
                start=time.perf_counter();source=np.load(r['source'],mmap_mode='r',allow_pickle=False)
                low=np.stack([cv2.resize(v,(256,144),interpolation=cv2.INTER_AREA) for v in source])
                common=[str(FFMPEG),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','256x144','-r','15','-i','pipe:0',
                    '-an','-c:v','libx265','-preset','slower','-b:v',str(rate)+'k','-pix_fmt','yuv420p','-threads','2']
                params='log-level=error:pools=2:frame-threads=1:keyint=150:min-keyint=150:scenecut=0:repeat-headers=1:info=0:stats=guide_pass.log'
                for n in (1,2):
                    p=subprocess.run([*common,'-x265-params',params+f':pass={n}','-f','hevc','pipe:1'],input=low.tobytes(),
                        stdout=subprocess.PIPE if n==2 else subprocess.DEVNULL,stderr=subprocess.PIPE,check=True,cwd=out)
                    (out/f'pass{n}.log').write_bytes(p.stderr)
                payload,_=guide.pack(p.stdout,file_sha(r['base']),(256,144));packet.write_bytes(payload)
                record=dict(reused=False,encode_seconds=time.perf_counter()-start,command=common,x265_params=params)
            header,_=guide.unpack(packet.read_bytes(),file_sha(r['base']))
            record.update(complete=True,bytes=packet.stat().st_size,kbps=packet.stat().st_size*8/10000,
                total_kbps=(Path(r['base']).stat().st_size+packet.stat().st_size)*8/10000,sha256=file_sha(packet),header=header)
            save_json(out/'send.json',record)
            progress(run,'encoded',case=case,quality=quality,total_kbps=record['total_kbps'])


def conservative_gate(images,mode):
    base=cached_texture_gate(images)
    def gate(values,i,key,back,forward):
        alpha,mx,my=base(values,i,key,back,forward)
        feather=np.minimum(cv2.distanceTransform((alpha>0).astype(np.uint8),cv2.DIST_L2,5)/8.,1.)
        weight=np.minimum(alpha,feather)
        if mode=='C8':
            gradient=np.maximum(np.linalg.norm(np.gradient(back,axis=0),axis=2),np.linalg.norm(np.gradient(back,axis=1),axis=2))
            gradient=cv2.GaussianBlur(gradient,(0,0),2)
            z=np.clip((gradient-.10)/(.50-.10),0,1);confidence=1-z*z*(3-2*z)
            weight*=confidence
        assert np.all(weight<=alpha+1e-7)
        return weight,mx,my
    return gate


def receive(run,case,quality):
    import torch
    from runtime_chain_worker import ReadGuard
    from raft_inference_runtime import load_model,estimate
    r=cases()[case];root=run/case/quality;out=root/'receive'
    guard=ReadGuard('generalization');guard.authorize(out,[r['base'],r['raw'],r['base_receipt'],root/'guide.glcguide'])
    blocked=[]
    for path in (Path(r['source']),HISTORY/case/'flow/to_anchor.npy',HISTORY/case/'flow/from_anchor.npy',root/'send.json',
        DATA/'support_refinement_20261111_01'/case/'K/receive/rgb.npy'):
        try:path.read_bytes()
        except RuntimeError:blocked.append(str(path))
        else:raise ValueError('Receiver source audit failed')
    out.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    receipt=load_json(r['base_receipt']);base_sha=file_sha(r['base'])
    assert receipt['reference_verified'] and receipt['source_media_reads_forbidden']
    assert base_sha==receipt['stream_sha256'] and file_sha(r['raw'])==receipt['decoded_rgb_sha256']
    raw=np.load(r['raw'],mmap_mode='r',allow_pickle=False)
    header,low,images=guide.decode((root/'guide.glcguide').read_bytes(),base_sha,FFMPEG)
    model=load_model();flows,flow_info=estimate(model,images,'feature_cache')
    for name,value in zip(('to_anchor','from_anchor','transition_flow'),flows):np.save(out/(name+'.npy'),value,allow_pickle=False)
    np.save(out/'guide_rgb.npy',images,allow_pickle=False)
    del model;torch.cuda.empty_cache();stats={}
    for mode in ('T3','F8','C8'):
        folder=out/mode;folder.mkdir();start=time.perf_counter()
        if mode=='T3':values=original_render(raw,images,*flows,cached=True)
        else:values=render(raw,images,*flows,gate_fn=conservative_gate(images,mode))
        elapsed=time.perf_counter()-start
        for name,value in values.items():np.save(folder/(name+'.npy'),value,allow_pickle=False)
        assert np.isfinite(values['alpha']).all() and np.all((values['alpha']>=0)&(values['alpha']<=1))
        stats[mode]=dict(render_seconds=elapsed,active_fraction=float((values['alpha']>0).mean()),mean_alpha=float(values['alpha'].mean()),
            outputs_sha256={name:file_sha(folder/(name+'.npy')) for name in values})
        progress(out,'rendered',case=case,quality=quality,mode=mode,seconds=elapsed)
    save_json(out/'receipt.json',dict(complete=True,case=case,quality=quality,base_sha256=base_sha,base_decode_verified=True,
        auxiliary_sha256=file_sha(root/'guide.glcguide'),source_reads_blocked=True,blocked=blocked,reads=sorted(guard.reads),
        flow=flow_info,guide_header=header,variants=stats,flow_sha256={n:file_sha(out/(n+'.npy')) for n in ('to_anchor','from_anchor','transition_flow','guide_rgb')},
        code_sha256=file_sha(__file__),no_per_clip_parameters=True,no_source_pixels=True))


def run_receivers(run):
    plan=load_json(run/'execution_plan.json');check_inputs(plan['input_sha256'])
    assert all(file_sha(STAGE/n)==h for n,h in plan['code_sha256'].items())
    for case,r in plan['cases'].items():
        for quality in ('Q6','Q12'):
            root=run/case/quality
            with (root/'receiver.log').open('w',encoding='utf-8') as f:
                subprocess.run([sys.executable,'-B','-u','-X','utf8',str(Path(__file__).resolve()),'worker','--run',str(run),
                    '--case',case,'--quality',quality],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=REPO)
            receipt=load_json(root/'receive/receipt.json')
            if quality=='Q6':
                for name in ('rgb','alpha','shared'):
                    assert receipt['variants']['T3']['outputs_sha256'][name]==file_sha(Path(r['current'])/(name+'.npy'))
            progress(run,'received',case=case,quality=quality,baseline_exact=quality=='Q6')


def evaluate(run):
    import evaluate_fullframe_reference as scoring
    from evaluate_structure_guide import motion_change_diagnostic
    from compare_deflicker import save_csv
    plan=load_json(run/'execution_plan.json');all_results={}
    for case,r in plan['cases'].items():
        root=run/case;out=root/'evaluation';out.mkdir(exist_ok=False)
        seq=dict(source=np.load(r['source'],mmap_mode='r'),raw=np.load(r['raw'],mmap_mode='r'),
            K=np.load(DATA/'support_refinement_20261111_01'/case/'K/receive/rgb.npy',mmap_mode='r'))
        rates={};coverage={}
        for name in VARIANTS:
            quality,mode=name.split('_');folder=root/quality/'receive'/mode
            seq[name]=np.load(folder/'rgb.npy',mmap_mode='r');alpha=np.load(folder/'alpha.npy',mmap_mode='r')
            rates[name]=load_json(root/quality/'send.json')['total_kbps']
            coverage[name]=dict(active_fraction=float((alpha>0).mean()),mean_alpha=float(alpha.mean()))
        scoring.SCOPES['reef']=tuple(r['reef']);scores,temporal,spatial=scoring.score(seq)
        save_csv(out/'temporal.csv',temporal);save_csv(out/'spatial.csv',spatial)
        dynamic,rows=motion_change_diagnostic(seq);save_csv(out/'dynamic.csv',rows)
        # Fixed support-border mask defined by the baseline receiver, never by the compared candidate.
        baseline_alpha=np.load(root/'Q6/receive/T3/alpha.npy',mmap_mode='r');boundary={n:0. for n in seq};count=0
        per_frame=[];kernel=np.ones((5,5),np.uint8)
        for i in range(150):
            support=(baseline_alpha[i]>.5).astype(np.uint8)
            mask=(cv2.dilate(support,kernel)-cv2.erode(support,kernel)).astype(bool)
            pixels=int(mask.sum());count+=pixels
            for name,value in seq.items():
                residual=value[i].astype(np.float32)-seq['source'][i].astype(np.float32)
                gradient=np.mean(np.abs(np.gradient(residual,axis=0))+np.abs(np.gradient(residual,axis=1)),axis=2)
                total=float(gradient[mask].sum());boundary[name]+=total
                per_frame.append(dict(frame=i,method=name,pixels=pixels,residual_gradient_sum=total))
        boundary={n:v/count for n,v in boundary.items()};save_csv(out/'boundary.csv',per_frame)
        p95={};peaks={}
        for name in seq:
            ts=[dict(frame=x['frame'],mae=x['coarse_sum']/x['pixels']) for x in temporal if x['scope']=='full' and x['method']==name]
            p95[name]=float(np.percentile([x['mae'] for x in ts],95));peaks[name]=sorted(ts,key=lambda x:x['mae'],reverse=True)[:5]
        result=dict(complete=True,case=case,scores=scores,rates=rates,coverage=coverage,dynamic=dynamic,
            boundary_residual_gradient=boundary,boundary_pixels=count,full_frame_p95=p95,worst_frames=peaks,
            boundary_note='Mean absolute spatial gradient of candidate-minus-source RGB on fixed Q6_T3 support borders; diagnostic, not perceptual seam ground truth')
        save_json(out/'summary.json',result);all_results[case]=result
        progress(run,'evaluated',case=case,full_mae={n:v['coarse_mae'] for n,v in scores['full'].items()})
    screens={}
    for name in VARIANTS[1:]:
        checks={}
        for case,d in all_results.items():
            base=d['scores']['full']['Q6_T3'];candidate=d['scores']['full'][name]
            c=dict(full_mae_ratio=candidate['coarse_mae']/base['coarse_mae'],
                reef_mae_ratio=d['scores']['reef'][name]['coarse_mae']/d['scores']['reef']['Q6_T3']['coarse_mae'],
                psnr_change_db=candidate['psnr_db']-base['psnr_db'],
                dynamic_error_ratio=d['dynamic'][name]['RGB_delta_error']/d['dynamic']['Q6_T3']['RGB_delta_error'],
                boundary_ratio=d['boundary_residual_gradient'][name]/d['boundary_residual_gradient']['Q6_T3'])
            if case=='video2':c['pass']=c['full_mae_ratio']<=.90 and c['boundary_ratio']<=.95 and c['dynamic_error_ratio']<=1.01 and c['psnr_change_db']>=-.05
            else:c['pass']=c['full_mae_ratio']<=1.05 and c['reef_mae_ratio']<=1.05 and c['psnr_change_db']>=-.10 and c['dynamic_error_ratio']<=1.03
            checks[case]=c
        screens[name]=dict(passed=all(c['pass'] for c in checks.values()),cases=checks)
    passing=[n for n,v in screens.items() if v['passed']]
    selected=min(passing,key=lambda n:all_results['video2']['scores']['full'][n]['coarse_mae']) if passing else None
    diagnostic=min(VARIANTS[1:],key=lambda n:all_results['video2']['scores']['full'][n]['coarse_mae'])
    summary=dict(complete=True,cases=all_results,numeric_screen=screens,selected=selected,
        best_video2_diagnostic=diagnostic,automatic_adoption=False,user_visual_review='pending',
        scope='Known development clips, not unseen holdout or long-video proof')
    save_json(run/'summary.json',summary);check_inputs(plan['input_sha256'])
    progress(run,'complete',selected=selected,best_video2_diagnostic=diagnostic,passed=passing)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','encode','run','worker','evaluate'))
    p.add_argument('--run',type=Path,required=True);p.add_argument('--case',choices=('video1','video2','video3'))
    p.add_argument('--quality',choices=('Q6','Q12'));a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    run=a.run.resolve()
    try:
        if a.action=='prepare':prepare(run)
        elif a.action=='encode':encode(run)
        elif a.action=='run':run_receivers(run)
        elif a.action=='worker':receive(run,a.case,a.quality)
        else:evaluate(run)
    except BaseException as e:
        if run.exists():save_json(run/f'failure_{a.action}_{a.case}_{a.quality}.json',dict(error=str(e),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
