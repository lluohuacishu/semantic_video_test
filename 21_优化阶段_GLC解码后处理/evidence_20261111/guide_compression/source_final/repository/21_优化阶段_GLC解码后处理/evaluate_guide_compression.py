"""Score lower-rate guides against the user-accepted 12.9496 kbps T3 baseline."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw

from receiver_structure import file_sha,load_json,save_json,render
from receive_guide_texture import texture_gate
from run_fullframe_reference import reliable as legacy_reliable
from evaluate_structure_guide import motion_change_diagnostic
import evaluate_fullframe_reference as scoring
from compare_deflicker import save_csv
from run_fullframe_factorial import BOX
from run_p0 import preview
from verify_video1_persistent_structure import inspect_preview


def verify_candidate(run,name,raw):
    folder=run/'video1'/name;rec=folder/'receive';flow=folder/'flow_receive'
    receipt=load_json(rec/'receipt.json');first=load_json(flow/'receipt.json');send=load_json(folder/'send.json')
    assert receipt['source_reads_blocked'] and len(receipt['reads'])==9 and len(receipt['blocked_inputs'])==7
    assert first['source_reads_blocked'] and len(first['reads'])==5 and len(first['blocked_inputs'])==7
    assert receipt['receiver_local_flow_receipt_sha256']==file_sha(flow/'receipt.json')
    assert receipt['auxiliary_sha256']==first['auxiliary_sha256']==send['sha256']==file_sha(folder/'guide.glcguide')
    assert receipt['auxiliary_bytes']==send['bytes']==(folder/'guide.glcguide').stat().st_size<=send['cap_bytes']
    for root,record in ((rec,receipt),(flow,first)):
        for filename,digest in record['outputs_sha256'].items():assert file_sha(root/filename)==digest
    guide=np.load(rec/'guide_rgb.npy',mmap_mode='r')
    assert np.array_equal(guide,np.load(folder/'sender_expected_guide.npy',mmap_mode='r'))
    assert np.array_equal(guide,np.load(flow/'guide_rgb.npy',mmap_mode='r'))
    motion=[np.load(flow/(n+'.npy'),mmap_mode='r') for n in ('to_anchor','from_anchor','transition_flow')]
    replay=render(raw,guide,*motion,gate_fn=texture_gate(guide,base_gate=legacy_reliable))
    for n,v in replay.items():assert np.array_equal(v,np.load(rec/(n+'.npy'),mmap_mode='r'))
    rgb=np.load(rec/'rgb.npy',mmap_mode='r');alpha=np.load(rec/'alpha.npy',mmap_mode='r')
    assert np.array_equal(rgb[alpha==0],raw[alpha==0])
    assert np.all(alpha<=np.load(flow/'alpha.npy',mmap_mode='r'))
    checks=dict(complete=True,actual_full_file_budget=True,source_reads_blocked=True,
        actual_guide_decode_exact=True,source_free_cache_provenance=True,
        unchanged_legacy_T3_replay_bit_exact=True,zero_alpha_unchanged=True,texture_alpha_only_reduces=True)
    save_json(folder/'verification.json',checks)
    return send,receipt,rgb,alpha


def evaluate(workspace,run,names,tag):
    data=run.parent;out=run/'evaluation'/tag;out.mkdir(parents=True,exist_ok=False)
    base=data/'source_dense_flow_three_clips_20261109_01/video1'
    prior=data/'structure_guidance_20261111_01/video1'
    raw=np.load(base/'base_receive/decoded_rgb.npy',mmap_mode='r')
    seq=dict(source=np.load(base/'source.npy',mmap_mode='r'),raw=raw,
        K=np.load(data/'support_refinement_20261111_01/video1/K/receive/rgb.npy',mmap_mode='r'),
        A12=np.load(prior/'G256_b12_T3/receive/rgb.npy',mmap_mode='r'))
    sends={};receipts={};alphas={}
    for name in names:
        sends[name],receipts[name],seq[name],alphas[name]=verify_candidate(run,name,raw)
        print(name,'source-free receiver and fixed T3 replay verified',flush=True)
    scoring.SCOPES['coral']=BOX
    scores,temporal,spatial=scoring.score(seq)
    archive=load_json(prior/'evaluation/summary.json')
    for scope,methods in scores.items():
        for current,previous in (('raw','raw'),('K','K'),('A12','G256_b12_T3')):
            assert methods[current]==archive['scores'][scope][previous],(scope,current)
    save_csv(out/'temporal.csv',temporal);save_csv(out/'spatial.csv',spatial)
    dynamic,rows=motion_change_diagnostic(seq);save_csv(out/'source_change.csv',rows)
    for current,previous in (('raw','raw'),('K','K'),('A12','G256_b12_T3')):
        assert dynamic[current]==archive['source_change_diagnostic']['scores'][previous]
    gate=load_json(run/'execution_plan.json')['quality_screen'];candidates={}
    for name in names:
        coral=scores['coral'][name];reference=scores['coral']['A12'];full=scores['full'][name]
        checks=dict(coral_MAE=coral['coarse_mae']<=reference['coarse_mae']*gate['coral_coarse_MAE_ratio_max'],
            full_MAE=full['coarse_mae']<=scores['full']['A12']['coarse_mae']*gate['full_coarse_MAE_ratio_max'],
            coral_NCC=coral['fine_ncc']>=reference['fine_ncc']-gate['coral_NCC_drop_max'],
            full_PSNR=full['psnr_db']>=scores['full']['A12']['psnr_db']-gate['full_PSNR_drop_max_db'],
            source_change=dynamic[name]['RGB_delta_error']<=dynamic['A12']['RGB_delta_error']*gate['source_change_error_ratio_max'])
        candidates[name]=dict(profile=name,bytes=sends[name]['bytes'],auxiliary_kbps=sends[name]['kbps'],
            base_bytes=sends[name]['base_bytes'],total_kbps=(sends[name]['base_bytes']+sends[name]['bytes'])*8/10000,
            auxiliary_saved_percent=(1-sends[name]['bytes']/16187)*100,
            preset=sends[name]['preset'],chroma_offset=sends[name]['chroma_offset'],
            coral_MAE=coral['coarse_mae'],coral_MAE_change_percent=(coral['coarse_mae']/reference['coarse_mae']-1)*100,
            coral_NCC=coral['fine_ncc'],full_MAE=full['coarse_mae'],
            full_MAE_change_percent=(full['coarse_mae']/scores['full']['A12']['coarse_mae']-1)*100,
            full_PSNR=full['psnr_db'],source_change_error=dynamic[name]['RGB_delta_error'],
            source_change_error_percent=(dynamic[name]['RGB_delta_error']/dynamic['A12']['RGB_delta_error']-1)*100,
            numeric_checks=checks,numeric_screen_pass=all(checks.values()),visual_acceptance='pending',
            quadrant_MAE_change_percent={scope:(scores[scope][name]['coarse_mae']/scores[scope]['A12']['coarse_mae']-1)*100
                for scope in ('top_left','top_right','bottom_left','bottom_right')},
            coverage={scope:float((alphas[name][:,y0:y1,x0:x1]>0).mean())
                for scope,(x0,y0,x1,y1) in scoring.SCOPES.items()},
            encoder_selected_seconds=sends[name]['selected_two_pass_encode_seconds'],
            flow_seconds=receipts[name]['flow_seconds'],render_seconds=receipts[name]['render_seconds'])
    save_json(out/'summary.json',dict(complete=True,profiles=names,scores=scores,candidates=candidates,
        dynamic_diagnostic=dynamic,archived_scores_exact=True,archived_dynamic_diagnostic_exact=True,
        receiver_rule_unchanged=True,source_only_evaluator=True,visual_acceptance='pending',
        code_sha256=file_sha(__file__)))
    for name,record in candidates.items():
        print('SCORE',name,{k:record[k] for k in ('auxiliary_kbps','auxiliary_saved_percent','coral_MAE',
            'coral_MAE_change_percent','coral_NCC','full_MAE_change_percent','numeric_screen_pass')},flush=True)
    aggregate={}
    for summary in sorted((run/'evaluation').glob('*/summary.json')):
        aggregate.update(load_json(summary)['candidates'])
    save_json(run/'evaluation_summary.json',dict(complete=True,candidates=aggregate,
        comparison='User-accepted A12 (G256_b12_T3); K retained as historical reference',
        note='Actual bytes differ within each common cap; do not claim exact equal-bitrate preset causality.'))


def previews(workspace,run,names):
    data=run.parent;out=run/'previews';out.mkdir(exist_ok=True)
    base=data/'source_dense_flow_three_clips_20261109_01/video1'
    prior=data/'structure_guidance_20261111_01/video1/G256_b12_T3/receive'
    seq=dict(source=np.load(base/'source.npy',mmap_mode='r'),raw=np.load(base/'base_receive/decoded_rgb.npy',mmap_mode='r'),
        A12=np.load(prior/'rgb.npy',mmap_mode='r'))
    summary=load_json(run/'evaluation_summary.json')['candidates']
    ffmpeg=workspace/'01_项目/运行环境/FFmpeg/ffmpeg.exe';checks={}
    for name in names:
        seq[name]=np.load(run/'video1'/name/'receive/rgb.npy',mmap_mode='r')
        labels=dict(source='Source RGB',raw='Original GLC',A12='Accepted T3 / aux 12.95 kbps')
        labels[name]=f'{name} / aux {summary[name]["auxiliary_kbps"]:.2f} kbps'
        specs=[(f'A12_vs_{name}_full.mp4',('A12',name),None),
            (f'A12_vs_{name}_coral.mp4',('A12',name),(250,0,512,132)),
            (f'source_raw_A12_{name}.mp4',('source','raw','A12',name),None)]
        for filename,keys,crop in specs:
            panels=[seq[k] for k in keys]
            preview(ffmpeg,out/filename,panels,15,0,[labels[k] for k in keys],
                crop=None if crop is None else np.s_[crop[1]:crop[3],crop[0]:crop[2]])
            checks[filename]=inspect_preview(ffmpeg,out/filename,panels,crop=crop)
        canvas=Image.new('RGB',(512*4,312*3),(24,35,48));draw=ImageDraw.Draw(canvas)
        for row,i in enumerate((20,80,149)):
            for col,key in enumerate(('source','raw','A12',name)):
                canvas.paste(Image.fromarray(np.asarray(seq[key][i])),(col*512,row*312+24))
                draw.text((col*512+6,row*312+5),f'{key} f{i}',fill='white')
        canvas.save(out/(name+'_contact.png'))
    save_json(out/'verification.json',dict(complete=True,profiles=names,previews=checks,
        frames=150,fps=15,seconds=10,lossless_RGB_scored=True,MP4_viewing_only=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('score','previews'))
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--profiles',nargs='+',required=True);p.add_argument('--tag',default='initial')
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    if a.action=='score':evaluate(a.workspace.resolve(),a.run.resolve(),a.profiles,a.tag)
    else:previews(a.workspace.resolve(),a.run.resolve(),a.profiles)
