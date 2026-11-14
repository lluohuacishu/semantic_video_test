"""Independently verify saved factorial results and build reviewable comparisons."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from runtime_speed_common import STAGE,WORKSPACE,FFMPEG,file_sha,load_json,save_json,check_inputs
from run_generalization_factorial import VARIANTS
from run_p0 import preview
from verify_video1_persistent_structure import inspect_preview
import structure_guide_stream as guide


def verify(run):
    plan=load_json(run/'execution_plan.json');check_inputs(plan['input_sha256'])
    assert all(file_sha(STAGE/n)==h for n,h in plan['code_sha256'].items())
    results={}
    for case,r in plan['cases'].items():
        raw=np.load(r['raw'],mmap_mode='r',allow_pickle=False);qualities={}
        for quality in ('Q6','Q12'):
            root=run/case/quality;receipt=load_json(root/'receive/receipt.json');send=load_json(root/'send.json')
            assert receipt['complete'] and receipt['source_reads_blocked'] and len(receipt['blocked'])==5
            assert receipt['code_sha256']==plan['code_sha256']['run_generalization_factorial.py']
            assert receipt['auxiliary_sha256']==send['sha256']==file_sha(root/'guide.glcguide')
            assert send['bytes']==(root/'guide.glcguide').stat().st_size
            assert receipt['base_sha256']==file_sha(r['base']) and receipt['flow']['pairs']==174
            allowed={str(Path(p).resolve()) for p in (r['base'],r['raw'],r['base_receipt'],str(root/'guide.glcguide'),str(STAGE/'not_used'))}
            assert str(Path(r['source']).resolve()) not in receipt['reads']
            header,low,full=guide.decode((root/'guide.glcguide').read_bytes(),file_sha(r['base']),FFMPEG)
            assert full.shape==(150,288,512,3) and header==receipt['guide_header']
            assert np.array_equal(full,np.load(root/'receive/guide_rgb.npy',mmap_mode='r'))
            for name,h in receipt['flow_sha256'].items():
                p=root/'receive'/(name+'.npy');assert file_sha(p)==h
                assert np.isfinite(np.load(p,mmap_mode='r')).all()
            baseline=np.load(root/'receive/T3/alpha.npy',mmap_mode='r');shared_base=np.load(root/'receive/T3/shared.npy',mmap_mode='r')
            variants={}
            for mode in ('T3','F8','C8'):
                folder=root/'receive'/mode;values={}
                for name,h in receipt['variants'][mode]['outputs_sha256'].items():
                    p=folder/(name+'.npy');assert file_sha(p)==h
                    a=np.load(p,mmap_mode='r',allow_pickle=False);values[name]=a
                    assert a.shape==({'rgb':(150,288,512,3),'alpha':(150,288,512),'shared':(30,288,512)}[name])
                    assert a.dtype==(np.uint8 if name=='rgb' else np.float32) and np.isfinite(a).all()
                alpha=values['alpha'];shared=values['shared']
                assert np.all((alpha>=0)&(alpha<=1)) and np.all((shared>=0)&(shared<=1))
                assert np.all(alpha<=baseline+1e-7) and np.all(shared<=shared_base+1e-7)
                assert np.array_equal(values['rgb'][alpha==0],raw[alpha==0])
                if quality=='Q6' and mode=='T3':
                    for name in values:assert file_sha(folder/(name+'.npy'))==file_sha(Path(r['current'])/(name+'.npy'))
                variants[mode]=dict(hashes_valid=True,zero_alpha_unchanged=True,no_expansion_beyond_T3_support=True)
            qualities[quality]=dict(variants=variants,read_guard_blocked=receipt['blocked'],base_binding=True,
                guide_decode_exact=True,actual_aux_bytes=send['bytes'])
        results[case]=qualities
    record=dict(complete=True,verified_variants=18,verified_guide_flows=6,frozen_inputs_and_inference_code_unchanged=True,
        exact_current_baselines=3,cases=results,verifier_sha256=file_sha(__file__))
    save_json(run/'verification.json',record);print('VERIFIED 18 variants, 6 flows, 3 exact baselines',flush=True)


def sample_sheet(out,case,sequences,labels,indices):
    cols=len(sequences);width=512*cols;height=314*len(indices)
    canvas=Image.new('RGB',(width,height),(24,35,48));draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    for row,i in enumerate(indices):
        for col,(name,array) in enumerate(sequences.items()):
            x,y=col*512,row*314
            draw.text((x+8,y+3),f'{case} · {labels[name]} · f{i}',font=font,fill='white')
            canvas.paste(Image.fromarray(array[i]),(x,y+26))
    canvas.save(out)


def previews(run):
    verification=load_json(run/'verification.json');assert verification['complete']
    summary=load_json(run/'summary.json');plan=load_json(run/'execution_plan.json')
    chosen=summary['selected'] or summary['best_video2_diagnostic']
    # Keep a zero-extra-byte control visible even if the numeric best uses more bytes.
    candidates=list(dict.fromkeys([chosen,'Q6_C8']))
    out=run/'previews';out.mkdir(exist_ok=False);records={}
    for case,r in plan['cases'].items():
        seq={'source':np.load(r['source'],mmap_mode='r'),'raw':np.load(r['raw'],mmap_mode='r')}
        for name in VARIANTS:
            q,m=name.split('_');seq[name]=np.load(run/case/q/'receive'/m/'rgb.npy',mmap_mode='r')
        rates=summary['cases'][case]['rates'];labels={n:f'{n} / {rates[n]:.2f} kbps' for n in VARIANTS}
        labels.update(source='Source RGB',raw='Original GLC')
        for candidate in candidates:
            names=('source','raw','Q6_T3',candidate);panels=[seq[n] for n in names]
            path=out/f'{case}_source_raw_current_{candidate}.mp4'
            preview(FFMPEG,path,panels,15,0,[labels[n] for n in names])
            records[path.name]=inspect_preview(FFMPEG,path,panels)
            path2=out/f'{case}_current_vs_{candidate}.mp4';names2=('Q6_T3',candidate)
            preview(FFMPEG,path2,[seq[n] for n in names2],15,0,[labels[n] for n in names2])
            records[path2.name]=inspect_preview(FFMPEG,path2,[seq[n] for n in names2])
        if case=='video2':
            path=out/'video2_full_2x3_matrix.mp4'
            preview(FFMPEG,path,[seq[n] for n in VARIANTS],15,0,[labels[n] for n in VARIANTS])
            records[path.name]=inspect_preview(FFMPEG,path,[seq[n] for n in VARIANTS])
        selected_seq={n:seq[n] for n in ('source','raw','Q6_T3',chosen)}
        zh=dict(source='原片',raw='原GLC',Q6_T3='当前方案');zh[chosen]='候选 '+chosen
        sample_sheet(out/(case+'_contact.png'),case,selected_seq,zh,(20,64,80,128,149))
        print('PREVIEWS',case,chosen,flush=True)
    result=dict(complete=True,selected_for_viewing=chosen,passed_numeric_screen=summary['selected'] is not None,
        candidates=candidates,videos=records,visual_acceptance='pending; numeric candidate is not adopted',
        layout='four-way: source top-left, raw GLC top-right, current bottom-left, candidate bottom-right; pair: current left, candidate right')
    save_json(run/'preview_verification.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('verify','previews'));p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    (verify if a.action=='verify' else previews)(a.run.resolve())
