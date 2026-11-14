"""Shared full-frame/quadrant scores and preview checks for the transfer run."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw

from audit_deflicker_texture import normalized,source_match
from compare_deflicker import save_csv
from coral_reference_sidecar import file_sha,load_json,save_json
from experiment_common import array_sha
from run_dense_batch import CASES
from run_fullframe_reference import get_inputs
from run_p0 import preview
from probe_reef import bands
from temporal_filter import warp
from verify_video1_persistent_structure import inspect_preview


SCOPES={'full':(0,0,512,288),'top_left':(0,0,256,144),'top_right':(256,0,512,144),
    'bottom_left':(0,144,256,288),'bottom_right':(256,144,512,288)}


def score(sequences):
    temporal=[];spatial=[];previous=None
    for i in range(150):
        current={k:bands(v[i]) for k,v in sequences.items()}
        if i:
            mx,my,common=source_match(sequences['source'][i-1],sequences['source'][i],
                previous['source'],current['source'],np.s_[:,:])
            aligned={k:warp(v,mx,my) for k,v in previous.items()}
            delta=current['source']-aligned['source']
        for scope,(x0,y0,x1,y1) in SCOPES.items():
            crop=np.s_[y0:y1,x0:x1]
            for name,video in sequences.items():
                error=video[i][crop].astype(np.float32)-sequences['source'][i][crop]
                spatial.append(dict(frame=i,scope=scope,method=name,pixels=(x1-x0)*(y1-y0),
                    rgb_sse=float(np.square(error).sum()),fine_sum=float(np.abs(current[name][crop][...,2]).sum())))
            if i:
                mask=common[crop];count=int(mask.sum())
                if count<64:raise ValueError(f'Insufficient shared texture: {scope} frame {i}')
                for name in sequences:
                    a=aligned[name][crop][mask];b=current[name][crop][mask]
                    error=np.abs(b-a-delta[crop][mask])
                    temporal.append(dict(frame=i,scope=scope,method=name,pixels=count,
                        coarse_sum=float(error[:,0].sum()),fine_sum=float(error[:,2].sum()),
                        full_sum=float(error[:,3].sum()),
                        fine_ncc=float((normalized(a[:,2:3])*normalized(b[:,2:3])).mean())))
        previous=current
    result={}
    for scope in SCOPES:
        result[scope]={}
        for method in sequences:
            rows=[r for r in temporal if r['scope']==scope and r['method']==method]
            images=[r for r in spatial if r['scope']==scope and r['method']==method]
            n=sum(r['pixels'] for r in rows);nimage=sum(r['pixels'] for r in images)
            mse=sum(r['rgb_sse'] for r in images)/(nimage*3)
            result[scope][method]=dict(pairs=len(rows),valid_pixels=n,
                coarse_mae=sum(r['coarse_sum'] for r in rows)/n,
                fine_mae=sum(r['fine_sum'] for r in rows)/n,
                full_mae=sum(r['full_sum'] for r in rows)/n,
                fine_ncc=sum(r['fine_ncc']*r['pixels'] for r in rows)/n,
                fine_amplitude=sum(r['fine_sum'] for r in images)/nimage,
                psnr_db=None if mse==0 else float(10*np.log10(255**2/mse)))
    return result,temporal,spatial


def evaluate(batch,run,case,ffmpeg,tone=False):
    root=run/case;out=root/('evaluation_tone' if tone else 'evaluation');out.mkdir(exist_ok=False)
    selection=load_json(run/('tone_selection.json' if tone else 'auxiliary_selection.json'));profile=selection['selected']
    source,raw,digest,size=get_inputs(batch,case)
    receipt=load_json(root/profile/'receive/receipt.json')
    send=load_json(root/profile/'send.json')
    if not receipt['source_files_unavailable'] or receipt['rgb_array_sha256']!=send['sender_expected_rgb_sha256']:
        raise ValueError('Independent receiver mismatch')
    for folder,record,key in ((root/'offline',load_json(root/'offline/summary.json'),'outputs_sha256'),
                             (root/profile/'receive',receipt,'outputs_sha256')):
        for name,d in record[key].items():
            if file_sha(folder/name)!=d:raise ValueError('Saved array changed')
    sequences=dict(source=source,raw=raw,
        dense=np.load(root/'offline/rgb.npy',mmap_mode='r'),
        actual=np.load(root/profile/'receive/rgb.npy',mmap_mode='r'))
    alpha=np.load(root/profile/'receive/alpha.npy',mmap_mode='r')
    if not np.array_equal(sequences['actual'][alpha==0],raw[alpha==0]):raise ValueError('Zero-weight changed')
    if case=='video1':
        prior=run.parent/'video1_persistent_aux_20261110_01/g17q4/receive'
        old=load_json(prior/'receipt.json')
        if file_sha(prior/'rgb.npy')!=old['output_sha256']['rgb.npy'] or old['base_sha256']!=digest:raise ValueError('Prior local U mismatch')
        sequences['prior_local_U']=np.load(prior/'rgb.npy',mmap_mode='r')
    if tone:
        initial=load_json(run/'auxiliary_selection.json')['selected']
        sequences['initial_fullframe']=np.load(root/initial/'receive/rgb.npy',mmap_mode='r')
    scores,rows,images=score(sequences)
    save_csv(out/'temporal.csv',rows);save_csv(out/'spatial.csv',images)
    labels=dict(source='Source RGB',raw='Original GLC',dense='F / full-frame dense auxiliary',
        actual='B / full-frame with tone matching' if tone else 'A / actual full-frame auxiliary',prior_local_U='Previous local coral U',
        initial_fullframe='A / initial full-frame')
    videos=[('full_pair.mp4',('raw','actual')),('dense_vs_actual.mp4',('dense','actual')),
        ('full_4way.mp4',('source','raw','dense','actual'))]
    if case=='video1':videos.append(('local_vs_fullframe.mp4',('prior_local_U','actual')))
    if tone:videos.append(('initial_vs_tone.mp4',('initial_fullframe','actual')))
    previews={};checks={}
    for name,keys in videos:
        panels=[sequences[k] for k in keys]
        previews[name]=preview(ffmpeg,out/name,panels,15,0,[labels[k] for k in keys])
        checks[name]=inspect_preview(ffmpeg,out/name,panels)
    cover={}
    for scope,(x0,y0,x1,y1) in SCOPES.items():
        active=alpha[:,y0:y1,x0:x1]
        cover[scope]=dict(active_fraction=float((active>0).mean()),unit_weight_fraction=float((active==1).mean()),
            non_key_active_fraction=float((active[[i for i in range(150) if i%30]]>0).mean()))
    canvas=Image.new('RGB',(512*3,312*3),(24,35,48));draw=ImageDraw.Draw(canvas)
    for row,i in enumerate((15,75,149)):
        a=alpha[i,...,None]*.45
        overlay=np.rint(source[i]*(1-a)+np.array([0,255,170])*a).astype(np.uint8)
        for col,(name,rgb) in enumerate((('Original GLC',raw[i]),('Actual full-frame result',sequences['actual'][i]),('Applied support',overlay))):
            canvas.paste(Image.fromarray(rgb),(col*512,row*312+24));draw.text((col*512+6,row*312+5),f'{case} {name} f{i}',fill='white')
    canvas.save(out/'actual_support.png')
    summary=dict(complete=True,case=case,frames=150,fps=15,billing_seconds=10,profile=profile,
        base_bytes=size,auxiliary_bytes=send['bytes'],base_kbps=size*8/10000,auxiliary_kbps=send['kbps'],
        total_kbps=(size+send['bytes'])*8/10000,scopes=SCOPES,scores=scores,coverage=cover,
        no_hand_selected_edit_ROI=True,common_parameters_all_cases=True,
        sender_receiver_bit_exact=True,zero_weight_pixels_unchanged=True,
        source_hash=array_sha(source),base_sha256=digest,previews=previews,
        preview_verification=checks,code_sha256=file_sha(__file__),
        interpretation='Transfer of new full-frame rules to three existing development sources, not a fresh unseen dataset or full-length test.')
    save_json(out/'summary.json',summary)
    print(case,'FULL',scores['full'],flush=True)
    print(case,'COVERAGE',cover,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--case',choices=CASES,required=True);parser.add_argument('--ffmpeg',type=Path,required=True)
    parser.add_argument('--tone',action='store_true')
    args=parser.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    evaluate(args.batch.resolve(),args.run.resolve(),args.case,args.ffmpeg.resolve(),args.tone)


if __name__=='__main__':main()
