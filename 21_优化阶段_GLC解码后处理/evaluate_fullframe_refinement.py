"""Evaluate denser motion and wider blending against the rejected full-frame run."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw

from compare_deflicker import save_csv
from coral_reference_sidecar import file_sha,load_json,save_json
from evaluate_fullframe_reference import SCOPES,score
from run_dense_batch import CASES
from run_fullframe_reference import get_inputs
from run_p0 import preview
from sparse_fullframe_stream import unpack
from verify_video1_persistent_structure import inspect_preview


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--initial',type=Path,required=True);parser.add_argument('--case',choices=CASES,required=True)
    parser.add_argument('--ffmpeg',type=Path,required=True)
    args=parser.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    run=args.run.resolve();case=args.case;root=run/case;out=root/'evaluation';out.mkdir(exist_ok=False)
    profile=load_json(run/'refinement_selection.json')['selected']
    source,raw,digest,size=get_inputs(args.batch.resolve(),case)
    initial=args.initial.resolve()/case
    seq=dict(source=source,raw=raw,A=np.load(initial/'grid16x9/receive/rgb.npy',mmap_mode='r'),
        B=np.load(initial/'grid16x9_tone12/receive/rgb.npy',mmap_mode='r'),
        dense=np.load(root/'offline/rgb.npy',mmap_mode='r'))
    receipts={};alphas={};decoded={}
    for variant in ('C','D'):
        folder=root/(profile+'_'+variant)
        send=load_json(folder/'send.json');receipt=load_json(folder/'receive/receipt.json')
        if receipt['rgb_array_sha256']!=send['expected_rgb_sha256'] or not receipt['source_and_sender_receipt_blocked']:
            raise ValueError('Independent reconstruction failed')
        for f,d in receipt['outputs_sha256'].items():
            if file_sha(folder/'receive'/f)!=d:raise ValueError('Output changed')
        seq[variant]=np.load(folder/'receive/rgb.npy',mmap_mode='r')
        alphas[variant]=np.load(folder/'receive/alpha.npy',mmap_mode='r')
        if not np.array_equal(seq[variant][alphas[variant]==0],raw[alphas[variant]==0]):raise ValueError('Unselected pixels changed')
        h,a=unpack((folder/'fullframe.glcfsp').read_bytes(),digest);decoded[variant]=a
        receipts[variant]=receipt
    if any(not np.array_equal(decoded['C'][k],decoded['D'][k]) for k in decoded['C']):
        raise ValueError('C/D motion or masks are not identical')
    if case=='video1':seq['local_U']=np.load(run.parent/'video1_persistent_aux_20261110_01/g17q4/receive/rgb.npy',mmap_mode='r')
    scores,rows,images=score(seq)
    save_csv(out/'temporal.csv',rows);save_csv(out/'spatial.csv',images)
    labels=dict(source='Source RGB',raw='Original GLC',A='A / initial full-frame',B='B / color only',
        C='C / denser motion, narrow blend',D='D / denser motion, wider blend',local_U='Previous local coral U')
    specs=[('raw_vs_D.mp4',('raw','D')),('A_vs_D.mp4',('A','D')),('C_vs_D.mp4',('C','D')),
        ('fourway.mp4',('source','raw','A','D'))]
    if case=='video1':specs.append(('local_U_vs_D.mp4',('local_U','D')))
    previews={}
    for name,names in specs:
        frames=[seq[k] for k in names]
        preview(args.ffmpeg,out/name,frames,15,0,[labels[k] for k in names])
        previews[name]=inspect_preview(args.ffmpeg,out/name,frames)
    coverage={}
    for variant,a in alphas.items():
        coverage[variant]={name:dict(active=float((a[:,y0:y1,x0:x1]>0).mean()),mean_weight=float(a[:,y0:y1,x0:x1].mean()))
            for name,(x0,y0,x1,y1) in SCOPES.items()}
    # Return to the same fixed coral frame used by the accepted short experiment.
    fixed_coral=None
    if case=='video1':
        from run_video1_persistent_structure import score as coral_score,FIXED_BOX
        fixed=np.zeros(raw.shape[:3],np.uint8);x0,y0,x1,y1=FIXED_BOX;fixed[:,y0:y1,x0:x1]=1
        coral,_,_=coral_score(seq,fixed,start=0)
        fixed_coral=coral['fixed']
    canvas=Image.new('RGB',(512*3,312*3),(24,35,48));draw=ImageDraw.Draw(canvas)
    for row,i in enumerate((15,75,149)):
        for col,name in enumerate(('A','C','D')):
            canvas.paste(Image.fromarray(np.asarray(seq[name][i])),(col*512,row*312+24))
            draw.text((col*512+6,row*312+5),f'{case} {name} f{i}',fill='white')
    canvas.save(out/'boundary_frames.png')
    save_json(out/'summary.json',dict(complete=True,case=case,profile=profile,frames=150,fps=15,
        scopes=SCOPES,scores=scores,coverage=coverage,fixed_video1_coral=fixed_coral,
        base_bytes=size,auxiliary_bytes={k:r['auxiliary_bytes'] for k,r in receipts.items()},
        receivers=receipts,C_D_auxiliary_arrays_bit_exact=True,previews=previews,
        code_sha256=file_sha(__file__),visual_review='pending'))
    print(case,'FULL',scores['full'],flush=True)
    if fixed_coral:print('VIDEO1 CORAL', {k:(v['coarse_temporal_mae'],v['fine_ncc']) for k,v in fixed_coral.items()},flush=True)


if __name__=='__main__':main()
