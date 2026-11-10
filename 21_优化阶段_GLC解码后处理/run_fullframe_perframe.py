"""Retain per-frame motion after sparse-motion reliability flicker failed."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from compare_deflicker import save_csv
from compare_source_dense_flow import load_flow
from coral_reference_sidecar import file_sha,load_json,save_json
from evaluate_fullframe_reference import score
from experiment_common import array_sha
from fullframe_structure_stream import pack,unpack,render
from run_dense_batch import CASES
from run_fullframe_auxiliary import compact_safety,quantize,receive
from run_fullframe_reference import get_inputs
from run_p0 import preview
from verify_video1_persistent_structure import inspect_preview


PROFILES=(('g24q2',(24,14)),('g20q2',(20,12)),('g16q2',(16,9)))


def prepare(batch,run):
    save_json(run/'perframe_plan.json',dict(reason='Reject 5fps motion interpolation after reliability coverage oscillated and video1 coral error exceeded raw.',
        profiles=PROFILES,selection='First shared profile whose E and F complete packets fit 50000 bytes for all cases.',
        scale=2,every_frame_motion=True,keys=[0,15,45,75,105,135],mask_block=8,
        E='Per-frame motion, 2px inward feather, no tone correction.',
        F='Same per-frame motion and masks; 8px inward feather and flat-area-only sigma12 tone correction.',
        annotation_free=True,receiver_source_media_forbidden=True))
    records={};selected=None
    for name,grid in PROFILES:
        records[name]={}
        for case in CASES:
            root=run/case;source,raw,digest,size=get_inputs(batch,case)
            keys=tuple(load_json(root/'plan.json')['keys']);events=[k+j for k in keys[1:] for j in range(6)]
            dense,_=load_flow(root/'flow',source,CASES[case][0],anchors=keys)
            record=load_json(root/'transition_flow.json')
            if file_sha(root/'transition_flow.npy')!=record['flow_sha256']:raise ValueError('Flow changed')
            old=np.load(root/'transition_flow.npy',mmap_mode='r')[:,0]
            alpha=np.load(root/'offline/alpha.npy');shared=np.load(root/'offline/shared.npy')
            new_q=quantize(dense,alpha,grid,scale=2);old_q=quantize(old,shared,grid,scale=2)
            new_mask=compact_safety(source,list(range(150)),[keys[np.searchsorted(keys,i,side='right')-1] for i in range(150)],
                dense,new_q.astype(np.float32)*2,alpha)
            old_mask=compact_safety(source,events,[keys[j//6] for j in range(len(events))],
                old,old_q.astype(np.float32)*2,shared)
            records[name][case]={}
            for variant,soft in (('E',False),('F',True)):
                folder=root/(name+'_'+variant);folder.mkdir(exist_ok=False)
                payload,header=pack(new_q,old_q,new_mask,old_mask,digest,keys=keys,scale=2,
                    feather=8 if soft else 2,tone_sigma=12 if soft else 0,flat_only=soft)
                (folder/'fullframe.glcfull').write_bytes(payload)
                info=dict(bytes=len(payload),kbps=len(payload)*8/10000,within_budget=len(payload)<=50000,
                    base_bytes=size,base_kbps=size*8/10000,header=header,sha256=file_sha(folder/'fullframe.glcfull'),
                    mean_safe_block_fraction=float(new_mask.mean()))
                if info['within_budget']:
                    h,a=unpack(payload,digest)
                    if not np.array_equal(a['new_motion'],new_q):raise ValueError('Quantized motion round-trip')
                    rgb,_=render(raw,h,a);info['sender_expected_rgb_sha256']=array_sha(rgb)
                save_json(folder/'send.json',info);records[name][case][variant]=info
            print(name,case,'E/F bytes',[records[name][case][v]['bytes'] for v in ('E','F')],flush=True)
        if all(v['F']['within_budget'] for v in records[name].values()):selected=name;break
    save_json(run/'perframe_selection.json',dict(complete=selected is not None,selected=selected,profiles=records,
        code_sha256=file_sha(__file__),codec_sha256=file_sha(Path(__file__).with_name('fullframe_structure_stream.py'))))
    if selected is None:raise ValueError('No per-frame common profile fits')
    print('PERFRAME SELECTED',selected,flush=True)


def evaluate(batch,run,initial,case,ffmpeg):
    root=run/case;out=root/'evaluation_perframe';out.mkdir(exist_ok=False)
    profile=load_json(run/'perframe_selection.json')['selected']
    source,raw,digest,size=get_inputs(batch,case)
    seq=dict(source=source,raw=raw,A=np.load(initial/case/'grid16x9/receive/rgb.npy',mmap_mode='r'))
    receipts={};alpha={};unpacked={}
    for v in ('E','F'):
        folder=root/(profile+'_'+v);send=load_json(folder/'send.json');r=load_json(folder/'receive/receipt.json')
        if r['rgb_array_sha256']!=send['sender_expected_rgb_sha256'] or not r['source_files_unavailable']:raise ValueError('Receiver mismatch')
        for n,d in r['outputs_sha256'].items():
            if file_sha(folder/'receive'/n)!=d:raise ValueError('Receiver output changed')
        seq[v]=np.load(folder/'receive/rgb.npy',mmap_mode='r');alpha[v]=np.load(folder/'receive/alpha.npy',mmap_mode='r')
        _,unpacked[v]=unpack((folder/'fullframe.glcfull').read_bytes(),digest);receipts[v]=r
    if any(not np.array_equal(unpacked['E'][k],v) for k,v in unpacked['F'].items()):raise ValueError('Blend comparison differs in grids')
    if case=='video1':seq['local_U']=np.load(run.parent/'video1_persistent_aux_20261110_01/g17q4/receive/rgb.npy',mmap_mode='r')
    scores,rows,images=score(seq);save_csv(out/'temporal.csv',rows);save_csv(out/'spatial.csv',images)
    labels=dict(source='Source RGB',raw='Original GLC',A='A / initial full-frame',
        E='E / per-frame motion, narrow blend',F='F / per-frame motion, wider blend',local_U='Previous local coral U')
    specs=[('raw_vs_F.mp4',('raw','F')),('A_vs_F.mp4',('A','F')),('E_vs_F.mp4',('E','F'))]
    if case=='video1':specs.append(('local_U_vs_F.mp4',('local_U','F')))
    previews={}
    for name,names in specs:
        panels=[seq[k] for k in names];preview(ffmpeg,out/name,panels,15,0,[labels[k] for k in names])
        previews[name]=inspect_preview(ffmpeg,out/name,panels)
    coral=None
    if case=='video1':
        from run_video1_persistent_structure import score as coral_score,FIXED_BOX
        fixed=np.zeros(raw.shape[:3],np.uint8);x0,y0,x1,y1=FIXED_BOX;fixed[:,y0:y1,x0:x1]=1
        coral,coral_rows,_=coral_score(seq,fixed,start=0);coral=coral['fixed']
        save_csv(out/'fixed_coral_temporal.csv',coral_rows)
    save_json(out/'summary.json',dict(complete=True,case=case,profile=profile,frames=150,fps=15,
        scores=scores,coral=coral,base_bytes=size,receivers=receipts,previews=previews,
        coverage={v:dict(active=float((a>0).mean()),mean_weight=float(a.mean())) for v,a in alpha.items()},
        E_F_motion_and_masks_identical=True,code_sha256=file_sha(__file__),visual_review='pending'))
    print(case,'PERFRAME FULL',scores['full'],flush=True)
    if coral:print('PERFRAME CORAL',{k:(v['coarse_temporal_mae'],v['fine_ncc']) for k,v in coral.items()},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','receive','evaluate'))
    parser.add_argument('--batch',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--case',choices=CASES);parser.add_argument('--variant',choices=('E','F'))
    parser.add_argument('--profile',choices=[p[0] for p in PROFILES]);parser.add_argument('--initial',type=Path)
    parser.add_argument('--ffmpeg',type=Path)
    args=parser.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    if args.action=='prepare':prepare(args.batch.resolve(),args.run.resolve())
    elif args.action=='receive':
        if not all((args.case,args.variant,args.profile)):parser.error('--case --variant --profile required')
        receive(args.batch.resolve(),args.run.resolve(),args.case,args.profile+'_'+args.variant)
    else:
        if not all((args.case,args.initial,args.ffmpeg)):parser.error('--case --initial --ffmpeg required')
        evaluate(args.batch.resolve(),args.run.resolve(),args.initial.resolve(),args.case,args.ffmpeg.resolve())


if __name__=='__main__':main()
