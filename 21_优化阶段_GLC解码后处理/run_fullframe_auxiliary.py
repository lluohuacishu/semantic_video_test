"""Full-frame auxiliary profiles selected only by bytes across all three clips."""

import argparse
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from cache_motion_inputs import base_frames
from coral_reference_sidecar import file_sha,load_json,save_json
from experiment_common import array_sha
from fullframe_structure_stream import EVENTS,KEYS,LIMIT,expanded_motion,pack,unpack,render
from run_dense_batch import CASES
from run_fullframe_reference import get_inputs


PROFILES=(('grid16x9',(16,9)),('grid12x7',(12,7)),('grid8x5',(8,5)))
PACKET='fullframe.glcfull'


def quantize(flow,alpha,grid,scale=4):
    gy,gx=np.mgrid[:grid[1],:grid[0]]
    basis=np.stack((gx,gy,np.ones_like(gx)),axis=-1)
    result=[]
    for f,a in zip(flow,alpha,strict=True):
        denominator=cv2.resize(a,grid,interpolation=cv2.INTER_AREA)
        numerator=cv2.resize(f*a[...,None],grid,interpolation=cv2.INTER_AREA)
        value=numerator/np.maximum(denominator[...,None],1e-6)
        valid=denominator>.01
        if valid.sum()>=3:
            fit=np.linalg.lstsq(basis[valid],value[valid],rcond=None)[0]
            value[~valid]=(basis@fit)[~valid]
        else:value[:]=0
        result.append(value)
    q=np.rint(np.stack(result)*scale)
    if not np.isfinite(q).all() or np.abs(q).max()>4096:raise ValueError('Grid flow out of range')
    return q.astype('<i2')


def compact_safety(source,indices,keys,dense,grid,alpha):
    result=[];y,x=np.mgrid[:288,:512].astype(np.float32)
    for j,(i,key) in enumerate(zip(indices,keys,strict=True)):
        qflow=expanded_motion(grid[j]);mx,my=x+qflow[...,0],y+qflow[...,1]
        difference=np.linalg.norm(qflow-dense[j],axis=2)
        current=cv2.GaussianBlur(source[i].astype(np.float32),(0,0),1)
        old=cv2.GaussianBlur(source[key].astype(np.float32),(0,0),1)
        photo=np.max(np.abs(current-cv2.remap(old,mx,my,cv2.INTER_LINEAR)),axis=2)
        safe=(alpha[j]>0)&(difference<1.25)&(photo<24)&(mx>=2)&(mx<510)&(my>=2)&(my<286)
        # Every source pixel in each 8x8 block must pass; no small holes are filled.
        blocks=safe.reshape(36,8,64,8).all(axis=(1,3))
        result.append(blocks.astype(np.uint8))
    return np.stack(result)


def prepare(batch,run):
    save_json(run/'auxiliary_plan.json',dict(common_profiles=PROFILES,
        selection='First profile fitting <=50000 bytes for every clip; no metric/visual selection.',
        full_frame=True,hand_annotations=False,mask_block=[8,8],motion_scale=4,
        quantized_motion_max_error_px=1.25,source_sigma1_max_rgb_error=24,
        mask_policy='All 64 pixels in each block must pass the source reliability and reconstructed-grid checks.',
        known_limit='Conservative blocks can lose coverage. Automatic correspondence confidence is not fish segmentation.'))
    all_records={}
    selected=None
    for name,grid in PROFILES:
        all_records[name]={}
        for case in CASES:
            root=run/case;out=root/name;out.mkdir(exist_ok=False)
            source,raw,digest,base_size=get_inputs(batch,case)
            record=load_json(root/'offline/summary.json')
            for filename,d in record['outputs_sha256'].items():
                if file_sha(root/'offline'/filename)!=d:raise ValueError('Offline output changed')
            alpha=np.load(root/'offline/alpha.npy');shared=np.load(root/'offline/shared.npy')
            dense=np.load(root/'flow/to_anchor.npy',mmap_mode='r')
            old=np.load(root/'transition_flow.npy',mmap_mode='r')[:,0]
            qnew=quantize(dense,alpha,grid);qold=quantize(old,shared,grid)
            mnew=compact_safety(source,list(range(150)),[i-i%30 for i in range(150)],dense,qnew,alpha)
            mold=compact_safety(source,EVENTS,[i-i%30-30 for i in EVENTS],old,qold,shared)
            payload,header=pack(qnew,qold,mnew,mold,digest)
            with (out/PACKET).open('xb') as f:f.write(payload)
            fits=len(payload)<=LIMIT
            info=dict(bytes=len(payload),kbps=len(payload)*8/10000,within_budget=fits,
                base_bytes=base_size,base_kbps=base_size*8/10000,header=header,sha256=file_sha(out/PACKET),
                mean_safe_block_fraction=float(mnew.mean()))
            if fits:
                h,a=unpack((out/PACKET).read_bytes(),digest)
                if not np.array_equal(a['new_motion'],qnew) or not np.array_equal(np.unpackbits(a['new_safe'],axis=2),mnew):
                    raise ValueError('Packet roundtrip failed')
                expected,_=render(raw,h,a)
                info['sender_expected_rgb_sha256']=array_sha(expected)
                checks={}
                damaged=bytearray(payload);damaged[-8]^=1
                for label,data,d in (('CRC',damaged,digest),('wrong_base',payload,'0'*64),
                        ('truncated',payload[:-1],digest),('over_budget',payload+b'0'*LIMIT,digest)):
                    try:unpack(data,d)
                    except ValueError:checks[label]=True
                    else:raise ValueError('Malformed packet accepted')
                info['negative_packet_checks']=checks
            save_json(out/'send.json',info);all_records[name][case]=info
            print(name,case,'bytes',len(payload),'block coverage',float(mnew.mean()),flush=True)
        if all(v['within_budget'] for v in all_records[name].values()):
            selected=name;break
    save_json(run/'auxiliary_selection.json',dict(complete=selected is not None,selected=selected,
        profiles=all_records,selection_uses_only_bytes=True,code_sha256=file_sha(__file__),
        codec_sha256=file_sha(Path(__file__).with_name('fullframe_structure_stream.py'))))
    if selected is None:raise ValueError('No common full-frame profile met the lane budget')
    print('COMMON PROFILE',selected,flush=True)


def receive(batch,run,case,profile):
    base=batch/case;folder=run/case/profile;output=folder/'receive';output.mkdir(exist_ok=False)
    allowed={base/'base.glcsnd',base/'base_cut.json',base/'base_receive/decoded_rgb.npy',base/'base_receive/result.json',folder/PACKET}
    observed=set()
    forbidden={'.json','.npy','.npz','.png','.jpg','.mp4','.mkv','.glcsnd','.glcfull','.glcflow','.glcgate','.glcstruct'}
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.suffix.lower() in forbidden:
                if p not in allowed and not p.is_relative_to(output):raise RuntimeError('Receiver blocked untransmitted input: '+str(p))
                if p in allowed:observed.add(str(p))
    sys.addaudithook(audit)
    blocked=[]
    for path in (base/'source.npy',folder/'send.json',run/case/'flow/to_anchor.npy'):
        try:path.read_bytes()
        except RuntimeError:blocked.append(str(path))
        else:raise ValueError('Receiver guard failed')
    raw,digest,size=base_frames(base,CASES[case][0])
    h,a=unpack((folder/PACKET).read_bytes(),digest)
    output_rgb,alpha=render(raw,h,a)
    np.save(output/'rgb.npy',output_rgb,allow_pickle=False);np.save(output/'alpha.npy',alpha,allow_pickle=False)
    save_json(output/'receipt.json',dict(complete=True,case=case,profile=profile,
        reads=sorted(observed),blocked_inputs=blocked,source_files_unavailable=True,
        independently_verified_base=True,base_sha256=digest,base_bytes=size,
        aux_bytes=(folder/PACKET).stat().st_size,aux_sha256=file_sha(folder/PACKET),
        rgb_array_sha256=array_sha(output_rgb),outputs_sha256={p.name:file_sha(p) for p in output.glob('*.npy')},
        code_sha256=file_sha(__file__),codec_sha256=file_sha(Path(__file__).with_name('fullframe_structure_stream.py'))))
    print(case,'full-frame independent receive PASS',flush=True)


def tone_packets(batch,run):
    selected=load_json(run/'auxiliary_selection.json')['selected']
    name=selected+'_tone12';records={}
    for case in CASES:
        source,raw,digest,_=get_inputs(batch,case)
        old=run/case/selected
        h,a=unpack((old/PACKET).read_bytes(),digest)
        payload,header=pack(a['new_motion'],a['old_motion'],np.unpackbits(a['new_safe'],axis=2),
            np.unpackbits(a['old_safe'],axis=2),digest,tone_sigma=12)
        folder=run/case/name;folder.mkdir(exist_ok=False)
        (folder/PACKET).write_bytes(payload)
        new_h,new_a=unpack(payload,digest)
        if any(not np.array_equal(new_a[k],v) for k,v in a.items()):raise ValueError('Tone control changed auxiliary grids')
        result,_=render(raw,new_h,new_a)
        receipt=dict(load_json(old/'send.json'),bytes=len(payload),kbps=len(payload)*8/10000,
            within_budget=len(payload)<=LIMIT,header=header,sha256=file_sha(folder/PACKET),
            sender_expected_rgb_sha256=array_sha(result),original_auxiliary=str(old/PACKET),
            only_change='Receiver matches sigma12 RGB color to current base frame before blending; all motion and mask arrays identical.')
        save_json(folder/'send.json',receipt);records[case]=receipt
        print(case,'tone control',len(payload),'bytes',flush=True)
    save_json(run/'tone_selection.json',dict(complete=True,selected=name,profiles={name:records},
        no_parameter_sweep=True,fixed_sigma=12,code_sha256=file_sha(__file__)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','receive','tone'))
    parser.add_argument('--batch',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--case',choices=CASES);parser.add_argument('--profile',choices=[x[0]+suffix for x in PROFILES for suffix in ('','_tone12')])
    args=parser.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    if args.action=='prepare':prepare(args.batch.resolve(),args.run.resolve())
    elif args.action=='tone':tone_packets(args.batch.resolve(),args.run.resolve())
    else:
        if not args.case or not args.profile:parser.error('--case and --profile required')
        receive(args.batch.resolve(),args.run.resolve(),args.case,args.profile)


if __name__=='__main__':main()
