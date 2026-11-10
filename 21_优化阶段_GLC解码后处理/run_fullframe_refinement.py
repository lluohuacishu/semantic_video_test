"""Denser temporally sampled motion and controlled boundary blending."""

import argparse
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from cache_motion_inputs import base_frames
from compare_source_dense_flow import load_flow
from coral_reference_sidecar import file_sha,load_json,save_json
from experiment_common import array_sha
from run_dense_batch import CASES
from run_fullframe_reference import get_inputs
from run_fullframe_auxiliary import quantize
from sparse_fullframe_stream import LIMIT,interpolate,pack,unpack,render,sampled_indices


PROFILES=(('g32b4',(32,18),4),('g32b8',(32,18),8),('g24b4',(24,14),4),('g24b8',(24,14),8))
PACKET='fullframe.glcfsp'


def safety(source,indices,keys,dense,reconstructed,alpha,block):
    result=[];y,x=np.mgrid[:288,:512].astype(np.float32)
    for j,(i,key) in enumerate(zip(indices,keys,strict=True)):
        flow=cv2.resize(reconstructed[j]/4.,(512,288),interpolation=cv2.INTER_LINEAR)
        mx,my=x+flow[...,0],y+flow[...,1]
        error=np.linalg.norm(flow-dense[j],axis=2)
        cur=cv2.GaussianBlur(source[i].astype(np.float32),(0,0),1)
        ref=cv2.GaussianBlur(source[key].astype(np.float32),(0,0),1)
        photo=np.max(np.abs(cur-cv2.remap(ref,mx,my,cv2.INTER_LINEAR)),axis=2)
        safe=(alpha[j]>0)&(error<1.25)&(photo<24)&(mx>=2)&(mx<510)&(my>=2)&(my<286)
        result.append(safe.reshape(288//block,block,512//block,block).all(axis=(1,3)).astype(np.uint8))
    return np.stack(result)


def prepare(batch,run):
    save_json(run/'refinement_plan.json',dict(profiles=PROFILES,selection='First common profile within 50000 B for all three clips; no metric-based per-case selection.',
        common_keys=[0,15,45,75,105,135],manual_edit_ROI=False,
        temporal_motion='Every third frame, plus segment endpoints. Interpolate within each reference segment only.',
        C='Denser motion; 2px inward feather; no tone compensation.',
        D='Exact same motion/masks as C; 8px inward feather and sigma12 current-base color matching only in low-texture areas.',
        diagnostic_limits='New schedule, grid, and temporal sampling change together versus A; C versus D isolates the combined blend change.',
        latency='Whole-file pilot; interpolation uses up to two future frames, not an online latency validation.'))
    results={};selected=None
    for name,grid,block in PROFILES:
        results[name]={}
        for case in CASES:
            root=run/case
            source,raw,digest,size=get_inputs(batch,case)
            keys=load_json(root/'plan.json')['keys'];events=[k+j for k in keys[1:] for j in range(6)]
            dense,_=load_flow(root/'flow',source,CASES[case][0],anchors=tuple(keys))
            trans_record=load_json(root/'transition_flow.json')
            if file_sha(root/'transition_flow.npy')!=trans_record['flow_sha256']:raise ValueError('Transition hash')
            old=np.load(root/'transition_flow.npy',mmap_mode='r')[:,0]
            record=load_json(root/'offline/summary.json')
            for f in ('alpha.npy','shared.npy'):
                if file_sha(root/'offline'/f)!=record['outputs_sha256'][f]:raise ValueError('Confidence changed')
            alpha=np.load(root/'offline/alpha.npy');shared=np.load(root/'offline/shared.npy')
            indices=sampled_indices(keys);old_indices=[j for j in range(len(events)) if j%6 in (0,3,5)]
            qnew=quantize(dense[indices],alpha[indices],grid)
            qold=quantize(old[old_indices],shared[old_indices],grid)
            new_expanded=interpolate(qnew,indices,150,keys)
            old_expanded=interpolate(qold,old_indices,len(events),list(range(0,len(events),6)))
            mnew=safety(source,list(range(150)),[keys[np.searchsorted(keys,i,side='right')-1] for i in range(150)],dense,new_expanded,alpha,block)
            mold=safety(source,events,[keys[j//6] for j in range(len(events))],old,old_expanded,shared,block)
            case_records={}
            for variant,soft in (('C',False),('D',True)):
                folder=root/(name+'_'+variant);folder.mkdir(exist_ok=False)
                payload,header=pack(qnew,qold,mnew,mold,digest,keys,soft=soft)
                (folder/PACKET).write_bytes(payload)
                info=dict(bytes=len(payload),kbps=len(payload)*8/10000,within_budget=len(payload)<=LIMIT,
                    base_bytes=size,base_kbps=size*8/10000,header=header,sha256=file_sha(folder/PACKET),
                    mask_coverage=float(mnew.mean()))
                if info['within_budget']:
                    h,a=unpack(payload,digest)
                    if not np.array_equal(a['new_motion'],qnew) or not np.array_equal(np.unpackbits(a['new_masks'],axis=2),mnew):raise ValueError('Wire roundtrip')
                    rgb,_=render(raw,h,a);info['expected_rgb_sha256']=array_sha(rgb)
                save_json(folder/'send.json',info);case_records[variant]=info
            results[name][case]=case_records
            print(name,case,'C/D bytes',[case_records[k]['bytes'] for k in ('C','D')],'coverage',float(mnew.mean()),flush=True)
        if all(v['D']['within_budget'] for v in results[name].values()):selected=name;break
    save_json(run/'refinement_selection.json',dict(complete=selected is not None,selected=selected,profiles=results,
        code_sha256=file_sha(__file__),codec_sha256=file_sha(Path(__file__).with_name('sparse_fullframe_stream.py'))))
    if selected is None:raise ValueError('No shared full-frame sparse profile fits')
    print('SELECTED',selected,flush=True)


def receive(batch,run,case,profile,variant):
    base=batch/case;folder=run/case/(profile+'_'+variant);out=folder/'receive';out.mkdir(exist_ok=False)
    allowed={base/'base.glcsnd',base/'base_cut.json',base/'base_receive/decoded_rgb.npy',base/'base_receive/result.json',folder/PACKET}
    observed=set()
    suffixes={'.json','.npy','.npz','.png','.jpg','.mp4','.mkv','.glcsnd','.glcfsp','.glcfull','.glcflow','.glcgate','.glcstruct'}
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.suffix.lower() in suffixes:
                if p not in allowed and not p.is_relative_to(out):raise RuntimeError('Untransmitted input blocked')
                if p in allowed:observed.add(str(p))
    sys.addaudithook(audit)
    for p in (base/'source.npy',folder/'send.json',run/case/'flow/to_anchor.npy'):
        try:p.read_bytes()
        except RuntimeError:pass
        else:raise ValueError('Input isolation failed')
    raw,digest,size=base_frames(base,CASES[case][0])
    h,a=unpack((folder/PACKET).read_bytes(),digest)
    rgb,alpha=render(raw,h,a)
    np.save(out/'rgb.npy',rgb,allow_pickle=False);np.save(out/'alpha.npy',alpha,allow_pickle=False)
    save_json(out/'receipt.json',dict(complete=True,source_and_sender_receipt_blocked=True,reads=sorted(observed),
        base_sha256=digest,base_bytes=size,auxiliary_sha256=file_sha(folder/PACKET),auxiliary_bytes=(folder/PACKET).stat().st_size,
        rgb_array_sha256=array_sha(rgb),outputs_sha256={p.name:file_sha(p) for p in out.glob('*.npy')},
        code_sha256=file_sha(__file__),codec_sha256=file_sha(Path(__file__).with_name('sparse_fullframe_stream.py'))))
    print(case,variant,'independent receiver PASS',flush=True)


def repack(batch,run):
    import json,lzma,shutil,zlib
    from sparse_fullframe_stream import HEADER,CRC
    selected='g32b8_opt';results={}
    for case in CASES:
        _,raw,digest,_=get_inputs(batch,case)
        source_folder=run/case/'g32b8_C'
        old_packet=(source_folder/PACKET).read_bytes()
        if file_sha(source_folder/PACKET)!=load_json(source_folder/'send.json')['sha256']:
            raise ValueError('Sender packet changed')
        # Sender-side inspection of its known over-budget diagnostic file; the
        # receiver unpacker retains the strict 50,000-byte admission limit.
        body=old_packet[:-4]
        if CRC.unpack(old_packet[-4:])[0]!=zlib.crc32(body):raise ValueError('Sender CRC')
        _,size=HEADER.unpack_from(body);h=json.loads(body[HEADER.size:HEADER.size+size]);pos=HEADER.size+size
        if h['base_sha256']!=digest:raise ValueError('Sender binding')
        arrays={}
        for name,spec in h['sections'].items():
            payload=lzma.decompress(body[pos:pos+spec['bytes']]);pos+=spec['bytes']
            if len(payload)!=spec['raw_bytes']:raise ValueError('Sender section length')
            a=np.frombuffer(payload,np.dtype(spec['dtype'])).reshape(spec['shape'])
            for axis in reversed(spec['axes']):a=np.cumsum(a,axis=axis,dtype=a.dtype)
            arrays[name]=a
        results[case]={}
        for variant,soft in (('C',False),('D',True)):
            folder=run/case/(selected+'_'+variant);folder.mkdir(exist_ok=False)
            payload,header=pack(arrays['new_motion'],arrays['old_motion'],
                np.unpackbits(arrays['new_masks'],axis=2),np.unpackbits(arrays['old_masks'],axis=2),digest,h['keys'],soft=soft)
            (folder/PACKET).write_bytes(payload)
            nh,na=unpack(payload,digest)
            if any(not np.array_equal(na[k],v) for k,v in arrays.items()):raise ValueError('Lossless repack changed grids')
            rgb,_=render(raw,nh,na)
            checks={};damaged=bytearray(payload);damaged[-8]^=1
            for label,data,binding in (('CRC',damaged,digest),('base',payload,'0'*64),('truncate',payload[:-1],digest),('budget',payload+b'0'*LIMIT,digest)):
                try:unpack(data,binding)
                except ValueError:checks[label]=True
                else:raise ValueError('Invalid packet accepted')
            record=dict(load_json(source_folder/'send.json'),bytes=len(payload),kbps=len(payload)*8/10000,
                within_budget=True,sha256=file_sha(folder/PACKET),header=header,expected_rgb_sha256=array_sha(rgb),
                lossless_motion_mask_repack=True,invalid_packet_checks=checks)
            save_json(folder/'send.json',record);results[case][variant]=record
        print(case,'32x18 lossless repack C/D',[results[case][v]['bytes'] for v in ('C','D')],flush=True)
    shutil.copy2(run/'refinement_selection.json',run/'refinement_selection_initial.json')
    save_json(run/'refinement_selection.json',dict(complete=True,selected=selected,profiles={selected:results},
        selection='Same 32x18/8px-mask profile across all clips after lossless predictor selection. No decoded motion or mask values changed.',
        code_sha256=file_sha(__file__),codec_sha256=file_sha(Path(__file__).with_name('sparse_fullframe_stream.py'))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','receive','repack'))
    parser.add_argument('--batch',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--case',choices=CASES);parser.add_argument('--profile',choices=[p[0] for p in PROFILES]+['g32b8_opt'])
    parser.add_argument('--variant',choices=('C','D'))
    args=parser.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    if args.action=='prepare':prepare(args.batch.resolve(),args.run.resolve())
    elif args.action=='repack':repack(args.batch.resolve(),args.run.resolve())
    else:
        if not all((args.case,args.profile,args.variant)):parser.error('--case --profile --variant required')
        receive(args.batch.resolve(),args.run.resolve(),args.case,args.profile,args.variant)


if __name__=='__main__':main()
