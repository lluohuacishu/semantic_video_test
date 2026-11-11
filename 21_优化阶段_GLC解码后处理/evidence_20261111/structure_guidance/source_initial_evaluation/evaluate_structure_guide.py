"""Evaluate actual guide-assisted GLC output separately from receiver inputs."""

import argparse
import copy
from pathlib import Path
import zlib

import cv2
import numpy as np
from PIL import Image,ImageDraw

import structure_guide_stream as guide
from receiver_structure import file_sha,load_json,save_json,render
from run_fullframe_reference import reliable as legacy_reliable
from compare_deflicker import save_csv
import evaluate_fullframe_reference as scoring
from run_fullframe_factorial import BOX
from run_p0 import preview
from verify_video1_persistent_structure import inspect_preview


def motion_change_diagnostic(sequences):
    """Source-selected high-change pixels, without candidate-dependent masks."""
    records=[]
    for i in range(1,150):
        source_delta=sequences['source'][i].astype(np.float32)-sequences['source'][i-1]
        magnitude=np.max(np.abs(source_delta),axis=2)
        mask=magnitude>=max(12,float(np.quantile(magnitude,.9)))
        n=int(mask.sum())
        for name,value in sequences.items():
            change=value[i].astype(np.float32)-value[i-1]
            records.append(dict(frame=i,method=name,pixels=n,
                delta_error_sum=float(np.abs(change-source_delta)[mask].sum()),
                change_sum=float(np.abs(change)[mask].sum()),
                source_change_sum=float(np.abs(source_delta)[mask].sum())))
    result={}
    for name in sequences:
        rows=[r for r in records if r['method']==name];n=sum(r['pixels'] for r in rows)
        result[name]=dict(pixels=n,RGB_delta_error=sum(r['delta_error_sum'] for r in rows)/(n*3),
            change_amplitude_ratio=sum(r['change_sum'] for r in rows)/sum(r['source_change_sum'] for r in rows))
    return result,records


def altered(data,field,value):
    import json
    magic,size=guide.PREFIX.unpack_from(data)
    h=json.loads(data[guide.PREFIX.size:guide.PREFIX.size+size]);h[field]=value
    encoded=json.dumps(h,separators=(',',':')).encode('ascii')
    body=guide.PREFIX.pack(magic,len(encoded))+encoded+data[guide.PREFIX.size+size:-4]
    return body+guide.CRC.pack(zlib.crc32(body))


def evaluate(workspace,run,case,profiles):
    data=run.parent;ffmpeg=workspace/'01_项目/运行环境/FFmpeg/ffmpeg.exe'
    base=data/'source_dense_flow_three_clips_20261109_01'/case
    old=data/'support_refinement_20261111_01'/case
    out=run/case/'evaluation';out.mkdir(exist_ok=False)
    source=np.load(base/'source.npy',mmap_mode='r')
    raw=np.load(base/'base_receive/decoded_rgb.npy',mmap_mode='r')
    sequences=dict(source=source,raw=raw,K=np.load(old/'K/receive/rgb.npy',mmap_mode='r'))
    if case=='video1':
        sequences['R0']=np.load(data/'receiver_structure_20261111_01/video1/RR/rgb.npy',mmap_mode='r')
        sequences['U']=np.load(data/'video1_persistent_aux_20261110_01/g17q4/receive/rgb.npy',mmap_mode='r')
    checks={};rates={};coverage={};receipts={}
    scoring.SCOPES.pop('coral',None)
    if case=='video1':scoring.SCOPES['coral']=BOX
    for name in profiles:
        folder=run/case/name;rec=folder/'receive'
        receipt=load_json(rec/'receipt.json');send=load_json(folder/'send.json')
        assert receipt['source_reads_blocked'] and len(receipt['reads'])==5 and len(receipt['blocked_inputs'])==7
        assert receipt['auxiliary_sha256']==send['sha256']==file_sha(folder/'guide.glcguide')
        assert receipt['auxiliary_bytes']==send['bytes']==(folder/'guide.glcguide').stat().st_size
        for filename,digest in receipt['outputs_sha256'].items():assert file_sha(rec/filename)==digest
        decoded=np.load(rec/'guide_rgb.npy',mmap_mode='r')
        assert np.array_equal(decoded,np.load(folder/'sender_expected_guide.npy',mmap_mode='r'))
        back,forward,transition=[np.load(rec/(n+'.npy'),mmap_mode='r')
            for n in ('to_anchor','from_anchor','transition_flow')]
        replay=render(raw,decoded,back,forward,transition,gate_fn=legacy_reliable)
        for field,value in replay.items():assert np.array_equal(value,np.load(rec/(field+'.npy'),mmap_mode='r'))
        del replay
        sequences[name]=np.load(rec/'rgb.npy',mmap_mode='r')
        alpha=np.load(rec/'alpha.npy',mmap_mode='r')
        assert np.array_equal(sequences[name][alpha==0],raw[alpha==0])
        coverage[name]={scope:dict(active_fraction=float((alpha[:,y0:y1,x0:x1]>0).mean()),
            mean_alpha=float(alpha[:,y0:y1,x0:x1].mean())) for scope,(x0,y0,x1,y1) in scoring.SCOPES.items()}
        packet=(folder/'guide.glcguide').read_bytes();digest=receipt['base_sha256']
        damaged=bytearray(packet);damaged[-10]^=1
        bad=[('CRC',damaged,digest),('truncated',packet[:-1],digest),('wrong_base',packet,'0'*64),
             ('over_cap',packet+bytes(guide.LIMIT),digest)]
        bad += [(field,altered(packet,field,value),digest) for field,value in
                (('frames',149),('fps',30),('guide_size',[512,288]),('target_size',[128,72]),
                 ('payload_bytes',1),('codec','unknown'),('upsample','nearest'))]
        negatives={}
        for label,payload,binding in bad:
            try:guide.unpack(payload,binding)
            except ValueError:negatives[label]=True
            else:raise ValueError('Malformed guide accepted: '+label)
        checks[name]=dict(receiver_source_blocked=True,guide_decode_exact=True,
            legacy_renderer_replay_exact=True,zero_alpha_unchanged=True,negative_packets=negatives)
        rates[name]=dict(base_bytes=receipt['base_bytes'],auxiliary_bytes=send['bytes'],
            auxiliary_kbps=send['kbps'],total_kbps=receipt['total_kbps'],K_aux_bytes=send['K_aux_bytes'],
            auxiliary_saving_percent=(1-send['bytes']/send['K_aux_bytes'])*100)
        receipts[name]=receipt
    scores,temporal,spatial=scoring.score(sequences)
    archived=load_json(old/'evaluation/summary.json')['scores']
    for scope in scoring.SCOPES:
        for name in ('raw','K'):
            for field in ('coarse_mae','fine_ncc','psnr_db','valid_pixels'):
                assert abs(scores[scope][name][field]-archived[scope][name][field])<1e-9
    save_csv(out/'temporal.csv',temporal);save_csv(out/'spatial.csv',spatial)
    dynamic,rows=motion_change_diagnostic(sequences);save_csv(out/'source_change_diagnostic.csv',rows)
    labels=dict(source='Source RGB [evaluation]',raw='Original GLC',K='K / original auxiliary',R0='Rejected R0 / aux 0',U='Prior local U')
    labels.update({n:f'{n} / guide {rates[n]["auxiliary_kbps"]:.2f} kbps' for n in profiles})
    specs=[]
    for name in profiles:
        specs.extend([(f'K_vs_{name}_full.mp4',('K',name),None),
            (f'source_raw_K_{name}.mp4',('source','raw','K',name),None)])
        if case=='video1':specs.append((f'K_vs_{name}_coral.mp4',('K',name),(250,0,512,132)))
    previews={}
    for filename,names,crop in specs:
        panels=[sequences[n] for n in names]
        region=None if crop is None else np.s_[crop[1]:crop[3],crop[0]:crop[2]]
        preview(ffmpeg,out/filename,panels,15,0,[labels[n] for n in names],crop=region)
        previews[filename]=inspect_preview(ffmpeg,out/filename,panels,crop=crop)
    for name in profiles:
        canvas=Image.new('RGB',(512*4,312*3),(24,35,48));draw=ImageDraw.Draw(canvas)
        for row,i in enumerate((20,80,149)):
            for col,label in enumerate(('source','raw','K',name)):
                canvas.paste(Image.fromarray(np.asarray(sequences[label][i])),(col*512,row*312+24))
                draw.text((col*512+6,row*312+5),f'{case} {label} f{i}',fill='white')
        canvas.save(out/(name+'_contact.png'))
    scope='coral' if case=='video1' else 'full'
    result=dict(complete=True,case=case,profiles=profiles,rates=rates,scores=scores,coverage=coverage,
        verification=checks,baseline_scores_exact=True,previews=previews,
        source_change_diagnostic=dict(explanation='Source-only top-decile temporal RGB change pixels (threshold at least 12), includes camera-motion edges; not semantic foreground segmentation or motion acceptance.',scores=dynamic),
        runtime={n:{k:v[k] for k in ('flow_seconds','render_seconds','guide_decode_seconds','peak_allocated_mib')} for n,v in receipts.items()},
        transfer_numeric_screen={n:bool(rates[n]['auxiliary_bytes']<rates[n]['K_aux_bytes'] and
            scores[scope][n]['coarse_mae']<=scores[scope]['K']['coarse_mae']*1.05) for n in profiles},
        visual_acceptance='pending',code_sha256=file_sha(__file__))
    save_json(out/'summary.json',result)
    print(case,scope,{n:(round(v['coarse_mae'],6),round(v['fine_ncc'],6)) for n,v in scores[scope].items()},flush=True)
    print('NUMERIC TRANSFER SCREEN',result['transfer_numeric_screen'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--case',choices=('video1','video2','video3'),required=True)
    p.add_argument('--profiles',nargs='+',required=True)
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    evaluate(a.workspace.resolve(),a.run.resolve(),a.case,a.profiles)
