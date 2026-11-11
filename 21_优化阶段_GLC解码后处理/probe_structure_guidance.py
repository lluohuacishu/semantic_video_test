"""Freeze two alternative structural representations before candidate rendering."""

import argparse
import json
import lzma
from pathlib import Path
import subprocess

import cv2
import numpy as np

import compact_support_stream as compact
import predictive_support_stream as codec
import structure_guide_stream as guide
from receiver_structure import file_sha,load_json,save_json,KEYS
from fullframe_structure_stream import unpack as unpack_full
from run_compact_support import safety


PROFILES={'G128':((128,72),24),'G256':((256,144),30)}


def probe(workspace,run):
    data=workspace/'02_实验数据/优化_GLC后处理'
    base=data/'source_dense_flow_three_clips_20261109_01/video1'
    prior=data/'fullframe_reference_refined_20261110_01/video1'
    k=data/'support_refinement_20261111_01/video1/K'
    ffmpeg=workspace/'01_项目/运行环境/FFmpeg/ffmpeg.exe'
    inputs=[base/'source.npy',base/'base.glcsnd',base/'base_receive/decoded_rgb.npy',
        k/'support.glccsp2',k/'receive/rgb.npy',k/'receive/alpha.npy',
        prior/'g24q2_E/fullframe.glcfull',prior/'flow/to_anchor.npy',
        prior/'transition_flow.npy',prior/'offline/alpha.npy',prior/'offline/shared.npy']
    run.mkdir(parents=True,exist_ok=False)
    save_json(run/'execution_plan.json',dict(date='2026-11-11',status='frozen_before_probe',
        authorization='User rejected R0 video1 as nearly unchanged from raw GLC and requested other ideas.',
        pilot='video1, first 150 frames, 15 FPS, 512x288',
        temporal_probe=dict(degrees=[1,3],representation='Per-grid-node Bernstein trajectory per reference interval; new reference starts at zero; old references have free intercept',coefficient_unit_pixels=1/8,
            screen='Compare retained source-safe K support and coefficient/support payload bytes. Payload probes alone are not deployed packet rates.'),
        guide_profiles={n:dict(size=size,crf=crf) for n,(size,crf) in PROFILES.items()},
        guide_encoding='Source RGB downsampled by area; libx265 medium, yuv420p, 15 FPS; full wrapper+HEVC+CRC bytes billed',
        guide_candidate='Infer motion and all reliability tests from the decoded transmitted guide; reference RGB remains historical raw GLC; same keys and full-RGB renderer as K dense framework.',
        render_selection='Render both frozen guide profiles if each complete file is below K auxiliary bytes; no parameter search or per-frame selection.',
        source_access='Only sender and independent evaluator; candidate receiver sees base, actual guide packet and preinstalled RAFT weights.',
        foreground_limit='Low-resolution guides may miss thin moving boundaries; full-frame source previews and source-relative temporal diagnostics required, no acceptance from temporal MAE alone.',
        comparison='K and original GLC on fixed external evaluation support; old R0 is a rejected reference',
        input_sha256={str(p):file_sha(p) for p in inputs},
        source_code_sha256={p.name:file_sha(p) for p in (Path(__file__),Path(guide.__file__))}))
    source=np.load(base/'source.npy',mmap_mode='r')
    digest=file_sha(base/'base.glcsnd')
    kh,ka=codec.unpack((k/'support.glccsp2').read_bytes(),digest)
    _,fa=unpack_full((prior/'g24q2_E/fullframe.glcfull').read_bytes(),digest)
    motion=np.concatenate([fa['new_motion'],fa['old_motion']])
    intervals=list(zip(KEYS,(*KEYS[1:],150)))+[(150+i,156+i) for i in range(0,30,6)]
    records={}
    for degree in (1,3):
        folder=run/f'trajectory_degree{degree}';folder.mkdir()
        result=np.zeros_like(motion);coefficients=[]
        for start,stop in intervals:
            t=np.linspace(0,1,stop-start)
            basis=np.stack([(1-t),t],axis=1) if degree==1 else np.stack([(1-t)**3,3*t*(1-t)**2,3*t*t*(1-t),t**3],axis=1)
            fixed_zero=start<150
            fit=basis[:,1:] if fixed_zero else basis
            control=np.linalg.lstsq(fit,motion[start:stop].reshape(stop-start,-1),rcond=None)[0]
            q=np.rint(control*4).astype('<i2')
            coefficients.append(q.reshape(-1,*motion.shape[1:]))
            result[start:stop]=np.rint((fit@(q.astype(np.float64)/4)).reshape(stop-start,*motion.shape[1:])).astype('<i2')
        coeff=np.concatenate(coefficients)
        safe=safety(source,prior,result)
        support=compact.reduce_support(safe,4,5)
        candidates=codec.support_candidates(support)
        sp,ss=min(candidates,key=lambda x:len(x[0])+len(json.dumps(x[1],separators=(',',':'))))
        cps=[lzma.compress(compact.motion_bytes(coeff,axes,layout)) for axes in ((),*compact.AXES) for layout in ('i16','zigzag')]
        old_pixels=int(compact.expand_support(ka['support'],5)[:150].sum())
        retained=int(compact.expand_support(support & ka['support'],5)[:150].sum())
        record=dict(degree=degree,coeff_shape=coeff.shape,coefficient_payload_bytes=min(map(len,cps)),
            support_payload_bytes=len(sp),payload_only_bytes=min(map(len,cps))+len(sp),
            K_needed_cells_retained_fraction=retained/old_pixels,
            support_fraction=float(support.mean()),K_support_fraction=float(ka['support'].mean()),
            actual_parametric_packet=False,source_safety_unchanged=True)
        np.save(folder/'motion.npy',result,allow_pickle=False)
        np.save(folder/'coefficients.npy',coeff,allow_pickle=False)
        np.save(folder/'support.npy',support,allow_pickle=False)
        save_json(folder/'summary.json',record);records[f'T{degree}']=record
        print('TEMPORAL PROBE',degree,record,flush=True)
    for name,(size,crf) in PROFILES.items():
        folder=run/'video1'/name;folder.mkdir(parents=True)
        low=np.stack([cv2.resize(v,size,interpolation=cv2.INTER_AREA) for v in source])
        encoded=subprocess.run([str(ffmpeg),'-v','error','-f','rawvideo','-pix_fmt','rgb24',
            '-s',f'{size[0]}x{size[1]}','-r','15','-i','pipe:0','-an','-c:v','libx265',
            '-preset','medium','-crf',str(crf),'-pix_fmt','yuv420p','-threads','2',
            '-x265-params','log-level=error:pools=2:frame-threads=1:keyint=150:min-keyint=150:scenecut=0:repeat-headers=1',
            '-f','hevc','pipe:1'],input=low.tobytes(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
        packet,h=guide.pack(encoded.stdout,digest,size)
        (folder/'guide.glcguide').write_bytes(packet)
        info=dict(complete=True,profile=name,source_assisted_sender=True,header=h,crf=crf,
            bytes=len(packet),kbps=len(packet)*8/10000,base_bytes=(base/'base.glcsnd').stat().st_size,
            K_aux_bytes=(k/'support.glccsp2').stat().st_size,below_K=len(packet)<(k/'support.glccsp2').stat().st_size,
            sha256=file_sha(folder/'guide.glcguide'),raw_HEVC_bytes=len(encoded.stdout))
        if len(packet)<=guide.LIMIT:
            dh,decoded,expanded=guide.decode(packet,digest,ffmpeg)
            assert dh==h
            info['decoded_low_psnr']=float(10*np.log10(255**2/np.mean((decoded.astype(np.float32)-low)**2)))
            np.save(folder/'sender_expected_guide.npy',expanded,allow_pickle=False)
        save_json(folder/'send.json',info);records[name]=info
        print('GUIDE BYTE PROBE',name,info,flush=True)
    selected=[n for n in PROFILES if records[n]['below_K']]
    save_json(run/'probe_summary.json',dict(complete=True,records=records))
    save_json(run/'render_selection.json',dict(frozen_before_receiver=True,profiles=selected,
        basis='Both preregistered guide profiles if complete auxiliary bytes below K; no candidate quality observed.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    probe(a.workspace.resolve(),a.run.resolve())
