"""Two fixed rate-targeted guide profiles; no candidate RGB used for selection."""

import argparse
from pathlib import Path
import subprocess

import cv2
import numpy as np

import structure_guide_stream as guide
from receiver_structure import file_sha,load_json,save_json

PROFILES={'G128_b8':((128,72),8),'G256_b12':((256,144),12)}


def encode(workspace,run,case):
    data=workspace/'02_实验数据/优化_GLC后处理'
    base=data/'source_dense_flow_three_clips_20261109_01'/case
    k=data/'support_refinement_20261111_01'/case/'K/support.glccsp2'
    ffmpeg=workspace/'01_项目/运行环境/FFmpeg/ffmpeg.exe'
    source=np.load(base/'source.npy',mmap_mode='r');digest=file_sha(base/'base.glcsnd')
    plan=run/'budget_guide_plan.json'
    if not plan.exists():
        save_json(plan,dict(frozen_before_budget_encode=True,date='2026-11-11',
            reason='Initial CRF profiles cost 44.07/54.73 kbps and failed the rate screen before any guide candidate was rendered; preserve initial probes.',
            profiles={n:dict(size=s,HEVC_target_kbps=b) for n,(s,b) in PROFILES.items()},
            encoding='libx265 two-pass average bitrate, medium, 15 FPS, 150 frames, yuv420p, one GOP; no informational encoder SEI',
            target_interpretation='8 and 12 kbps are experimental HEVC rate targets, not guaranteed whole-file rates or new hard requirements; bill the complete wrapper.',
            scope='video1 pilot; transfer a promising unchanged profile to video2/3 only if pilot quality warrants it',
            source_separation='Receiver only actual base and guide stream, decoded locally; source used by sender/evaluator',
            no_quality_search=True,transfer_screen='Lower full-file bytes than K and primary video1 coral coarse MAE within 5% of K; screen is not visual acceptance.',
            code_sha256=file_sha(__file__)))
    else:
        assert load_json(plan)['code_sha256']==file_sha(__file__)
    records={}
    for name,(size,bitrate) in PROFILES.items():
        folder=run/case/name;folder.mkdir(parents=True,exist_ok=False)
        low=np.stack([cv2.resize(v,size,interpolation=cv2.INTER_AREA) for v in source])
        common=[str(ffmpeg),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}',
                '-r','15','-i','pipe:0','-an','-c:v','libx265','-preset','medium','-b:v',f'{bitrate}k',
                '-pix_fmt','yuv420p','-threads','2']
        params='log-level=error:pools=2:frame-threads=1:keyint=150:min-keyint=150:scenecut=0:repeat-headers=1:info=0:stats=guide_pass.log'
        subprocess.run([*common,'-x265-params',params+':pass=1','-f','hevc','pipe:1'],
            input=low.tobytes(),stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,cwd=folder,check=True)
        encoded=subprocess.run([*common,'-x265-params',params+':pass=2','-f','hevc','pipe:1'],
            input=low.tobytes(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=folder,check=True)
        packet,h=guide.pack(encoded.stdout,digest,size)
        (folder/'guide.glcguide').write_bytes(packet)
        dh,decoded,expanded=guide.decode(packet,digest,ffmpeg)
        assert dh==h
        np.save(folder/'sender_expected_guide.npy',expanded,allow_pickle=False)
        info=dict(complete=True,profile=name,source_assisted_sender=True,header=h,
            HEVC_target_kbps=bitrate,bytes=len(packet),kbps=len(packet)*8/10000,
            base_bytes=(base/'base.glcsnd').stat().st_size,K_aux_bytes=k.stat().st_size,
            below_K=len(packet)<k.stat().st_size,sha256=file_sha(folder/'guide.glcguide'),
            raw_HEVC_bytes=len(encoded.stdout),decoded_guide_sha256=file_sha(folder/'sender_expected_guide.npy'),
            decoded_low_psnr=float(10*np.log10(255**2/np.mean((decoded.astype(np.float32)-low)**2))),
            frozen_source_sha256=file_sha(base/'source.npy'),code_sha256=file_sha(__file__))
        save_json(folder/'send.json',info);records[name]=info
        print(case,name,'complete bytes',len(packet),'aux kbps',info['kbps'],'guide PSNR',info['decoded_low_psnr'],flush=True)
    save_json(run/case/'budget_render_selection.json',dict(frozen_before_guide_receiver=True,
        profiles=[n for n,r in records.items() if r['below_K']],selection='Both fixed rate-targeted profiles whose complete bytes beat K; no candidate quality observed.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--case',choices=('video1','video2','video3'),required=True)
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    encode(a.workspace.resolve(),a.run.resolve(),a.case)
