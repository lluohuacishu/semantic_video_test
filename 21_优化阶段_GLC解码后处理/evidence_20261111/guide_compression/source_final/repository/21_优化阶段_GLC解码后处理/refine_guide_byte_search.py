"""Resolve integer-kbps gaps using file bytes, before downstream quality selection."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from receiver_structure import file_sha,load_json,save_json
from compress_structure_guide import receive
import structure_guide_stream as guide


def encode(workspace,run,name,chroma):
    folder=run/'video1'/name;folder.mkdir(parents=True,exist_ok=False)
    base=run.parent/'source_dense_flow_three_clips_20261109_01/video1'
    ffmpeg=workspace/'01_项目/运行环境/FFmpeg/ffmpeg.exe'
    save_json(folder/'plan.json',dict(frozen_before_encode=True,profile=name,preset='slower',
        integer_HEVC_kbps_candidates=[6,7,5],cap_bytes=10000,chroma_offset=chroma,
        cause='x265 log reports integer bitrate; previous multiplicative search tried 7,5,9,7 and skipped 6. Resolve the skipped integer level; retain all original attempts.',
        selection='First level within 97%-100% cap, otherwise largest of this bounded list below cap. Bytes only.',
        unchanged='256x144/15FPS, no preprocessing, exact accepted T3 receiver',code_sha256=file_sha(__file__)))
    source=np.load(base/'source.npy',mmap_mode='r')
    low=np.stack([cv2.resize(v,(256,144),interpolation=cv2.INTER_AREA) for v in source])
    digest=file_sha(base/'base.glcsnd');attempts=[];started=time.perf_counter()
    for rate in (6,7,5):
        trial=folder/f'encode_{rate}';trial.mkdir()
        params=('log-level=error:pools=2:frame-threads=1:keyint=150:min-keyint=150:'
            'scenecut=0:repeat-headers=1:info=0:stats=guide_pass.log')
        if chroma:params+=f':cbqpoffs={chroma}:crqpoffs={chroma}'
        common=[str(ffmpeg),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','256x144',
            '-r','15','-i','pipe:0','-an','-c:v','libx265','-preset','slower','-b:v',f'{rate}k',
            '-pix_fmt','yuv420p','-threads','2']
        tick=time.perf_counter()
        for pass_number in (1,2):
            r=subprocess.run([*common,'-x265-params',params+f':pass={pass_number}','-f','hevc','pipe:1'],
                input=low.tobytes(),stdout=subprocess.PIPE if pass_number==2 else subprocess.DEVNULL,
                stderr=subprocess.PIPE,cwd=trial,check=True)
            (trial/f'pass{pass_number}_stderr.txt').write_bytes(r.stderr)
        packet,h=guide.pack(r.stdout,digest,(256,144));(trial/'guide.glcguide').write_bytes(packet)
        row=dict(HEVC_target_kbps=rate,bytes=len(packet),payload_bytes=len(r.stdout),
            seconds=time.perf_counter()-tick,within_cap=len(packet)<=10000,
            command=common,x265_params=params,sha256=file_sha(trial/'guide.glcguide'))
        save_json(trial/'encode.json',row);attempts.append(row)
        print(name,'integer HEVC',rate,'full bytes',len(packet),flush=True)
        if 9700<=len(packet)<=10000:break
    valid=[a for a in attempts if a['within_cap']]
    if not valid:raise ValueError('No supplementary encoding below cap')
    selected=max(valid,key=lambda a:a['bytes'])
    shutil.copy2(folder/f"encode_{selected['HEVC_target_kbps']}"/'guide.glcguide',folder/'guide.glcguide')
    data=(folder/'guide.glcguide').read_bytes();h,decoded,expanded=guide.decode(data,digest,ffmpeg)
    np.save(folder/'sender_expected_guide.npy',expanded,allow_pickle=False)
    save_json(folder/'send.json',dict(complete=True,profile=name,source_assisted_sender=True,header=h,
        bytes=len(data),kbps=len(data)*8/10000,cap_bytes=10000,preset='slower',chroma_offset=chroma,
        base_bytes=(base/'base.glcsnd').stat().st_size,reference_aux_bytes=16187,
        saving_reference_percent=(1-len(data)/16187)*100,K_aux_bytes=24649,
        sha256=file_sha(folder/'guide.glcguide'),decoded_guide_sha256=file_sha(folder/'sender_expected_guide.npy'),
        decoded_low_psnr=float(10*np.log10(255**2/np.mean((decoded.astype(np.float32)-low)**2))),
        attempts=attempts,selected_HEVC_target_kbps=selected['HEVC_target_kbps'],
        selection='Bounded integer byte-only search; no downstream quality used',
        encode_seconds_all_trials=time.perf_counter()-started,selected_two_pass_encode_seconds=selected['seconds'],
        frozen_source_sha256=file_sha(base/'source.npy'),code_sha256=file_sha(__file__)))
    return name


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--chroma',type=int,choices=(0,4),default=0)
    p.add_argument('--encode-only',action='store_true')
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    name='S8C4' if a.chroma else 'S8I'
    encode(a.workspace.resolve(),a.run.resolve(),name,a.chroma)
    if not a.encode_only:receive(a.workspace.resolve(),a.run.resolve(),name)
