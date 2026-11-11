"""Controlled guide rate/preset experiment, with the accepted T3 receiver fixed."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from receiver_structure import file_sha,load_json,save_json
import structure_guide_stream as guide


PROFILES = {
    'M10': dict(preset='medium',cap_bytes=12500,chroma_offset=0),
    'S10': dict(preset='slower',cap_bytes=12500,chroma_offset=0),
    'M8': dict(preset='medium',cap_bytes=10000,chroma_offset=0),
    'S8': dict(preset='slower',cap_bytes=10000,chroma_offset=0),
}


def freeze(workspace,run):
    stage=Path(__file__).resolve().parent
    data=workspace/'02_实验数据/优化_GLC后处理'
    base=data/'source_dense_flow_three_clips_20261109_01/video1'
    prior=data/'structure_guidance_20261111_01/video1'
    k=data/'support_refinement_20261111_01/video1/K'
    paths=[base/'source.npy',base/'base.glcsnd',base/'base_receive/decoded_rgb.npy',base/'base_receive/result.json',
        prior/'G256_b12_T3/guide.glcguide',prior/'G256_b12_T3/receive/rgb.npy',
        prior/'G256_b12_T3/receive/alpha.npy',prior/'G256_b12_T3/receive/receipt.json',prior/'evaluation/summary.json',
        k/'support.glccsp2',k/'receive/rgb.npy',k/'receive/alpha.npy',
        data/'deflicker_reef80_20261108_01/upstream/pretrained_weights/raft-things.pth']
    run.mkdir(parents=True,exist_ok=False)
    save_json(run/'execution_plan.json',dict(date='2026-11-11',frozen_before_encoding=True,
        authorization='User accepted the current video1 guide result and asked to proceed with bitrate compression before visual improvements.',
        user_accepted_baseline='G256_b12_T3: 16187 B, 12.9496 kbps; this is the new rate comparison, keep K as historical regression.',
        case='video1',frames=150,fps=15,guide_size=[256,144],base_size=[512,288],
        initial_profiles=PROFILES,
        receiver='Identical RAFT-Things FP32/24 iterations; same six anchors, same T3 texture std>=3, 2px feather, old-reference transition and RGB reuse.',
        unchanged=['resolution','frame rate','source downsampling','chroma 4:2:0','receiver gate thresholds','reference RGB','blend and rounding','GLC base'],
        encoder='libx265 two-pass ABR; medium versus slower as complete presets; 150-frame GOP; no encoder info SEI',
        byte_selection='At most four two-pass encodes per profile to approach but not exceed the complete-file cap. Select largest <=cap using bytes only, without candidate RGB or metrics.',
        target_cap_interpretation='10/8 kbps are experimental operating points, not new user hard limits; all wrapper, HEVC and CRC bytes counted.',
        quality_screen=dict(reference='accepted T3',coral_coarse_MAE_ratio_max=1.05,full_coarse_MAE_ratio_max=1.05,
            coral_NCC_drop_max=.02,full_PSNR_drop_max_db=.15,source_change_error_ratio_max=1.03,
            interpretation='Predeclared candidate prioritization only, never visual acceptance or a newly imposed user requirement.'),
        followup='After initial 2x2, one fixed chroma-offset +4 at slower/8kbps may be evaluated as a separate factor if useful; no resolution, frame dropping, smoothing or T3 retuning in this run.',
        evaluation='Fixed external source correspondence; full/quadrants/coral plus source-selected high-change pixels; source only in independent evaluation.',
        limits='Video1 development pilot; no fresh-data, long-video, realtime or equal-total-byte H265 claim.',
        inputs_sha256={str(p):file_sha(p) for p in paths},
        code_sha256={n:file_sha(stage/n) for n in ('compress_structure_guide.py','structure_guide_stream.py',
            'receive_structure_guide.py','receive_guide_texture.py','receiver_structure.py')}))


def encode(workspace,run,name,spec):
    data=workspace/'02_实验数据/优化_GLC后处理';base=data/'source_dense_flow_three_clips_20261109_01/video1'
    ffmpeg=workspace/'01_项目/运行环境/FFmpeg/ffmpeg.exe';folder=run/'video1'/name
    folder.mkdir(parents=True,exist_ok=False)
    source=np.load(base/'source.npy',mmap_mode='r')
    low=np.stack([cv2.resize(v,(256,144),interpolation=cv2.INTER_AREA) for v in source])
    digest=file_sha(base/'base.glcsnd');cap=spec['cap_bytes']
    target=(cap-252)*8/10000*.98
    attempts=[];start=time.perf_counter()
    for j in range(4):
        trial=folder/f'encode_{j}';trial.mkdir()
        params=('log-level=error:pools=2:frame-threads=1:keyint=150:min-keyint=150:'
                'scenecut=0:repeat-headers=1:info=0:stats=guide_pass.log')
        if spec['chroma_offset']:
            params+=f":cbqpoffs={spec['chroma_offset']}:crqpoffs={spec['chroma_offset']}"
        common=[str(ffmpeg),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','256x144','-r','15',
            '-i','pipe:0','-an','-c:v','libx265','-preset',spec['preset'],'-b:v',f'{target:.4f}k',
            '-pix_fmt','yuv420p','-threads','2']
        tick=time.perf_counter()
        for pass_number in (1,2):
            command=[*common,'-x265-params',params+f':pass={pass_number}','-f','hevc','pipe:1']
            result=subprocess.run(command,input=low.tobytes(),stdout=subprocess.PIPE if pass_number==2 else subprocess.DEVNULL,
                stderr=subprocess.PIPE,cwd=trial,check=True)
            (trial/f'pass{pass_number}_stderr.txt').write_bytes(result.stderr)
        packet,h=guide.pack(result.stdout,digest,(256,144))
        (trial/'guide.glcguide').write_bytes(packet)
        row=dict(index=j,HEVC_target_kbps=target,bytes=len(packet),payload_bytes=len(result.stdout),
            kbps=len(packet)*8/10000,seconds=time.perf_counter()-tick,within_cap=len(packet)<=cap,
            command=common,x265_params=params,sha256=file_sha(trial/'guide.glcguide'))
        save_json(trial/'encode.json',row);attempts.append(row)
        print(name,'byte-only attempt',j,'bytes',len(packet),'cap',cap,flush=True)
        if .97*cap<=len(packet)<=cap:break
        target*=((cap-(len(packet)-len(result.stdout)))/len(result.stdout))*.995
    valid=[r for r in attempts if r['within_cap']]
    if not valid:raise ValueError('No profile encode under frozen cap')
    selected=max(valid,key=lambda r:r['bytes'])
    shutil.copy2(folder/f"encode_{selected['index']}"/'guide.glcguide',folder/'guide.glcguide')
    payload=(folder/'guide.glcguide').read_bytes()
    header,decoded,expanded=guide.decode(payload,digest,ffmpeg)
    np.save(folder/'sender_expected_guide.npy',expanded,allow_pickle=False)
    send=dict(complete=True,profile=name,source_assisted_sender=True,header=header,bytes=len(payload),
        kbps=len(payload)*8/10000,cap_bytes=cap,preset=spec['preset'],chroma_offset=spec['chroma_offset'],
        base_bytes=(base/'base.glcsnd').stat().st_size,reference_aux_bytes=16187,
        saving_reference_percent=(1-len(payload)/16187)*100,
        K_aux_bytes=(data/'support_refinement_20261111_01/video1/K/support.glccsp2').stat().st_size,
        sha256=file_sha(folder/'guide.glcguide'),decoded_guide_sha256=file_sha(folder/'sender_expected_guide.npy'),
        decoded_low_psnr=float(10*np.log10(255**2/np.mean((decoded.astype(np.float32)-low)**2))),
        attempts=attempts,selected_attempt=selected['index'],selection='Largest <=cap; bytes only, no quality selection',
        encode_seconds_all_trials=time.perf_counter()-start,selected_two_pass_encode_seconds=selected['seconds'],
        frozen_source_sha256=file_sha(base/'source.npy'),code_sha256=file_sha(__file__))
    save_json(folder/'send.json',send)
    print(name,'SELECTED aux',send['kbps'],'kbps, savings',send['saving_reference_percent'],'%',flush=True)


def receive(workspace,run,name):
    stage=Path(__file__).resolve().parent;data=workspace/'02_实验数据/优化_GLC后处理'
    folder=run/'video1'/name;ffmpeg=workspace/'01_项目/运行环境/FFmpeg/ffmpeg.exe'
    base=data/'source_dense_flow_three_clips_20261109_01/video1'
    command=[sys.executable,'-B','-u','-X','utf8',str(stage/'receive_structure_guide.py'),
        '--base',str(base),'--packet',str(folder/'guide.glcguide'),
        '--upstream',str(data/'deflicker_reef80_20261108_01/upstream'),'--ffmpeg',str(ffmpeg),
        '--output',str(folder/'flow_receive')]
    subprocess.run(command,check=True)
    subprocess.run([sys.executable,'-B','-u','-X','utf8',str(stage/'receive_guide_texture.py'),
        '--base',str(base),'--packet',str(folder/'guide.glcguide'),'--cache',str(folder/'flow_receive'),
        '--ffmpeg',str(ffmpeg),'--output',str(folder/'receive')],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('initial','chroma'))
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();w=a.workspace.resolve();r=a.run.resolve();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    if a.action=='initial':
        freeze(w,r)
        for name,spec in PROFILES.items():encode(w,r,name,spec)
        save_json(r/'render_selection.json',dict(frozen_before_receiver=True,profiles=list(PROFILES),all_frozen_profiles=True))
        for name in PROFILES:receive(w,r,name)
    else:
        spec=dict(PROFILES['S8'],chroma_offset=4)
        save_json(r/'chroma_plan.json',dict(frozen_before_chroma_encoding=True,profile='S8C4',spec=spec,
            only_factor='Cb/Cr QP offset +4 versus S8; identical target, input size/fps and T3 receiver'))
        encode(w,r,'S8C4',spec);receive(w,r,'S8C4')
