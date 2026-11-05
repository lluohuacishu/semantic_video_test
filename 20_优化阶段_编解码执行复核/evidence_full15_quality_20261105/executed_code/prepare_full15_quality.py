"""Prepare FP32 controls for both complete 15FPS videos; reuse only verified data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parent
DATA=REPO.parents[2]/'02_实验数据'
CURRENT=DATA/'优化_编解码执行复核/full_video_15fps_20261105_01'
OLD=DATA/'C_编解码加速与网页展示/C5_loopback/sender/inference_lab/9223a0d0559c1475a7087e8df16d1f5a'
OLD_C=DATA/'优化_解码器推理/decoder_20261031_05/full1239/eager'


def load(p):
    return json.loads(Path(p).read_text('utf-8-sig'))


def save(p,d):
    Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def run(folder,name,entry,argv):
    print('START '+name,flush=True)
    command=[sys.executable,'-X','utf8','-B',str(entry),*map(str,argv)]
    with (folder/(name+'.log')).open('wb') as f:
        subprocess.run(command,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True,
                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    print('DONE '+name,flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    sender_entry=REPO/'16_优化阶段_输入编码器推理加速/web_video_worker.py'
    decoder_entry=REPO/'17_优化阶段_解码器推理加速/decoder_worker.py'
    code={str(p.relative_to(REPO)):sha(p) for p in (sender_entry,decoder_entry)}
    # The 640 input chunks were removed after the old job. The current files
    # exactly match every historical chunk digest, so no source restoration is needed.
    old_source=load(OLD/'prepare/result.json')
    new_source=load(CURRENT/'width640/prepare/result.json')
    if len(old_source['chunks'])!=len(new_source['chunks']):raise ValueError('Source chunk count differs')
    for a,b in zip(old_source['chunks'],new_source['chunks']):
        if (a['start'],a['frames'],a['sha256'])!=(b['start'],b['frames'],b['sha256']) or sha(b['path'])!=a['sha256']:
            raise ValueError('Current 640 source differs from historical source')
    old_c=load(OLD_C/'result.json')
    d640=load(CURRENT/'width640/receiver/result.json')
    if sha(OLD_C/'decoded_rgb.npy')!=old_c['decoded_rgb_sha256']:raise ValueError('Stored FP32 decoder RGB changed')
    for key in ('stream_sha256','reference_checksums','entropy_symbols_and_cdf_sha256'):
        if old_c[key]!=d640[key]:raise ValueError('Stored C640 is a different stream/reference/entropy sequence')
    for rel,h in old_c['code_sha256'].items():
        if sha(REPO/rel)!=h:raise ValueError('Frozen decoder differs')
    plan=dict(source_current=str(CURRENT),frames=1239,fps=15,duration_seconds=82.6,q=0,
              variants=['fp32','encoder_only','accelerated'],clips=[],code_sha256=code,
              reuse640=dict(current_source_matches_all_historical_chunk_hashes=True,
                  encoder_only_result=str(OLD_C/'result.json'),encoder_only_rgb=str(OLD_C/'decoded_rgb.npy')))
    for width in (512,640):
        folder=args.output/f'width{width}';folder.mkdir()
        prepared=CURRENT/f'width{width}/prepare/result.json'
        meta=load(prepared)
        if meta['frames']!=1239 or meta['fps']!=15 or not meta['read_until_eof']:
            raise ValueError('Expected complete, aligned 15FPS source')
        if width==512:
            cfg=dict(action='sender',source=meta,assets=str(DATA/'优化_部署精简/role_assets_20261029_01'),
                output=str(folder/'fp32_sender'),stop_file=str(folder/'STOP'),variant='eager',q=0)
            save(folder/'fp32_sender.config.json',cfg)
            run(folder,'fp32_sender',sender_entry,['--config',folder/'fp32_sender.config.json'])
            fp32_stream=folder/'fp32_sender/stream.glcsnd'
            fp32_sender=folder/'fp32_sender/result.json'
        else:
            fp32_stream=OLD/'eager_sender/stream.glcsnd'
            fp32_sender=OLD/'eager_sender/result.json'
        encoded=load(fp32_sender)
        if encoded['frames']!=1239 or encoded['stream_sha256']!=sha(fp32_stream):
            raise ValueError('FP32 source stream differs')
        fp32_out=folder/'fp32_receiver'
        run(folder,'fp32_receiver',decoder_entry,['--stream',fp32_stream,'--output',fp32_out,'--backend','eager','--video'])
        fp32_result=load(fp32_out/'result.json')
        if encoded['reference_checksums']!=fp32_result['reference_checksums']:
            raise ValueError('FP32 independent reference recovery differs')
        d_path=CURRENT/f'width{width}/receiver/result.json'
        d=load(d_path)
        if sha(CURRENT/f'width{width}/receiver/decoded_rgb.npy')!=d['rgb_sha256']:
            raise ValueError('Existing accelerated RGB changed')
        if width==512:
            c_out=folder/'encoder_only_receiver'
            run(folder,'encoder_only_receiver',decoder_entry,['--stream',CURRENT/'width512/sender/stream.glcsnd',
                '--output',c_out,'--backend','eager'])
            c_result_path=c_out/'result.json'
            c_rgb=c_out/'decoded_rgb.npy'
        else:
            c_result_path=OLD_C/'result.json'
            c_rgb=OLD_C/'decoded_rgb.npy'
        c=load(c_result_path)
        for key in ('stream_sha256','reference_checksums','entropy_symbols_and_cdf_sha256'):
            if c[key]!=d[key]:raise ValueError('Decoder-only comparison changed reference/entropy: '+key)
        if [r['latent_sha256'] for r in c['records']]!=[r['latent_sha256'] for r in d['records']]:
            raise ValueError('Per-frame reference sequence differs')
        for result in (fp32_result,c,d):
            if not result['reference_verified'] or not result['source_media_reads_forbidden']:
                raise ValueError('Independent recovery evidence missing')
        clip=dict(width=width,height=meta['height'],frames=1239,fps=15,source=str(prepared),
            source_sha256=sha(prepared),rgb=dict(fp32=str(fp32_out/'decoded_rgb.npy'),encoder_only=str(c_rgb),
                accelerated=str(CURRENT/f'width{width}/receiver/decoded_rgb.npy')),
            receiver_results=dict(fp32=str(fp32_out/'result.json'),encoder_only=str(c_result_path),accelerated=str(d_path)),
            fp32_sender_result=str(fp32_sender),stream_rates_kbps=dict(fp32=encoded['stream_bytes']*8/82.6/1000,
                accelerated=(CURRENT/f'width{width}/sender/stream.glcsnd').stat().st_size*8/82.6/1000),
            decoder_only_reference_and_entropy_identical=True)
        plan['clips'].append(clip)
        save(args.output/'plan.partial.json',plan)
        print(f'PASS controls {width}: FP32 + encoder-only + full acceleration, 1239 frames',flush=True)
    for rel,h in code.items():
        if sha(REPO/rel)!=h:raise ValueError('Frozen producer changed')
    plan.update(complete=True,prepare_elapsed_seconds=time.perf_counter()-started,
                producer_sha256=sha(Path(__file__)))
    save(args.output/'plan.json',plan)
    print('COMPLETE FP32 controls for both full videos',flush=True)


if __name__=='__main__':
    main()
