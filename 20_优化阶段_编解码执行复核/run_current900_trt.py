"""Current 900-frame source: existing TRT input encoder and independent safe decoder."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parent
DATA=REPO.parents[2]/'02_实验数据'


def load(p):
    return json.loads(p.read_text('utf-8-sig'))


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    source_config=DATA/'GVC1D/long_20261102_01/glc_sender_config.json'
    cfg=load(source_config)
    cfg.update(action='sender',output=str(args.output/'sender'),stop_file=str(args.output/'STOP'),
        assets=str(DATA/'优化_部署精简/role_assets_20261029_01'),variant='trt_fp16_stream',
        engine_cache=str(DATA/'优化_输入编码器推理/inference_20261030_01/engines'))
    if cfg['source']['frames']!=900 or cfg['source']['fps']!=30 or cfg['q']!=0:
        raise ValueError('Expected the existing 900-frame q0 input')
    path=args.output/'sender_config.json'
    path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    entry=REPO/'16_优化阶段_输入编码器推理加速/web_video_worker.py'
    code_sha=sha(entry)
    print('START current900 TRT FP16 sender',flush=True)
    with (args.output/'sender.log').open('wb') as log:
        subprocess.run([sys.executable,'-X','utf8','-B',str(entry),'--config',str(path)],
            stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    encoded=load(args.output/'sender/result.json')
    if sha(entry)!=code_sha:
        raise ValueError('Frozen sender entry changed')
    print(f'DONE encode {encoded["core_seconds"]:.3f}s {encoded["fps"]:.3f}FPS',flush=True)
    cmd=[sys.executable,'-X','utf8','-B',str(HERE/'candidate_receiver.py'),'--mode','baseline',
        '--stream',str(args.output/'sender/stream.glcsnd'),'--cache',str(args.output/'unused_candidate_cache'),
        '--output',str(args.output/'receiver'),'--save-rgb','--video']
    with (args.output/'receiver.log').open('wb') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    decoded=load(args.output/'receiver/result.json')
    if (encoded['frames']!=decoded['frames'] or encoded['reference_checksums']!=decoded['reference_checksums'] or
        encoded['stream_sha256']!=decoded['stream_sha256'] or not decoded['reference_verified']):
        raise ValueError('Independent new-stream recovery differs')
    duration=encoded['frames']/cfg['source']['fps']
    summary=dict(complete=True,frames=encoded['frames'],fps=cfg['source']['fps'],duration_seconds=duration,
        width=cfg['source']['width'],height=cfg['source']['height'],q=cfg['q'],
        encoder='trt_fp16_stream',decoder='trt_fp16_safe',encode_core_seconds=encoded['core_seconds'],
        encode_fps=encoded['fps'],decode_core_seconds=decoded['core_seconds'],decode_fps=decoded['fps'],
        stream_bytes=encoded['stream_bytes'],bitrate_kbps=encoded['stream_bytes']*8/duration/1000,
        stream_sha256=encoded['stream_sha256'],reference_verified=True,
        source_config_sha256=sha(source_config),sender_code_sha256=code_sha,
        producer_sha256=sha(Path(__file__)),timing_repeats=1,
        source='reuse aligned lossless RGB chunks; source preparation excluded',
        precision='input image encoder FP16; reference/entropy FP32; RGB safe mixed precision')
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PASS '+json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
