"""Read the full original video to EOF, encode at half rate, independently decode.

Each requested resolution uses the existing TRT FP16 input encoder and protected
mixed-precision RGB decoder. No experimental normalization/BF16/Graph backend.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parent
WORKSPACE=REPO.parents[2]
DATA=WORKSPACE/'02_实验数据'
FFPROBE=Path('C:/Users/lenovo/Desktop/works/Python/video-baseline/tools/ffmpeg-9.0.1-essentials_build/bin/ffprobe.exe')


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def probe(path,count=True):
    argv=[str(FFPROBE),'-v','error','-select_streams','v:0']
    if count:argv.append('-count_frames')
    argv.extend(['-show_entries','stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_read_frames,duration:format=duration,size',
                 '-of','json',str(path)])
    return json.loads(subprocess.check_output(argv,text=True))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--viewer',type=Path,required=True)
    ap.add_argument('--sizes',type=int,nargs='+',default=[512,640])
    args=ap.parse_args()
    if args.sizes!=[512,640]:raise ValueError('This run verifies the requested 512 and 640 pair')
    args.output.mkdir(parents=True,exist_ok=False)
    args.viewer.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    original_sha=sha(args.input)
    source_probe=probe(args.input)
    stream=source_probe['streams'][0]
    if stream['r_frame_rate']!='30/1':raise ValueError('Expected the original 30FPS upload')
    expected=(int(stream['nb_read_frames'])+1)//2
    if expected!=1239:raise ValueError('Unexpected complete source length')
    source_receipt=dict(input=str(args.input),sha256=original_sha,probe=source_probe,
        expected_output_frames=expected,frame_stride=2,source_indices=dict(first=0,last=2*(expected-1),step=2))
    write(args.output/'source.json',source_receipt)
    worker=REPO/'16_优化阶段_输入编码器推理加速/web_video_worker.py'
    receiver=HERE/'candidate_receiver.py'
    tracked=[worker,receiver,REPO/'16_优化阶段_输入编码器推理加速/encoder_variants.py',
             REPO/'16_优化阶段_输入编码器推理加速/extra_variants.py',
             REPO/'16_优化阶段_输入编码器推理加速/web_engine_cache.py']
    for rel,digest in read(HERE/'evidence_candidates_20261105/execution_manifest.json')['final_code_sha256'].items():
        if sha(HERE/rel)!=digest:raise ValueError('Previously verified producer changed: '+rel)
    code={str(p.relative_to(REPO)):sha(p) for p in tracked}
    all_rows=[]
    for size in args.sizes:
        folder=args.output/f'width{size}'
        folder.mkdir()
        tick=time.perf_counter()

        def run(stage,entry,argv):
            print(f'START {size} {stage}',flush=True)
            launch=dict(command=[sys.executable,'-X','utf8','-B',str(entry),*argv],started_unix=time.time())
            write(folder/(stage+'.launch.json'),launch)
            with (folder/(stage+'.log')).open('wb') as log:
                process=subprocess.run(launch['command'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            launch.update(completed_unix=time.time(),returncode=process.returncode)
            write(folder/(stage+'.launch.json'),launch)
            if process.returncode:raise RuntimeError(f'{size} {stage} failed; see {folder/(stage+".log")}')

        prepare_cfg=dict(action='prepare',frames=None,size=size,fps_mode='half',input=str(args.input),
                         output=str(folder/'prepare'),stop_file=str(folder/'STOP'))
        write(folder/'prepare.config.json',prepare_cfg)
        run('prepare',worker,['--config',str(folder/'prepare.config.json')])
        source=read(folder/'prepare/result.json')
        if not source['read_until_eof'] or source['frames']!=expected or source['fps']!=15.0 or source['frame_stride']!=2:
            raise ValueError('Video was not completely sampled at 15FPS')
        if (source['width'],source['height'])!=(size,size*9//16):raise ValueError('Unexpected geometry')
        sender_cfg=dict(action='sender',source=source,assets=str(DATA/'优化_部署精简/role_assets_20261029_01'),
            output=str(folder/'sender'),stop_file=str(folder/'STOP'),variant='trt_fp16_stream',q=0,
            engine_cache=str(DATA/'优化_输入编码器推理/inference_20261030_01/engines'))
        write(folder/'sender.config.json',sender_cfg)
        run('sender',worker,['--config',str(folder/'sender.config.json')])
        encoded=read(folder/'sender/result.json')
        run('receiver',receiver,['--mode','baseline','--stream',str(folder/'sender/stream.glcsnd'),
            '--cache',str(folder/'unused_candidate_cache'),'--output',str(folder/'receiver'),'--save-rgb','--video'])
        decoded=read(folder/'receiver/result.json')
        if (encoded['frames']!=expected or decoded['frames']!=expected or not decoded['reference_verified'] or
            encoded['reference_checksums']!=decoded['reference_checksums'] or
            encoded['stream_sha256']!=decoded['stream_sha256'] or not decoded['source_media_reads_forbidden']):
            raise ValueError('Independent full-stream recovery failed')
        video_probe=probe(folder/'receiver/video.mp4')
        video=video_probe['streams'][0]
        duration=expected/15
        if (int(video['nb_read_frames'])!=expected or video['r_frame_rate']!='15/1' or
            (video['width'],video['height'])!=(source['width'],source['height']) or
            abs(float(video_probe['format']['duration'])-duration)>.001):
            raise ValueError('Actual MP4 cadence/length/geometry differs')
        elapsed=time.perf_counter()-tick
        name=f'{size}x{source["height"]}_15FPS_GLC_TRT_q0.mp4'
        dest=args.viewer/name
        shutil.copy2(folder/'receiver/video.mp4',dest)
        if sha(dest)!=decoded['video_sha256']:raise ValueError('Watch copy differs')
        source_dest=args.viewer/f'{size}x{source["height"]}_15FPS_原参照.mp4'
        shutil.copy2(folder/'prepare/source.mp4',source_dest)
        row=dict(width=size,height=source['height'],frames=expected,fps=15,duration_seconds=duration,q=0,
            encoder='trt_fp16_stream',decoder='trt_fp16_safe',
            encode_core_seconds=encoded['core_seconds'],encode_fps=encoded['fps'],
            decode_core_seconds=decoded['core_seconds'],decode_fps=decoded['fps'],
            stream_bytes=encoded['stream_bytes'],bitrate_kbps=encoded['stream_bytes']*8/duration/1000,
            stream_sha256=encoded['stream_sha256'],reference_checksums=decoded['reference_checksums'],
            reference_verified=True,source_media_reads_forbidden=True,read_until_eof=True,
            prepare_seconds=source['seconds'],encode_worker_seconds=encoded['worker_seconds'],
            decode_worker_seconds=decoded['worker_seconds'],total_process_wall_seconds=elapsed,
            video=str(dest),video_sha256=sha(dest),source_preview=str(source_dest),
            source_preview_sha256=sha(source_dest),video_probe=video_probe)
        write(folder/'summary.json',row)
        all_rows.append(row)
        write(args.output/'progress.json',dict(complete=False,source=source_receipt,results=all_rows))
        print('PASS '+json.dumps(row,ensure_ascii=False),flush=True)
    for rel,digest in code.items():
        if sha(REPO/rel)!=digest:raise ValueError('Producer changed during run')
    if sha(args.input)!=original_sha:raise ValueError('Original upload changed')
    summary=dict(complete=True,source=source_receipt,results=all_rows,
        total_wall_seconds=time.perf_counter()-started,code_sha256=code,
        runner_sha256=sha(Path(__file__)),training_steps=0,production_changed=False,
        timing_repeats=1,perceptual_metrics_recomputed=False)
    write(args.output/'summary.json',summary)
    write(args.viewer/'结果与来源.json',summary)
    snapshot=args.output/'executed_code'
    snapshot.mkdir()
    for p in HERE.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    print('COMPLETE both resolutions, actual EOF, 15FPS, independent references and MP4 verified',flush=True)


if __name__=='__main__':
    main()
