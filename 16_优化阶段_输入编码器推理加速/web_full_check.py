"""Explicit GPU acceptance: force a natural-GOP boundary, then check every decoded pixel."""
from __future__ import annotations
import argparse
import subprocess
from experiment_common import *
from web_frames import Frames,discard_working_frames


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='Previously prepared 66-frame long108 result.json')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    trial=WORKSPACE/'02_实验数据/优化_输入编码器推理/inference_20261030_01'
    plan=load_json(trial/'plan.json');source=load_json(args.source)
    if source['frames']!=66 or source['width']!=640 or source['height']!=360:raise ValueError('Use the frozen long108 66-frame input')
    def run(name,cfg):
        cfg.update(output=str(args.output/name),stop_file=str(args.output/'STOP'))
        path=args.output/(name+'.config.json');save_json(path,cfg)
        with (args.output/(name+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run([str(load_settings()['runtime']['python']),'-X','utf8','-u',str(HERE/'web_video_worker.py'),
                '--config',str(path)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        return load_json(args.output/name/'result.json')
    send=run('sender',dict(action='sender',assets=plan['assets'],source=source,variant='trt_fp16_stream',
        q=0,segment_frames=64,engine_cache=str(trial/'engines')))
    receive=run('receiver',dict(action='receiver',assets=plan['assets'],input=str(args.output/'sender'/send['stream_file'])))
    expected=load_json(trial/'full/long108_q0_640/trt_fp16_stream/sender1/result.json')
    if send['segments']!=2 or receive['segments']!=2 or send['frames']!=66 or receive['frames']!=66:
        raise ValueError('Segment boundary lost frames')
    if not receive['reference_verified'] or send['reference_checksums']!=receive['reference_checksums'] or send['reference_checksums']!=expected['reference_checksums']:
        raise ValueError('Global reference chain changed at the natural I-frame boundary')
    actual=Frames(receive['decoded_storage'])
    prior=np.load(trial/'full/long108_q0_640/trt_fp16_stream/receiver/decoded_rgb.npy',mmap_mode='r')
    try:
        for i in range(66):
            if not np.array_equal(actual[i],prior[i]):raise ValueError('Decoded RGB changed at frame '+str(i))
    finally:actual.close()
    discard_working_frames(receive['decoded_storage'],args.output)
    save_json(args.output/'acceptance.json',dict(passed=True,frames=66,segment_frames=[64,2],
        independently_decoded=True,all_rgb_frames_bit_exact_to_prior=True,reference_chain_bit_exact_to_prior=True,
        transport=send['stream_file'],stream_bytes=send['stream_bytes'],stream_sha256=send['stream_sha256']))
    print('PASS: two segments (64+2), all 66 RGB frames and global references match prior verified output',flush=True)


if __name__=='__main__':main()
