"""Prepare the independent shipwreck clip and verify an eager GLC round trip."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from cache_motion_inputs import Clip, DECODER, FPS, FRAME_SHAPE, HEIGHT, REPO, WIDTH, base_frames
from coral_reference_sidecar import file_sha, load_json, save_json
from experiment_common import array_sha


CLIP = Clip(30, 30, 120, (330, 24, 502, 168))
ENCODER = REPO/'16_优化阶段_输入编码器推理加速/codec_worker.py'
SOURCE_SHA = '0351a2661573582477bb1ffd1ac809ac5364185578e2485cb0950715203d3ce9'


def prepare(video, ffmpeg, assets, output, *, clip=CLIP, source_sha=SOURCE_SHA, clip_id='shipwreck'):
    if file_sha(video) != source_sha:
        raise ValueError(f'Unexpected source file for {clip_id}')
    output.mkdir(parents=True, exist_ok=False)
    source_path = output/'source.npy'
    extract = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin',
        '-ss', str(clip.start/FPS), '-i', str(video), '-map', '0:v:0', '-an',
        '-vf', f'fps={FPS},scale={WIDTH}:{HEIGHT}:flags=area,format=rgb24',
        '-frames:v', str(clip.count), '-f', 'rawvideo', 'pipe:1']
    pixels = subprocess.check_output(extract)
    source = np.frombuffer(pixels, np.uint8).reshape((clip.count, *FRAME_SHAPE))
    distinct = len({array_sha(frame) for frame in source})
    if distinct != clip.count:
        raise ValueError('Source sampling introduced duplicate RGB frames')
    np.save(source_path, source, allow_pickle=False)
    plan = output/'encoder_plan.json'
    code_hashes = {path.relative_to(REPO).as_posix(): file_sha(path) for path in (ENCODER, DECODER)}
    save_json(plan, dict(assets=str(assets), assets_manifest_sha256=file_sha(assets/'manifest.json'),
        code_sha256=code_hashes, warmup_frames=8, clips=[dict(id=clip_id, source=str(source_path),
            source_sha256=file_sha(source_path), width=WIDTH, height=HEIGHT, frames=clip.count, fps=FPS, q=0)]))
    print(f'{clip_id}: sampled {clip.count} distinct source frames at {FPS} FPS', flush=True)

    python = [sys.executable, '-B', '-u', '-X', 'utf8']
    sender = output/'sender'
    subprocess.run(python + [str(ENCODER), '--plan', str(plan), '--clip', clip_id,
        '--role', 'sender', '--variant', 'eager', '--fill', 'on', '--output', str(sender)], check=True)
    stream = output/'base.glcsnd'
    shutil.copyfile(sender/'stream.glcsnd', stream)
    sent = load_json(sender/'result.json')
    save_json(output/'base_cut.json', dict(source_start=clip.base_start, source_stop=clip.stop,
        display_start=clip.start, display_stop=clip.stop, bytes=sent['stream_bytes'], fps=FPS,
        base_stream_sha256=sent['stream_sha256'], reference_checksums=sent['reference_checksums']))
    subprocess.run(python + [str(DECODER), '--stream', str(stream), '--output',
        str(output/'base_receive'), '--backend', 'eager', '--assets', str(assets)], check=True)
    raw, digest, size = base_frames(output, clip)
    save_json(output/'inputs.json', dict(source_filename=video.name, source_file_sha256=source_sha,
        source_rgb_sha256=array_sha(source), source_npy_sha256=file_sha(source_path),
        source_range_15fps=[clip.start, clip.stop], source_seconds=[clip.start/FPS, clip.stop/FPS],
        background_roi=clip.coral, distinct_sampled_frames=distinct, fps=FPS, shape=source.shape,
        base_bytes=size, base_kbps=size*8*FPS/clip.count/1000, base_sha256=digest,
        decoded_rgb_sha256=array_sha(raw), header=sent['header'], code_sha256=file_sha(__file__)))
    print(f'Independent GLC base verified, {size} bytes', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'ffmpeg', 'assets', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.input.resolve(), args.ffmpeg.resolve(), args.assets.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
