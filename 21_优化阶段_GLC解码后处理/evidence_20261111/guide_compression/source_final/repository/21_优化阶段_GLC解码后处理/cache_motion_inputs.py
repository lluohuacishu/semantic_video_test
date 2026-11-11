"""Frozen clips and independently decoded GLC inputs for the cache controls."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import numpy as np

from coral_reference_sidecar import (
    BASE_START, START, STOP, FPS, HEIGHT, WIDTH, REPO,
    baseline, cut, file_sha, load_json, load_range, open_baseline, save_json,
)
from experiment_common import array_sha


@dataclass(frozen=True)
class Clip:
    base_start: int
    start: int
    stop: int
    coral: tuple[int, int, int, int]

    @property
    def count(self):
        return self.stop - self.start


CLIPS = {
    'moving_reef': Clip(BASE_START, START, STOP, (360, 24, 510, 272)),
    'holdout_reef': Clip(832, 832, 922, (145, 190, 510, 286)),
}
CODEC = dict(q=0, intra_period=64, reset_interval=32)
FRAME_SHAPE = (HEIGHT, WIDTH, 3)
LANE_LIMIT = 40000
DECODER = REPO/'17_优化阶段_解码器推理加速/decoder_worker.py'


def source_frames(workspace, clip):
    _, source, _ = open_baseline(workspace, baseline(workspace), CODEC)
    return load_range(source, clip.start, clip.stop, FRAME_SHAPE)


def base_frames(folder, clip):
    cut_record = load_json(folder/'base_cut.json')
    path = folder/'base_receive/decoded_rgb.npy'
    receipt = load_json(path.with_name('result.json'))
    expected = (clip.base_start, clip.start, clip.stop)
    actual = tuple(cut_record[key] for key in ('source_start', 'display_start', 'display_stop'))
    stream = folder/'base.glcsnd'
    stream_sha = file_sha(stream)
    size = stream.stat().st_size
    if (actual != expected or cut_record['fps'] != FPS or cut_record['bytes'] != size
            or not stream_sha == receipt['stream_sha256'] == cut_record['base_stream_sha256']
            or not receipt['reference_verified'] or not receipt['source_media_reads_forbidden']
            or receipt['reference_checksums'] != cut_record['reference_checksums']
            or file_sha(path) != receipt['decoded_rgb_sha256']
            or size*8*FPS > LANE_LIMIT*clip.count):
        raise ValueError('Invalid independently received base stream or base lane budget')
    frames = np.load(path, mmap_mode='r', allow_pickle=False)
    if frames.shape != (clip.stop-clip.base_start, *FRAME_SHAPE) or frames.dtype != np.uint8:
        raise ValueError('Unexpected received RGB geometry')
    return frames[clip.start-clip.base_start:], stream_sha, size


def prepare(workspace, output, assets):
    output.mkdir(parents=True, exist_ok=False)
    original, _, _ = open_baseline(workspace, baseline(workspace), CODEC)
    prepared = {}
    for name, clip in CLIPS.items():
        folder = output/name
        folder.mkdir()
        cut(workspace, folder, base_start=clip.base_start, start=clip.start, stop=clip.stop)
        subprocess.run([sys.executable, '-B', '-u', '-X', 'utf8', str(DECODER),
            '--stream', str(folder/'base.glcsnd'), '--output', str(folder/'base_receive'),
            '--backend', 'eager', '--assets', str(assets)], check=True)
        frames, digest, size = base_frames(folder, clip)
        entire = np.load(folder/'base_receive/decoded_rgb.npy', mmap_mode='r', allow_pickle=False)
        if not np.array_equal(entire, load_range(original, clip.base_start, clip.stop, FRAME_SHAPE)):
            raise ValueError('The cut and the archived GLC decode differ')
        prepared[name] = dict(base_bytes=size, base_kbps=size*8*FPS/clip.count/1000,
            base_sha256=digest, display_rgb_sha256=array_sha(frames), decoded_rgb_bit_exact=True)
        print(f'{name}: verified {clip.count} display frames from {size} base bytes', flush=True)
    save_json(output/'prepare.json', dict(clips=prepared, code_sha256=file_sha(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.workspace.resolve(), args.output.resolve(), args.assets.resolve())


if __name__ == '__main__':
    main()
