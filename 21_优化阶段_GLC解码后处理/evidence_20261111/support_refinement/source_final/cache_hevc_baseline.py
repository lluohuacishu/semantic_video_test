"""Encode byte-matched H.265 controls from the frozen 15 FPS source clips."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess

import numpy as np

from cache_motion_inputs import CLIPS, FPS, FRAME_SHAPE, HEIGHT, REPO, WIDTH, base_frames, source_frames
from coral_reference_sidecar import file_sha, save_json
from experiment_common import array_sha


FIXED_BUDGETS = {'h265_40': 30000, 'h265_dual': 52754}
X265_PARAMS = 'pools=2:frame-threads=1:keyint=64:scenecut=0:log-level=error'
CRF_STEP = 0.0625


def encode_controls(ffmpeg, folder, clip, source, budgets):
    folder.mkdir(exist_ok=False)
    pixels = source.tobytes()
    candidates = {}
    common = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin']

    def encode(tick):
        if tick not in candidates:
            crf = tick * CRF_STEP
            path = folder / f'crf_{tick}.mp4'
            subprocess.run(common + [
                '-f', 'rawvideo', '-pixel_format', 'rgb24',
                '-video_size', f'{WIDTH}x{HEIGHT}', '-framerate', str(FPS),
                '-i', 'pipe:0', '-an', '-vf', 'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p',
                '-color_range', 'tv', '-colorspace', 'bt709',
                '-color_primaries', 'bt709', '-color_trc', 'bt709',
                '-c:v', 'libx265', '-preset', 'medium', '-crf', str(crf),
                '-x265-params', X265_PARAMS, '-tag:v', 'hvc1', '-movflags', '+faststart',
                '-n', str(path)], input=pixels, check=True)
            candidates[tick] = (path.stat().st_size, path)
        return candidates[tick][0]

    choices = {}
    for name, budget in budgets.items():
        # Search by the entire MP4 size, including parameter sets, headers and muxing.
        low, high = 0, int(51 / CRF_STEP)
        if encode(high) > budget:
            raise ValueError(f'Cannot fit the H.265 target of {budget} bytes')
        while low < high:
            tick = (low + high) // 2
            if encode(tick) <= budget:
                high = tick
            else:
                low = tick + 1
        for tick in range(max(0, low - 4), min(int(51 / CRF_STEP), low + 5) + 1):
            encode(tick)
        choices[name] = max(
            (tick for tick, (size, _) in candidates.items() if size <= budget),
            key=lambda tick: (candidates[tick][0], -tick))

    ffprobe = ffmpeg.with_name('ffprobe' + ffmpeg.suffix)
    results = {}
    for name, budget in budgets.items():
        tick = choices[name]
        size, candidate = candidates[tick]
        path = folder / f'{name}.mp4'
        path.write_bytes(candidate.read_bytes())
        stream = json.loads(subprocess.check_output([
            str(ffprobe), '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries',
            'stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration,pix_fmt,color_space',
            '-of', 'json', str(path)]))['streams'][0]
        if (stream['codec_name'] != 'hevc' or (stream['width'], stream['height']) != (WIDTH, HEIGHT)
                or Fraction(stream['avg_frame_rate']) != FPS or int(stream['nb_read_frames']) != clip.count
                or Fraction(stream['duration']) != Fraction(clip.count, FPS)
                or path.stat().st_size != size or size > budget):
            raise ValueError(f'H.265 control violates the budget or display timeline: {path}')
        decoded = subprocess.check_output(common + [
            '-i', str(path), '-map', '0:v:0',
            '-vf', 'scale=in_color_matrix=bt709:in_range=tv,format=rgb24',
            '-f', 'rawvideo', 'pipe:1'])
        rgb = np.frombuffer(decoded, np.uint8).reshape((clip.count, *FRAME_SHAPE))
        np.save(path.with_suffix('.npy'), rgb, allow_pickle=False)
        results[name] = dict(target_bytes=budget, bytes=size, kbps=size*8*FPS/clip.count/1000,
            crf=tick*CRF_STEP, sha256=file_sha(path), rgb_sha256=array_sha(rgb), stream=stream)
        print(f'{folder.parent.name} {name}: {size}/{budget} B', flush=True)

    for _, path in candidates.values():
        path.unlink()
    save_json(folder/'result.json', dict(controls=results,
        source_rgb_sha256=array_sha(source), preset='medium', x265_params=X265_PARAMS,
        candidates={str(tick*CRF_STEP): size for tick, (size, _) in sorted(candidates.items())},
        code_sha256=file_sha(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    for name, clip in CLIPS.items():
        folder = args.run.resolve()/name
        _, _, size = base_frames(folder, clip)
        source = source_frames(args.workspace.resolve(), clip)
        encode_controls(args.ffmpeg.resolve(), folder/'hevc', clip, source,
                        {'h265_base': size, **FIXED_BUDGETS})


if __name__ == '__main__':
    main()
