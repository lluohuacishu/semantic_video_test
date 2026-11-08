"""Run the pinned low-memory SeedVR2 CLI on the existing lossless GLC pilot."""

import argparse
from contextlib import chdir
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image

from run_deflicker import FPS, REEF_NAME, SHAPE, START, STOP, git_commit, png_frames

DIT = 'seedvr2_ema_3b-Q8_0.gguf'
VAE = 'ema_vae_fp16.safetensors'
WEIGHTS = {
    DIT: 'be0d60083a2051a265eb4b77f28edf494e6db67ffc250216f32b72292e5cbd96',
    VAE: '20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1',
}
RANGES = {'smoke': (188, 197), 'full': (START, STOP)}


def encode_lossless(frames, output, ffmpeg):
    height, width, _ = SHAPE
    subprocess.run([str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin',
                    '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
                    '-framerate', str(FPS), '-i', 'pipe:0', '-an', '-c:v', 'ffv1',
                    '-pix_fmt', 'bgr0', '-n', str(output)], input=frames.tobytes(), check=True)
    # SeedVR2 decodes video via OpenCV. Verify that exact route before
    # passing the clip to the model, including RGB order and frame cadence.
    with chdir(output.parent):
        reader = cv2.VideoCapture(output.name)
        try:
            if not reader.isOpened() or abs(reader.get(cv2.CAP_PROP_FPS) - FPS) > 1e-6:
                raise ValueError('OpenCV did not preserve the input timebase')
            for i, expected in enumerate(frames):
                ok, bgr = reader.read()
                if not ok or not np.array_equal(bgr[..., ::-1], expected):
                    raise ValueError(f'Lossless video round-trip differs at input frame {i}')
            if reader.read()[0]:
                raise ValueError('Unexpected extra video frames')
        finally:
            reader.release()


def read_output(png_root, count):
    files = sorted(png_root.rglob('*.png'))
    if [file.name for file in files] != [f'reef_glc_{i:06d}.png' for i in range(count)]:
        raise ValueError(f'SeedVR2 emitted unexpected frame names or counts: {len(files)}')
    frames = []
    for path in files:
        with Image.open(path) as image:
            frame = np.array(image)
            if image.mode != 'RGB' or frame.shape != SHAPE or frame.dtype != np.uint8:
                raise ValueError(f'Unexpected lossless output geometry or RGB precision: {path}')
        frames.append(frame)
    return np.stack(frames)


def profile_cli(cli, profile, arguments):
    import runpy
    import torch
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.cuda.set_device(0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    free, total = torch.cuda.mem_get_info()
    tick = time.perf_counter()
    sys.argv = [str(cli), *arguments]
    try:
        runpy.run_path(str(cli), run_name='__main__')
    finally:
        torch.cuda.synchronize()
        result = dict(
            wall_seconds=time.perf_counter() - tick, gpu=torch.cuda.get_device_name(0),
            torch=str(torch.__version__), cuda=torch.version.cuda,
            torch_allocated_at_exit_mib=torch.cuda.memory_allocated() / 1024**2,
            torch_reserved_at_exit_mib=torch.cuda.memory_reserved() / 1024**2,
            free_before_mib=free / 1024**2, device_total_mib=total / 1024**2,
            torch_allocator='native')
        profile.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


def phase_memory(log_path):
    # The upstream debug logger resets allocator peaks between phases.
    # Preserve its phase peaks instead of reading the reset counter at exit.
    pattern = re.compile(
        r'\[VRAM\] [0-9.]+GB allocated / ([0-9.]+)GB reserved / Peak: ([0-9.]+)GB / ([0-9.]+)GB free')
    rows = [tuple(map(float, match)) for match in pattern.findall(log_path.read_text(encoding='utf-8'))]
    if not rows:
        raise ValueError('No phase memory measurements in the SeedVR2 log')
    return dict(phase_peak_allocated_gib=max(row[1] for row in rows),
                max_reported_reserved_gib=max(row[0] for row in rows),
                min_reported_free_gib=min(row[2] for row in rows), report_count=len(rows))


def run(args):
    from run_p0 import file_sha, load_json, save_json
    from experiment_common import array_sha

    upstream = args.pilot / 'upstream'
    model_dir = args.pilot / 'models'
    weights = {}
    for name, expected in WEIGHTS.items():
        path = model_dir / name
        if file_sha(path) != expected:
            raise ValueError(f'Published weight checksum differs: {name}')
        weights[name] = dict(bytes=path.stat().st_size, sha256=expected)
    if not args.ffmpeg.is_file():
        raise FileNotFoundError(args.ffmpeg)
    previous = args.deflicker_pilot
    metadata = load_json(previous / 'trial/inputs.json')
    raw = png_frames(previous / 'upstream/data/test' / REEF_NAME)
    if array_sha(raw) != metadata['raw_rgb_sha256']:
        raise ValueError('Earlier lossless GLC input changed')
    first, stop = RANGES[args.case]
    frames = raw[first - START:stop - START]
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    video = output / 'reef_glc.mkv'
    encode_lossless(frames, video, args.ffmpeg)

    overlap = 0 if args.case == 'smoke' and args.batch_size >= len(frames) else 3
    cli = upstream / 'inference_cli.py'
    arguments = [
        os.path.relpath(video, upstream), '--output', os.path.relpath(output / 'png', upstream),
        '--output_format', 'png', '--model_dir', os.path.relpath(model_dir, upstream),
        '--dit_model', DIT, '--resolution', '288', '--max_resolution', '512',
        '--batch_size', str(args.batch_size), '--uniform_batch_size', '--temporal_overlap', str(overlap),
        '--seed', '2023', '--cuda_device', '0', '--dit_offload_device', 'cpu',
        '--vae_offload_device', 'cpu', '--tensor_offload_device', 'cpu',
        '--blocks_to_swap', '32', '--swap_io_components', '--attention_mode', 'sdpa',
        '--color_correction', 'lab', '--debug',
    ]
    if args.vae_tiled:
        arguments += ['--vae_encode_tiled', '--vae_decode_tiled',
                      '--vae_encode_tile_size', '256', '--vae_decode_tile_size', '256',
                      '--vae_encode_tile_overlap', '64', '--vae_decode_tile_overlap', '64']
    save_json(output / 'inputs.json', dict(
        start=first, stop=stop, fps=FPS, shape=SHAPE, frames=len(frames), input_rgb_sha256=array_sha(frames),
        original_pilot_input_sha256=metadata['raw_rgb_sha256'], ffv1_rgb_round_trip_bit_exact=True,
        video_sha256=file_sha(video), upstream_commit=git_commit(upstream), upstream_cli_sha256=file_sha(cli),
        weights=weights, cli_arguments=arguments, batch_size=args.batch_size, overlap=overlap,
        packages={name: version(name) for name in ('torch', 'torchvision', 'numpy', 'opencv-python',
                  'diffusers', 'peft', 'transformers', 'gguf', 'safetensors', 'omegaconf', 'rotary-embedding-torch')}))
    print(f'{args.case}: exact {len(frames)}-frame RGB video at {FPS} FPS; launching SeedVR2', flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTORCH_CUDA_ALLOC_CONF='backend:native',
               MPLBACKEND='Agg', MPLCONFIGDIR=str(output / 'mpl_cache'))
    log_path = output / 'seedvr2.log'
    with log_path.open('xb') as log:
        subprocess.run([sys.executable, '-B', '-u', '-X', 'utf8', str(Path(__file__).resolve()),
                        '--profile-cli', str(cli), str(output / 'profile.json'), *arguments],
                       cwd=upstream, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    restored = read_output(output / 'png', len(frames))
    np.save(output / 'seedvr2_rgb.npy', restored, allow_pickle=False)
    profile = load_json(output / 'profile.json')
    profile.update(phase_memory(log_path))
    save_json(output / 'run.json', dict(
        complete=True, frames=len(restored), fps=FPS, shape=SHAPE, dtype=str(restored.dtype),
        inputs_sha256=file_sha(output / 'inputs.json'), output_sha256=file_sha(output / 'seedvr2_rgb.npy'),
        runner_sha256=file_sha(__file__), profile=profile, log_sha256=file_sha(log_path)))
    print(f'PASS: {len(restored)} verified SeedVR2 RGB frames at {SHAPE[1]}x{SHAPE[0]}', flush=True)


def main():
    if sys.argv[1:2] == ['--profile-cli']:
        profile_cli(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--deflicker-pilot', type=Path, required=True)
    parser.add_argument('--case', choices=tuple(RANGES), required=True)
    parser.add_argument('--batch-size', choices=(5, 9, 13, 17, 21), type=int, default=9)
    parser.add_argument('--vae-tiled', action='store_true')
    parser.add_argument('--ffmpeg', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for name in ('pilot', 'deflicker_pilot', 'ffmpeg', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    run(args)


if __name__ == '__main__':
    main()
