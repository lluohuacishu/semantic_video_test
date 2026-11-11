"""Run the official All-In-One-Deflicker stages on lossless GLC reef RGB."""

import argparse
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

REEF_NAME = 'glc_reef80'
START, STOP = 152, 232
FPS = 30
SHAPE = (288, 512, 3)
WEIGHTS = ('raft-things.pth', 'neural_filter.pth', 'local_refinement_net.pth')
STAGES = ('flow', 'atlas', 'refinement')


def png_frames(folder):
    expected = [f'{i:05d}.png' for i in range(STOP - START)]
    files = sorted(folder.glob('*.png'))
    if [path.name for path in files] != expected:
        raise ValueError(f'Unexpected PNG names or frame count in {folder}')
    frames = []
    for path in files:
        with Image.open(path) as image:
            if image.mode != 'RGB':
                raise ValueError(f'Expected an RGB PNG: {path}')
            frame = np.array(image)
        if frame.dtype != np.uint8 or frame.shape != SHAPE:
            raise ValueError(f'Unexpected frame shape or dtype: {path}')
        frames.append(frame)
    return np.stack(frames)


def prepare(workspace, upstream, weights_repo, output):
    from run_p0 import HERE, file_sha, load_json, load_range, open_baseline, save_json
    from experiment_common import array_sha
    plan = load_json(HERE / 'plan_p0.json')
    clip = next(row for row in plan['clips'] if row['id'] == '01_coral_wall')
    manifest = workspace / '01_项目/GLC后处理/接口与基线.json'
    baseline = next(row for row in load_json(manifest)['baselines'] if row['id'] == clip['baseline'])
    if baseline['fps'] != FPS or baseline['shape'][1:] != list(SHAPE) or not (
            clip['start'] <= START < STOP <= clip['stop']):
        raise ValueError('The fixed reef pilot does not match the baseline geometry or cadence')
    decoded, _, receipt = open_baseline(workspace, baseline, plan['codec'])
    raw = load_range(decoded, START, STOP, SHAPE)
    output.mkdir(parents=True, exist_ok=False)
    folder = upstream / 'data/test' / REEF_NAME
    folder.mkdir(parents=True, exist_ok=False)
    for i, frame in enumerate(raw):
        Image.fromarray(frame).save(folder / f'{i:05d}.png')
    if not np.array_equal(png_frames(folder), raw):
        raise ValueError('PNG export changed the decoded RGB')
    local_weights = upstream / 'pretrained_weights'
    local_weights.mkdir(exist_ok=True)
    weights = {}
    for name in WEIGHTS:
        source = weights_repo / 'pretrained_weights' / name
        destination = local_weights / name
        if not destination.exists():
            try:
                os.link(source, destination)
            except OSError:
                shutil.copyfile(source, destination)
        if file_sha(source) != file_sha(destination):
            raise ValueError(f'Official weight mismatch: {name}')
        weights[name] = dict(bytes=destination.stat().st_size, sha256=file_sha(destination))
    config = upstream / 'src/config/config_flow_100.json'
    save_json(output / 'inputs.json', dict(
        clip=clip, source_start=START, source_stop=STOP, frame_count=STOP - START, fps=FPS, shape=SHAPE,
        raw_rgb_sha256=array_sha(raw), input_png_round_trip_bit_exact=True, glc_baseline=receipt,
        plan_sha256=file_sha(HERE / 'plan_p0.json'), baseline_manifest_sha256=file_sha(manifest),
        upstream_commit=git_commit(upstream), weights_commit=git_commit(weights_repo), weights=weights,
        atlas_config=load_json(config), atlas_config_sha256=file_sha(config), atlas_downsample=4, seed=2023))
    print(f'Exported {len(raw)} exact RGB PNGs from GLC frames {START}–{STOP-1} at {FPS} FPS', flush=True)


def git_commit(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def worker(stage, output):
    import json
    import cv2
    import torch
    sys.path.insert(0, str(Path.cwd()))
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for this pilot')
    random.seed(2023)
    np.random.seed(2023)
    torch.manual_seed(2023)
    torch.cuda.set_device(0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    free, total = torch.cuda.mem_get_info()
    start = time.perf_counter()

    if stage == 'flow':
        from src.preprocess_optical_flow import preprocess
        with torch.inference_mode():
            preprocess(SimpleNamespace(vid_path=Path('data/test') / REEF_NAME, max_long_edge=2000))
    elif stage == 'atlas':
        from src.stage1_neural_atlas import main as fit_atlas
        config = json.loads(Path('src/config/config_flow_100.json').read_text(encoding='utf-8'))
        fit_atlas(config, SimpleNamespace(vid_path=str(Path('data/test') / REEF_NAME), down=4))
    else:
        # The upstream script also saves 12-FPS internal preview files.
        # Only its final numbered PNGs are used by the GLC evaluation.
        import runpy
        path = 'src/neural_filter_and_refinement.py'
        sys.argv = [path, '--video_name', REEF_NAME, '--fps', str(FPS)]
        runpy.run_path(path, run_name='__main__')

    torch.cuda.synchronize()
    (output / f'{stage}.json').write_text(json.dumps(dict(
        stage=stage, wall_seconds=time.perf_counter() - start,
        torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
        torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
        free_before_mib=free / 1024**2, device_total_mib=total / 1024**2,
        gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__), cuda=torch.version.cuda,
    ), indent=2) + '\n', encoding='utf-8')


def run(workspace, upstream, weights_repo, ffmpeg, output):
    from run_p0 import file_sha, load_json, save_json
    prepare(workspace, upstream, weights_repo, output)
    if not ffmpeg.is_file():
        raise FileNotFoundError(ffmpeg)
    env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(output / 'mpl_cache'),
               IMAGEIO_FFMPEG_EXE=str(ffmpeg), CUDA_VISIBLE_DEVICES='0',
               TORCH_FORCE_WEIGHTS_ONLY_LOAD='1')
    env['PATH'] = str(ffmpeg.parent) + os.pathsep + env['PATH']
    started = time.perf_counter()
    for stage in STAGES:
        log_path = output / f'{stage}.log'
        print(f'{stage}: running (see {log_path.name})', flush=True)
        with log_path.open('xb') as log:
            subprocess.run([sys.executable, '-B', '-u', '-X', 'utf8', str(Path(__file__).resolve()),
                            '--worker', stage, '--output', str(output)], cwd=upstream, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        if stage == 'flow':
            names = sorted((upstream / 'data/test' / f'{REEF_NAME}_flow').glob('*.npy'))
            if len(names) != 2 * (STOP - START - 1):
                raise ValueError('Optical flow results do not cover every adjacent pair')
        elif stage == 'atlas':
            files = sorted((upstream / 'results' / REEF_NAME / 'stage_1/output').glob('*.png'))
            if [p.name for p in files] != [f'{i:05d}.png' for i in range(STOP - START)]:
                raise ValueError('The fitted atlas did not render all input frames')
        receipt = load_json(output / f'{stage}.json')
        print(f'{stage}: {receipt["wall_seconds"]:.1f}s, '
              f'peak Torch allocation {receipt["torch_peak_allocated_mib"]:.0f} MiB', flush=True)
    restored = png_frames(upstream / 'results' / REEF_NAME / 'final/output')
    np.save(output / 'deflicker_rgb.npy', restored, allow_pickle=False)
    save_json(output / 'run.json', dict(
        complete=True, wall_seconds=time.perf_counter() - started,
        frames=len(restored), fps=FPS, shape=restored.shape[1:], output_dtype=str(restored.dtype),
        output_rgb_sha256=file_sha(output / 'deflicker_rgb.npy'),
        runner_sha256=file_sha(__file__), inputs_sha256=file_sha(output / 'inputs.json'),
        stages=[load_json(output / f'{stage}.json') for stage in STAGES],
        upstream_commit_after=git_commit(upstream)))
    print('PASS: 80 restored PNGs retained at original geometry', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--weights-repo', type=Path)
    parser.add_argument('--ffmpeg', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=STAGES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output.resolve())
        return
    if any(path is None for path in (args.upstream, args.weights_repo, args.ffmpeg)):
        parser.error('--upstream, --weights-repo and --ffmpeg are required')
    run(args.workspace.resolve(), args.upstream.resolve(), args.weights_repo.resolve(),
        args.ffmpeg.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
