"""Apply published underwater enhancement weights to the fixed GLC RGB clips."""

import argparse
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType

import numpy as np

from run_p0 import HERE, REPO, file_sha, load_json, load_range, open_baseline, save_json
from experiment_common import array_sha

UVE_REVISION = 'a9c38b284aa9fca609b443c1857ecd8468ce7a7d'
UVE_WEIGHT_SHA = '68114f50e48e400f9affbeaf6ad640745a2e7733bcd8c6b40821425e4e1c31bc'
UNDIVE_REVISION = '7fac42cb01e5e40cc55470b56c4491162f80d9b9'


def fixed_clips(workspace):
    plan = load_json(HERE / 'plan_p0.json')
    manifest = workspace / '01_项目/GLC后处理/接口与基线.json'
    baselines = {item['id']: item for item in load_json(manifest)['baselines']}
    opened = {}
    for clip in plan['clips']:
        key = clip['baseline']
        baseline = baselines[key]
        if key not in opened:
            opened[key] = open_baseline(workspace, baseline, plan['codec'])
        yield clip, baseline, *opened[key]


def revision(upstream, expected):
    git = ['git', '-C', str(upstream)]
    head = subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip()
    if head != expected or subprocess.run([*git, 'diff', '--quiet', 'HEAD', '--']).returncode:
        raise ValueError(f'Expected the unmodified upstream revision {expected}')
    return head


def load_uve(upstream):
    import torch

    code = upstream / 'UVE-Net/basicsr'
    weight = upstream / 'UVE-Net/pretrained/large_net_g.pth'
    head = revision(upstream, UVE_REVISION)
    if file_sha(weight) != UVE_WEIGHT_SHA:
        raise ValueError('UVEB checkpoint differs from the published Git artifact')
    # Import the official inference modules without BasicSR's training registry
    # scan, which would import unrelated Linux CUDA extensions on Windows.
    for name, path in (('basicsr', code), ('basicsr.utils', code / 'utils'),
                       ('basicsr.archs', code / 'archs')):
        module = ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    from basicsr.utils.logger import get_root_logger
    sys.modules['basicsr.utils'].get_root_logger = get_root_logger
    from basicsr.archs.deblur_arch import Deblur

    model = Deblur(num_feat=24, num_block=15)
    state = torch.load(weight, map_location='cpu', weights_only=True)
    model.load_state_dict(state['params'], strict=True)
    return model.eval().cuda(), dict(
        upstream_commit=head, weights={weight.name: UVE_WEIGHT_SHA},
        window=3, stride=1, overlap_weights=[1, 2, 1], batch=4)


def uve_video(model, raw):
    import torch

    if any(side % 16 for side in raw.shape[1:3]):
        raise ValueError('UVE-Net requires frame sides divisible by 16')
    accumulated = np.zeros(raw.shape, np.float32)
    weight_sum = np.zeros(len(raw), np.float32)
    offsets = np.arange(-1, 2)
    for begin in range(0, len(raw), 4):
        centers = np.arange(begin, min(begin + 4, len(raw)))
        targets = centers[:, None] + offsets
        indexes = targets.clip(0, len(raw) - 1)
        video = torch.from_numpy(raw[indexes]).to('cuda').permute(0, 1, 4, 2, 3).float() / 255
        output, _ = model(video)
        if output.shape != video.shape:
            raise ValueError('UVE-Net changed the frame geometry')
        rgb = output.permute(0, 1, 3, 4, 2).cpu().numpy()
        for col, weight in enumerate((1, 2, 1)):
            valid = targets[:, col] == indexes[:, col]
            accumulated[indexes[valid, col]] += rgb[valid, col] * weight
            weight_sum[indexes[valid, col]] += weight
    averaged = accumulated / weight_sum[:, None, None, None]
    if not np.isfinite(averaged).all():
        raise ValueError('UVE-Net produced non-finite RGB')
    return np.rint(averaged.clip(0., 1.) * 255).astype(np.uint8)


def load_undive(upstream):
    import torch

    head = revision(upstream, UNDIVE_REVISION)
    sys.path.insert(0, str(upstream))
    from model_depth import HDRPointwiseNN_depth
    from utils import resize

    weights = upstream / 'PretrainedModels'
    ddpm, checkpoint = weights / 'DDPM_100.pth', weights / 'UnDIVE_100.pth'
    params = dict(luma_bins=8, channel_multiplier=1, spatial_bin=16, guide_complexity=16,
                  batch_norm=False, net_input_size=256, net_output_size=512)
    model = HDRPointwiseNN_depth(params=params, ckpt=str(ddpm))
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    return (model.eval().cuda(), resize), dict(
        upstream_commit=head, weights={path.name: file_sha(path) for path in (ddpm, checkpoint)},
        rgb_input=True, backscatter_removal=False, network_input=256, batch=1,
        output_mapping='official per-frame global min/max')


def undive_video(bundle, raw):
    import torch

    model, resize = bundle
    restored = np.empty_like(raw)
    for i, frame in enumerate(raw):
        low = resize(frame, 256, strict=True)
        low, full = (torch.from_numpy(image).to('cuda').permute(2, 0, 1)[None].float() / 255
                     for image in (low, frame))
        res, _ = model(low, full)
        rgb = full / (torch.maximum(res, full).clamp(max=1.) + .001)
        rgb = rgb[0].permute(1, 2, 0).cpu().numpy()
        if rgb.shape != frame.shape or not np.isfinite(rgb).all():
            raise ValueError(f'UnDIVE RGB differs in geometry or is non-finite at frame {i}')
        low, high = float(rgb.min()), float(rgb.max())
        if high > low:
            rgb = (rgb - low) / (high - low) * 255.
        restored[i] = rgb.clip(0., 255.).astype(np.uint8)
    return restored


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--method', choices=('uve', 'undive'), required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    load_model, enhance = (load_uve, uve_video) if args.method == 'uve' else (load_undive, undive_video)
    model, model_info = load_model(args.upstream.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    with torch.inference_mode():
        for clip, baseline, decoded, _, receipt in fixed_clips(args.workspace):
            start, stop = clip['start'], clip['stop']
            raw = load_range(decoded, start, stop, baseline['shape'][1:])
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            tick = time.perf_counter()
            output = enhance(model, raw)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - tick
            if output.shape != raw.shape or output.dtype != np.uint8:
                raise ValueError('Enhanced output changed RGB format, frame count or geometry')
            path = args.output / f'{clip["id"]}.npy'
            np.save(path, output, allow_pickle=False)
            records.append(dict(
                clip=clip, glc_baseline=receipt, fps=baseline['fps'], frames=len(raw), shape=list(raw.shape),
                input_rgb_sha256=array_sha(raw), output_npy_sha256=file_sha(path), seconds=seconds,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2))
            print(f'{args.method} {clip["id"]}: {len(raw)} frames in {seconds:.2f}s', flush=True)
    save_json(args.output / 'run.json', dict(
        complete=True, method=args.method, model=model_info, clips=records, torch=str(torch.__version__),
        gpu=torch.cuda.get_device_name(), precision='FP32', runner_sha256=file_sha(__file__)))


if __name__ == '__main__':
    main()
