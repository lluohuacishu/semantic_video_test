"""Diagnose GLC E->G reconstruction without latent compression or prediction."""

import argparse
import gc
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from cache_motion_inputs import FPS, FRAME_SHAPE, REPO, base_frames
from compare_boundary_codec import frame_kind
from compare_deflicker import save_csv
from coral_reference_sidecar import file_sha, load_json, save_json
from experiment_common import array_sha, input_tensor
from run_dense_transfer import source_input
from run_p0 import preview, roi
from run_structure_memory import CLIP
from run_video1_persistent_structure import FIXED_BOX, VIEW_BOX, START, STOP, score


def load_pair(assets, name, torch):
    from role_codec import models_on_meta, ref
    from sender_common import load_settings
    runtime = load_settings()['runtime']
    ref.check_source(runtime['repo'])
    manifest = load_json(assets/'manifest.json')
    if (manifest['revision'] != ref.REVISION or manifest['original_weights'] !=
            {k: v['sha256'] for k, v in ref.ASSETS.items()}):
        raise ValueError('Unexpected autoencoder provenance')
    models = models_on_meta(runtime['repo'])
    modules = dict(encoder=models[name].vqgan.encoder, generator=models[name].vqgan.generator)
    del models
    files = {}
    for module, role in (('encoder', 'sender'), ('generator', 'receiver')):
        info = manifest['roles'][role]['files'][name]
        path = assets/info['file']
        if file_sha(path) != info['sha256']:
            raise ValueError('Role weight hash mismatch')
        state = torch.load(path, map_location='cpu', weights_only=True)
        prefix = f'vqgan.{module}.'
        selected = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        modules[module].load_state_dict(selected, strict=True, assign=True)
        modules[module].requires_grad_(False).eval().cuda()
        files[module] = dict(**info, loaded_keys=len(selected), prefix=prefix)
        del state, selected
    return modules['encoder'], modules['generator'], files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/'plan.json', dict(frames=150, fps=15, source='video1/source.npy',
        fixed_video_model='E_video(x) -> G_video; all 150 frames',
        switched_model='same, replacing frames 0,64,128 with E_image(x) -> G_image',
        precision='FP32, deterministic, TF32 disabled', padding='replicate to 64, identical input_tensor',
        excluded=['latent transform codec', 'hyperprior', 'quantization', 'entropy coding', 'temporal context'],
        diagnostic_only=True, bitrate=None,
        limitation='E->G is an intervention outside the jointly trained full compression path; it does not isolate quantization from prediction or imply a deployable zero-rate decoder.'))
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    workspace = args.workspace.resolve()
    base = workspace/'02_实验数据/优化_GLC后处理/source_dense_flow_three_clips_20261109_01/video1'
    assets = workspace/'02_实验数据/优化_部署精简/role_assets_20261029_01'
    source = source_input(base, CLIP)
    raw, base_sha, _ = base_frames(base, CLIP)
    sys.path.insert(0, str(REPO/'14_优化阶段_部署精简'))
    from role_codec import ref
    import torch
    profile = ref.configure('cuda')
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    fixed = np.empty_like(source)
    weights = {}
    with torch.inference_mode():
        encoder, generator, weights['video'] = load_pair(assets, 'video', torch)
        for i, rgb in enumerate(source):
            fixed[i] = ref.rgb_array(generator(encoder(input_tensor(rgb))), FRAME_SHAPE[1], FRAME_SHAPE[0])
            if (i+1) % 30 == 0:
                print(f'Fixed video E->G: {i+1}/150 frames', flush=True)
        repeat = ref.rgb_array(generator(encoder(input_tensor(source[105]))), FRAME_SHAPE[1], FRAME_SHAPE[0])
        if not np.array_equal(repeat, fixed[105]):
            raise ValueError('Repeated bypass reconstruction changed')
        del encoder, generator
        gc.collect()
        torch.cuda.empty_cache()
        switched = fixed.copy()
        encoder, generator, weights['image'] = load_pair(assets, 'image', torch)
        for i in (0, 64, 128):
            switched[i] = ref.rgb_array(generator(encoder(input_tensor(source[i]))), FRAME_SHAPE[1], FRAME_SHAPE[0])
        del encoder, generator
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter()-tick
    peak_mib = torch.cuda.max_memory_allocated()/2**20
    torch.cuda.empty_cache()
    np.save(output/'fixed_video_rgb.npy', fixed, allow_pickle=False)
    np.save(output/'switched_IP_rgb.npy', switched, allow_pickle=False)
    sequences = dict(source=source, raw=raw, AE=fixed, AE_IP=switched)
    area = np.zeros(FRAME_SHAPE[:2], np.uint8)
    area[roi(FIXED_BOX, FRAME_SHAPE)] = 1
    summary, temporal, spatial = score(sequences, np.broadcast_to(area, source.shape[:3]), start=0)
    temporal = [r for r in temporal if r['scope'] == 'fixed']
    spatial = [r for r in spatial if r['scope'] == 'fixed']
    for row in temporal:
        row['kind'] = frame_kind(row['frame'], 64, 32)
        row['previous_kind'] = frame_kind(row['frame']-1, 64, 32)
    save_csv(output/'temporal.csv', temporal)
    save_csv(output/'spatial.csv', spatial)
    late = {}
    for name in sequences:
        rows = [r for r in temporal if r['method'] == name and r['frame'] >= 90
                and r['kind'] == r['previous_kind'] == 'ordinary_p']
        n = sum(r['valid_pixels'] for r in rows)
        late[name] = dict(pairs=len(rows), coarse_temporal_mae=sum(r['coarse_sum'] for r in rows)/n,
            fine_ncc=sum(r['fine_ncc']*r['valid_pixels'] for r in rows)/n)
    masks = np.load(args.pilot/'tracked.npy', allow_pickle=False)
    pilot_scores, pilot_rows, _ = score({k: v[START:STOP] for k, v in sequences.items()}, masks)
    save_csv(output/'pilot_temporal.csv', pilot_rows)
    previews = {}
    labels = ['Source RGB', 'GLC raw', 'AE / video E->G', 'AE / switched I and P']
    for name, crop in (('bypass_coral_10s.mp4', roi(VIEW_BOX, FRAME_SHAPE)), ('bypass_full_10s.mp4', None)):
        previews[name] = preview(args.ffmpeg, output/name, list(sequences.values()), FPS, 0, labels, crop)
    save_json(output/'summary.json', dict(complete=True, diagnostic_only=True, bitrate=None,
        base_sha256=base_sha, source_rgb_sha256=array_sha(source), profile=profile, weights=weights,
        code_sha256=file_sha(__file__), input_tensor_code_sha256=file_sha(REPO/'16_优化阶段_输入编码器推理加速/experiment_common.py'),
        output_sha256={name: file_sha(output/name) for name in ('fixed_video_rgb.npy', 'switched_IP_rgb.npy')},
        full_fixed_scores=summary['fixed'], late_ordinary_p=late, pilot_scores=pilot_scores,
        repeated_frame105_bit_exact=True, peak_allocated_mib=peak_mib,
        loading_and_inference_seconds=inference_seconds, total_seconds=time.perf_counter()-tick, previews=previews))
    print('PASS', flush=True)
    for name, values in late.items():
        print(name, values, flush=True)


if __name__ == '__main__':
    main()
