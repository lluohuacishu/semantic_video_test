"""Screen RGB-only candidates on the existing twenty true P latents."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import statistics
import time
from candidate_backend import *
from probe_cached_engines import check_frozen, gpu_status


@torch.inference_mode()
def main(backend_builder=build_candidate, modes=MODES, backend_source=HERE/'candidate_backend.py'):
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=modes, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = check_frozen()
    capture_dir = DATA / '优化_解码批量/batch_pilot_20261031_01/capture'
    captured = load_json(capture_dir / 'result.json')
    if file_sha(capture_dir / 'latents.npy') != captured['latents_sha256']:
        raise ValueError('Captured input changed')
    x = torch.from_numpy(np.load(capture_dir / 'latents.npy')).cuda()
    source_x = x.clone()
    from role_codec import RoleCodec
    codec = RoleCodec(ASSETS, 'receiver')
    torch.utils.deterministic.fill_uninitialized_memory = False
    candidate = backend_builder(codec.video.vqgan.generator, x[:1], args.mode, 'video', args.cache)
    receipt = candidate.setup_receipt
    del codec
    gc.collect()
    torch.cuda.empty_cache()
    baseline = EngineRunner(BASE_ENGINE, load_json(BASE_ENGINE.with_name('build.json'))['engine_sha256'])
    h, w = captured['header']['height'], captured['header']['width']

    def rgb(output):
        if not torch.isfinite(output).all():
            raise ValueError('Non-finite candidate RGB')
        value = output[0, :, :h, :w].clamp(-1, 1).add(1).mul(127.5)
        return value.round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()

    renderers = dict(baseline=baseline, candidate=candidate)
    pixels = {}
    for name, renderer in renderers.items():
        pixels[name] = np.stack([rgb(renderer(x[i:i+1])) for i in range(len(x))])
        np.save(args.output / f'{name}_rgb.npy', pixels[name])
    old_rgb = np.load(capture_dir.parent / 'batch1/run1/rgb.npy')
    if not np.array_equal(pixels['baseline'], old_rgb):
        raise ValueError('Baseline no longer matches its historical RGB')
    timings = []
    for renderer in renderers.values():
        for i in range(4):
            rgb(renderer(x[i:i+1]))
    for number, name in enumerate(['baseline', 'candidate', 'candidate', 'baseline'] * 2):
        renderer = renderers[name]
        torch.cuda.synchronize()
        tick = time.perf_counter()
        for i in range(len(x)):
            rgb(renderer(x[i:i+1]))
        torch.cuda.synchronize()
        ms = (time.perf_counter()-tick)*1000/len(x)
        timings.append(dict(pass_number=number+1, name=name, ms_per_frame=ms))
        print(f'{args.mode} {name}: {ms:.3f}ms/frame including RGB', flush=True)
    if not torch.equal(source_x, x) or check_frozen() != frozen:
        raise ValueError('Input or frozen implementation changed')
    baseline_ms = statistics.median(r['ms_per_frame'] for r in timings if r['name']=='baseline')
    candidate_ms = statistics.median(r['ms_per_frame'] for r in timings if r['name']=='candidate')
    diff = pixels['candidate'].astype(np.int16)-pixels['baseline'].astype(np.int16)
    save_json(args.output / 'result.json', dict(mode=args.mode, frames=len(x), indices=captured['selected_indices'],
        baseline_ms=baseline_ms, candidate_ms=candidate_ms, speedup=baseline_ms/candidate_ms, timings=timings,
        rgb_changed_values=int(np.count_nonzero(diff)), max_rgb_difference=int(np.abs(diff).max()),
        mean_abs_rgb_difference=float(np.abs(diff).mean()), candidate=receipt,
        latents_sha256=captured['latents_sha256'], baseline_engine_sha256=baseline.engine_sha256,
        baseline_rgb_matches_history=True, source_tensor_unchanged=True, frozen_code_sha256=frozen,
        producer_sha256=file_sha(__file__), backend_sha256=file_sha(backend_source),
        full_codec=False, reference_path_changed=False, gpu_after=gpu_status()))
    print(f'PASS {args.mode}: {baseline_ms/candidate_ms:.4f}x module throughput; max RGB delta {np.abs(diff).max()}', flush=True)


if __name__ == '__main__':
    main()
