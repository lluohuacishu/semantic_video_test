"""P-generator Batch 1/2/4 only: real latents, explicit mixed precision, no decoder refactor."""
from __future__ import annotations

import argparse
import gc
import statistics
import threading
import time
from batch_common import *
import numpy as np
import torch
from role_codec import RoleCodec
from generator_backend import GeneratorTensorRT


def rgb_batch(output, width, height):
    if output.dtype != torch.float32 or output.ndim != 4 or output.shape[1] != 3:
        raise ValueError('RGB conversion requires NCHW FP32')
    if not torch.isfinite(output).all():
        raise ValueError('Batched generator returned nonfinite RGB')
    value = output[:, :, :height, :width].clamp(-1, 1).add(1).mul(127.5)
    return value.round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, choices=(1, 2, 4), required=True)
    parser.add_argument('--repeat', type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    idle()
    hashes = code_hashes()
    output = TRIAL / f'batch{args.batch}' / f'run{args.repeat}'
    output.mkdir(parents=True, exist_ok=False)
    captured = load_json(TRIAL / 'capture/result.json')
    if captured['code_sha256'] != hashes or not captured['reference_verified']:
        raise ValueError('Capture provenance differs')
    for filename, key in (('latents.npy', 'latents_sha256'), ('teacher_rgb.npy', 'teacher_rgb_sha256')):
        if file_sha(TRIAL / 'capture' / filename) != captured[key]:
            raise ValueError('Captured inputs changed')
    array = np.load(TRIAL / 'capture/latents.npy')
    count = len(array)
    partitions = groups(count, args.batch)
    if any(end - begin != args.batch for begin, end in partitions):
        raise ValueError('This module test intentionally uses complete batches only')
    tick = time.perf_counter()
    codec = RoleCodec(ASSETS, 'receiver')
    torch.utils.deterministic.fill_uninitialized_memory = False
    model = codec.video.vqgan.generator
    x = torch.from_numpy(array).cuda()
    cache = DECODER_TRIAL / 'engines' if args.batch == 1 else TRIAL / 'engines'
    renderer = GeneratorTensorRT(model, x[:args.batch], 'trt_fp16_safe', 'video', cache, idle)
    if args.batch == 1 and renderer.built:
        raise ValueError('Batch-1 must reuse the validated stage 17 P engine')
    receipt = renderer.receipt()
    del codec, model
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - tick
    width, height = captured['header']['width'], captured['header']['height']
    def forward(begin, end):
        # Latents are already on GPU. Include gathering/copying needed to form
        # a true batch, rather than timing only a prepacked tensor view.
        value = x[begin:end] if args.batch == 1 else torch.cat([x[i:i+1] for i in range(begin, end)], dim=0)
        return renderer(value)
    timings = {}
    for mode in ('generator', 'rgb_output'):
        idle()
        for _ in range(2):
            for begin, end in partitions:
                value = forward(begin, end)
                if mode == 'rgb_output':rgb_batch(value, width, height)
        torch.cuda.synchronize()
        rows = []
        for repeat in range(3):
            idle();torch.cuda.synchronize()
            start, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            tick = time.perf_counter();start.record()
            for begin, end in partitions:
                value = forward(begin, end)
                if mode == 'rgb_output':rgb_batch(value, width, height)
            finish.record();torch.cuda.synchronize()
            seconds = time.perf_counter() - tick
            idle()
            rows.append(dict(seconds=seconds, ms_per_frame=seconds*1000/count,
                             fps=count/seconds, cuda_event_ms=start.elapsed_time(finish)))
        timings[mode] = rows
        print(f'B{args.batch} {mode}: {statistics.median(r["ms_per_frame"] for r in rows):.3f}ms/frame', flush=True)
    # Numerical checks and memory sampling are outside the timing intervals.
    reference = x.clone()
    decoded = []
    for begin, end in partitions:
        decoded.append(rgb_batch(forward(begin, end), width, height))
    rgb = np.concatenate(decoded)
    if not torch.equal(reference, x):
        raise ValueError('Generator changed its input latent')
    np.save(output / 'rgb.npy', rgb)
    del reference, value
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = [];stop = threading.Event()
    def sample():
        while not stop.wait(.02):
            free, total = torch.cuda.mem_get_info()
            samples.append((total-free)/1024**2)
    thread = threading.Thread(target=sample, daemon=True)
    free, total = torch.cuda.mem_get_info();samples.append((total-free)/1024**2)
    thread.start()
    try:
        for _ in range(2):
            for begin, end in partitions:
                rgb_batch(forward(begin, end), width, height)
        torch.cuda.synchronize()
    finally:
        stop.set();thread.join()
    idle()
    if hashes != code_hashes():
        raise ValueError('Benchmark producer changed during run')
    save_json(output / 'result.json', dict(batch=args.batch, repeat=args.repeat, model='video', frames=count,
        selected_indices=captured['selected_indices'], complete_batches_only=True,
        input_shape=list(x.shape), engine=receipt, setup_seconds=setup_seconds,
        timings=timings, rgb_sha256=file_sha(output/'rgb.npy'), reference_input_unchanged=True,
        latent_source_sha256=captured['latents_sha256'], input_stream_sha256=captured['input_stream_sha256'],
        device_used_sampled_peak_mib=max(samples), memory_samples=len(samples),
        memory_scope='Device-wide usage including desktop applications; 20ms sampling in a separate untimed pass, not a process exact peak',
        torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        engine_context_bytes=int(renderer.engine.device_memory_size),
        code_sha256=hashes, full_decoder_batched=False, training_steps=0, Orin_tested=False))
    print(f'PASS B{args.batch}, run{args.repeat}: finite RGB, original latent unchanged', flush=True)


if __name__ == '__main__':main()
