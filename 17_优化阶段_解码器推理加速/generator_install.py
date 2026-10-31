"""Attach a validated display backend to an independent receiver, outside core timing."""
from __future__ import annotations

import time
import torch
from decoder_common import BACKENDS
from generator_backend import GeneratorTensorRT


@torch.inference_mode()
def install(codec, header, packets, backend, cache, check=lambda: None):
    if backend not in BACKENDS or codec.role != 'receiver':
        raise ValueError('Invalid receiver display backend')
    if backend == 'eager':
        return dict(capture_seconds=0., setup_seconds=0., generators={})
    examples, handles = {}, []
    def capture(name):
        def hook(_module, args):
            if name not in examples:
                examples[name] = args[0].detach().clone()
        return hook
    started = time.perf_counter()
    codec.dpb = dict(ref_latent=None, ref_y=None)
    for name in ('image', 'video'):
        handles.append(getattr(codec, name).vqgan.generator.register_forward_pre_hook(capture(name)))
    try:
        for i in range(min(2, header['frames'])):
            check()
            codec.frame(header, i, packet=packets[i])
    finally:
        for hook in handles:
            hook.remove()
        codec.dpb = dict(ref_latent=None, ref_y=None)
    torch.cuda.synchronize()
    capture_seconds = time.perf_counter() - started
    generators = {}
    for name, example in examples.items():
        check()
        target = getattr(codec, name).vqgan
        wrapper = GeneratorTensorRT(target.generator, example, backend, name, cache, check)
        target.generator = wrapper
        generators[name] = wrapper.receipt()
    return dict(capture_seconds=capture_seconds, setup_seconds=sum(v['setup_seconds'] for v in generators.values()),
                generators=generators)
