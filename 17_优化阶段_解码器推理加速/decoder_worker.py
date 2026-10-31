"""Independent GLCSND receiver. No source/GT input; replace only final I/P RGB generators."""
from __future__ import annotations

import argparse
import hashlib
import sys
import time

from decoder_common import ASSETS, BACKENDS, TRIAL, Path, code_hashes, file_sha, load_settings, save_json
import numpy as np
import torch
from role_codec import RoleCodec, ref
from sender_container import ReferenceChecksums, read_sender_stream


def audit_media_reads(output):
    root = output.resolve()
    def audit(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes)):
            path = Path(str(args[0]))
            if path.suffix.lower() in ('.npy', '.npz', '.png', '.jpg', '.jpeg', '.mp4', '.mkv', '.avi'):
                if not path.resolve().is_relative_to(root):
                    raise RuntimeError('Independent receiver cannot read source media or GT')
    sys.addaudithook(audit)


def validate_stream(header, codec):
    if (header['arithmetic'] != codec.profile or header['table_sha256'] != codec.entropy.table_sha256 or
            header['revision'] != ref.REVISION or header['entropy'] != ref.ENTROPY_ID or
            header['weights'] != {key: value['sha256'] for key, value in ref.ASSETS.items()}):
        raise ValueError('Receiver arithmetic, entropy, revision or original weight identity differs')


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=BACKENDS, required=True)
    parser.add_argument('--assets', type=Path, default=ASSETS)
    parser.add_argument('--engine-cache', type=Path, default=TRIAL / 'engines')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--microbenchmark', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    hashes = code_hashes()
    stream_sha = file_sha(args.stream)
    audit_media_reads(args.output)
    header, packets, integrity = read_sender_stream(args.stream)
    tick = time.perf_counter()
    codec = RoleCodec(args.assets, 'receiver')
    torch.utils.deterministic.fill_uninitialized_memory = False
    if not torch.are_deterministic_algorithms_enabled():
        raise ValueError('Reference determinism must remain enabled')
    validate_stream(header, codec)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - tick
    samples, expected, setup, differences, micro = {}, {}, {}, {}, {}
    handles = []
    def capture(name):
        def hook(module, inputs, output):
            if name not in samples:
                samples[name] = inputs[0].detach().clone()
                expected[name] = output.detach().cpu().numpy().copy()
        return hook
    tick = time.perf_counter()
    for name in ('image', 'video'):
        handles.append(getattr(codec, name).vqgan.generator.register_forward_hook(capture(name)))
    try:
        for i in range(min(2, header['frames'])):
            codec.frame(header, i, packet=packets[i])
    finally:
        for handle in handles:
            handle.remove()
    torch.cuda.synchronize()
    capture_seconds = time.perf_counter() - tick
    if args.backend != 'eager':
        from generator_backend import GeneratorTensorRT
        for name, example in samples.items():
            target = getattr(codec, name).vqgan
            wrapped = GeneratorTensorRT(target.generator, example, args.backend, name, args.engine_cache)
            target.generator = wrapped
            setup[name] = wrapped.receipt()
    for name, example in samples.items():
        target = getattr(codec, name).vqgan.generator
        before = example.clone()
        actual = target(example).float().cpu().numpy()
        error = actual.astype(np.float64) - expected[name]
        unchanged = bool(torch.equal(before, example))
        if not np.isfinite(actual).all() or not unchanged:
            save_json(args.output / 'numerical_failure.json', dict(model=name, backend=args.backend,
                nonfinite_values=int(np.count_nonzero(~np.isfinite(actual))), reference_input_unchanged=unchanged))
            raise ValueError(f'{name} generator output is nonfinite or changed its reference input')
        differences[name] = dict(shape=list(actual.shape), max_abs=float(np.abs(error).max()),
                                 mean_abs=float(np.abs(error).mean()), rmse=float(np.sqrt(np.mean(error**2))),
                                 reference_input_unchanged=True, output_dtype=str(actual.dtype))
        if args.microbenchmark:
            for _ in range(4):
                target(example)
            torch.cuda.synchronize()
            rows = []
            for _ in range(12):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                tick = time.perf_counter()
                start.record()
                target(example)
                end.record()
                torch.cuda.synchronize()
                rows.append(dict(wall_ms=(time.perf_counter() - tick) * 1000, cuda_event_ms=start.elapsed_time(end)))
            micro[name] = rows
    tick = time.perf_counter()
    codec.dpb = dict(ref_latent=None, ref_y=None)
    for i in range(min(8, header['frames'])):
        codec.frame(header, i, packet=packets[i])
    torch.cuda.synchronize()
    warm_seconds = time.perf_counter() - tick
    codec.dpb = dict(ref_latent=None, ref_y=None)
    torch.cuda.reset_peak_memory_stats()
    free_before, total_memory = torch.cuda.mem_get_info()
    entropy_digest = hashlib.sha256()
    decode = codec.entropy.decode
    def checked_decode(stream, indexes):
        symbols = decode(stream, indexes)
        entropy_digest.update(np.asarray(indexes, dtype='<i4').tobytes())
        entropy_digest.update(np.asarray(symbols, dtype='<i4').tobytes())
        return symbols
    codec.entropy.decode = checked_decode
    rgb_path = args.output / 'decoded_rgb.npy'
    preview = None
    if args.video:
        from c1_io import RawPreview
        preview = RawPreview(load_settings()['runtime']['ffmpeg'], args.output / 'video.mp4',
                             header['width'], header['height'], header['fps'])
    checks = ReferenceChecksums()
    rows = []
    operation = time.perf_counter()
    try:
        with rgb_path.open('xb') as output:
            np.lib.format.write_array_header_1_0(output, dict(descr='|u1', fortran_order=False,
                shape=(header['frames'], header['height'], header['width'], 3)))
            for i, packet in enumerate(packets):
                torch.cuda.synchronize()
                tick = time.perf_counter()
                rgb, result = codec.frame(header, i, packet=packet)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - tick
                checks.add(i, result)
                output.write(rgb.tobytes())
                if preview:
                    preview.add(rgb, lambda: None)
                rows.append(dict(index=i, kind=result['kind'], seconds=elapsed,
                                 latent_sha256=result['latent_sha256'], rgb_sha256=result['rgb_sha256']))
                if (i + 1) % 64 == 0 or i == len(packets) - 1:
                    print(f'{args.backend}: {i+1}/{len(packets)}, core={sum(r["seconds"] for r in rows):.3f}s', flush=True)
        checks.verify(integrity['expected_reference_checksums'])
        if preview:
            preview.finish(lambda: None)
    except BaseException:
        if preview:
            preview.abort()
        raise
    operation_seconds = time.perf_counter() - operation
    if stream_sha != file_sha(args.stream) or hashes != code_hashes():
        raise ValueError('Input bitstream or implementation changed during the receiver task')
    core = sum(row['seconds'] for row in rows)
    save_json(args.output / 'result.json', dict(backend=args.backend, header=header, frames=len(rows),
        stream_sha256=stream_sha, stream_bytes=args.stream.stat().st_size, records=rows,
        model_load_seconds=load_seconds, capture_seconds=capture_seconds,
        engine_setup_seconds=sum(v['setup_seconds'] for v in setup.values()), setup=setup,
        generator_sample_difference=differences, generator_microbenchmark=micro,
        warmup_seconds=warm_seconds, core_seconds=core, fps=len(rows) / core,
        file_operation_seconds=operation_seconds, worker_seconds=time.perf_counter() - began,
        reference_checksums=checks.values(), reference_verified=True, source_media_reads_forbidden=True,
        entropy_symbols_and_cdf_sha256=entropy_digest.hexdigest(), decoded_rgb_sha256=file_sha(rgb_path),
        video_sha256=file_sha(args.output / 'video.mp4') if preview else None,
        peak_torch_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
        device_free_before_mib=free_before / 1024**2, device_free_after_mib=torch.cuda.mem_get_info()[0] / 1024**2,
        device_total_mib=total_memory / 1024**2, code_sha256=hashes,
        deterministic_reference_fp32=True, tf32=False, fill_uninitialized_memory=False,
        weights_changed=False, training_steps=0, Orin_tested=False))
    print(f'PASS {args.backend}: {core:.3f}s, {len(rows)/core:.3f} fps; all references verified', flush=True)


if __name__ == '__main__':
    main()
