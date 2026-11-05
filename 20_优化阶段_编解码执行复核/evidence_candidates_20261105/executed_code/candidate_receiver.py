"""Independent full-stream check for an experimental RGB generator backend."""
from __future__ import annotations
import argparse
import hashlib
import time
from candidate_backend import *
from probe_cached_engines import check_frozen, gpu_status
from role_codec import RoleCodec, ref
from sender_container import read_sender_stream, ReferenceChecksums
from decoder_worker import validate_stream, audit_media_reads
from generator_backend import GeneratorTensorRT


@torch.inference_mode()
def main(backend_builder=build_candidate, modes=MODES, backend_source=HERE/'candidate_backend.py'):
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('baseline',)+modes, required=True)
    ap.add_argument('--stream', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--save-rgb', action='store_true')
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--cuda-graph', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    frozen = check_frozen()
    stream_sha = file_sha(args.stream)
    audit_media_reads(args.output)
    header, packets, integrity = read_sender_stream(args.stream)
    before_gpu = gpu_status()
    tick = time.perf_counter()
    codec = RoleCodec(ASSETS, 'receiver')
    torch.utils.deterministic.fill_uninitialized_memory = False
    validate_stream(header, codec)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter()-tick
    samples, handles = {}, []

    def capture(name):
        def hook(_module, values):
            if name not in samples:
                samples[name] = values[0].detach().clone()
        return hook

    for role in ('image', 'video'):
        handles.append(getattr(codec, role).vqgan.generator.register_forward_pre_hook(capture(role)))
    try:
        for i in range(min(2, len(packets))):
            codec.frame(header, i, packet=packets[i])
    finally:
        for handle in handles:
            handle.remove()
    setup = {}
    for role, sample in samples.items():
        target = getattr(codec, role).vqgan
        if args.mode == 'baseline':
            wrapped = GeneratorTensorRT(target.generator, sample, 'trt_fp16_safe', role,
                DATA/'优化_解码器推理/decoder_20261031_05/engines')
            if wrapped.built:
                raise ValueError('Baseline must reuse the historical verified engine')
            setup[role] = wrapped.receipt()
        else:
            wrapped = backend_builder(target.generator, sample, args.mode, role, args.cache)
            setup[role] = wrapped.setup_receipt
        if args.cuda_graph:
            from graph_rgb import GraphRGB
            wrapped = GraphRGB(wrapped, sample)
        target.generator = wrapped
    codec.dpb = dict(ref_latent=None, ref_y=None)
    for i in range(min(8, len(packets))):
        codec.frame(header, i, packet=packets[i])
    torch.cuda.synchronize()
    codec.dpb = dict(ref_latent=None, ref_y=None)
    digest = hashlib.sha256()
    original_decode = codec.entropy.decode

    def decoded(payload, indexes):
        symbols = original_decode(payload, indexes)
        digest.update(np.asarray(indexes, dtype='<i4').tobytes())
        digest.update(np.asarray(symbols, dtype='<i4').tobytes())
        return symbols

    codec.entropy.decode = decoded
    preview = None
    if args.video:
        from c1_io import RawPreview
        from sender_common import load_settings
        preview = RawPreview(load_settings()['runtime']['ffmpeg'], args.output/'video.mp4',
            header['width'], header['height'], header['fps'])
    output = None
    if args.save_rgb:
        output = (args.output/'decoded_rgb.npy').open('xb')
        np.lib.format.write_array_header_1_0(output, dict(descr='|u1', fortran_order=False,
            shape=(len(packets), header['height'], header['width'], 3)))
    checks = ReferenceChecksums()
    rows = []
    operation = time.perf_counter()
    try:
        for i, packet in enumerate(packets):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            rgb, row = codec.frame(header, i, packet=packet)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-tick
            checks.add(i, row)
            if output is not None:
                output.write(rgb.tobytes())
            if preview is not None:
                preview.add(rgb)
            rows.append(dict(index=i, kind=row['kind'], seconds=elapsed,
                latent_sha256=row['latent_sha256'], rgb_sha256=row['rgb_sha256']))
            if (i+1)%64 == 0 or i == len(packets)-1:
                print(f'{args.mode}: {i+1}/{len(packets)} core={sum(r["seconds"] for r in rows):.3f}s', flush=True)
        checks.verify(integrity['expected_reference_checksums'])
        if preview is not None:
            preview.finish()
    except BaseException:
        if preview is not None:
            preview.abort()
        raise
    finally:
        if output is not None:
            output.close()
    operation_seconds = time.perf_counter()-operation
    if file_sha(args.stream) != stream_sha or check_frozen() != frozen:
        raise ValueError('Stream or frozen reference code changed')
    core = sum(r['seconds'] for r in rows)
    save_json(args.output/'result.json', dict(mode=args.mode, frames=len(rows), header=header,
        stream_sha256=stream_sha, reference_verified=True, source_media_reads_forbidden=True,
        core_seconds=core, fps=len(rows)/core, file_operation_seconds=operation_seconds,
        model_load_seconds=load_seconds, worker_seconds=time.perf_counter()-started,
        reference_checksums=checks.values(), entropy_symbols_and_cdf_sha256=digest.hexdigest(),
        records=rows, setup=setup, frozen_code_sha256=frozen,
        producer_sha256=file_sha(__file__), backend_sha256=file_sha(backend_source),
        experiment_code_sha256={p.name:file_sha(p) for p in HERE.glob('*.py')},
        rgb_saved=args.save_rgb, rgb_sha256=file_sha(args.output/'decoded_rgb.npy') if args.save_rgb else None,
        video_sha256=file_sha(args.output/'video.mp4') if args.video else None,
        reference_tf32=False, reference_dtype='float32', rgb_backend=args.mode, rgb_cuda_graph=args.cuda_graph,
        gpu_before=before_gpu, gpu_after=gpu_status(), weights_changed=False, training_steps=0))
    print(f'PASS {args.mode}: {core:.3f}s, {len(rows)/core:.3f}FPS, all reference states verified', flush=True)


if __name__ == '__main__':
    main()
