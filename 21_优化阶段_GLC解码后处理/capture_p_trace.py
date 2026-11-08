"""Record P-frame activations while replaying the original GLC bitstream."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

from run_boundary_codec import REPO, file_sha, load_json, reef_inputs, save_json
from run_deflicker import START, STOP
from sender_container import read_sender_stream

STAGES = ('context', 'y_hat', 'latent', 'generator_rgb')


def capture(workspace, assets, prefix_run, output):
    sys.path.insert(0, str(REPO / '17_优化阶段_解码器推理加速'))
    from decoder_worker import audit_media_reads, validate_stream
    from role_codec import RoleCodec, ref
    import torch

    _, _, baseline = reef_inputs(workspace)
    baseline_result = workspace / baseline['metadata']
    original = load_json(baseline_result)
    stream = baseline_result.parent.parent / 'glc_sender/stream.glcsnd'
    stream_sha = file_sha(stream)
    if stream_sha != original['stream_sha256']:
        raise ValueError('Original GLC stream changed')
    header, packets, integrity = read_sender_stream(stream)
    prefix_stream = prefix_run / 'stream.glcsnd'
    prefix_header, prefix_packets, prefix_integrity = read_sender_stream(prefix_stream)
    receipt_path = prefix_run / 'receiver/result.json'
    receipt = load_json(receipt_path)
    if (original['reference_checksums'] != integrity['expected_reference_checksums']
            or prefix_header != dict(header, frames=len(prefix_packets)) or len(prefix_packets) < STOP
            or prefix_packets != packets[:len(prefix_packets)] or receipt['header'] != prefix_header
            or receipt['stream_sha256'] != file_sha(prefix_stream) or not receipt['reference_verified']
            or receipt['reference_checksums'] != prefix_integrity['expected_reference_checksums']):
        raise ValueError('Expected the independently verified bit-exact GLC prefix')

    output.mkdir(parents=True, exist_ok=False)
    audit_media_reads(output)
    codec = RoleCodec(assets, 'receiver')
    torch.utils.deterministic.fill_uninitialized_memory = False
    validate_stream(header, codec)
    i_period = header['intra_period']
    warm_start = START - START % i_period
    indexes = [i for i in range(START, STOP) if i % i_period]
    p_slot = {i: slot for slot, i in enumerate(indexes)}
    stage_arrays = {}
    recorded = set()
    slot = None
    height, width = header['height'], header['width']
    rgb = np.lib.format.open_memmap(output / 'decoded_rgb.npy', mode='w+', dtype=np.uint8,
                                    shape=(STOP - START, height, width, 3))

    def record(name, value):
        if slot is None:
            return
        if name in recorded:
            raise ValueError(f'Duplicate stage capture for {name}')
        array = value.detach().cpu().numpy()
        if array.ndim != 4 or array.shape[0] != 1 or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f'Expected one finite FP32 tensor for {name}')
        if name not in stage_arrays:
            stage_arrays[name] = np.lib.format.open_memmap(
                output / f'{name}.npy', mode='w+', dtype=np.float32,
                shape=(len(indexes), *array.shape[1:]))
        stage_arrays[name][slot] = array[0]
        recorded.add(name)

    def record_decoder_input(_module, args):
        record('y_hat', args[0])
        record('context', args[1])

    def record_generator_input(_module, args):
        record('latent', args[0])

    def record_generator_output(_module, _args, value):
        record('generator_rgb', value)

    decoder, generator = codec.video.decoder, codec.video.vqgan.generator
    hooks = [decoder.register_forward_pre_hook(record_decoder_input),
             generator.register_forward_pre_hook(record_generator_input),
             generator.register_forward_hook(record_generator_output)]
    tick = time.perf_counter()
    print(f'Replay original bitstream from I frame {warm_start}; capture P tensors in [{START}, {STOP})', flush=True)
    try:
        with torch.inference_mode():
            for i in range(warm_start, STOP):
                slot = p_slot.get(i)
                recorded.clear()
                decoded, result = codec.frame(header, i, packet=packets[i])
                if slot is not None and recorded != set(STAGES):
                    raise ValueError(f'Missing stage capture at frame {i}')
                expected = receipt['records'][i]
                if expected['index'] != i or any(result[key] != expected[key] for key in
                                                  ('kind', 'latent_sha256', 'rgb_sha256')):
                    raise ValueError(f'Trace altered reference or RGB at frame {i}')
                if i >= START:
                    rgb[i - START] = decoded
                if (i + 1 - warm_start) % 24 == 0 or i + 1 == STOP:
                    print(f'Trace verified through original frame {i}', flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    elapsed = time.perf_counter() - tick
    if set(stage_arrays) != set(STAGES):
        raise ValueError('One or more P-frame stages were not captured')
    for value in (*stage_arrays.values(), rgb):
        value.flush()
    files = {name: dict(shape=list(array.shape), dtype=str(array.dtype),
                        sha256=file_sha(output / f'{name}.npy'))
             for name, array in dict(stage_arrays, decoded_rgb=rgb).items()}
    save_json(output / 'capture.json', dict(
        complete=True, start=START, stop=STOP, warm_start=warm_start, p_frames=indexes,
        header=header, stream_sha256=stream_sha, prefix_receipt_sha256=file_sha(receipt_path),
        replay_frames_verified=STOP - warm_start, source_media_reads_forbidden=True,
        files=files, decode_and_capture_seconds=elapsed, code_sha256=file_sha(__file__),
        reference_adaptor_schedule=ref.SCHEDULE))
    print(f'PASS: {len(indexes)} P-frame traces with bit-exact RGB and reference state', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--prefix-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assets = args.assets or args.workspace / '02_实验数据/优化_部署精简/role_assets_20261029_01'
    capture(args.workspace.resolve(), assets.resolve(), args.prefix_run.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
