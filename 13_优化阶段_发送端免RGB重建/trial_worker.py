"""One fresh GPU process: matched sender timing or source-blind full decoding."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import struct
import sys
import time

import numpy as np

from sender_common import file_sha, load_json, new_codec, save_json
from sender_container import PROFILE, ReferenceChecksums, SenderWriter, read_sender_stream


def payload_fingerprint(packet):
    digest = hashlib.sha256()
    for part in [packet['z'], *packet['y']]:
        digest.update(struct.pack('>I', len(part)))
        digest.update(part)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--clip', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--role', choices=['baseline', 'no_rgb', 'decode'], required=True)
    parser.add_argument('--stream', type=Path)
    args = parser.parse_args()
    process_start = time.perf_counter()
    plan = load_json(args.plan)
    clip = next(c for c in plan['clips'] if c['id'] == args.clip)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.role == 'decode':
        forbidden = {str(Path(c['source']).resolve()).casefold() for c in plan['clips']}
        def audit(event, values):
            if event == 'open' and values and isinstance(values[0], (str, bytes)):
                path = str(values[0])
                normalized = path.replace('\\', '/').lower()
                if (str(Path(path).resolve()).casefold() in forbidden or
                        normalized.endswith(('/source.npy', '/teacher.npy', '/fish_mask.npy')) or
                        normalized.endswith(('.mp4', '.mkv', '.png', '.jpg', '.jpeg'))):
                    raise RuntimeError('Independent receiver tried to read source/GT media')
        sys.addaudithook(audit)
    import torch
    import torch.nn.functional as F
    init_start = time.perf_counter()
    codec, ref = new_codec(decoder_only=args.role == 'decode')
    torch.cuda.synchronize()
    init_seconds = time.perf_counter() - init_start
    header = {key: clip[key] for key in ('width', 'height', 'frames', 'fps', 'q')}
    header.update(intra_period=64, reset_interval=32, revision=ref.REVISION,
                  entropy=ref.ENTROPY_ID, table_sha256=codec.entropy.table_sha256,
                  arithmetic=codec.profile,
                  weights={key: value['sha256'] for key, value in ref.ASSETS.items()},
                  container_version=2)
    generator_calls = dict(image=0, video=0)
    def count_generator(name):
        def hook(_module, _inputs):
            generator_calls[name] += 1
            if args.role == 'no_rgb':
                raise RuntimeError('No-RGB sender executed a terminal RGB generator')
        return hook
    handles = [getattr(codec, name).vqgan.generator.register_forward_pre_hook(count_generator(name))
               for name in generator_calls]
    frame_entropy = []
    operation = codec.entropy.decode if args.role == 'decode' else codec.entropy.encode
    def entropy_operation(values, indexes):
        result = operation(values, indexes)
        symbols = result if args.role == 'decode' else values
        frame_entropy.append({
            'symbols_sha256': hashlib.sha256(np.asarray(symbols, dtype='<i4').tobytes()).hexdigest(),
            'cdf_indexes_sha256': hashlib.sha256(np.asarray(indexes, dtype='<i4').tobytes()).hexdigest(),
            'count': int(np.asarray(symbols).size),
        })
        return result
    if args.role == 'decode':
        codec.entropy.decode = entropy_operation
        if args.stream is None:
            raise ValueError('Receiver requires an explicit sender stream')
        stream_header, packets, integrity = read_sender_stream(args.stream)
        expected_header = dict(header, container_version=3, integrity_profile=PROFILE,
                               sender_rgb_reconstructed=False)
        if stream_header != expected_header:
            raise ValueError('Independent receiver model/table/arithmetic/header mismatch')
        header = stream_header
    else:
        codec.entropy.encode = entropy_operation
        if file_sha(clip['source']) != clip['source_sha256']:
            raise ValueError('Frozen source array changed')
        source = np.load(clip['source'], mmap_mode='r')
        if tuple(source.shape) != (clip['frames'], clip['height'], clip['width'], 3) or source.dtype != np.uint8:
            raise ValueError('Source array shape/type mismatch')
    def input_tensor(rgb):
        value = torch.from_numpy(np.array(rgb)).cuda().permute(2, 0, 1)[None].float().div(255)
        return F.pad(value, (0, (-header['width']) % 64, 0, (-header['height']) % 64), mode='replicate').mul(2).sub(1)
    # Eight real frames, the same mode and geometry as the measured task.
    warm_start = time.perf_counter()
    for i in range(min(8, header['frames'])):
        if args.role == 'decode':
            codec.frame(header, i, packet=packets[i])
        else:
            codec.frame(header, i, x=input_tensor(source[i]), render_rgb=args.role == 'baseline')
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warm_start
    codec.dpb = dict(ref_latent=None, ref_y=None)
    generator_calls.update(image=0, video=0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    resident_mib = torch.cuda.memory_allocated() / 1024**2
    records, core_seconds = [], 0.0
    task_start = time.perf_counter()
    if args.role == 'decode':
        references = ReferenceChecksums()
        rgb_output = np.lib.format.open_memmap(args.output / 'decoded_rgb.npy', mode='w+', dtype='uint8',
                        shape=(header['frames'], header['height'], header['width'], 3))
        writer = None
        stream_path = args.stream
    else:
        stream_path = args.output / ('stream.glcv' if args.role == 'baseline' else 'stream.glcsnd')
        if args.role == 'no_rgb':
            header = dict(header, container_version=3, integrity_profile=PROFILE,
                          sender_rgb_reconstructed=False)
            writer = SenderWriter(stream_path, header)
        else:
            writer = ref.StreamWriter(stream_path, header, version=2)
    try:
        for index in range(header['frames']):
            frame_entropy.clear()
            x = None if args.role == 'decode' else input_tensor(source[index])
            torch.cuda.synchronize()
            start = time.perf_counter()
            if args.role == 'decode':
                rgb, packet = codec.frame(header, index, packet=packets[index])
            else:
                rgb, packet = codec.frame(header, index, x=x, render_rgb=args.role == 'baseline')
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            core_seconds += seconds
            if writer is not None:
                writer.add(packet)
            else:
                references.add(index, packet)
                rgb_output[index] = rgb
            row = dict(index=index, kind=packet['kind'], seconds=seconds,
                       latent_sha256=packet['latent_sha256'], payload_sha256=payload_fingerprint(packet),
                       entropy=list(frame_entropy),
                       payload_bytes=len(packet['z']) + sum(map(len, packet['y'])))
            if 'rgb_sha256' in packet:
                row['rgb_sha256'] = packet['rgb_sha256']
            records.append(row)
            if (index + 1) % 32 == 0:
                print(f'{args.clip} {args.role} {index+1}/{header["frames"]}', flush=True)
        if writer is not None:
            writer.finish()
        else:
            references.verify(integrity['expected_reference_checksums'])
            rgb_output.flush()
        torch.cuda.synchronize()
        task_seconds = time.perf_counter() - task_start
    finally:
        if writer is not None:
            writer.close()
        for handle in handles:
            handle.remove()
    result = dict(role=args.role, clip=args.clip, header=header, records=records,
                  stream_sha256=file_sha(stream_path), stream_bytes=stream_path.stat().st_size,
                  payload_bytes=sum(r['payload_bytes'] for r in records), generator_calls=generator_calls,
                  core_seconds=core_seconds, task_seconds=task_seconds, init_seconds=init_seconds,
                  warmup_seconds=warmup_seconds, warmup_frames=min(8, header['frames']),
                  resident_allocated_mib=resident_mib,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                  peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
                  process_seconds_before_report=time.perf_counter()-process_start,
                  measurement='FP32/native rANS, synchronized wall clock; all frames/stalls retained; includes audit hooks')
    if args.role == 'decode':
        result.update(reference_verified=True, verified_reference_checksums=references.values(),
                      sender_rgb_checksum_available=False, sender_rgb_verified=None,
                      receiver_rgb_sha256=file_sha(args.output / 'decoded_rgb.npy'),
                      source_reads_forbidden=True, source_encoder_forbidden=True)
    elif args.role == 'no_rgb' and generator_calls != dict(image=0, video=0):
        raise AssertionError('Sender ran a generator')
    save_json(args.output / 'result.json', result)
    print(f'completed {args.clip} {args.role}: core={core_seconds:.3f}s task={task_seconds:.3f}s', flush=True)


if __name__ == '__main__':
    main()
