"""Source-blind experimental decoder; preserves transport and latent verification.

Only the terminal P RGB generator is substituted. This does not modify the
production codec or declare student RGB equal to the original teacher checksum.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

from pilot_common import file_sha, load_json, model_sha, new_codec, save_json, sha, student


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clip', default='v007')
    parser.add_argument('--mode', choices=['teacher', 'kd', 'gt'], required=True)
    parser.add_argument('--trial', type=int, default=0)
    parser.add_argument('--save-rgb', action='store_true')
    args = parser.parse_args()
    folder = args.output / args.clip
    target = args.output / 'evaluation' / 'full_decode' / f'{args.clip}_{args.mode}_{args.trial}'
    target.mkdir(parents=True, exist_ok=False)

    def audit(event, values):
        if event == 'open' and values and isinstance(values[0], (str, bytes)):
            name = str(values[0]).replace('\\', '/').lower()
            if any(name.endswith('/' + n) for n in ['source.npy', 'fish_mask.npy', 'latent.npy', 'teacher.npy']) or '/timing_rgb/' in name:
                raise RuntimeError('Independent student decoder tried to read source/GT/teacher/latent cache')
    sys.addaudithook(audit)
    codec, ref = new_codec(decoder_only=True)
    model_identity = dict(mode=args.mode, image_rgb='original teacher', p_rgb='original teacher', checkpoint_sha256=None)
    if args.mode != 'teacher':
        checkpoint_path = args.output / f'train_{args.mode}' / 'best.pt'
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if state['plan_sha256'] != file_sha(args.output / 'plan.json') or state['mode'] != args.mode:
            raise ValueError('Student checkpoint protocol mismatch')
        renderer = student()
        renderer.load_state_dict(state['model'], strict=True)
        renderer = renderer.eval().cuda()
        model_identity.update(p_rgb='experimental 8800707-parameter student; FP32', step=state['step'],
                              checkpoint_sha256=file_sha(checkpoint_path), state_sha256=model_sha(renderer))
        codec.video.vqgan.generator = renderer
        del state, renderer
        torch.cuda.empty_cache()
    header, packets, integrity = ref.read_stream(folder / 'stream.glcv', with_integrity=True)
    sent = load_json(folder / 'encode_cache.json')
    if file_sha(folder / 'stream.glcv') != sent['stream_sha256'] or header != sent['header']:
        raise ValueError('Stream changed from recorded teacher cache')
    if header['arithmetic'] != codec.profile or header['table_sha256'] != codec.entropy.table_sha256:
        raise ValueError('Arithmetic profile mismatch')
    # Two real frames warm all I/P kernels and CPU entropy tables, then reset DPB.
    for i in range(min(2, len(packets))):
        codec.frame(header, i, packet=packets[i])
    codec.dpb = dict(ref_latent=None, ref_y=None)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rgb_array = np.empty((header['frames'], header['height'], header['width'], 3), dtype='uint8')
    checksums = ref.SequenceChecksums()
    records, durations = [], []
    start = time.perf_counter()
    for i, packet in enumerate(packets):
        tick = time.perf_counter()
        rgb, checked = codec.frame(header, i, packet=packet)
        duration = time.perf_counter() - tick
        expected = sent['records'][i]
        if checked['latent_sha256'] != expected['latent_sha256'] or checked['kind'] != expected['kind']:
            raise ValueError(f'Original reference chain changed at frame {i}')
        if (args.mode == 'teacher' or checked['kind'] == 0) and checked['rgb_sha256'] != expected['rgb_sha256']:
            raise ValueError(f'Original teacher RGB changed at frame {i}')
        checksums.add(i, checked)
        rgb_array[i] = rgb
        durations.append(duration)
        records.append(dict(index=i, kind=checked['kind'], seconds=duration,
                            reference_sha256=checked['latent_sha256'], output_rgb_sha256=checked['rgb_sha256']))
    core_seconds = time.perf_counter() - start
    expected_checksums = integrity['sequence_reconstruction_checksums']
    if checksums.values()['latent_sha256'] != expected_checksums['latent_sha256']:
        raise ValueError('Ordered original latent checksum changed')
    if args.mode == 'teacher':
        checksums.verify(expected_checksums)
    report = dict(mode=args.mode, clip=args.clip, trial=args.trial, frames=header['frames'],
                  precision='FP32; TF32 disabled; original deterministic codec profile',
                  model_identity=model_identity, source_cache_reads_forbidden=True,
                  stream_sha256=sent['stream_sha256'], transport_sha256_verified=True,
                  original_ordered_latent_sha256_verified=True, original_i_rgb_verified=True,
                  original_teacher_ordered_rgb_sha256=expected_checksums['rgb_sha256'],
                  output_ordered_rgb_sha256=checksums.values()['rgb_sha256'],
                  student_rgb_is_different_model_output=args.mode != 'teacher',
                  core_seconds=core_seconds, sum_frame_seconds=float(sum(durations)),
                  mean_frame_seconds=float(np.mean(durations)), median_frame_seconds=float(np.median(durations)),
                  p_mean_frame_seconds=float(np.mean([r['seconds'] for r in records if r['kind'] == 1])),
                  timing_scope='warm model: entropy decode + neural reference chain + RGB renderer + CPU RGB transfer/checksums; excludes model loading, stream disk read, output disk write and MP4 encoding',
                  warmup_frames=2, records=records,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
    if args.save_rgb:
        np.save(target / 'rgb.npy', rgb_array)
        report['rgb_array_sha256'] = file_sha(target / 'rgb.npy')
    save_json(target / 'decode.json', report)
    print(f'{args.clip} {args.mode} trial{args.trial}: {core_seconds:.3f}s / {header["frames"]} frames; reference chain exact', flush=True)


if __name__ == '__main__':
    main()
