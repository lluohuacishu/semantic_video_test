"""Source-blind same-stream full decoding with explicitly identified P renderer."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot_common import file_sha, load_json, model_sha, new_codec, save_json
from model_transfer import architecture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clip', required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--trial', type=int, default=0)
    parser.add_argument('--save-rgb', action='store_true')
    args = parser.parse_args()
    plan = load_json(args.output / 'plan.json')
    if args.case != 'teacher' and args.case not in plan['training']['cases']:
        raise ValueError('Unknown case')
    target = args.output / 'evaluation/full_decode' / f'{args.clip}_{args.case}_{args.trial}'
    target.mkdir(parents=True, exist_ok=False)
    def audit(event, values):
        if event == 'open' and values and isinstance(values[0], (str, bytes)):
            name = str(values[0]).replace('\\', '/').lower()
            if any(name.endswith('/'+n) for n in ['source.npy', 'fish_mask.npy', 'gt_available.npy', 'teacher.npy', 'latent.npy']):
                raise RuntimeError('Independent decoder tried to read source/GT/teacher/latent cache')
    sys.addaudithook(audit)
    codec, ref = new_codec(decoder_only=True)
    identity = dict(case=args.case, p_rgb='original teacher', i_rgb='original teacher', checkpoint_sha256=None)
    if args.case != 'teacher':
        path = args.output / f'train_{args.case}/best.pt'
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['case'] != args.case or state['plan_sha256'] != file_sha(args.output / 'plan.json'):
            raise ValueError('Student checkpoint protocol mismatch')
        renderer = architecture()
        renderer.load_state_dict(state['model'], strict=True)
        renderer = renderer.eval().cuda()
        identity.update(p_rgb='experimental 10985155-parameter student, FP32', step=state['step'],
                        state_sha256=model_sha(renderer), checkpoint_sha256=file_sha(path))
        codec.video.vqgan.generator = renderer
        del state, renderer
        torch.cuda.empty_cache()
    folder = args.output / args.clip
    header, packets, integrity = ref.read_stream(folder / 'stream.glcv', with_integrity=True)
    expected = load_json(folder / 'encode_cache.json')
    if header != expected['header'] or file_sha(folder / 'stream.glcv') != expected['stream_sha256']:
        raise ValueError('Stream changed')
    if header['arithmetic'] != codec.profile or header['table_sha256'] != codec.entropy.table_sha256:
        raise ValueError('Decoder arithmetic mismatch')
    for i in range(2):
        codec.frame(header, i, packet=packets[i])
    codec.dpb = dict(ref_latent=None, ref_y=None)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rgb_array = np.empty((header['frames'], 288, 512, 3), dtype='uint8')
    checksums, rows = ref.SequenceChecksums(), []
    start = time.perf_counter()
    for i, packet in enumerate(packets):
        tick = time.perf_counter()
        rgb, checked = codec.frame(header, i, packet=packet)
        duration = time.perf_counter()-tick
        sent = expected['records'][i]
        if checked['latent_sha256'] != sent['latent_sha256'] or checked['kind'] != sent['kind']:
            raise ValueError('Original reference chain diverged')
        if (args.case == 'teacher' or checked['kind'] == 0) and checked['rgb_sha256'] != sent['rgb_sha256']:
            raise ValueError('Original teacher/I RGB diverged')
        checksums.add(i, checked)
        rgb_array[i] = rgb
        rows.append(dict(index=i, kind=checked['kind'], seconds=duration,
                         reference_sha256=checked['latent_sha256'], output_rgb_sha256=checked['rgb_sha256']))
    seconds = time.perf_counter()-start
    if checksums.values()['latent_sha256'] != integrity['sequence_reconstruction_checksums']['latent_sha256']:
        raise ValueError('Original ordered latent checksum differs')
    if args.case == 'teacher':
        checksums.verify(integrity['sequence_reconstruction_checksums'])
    report = dict(clip=args.clip, case=args.case, trial=args.trial, frames=header['frames'], model_identity=identity,
                  stream_sha256=expected['stream_sha256'], transport_verified=True, original_reference_chain_verified=True,
                  original_i_rgb_verified=True, source_and_cache_reads_forbidden=True,
                  original_teacher_ordered_rgb_sha256=integrity['sequence_reconstruction_checksums']['rgb_sha256'],
                  output_ordered_rgb_sha256=checksums.values()['rgb_sha256'], student_rgb_is_distinct_model_output=args.case!='teacher',
                  core_seconds=seconds, mean_frame_seconds=float(np.mean([r['seconds'] for r in rows])),
                  p_mean_frame_seconds=float(np.mean([r['seconds'] for r in rows if r['kind']==1])),
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                  timing_scope='warm entropy+latent reference chain+RGB+CPU transfer/checksums; excludes model loading, disk I/O, MP4 and network',
                  precision='FP32; TF32 off; original deterministic codec profile', warmup_frames=2, records=rows)
    if args.save_rgb:
        np.save(target / 'rgb.npy', rgb_array)
        report['rgb_array_sha256'] = file_sha(target / 'rgb.npy')
    save_json(target / 'decode.json', report)
    print(f'decode {args.clip} {args.case} trial{args.trial}: {seconds:.3f}s; original references exact', flush=True)


if __name__ == '__main__':
    main()
