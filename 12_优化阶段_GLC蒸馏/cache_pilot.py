"""Real byte-stream cache: encode and source-blind decode in separate processes."""
from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from pilot_common import check_stop, file_sha, load_json, new_codec, save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--role', choices=['encode', 'decode'], required=True)
    args = parser.parse_args()
    plan = load_json(args.output / 'plan.json')
    if not plan['complete'] or file_sha(args.output / 'plan.json') != load_json(args.output / 'plan_identity.json')['sha256']:
        raise ValueError('Frozen plan is incomplete or changed')
    if args.role == 'decode':
        def audit(event, values):
            if event == 'open' and values and isinstance(values[0], (str, bytes)):
                name = str(values[0]).replace('\\', '/').lower()
                if name.endswith('/source.npy') or '/timing_rgb/' in name or name.endswith('/fish_mask.npy'):
                    raise RuntimeError('Independent cache decoder tried to read source/GT media')
        sys.addaudithook(audit)
    codec, ref = new_codec(decoder_only=args.role == 'decode')
    for clip in plan['clips']:
        check_stop(args.output)
        folder = args.output / clip['id']
        done_path = folder / f'{args.role}_cache.json'
        if done_path.exists():
            done = load_json(done_path)
            if done['stream_sha256'] != file_sha(folder / 'stream.glcv'):
                raise ValueError('Previously completed cache stream changed')
            if args.role == 'decode':
                for key in ['latent', 'teacher']:
                    if done[f'{key}_array_sha256'] != file_sha(folder / f'{key}.npy'):
                        raise ValueError('Previously completed cache array changed')
            print(f'skip verified {args.role} {clip["id"]}', flush=True)
            continue
        codec.dpb = dict(ref_latent=None, ref_y=None)
        records = []
        started = time.perf_counter()
        if args.role == 'encode':
            if (folder / 'stream.glcv').exists():
                raise FileExistsError('Incomplete encode exists; preserve it and use a new pilot directory')
            if file_sha(folder / 'source.npy') != clip['source_array_sha256']:
                raise ValueError('Source cache changed')
            source = np.load(folder / 'source.npy', mmap_mode='r')
            header = {k: v for k, v in plan['codec'].items() if k != 'padded_height'}
            header.update(frames=clip['frames'], fps=float(Fraction(clip['fps'] or '30')),
                          revision=ref.REVISION, entropy=ref.ENTROPY_ID, table_sha256=codec.entropy.table_sha256,
                          arithmetic=codec.profile, weights={k: v['sha256'] for k, v in ref.ASSETS.items()}, container_version=2)
            # For index-only training and VFR static test, header FPS is nominal;
            # exact available source PTS stay in source_manifest, no bitrate/time claim.
            writer = ref.StreamWriter(folder / 'stream.glcv', header, version=2)
            try:
                for i in range(clip['frames']):
                    check_stop(args.output)
                    x = torch.from_numpy(np.array(source[i])).cuda().permute(2, 0, 1)[None].float() / 255
                    x = F.pad(x, (0, 0, 0, 32), mode='replicate').mul(2).sub(1)
                    rgb, packet = codec.frame(header, i, x=x)
                    writer.add(packet)
                    records.append({k: packet[k] for k in ('kind', 'latent_sha256', 'rgb_sha256')})
                    if (i + 1) % 16 == 0:
                        print(f'encode {clip["id"]} {i+1}/{clip["frames"]} {(time.perf_counter()-started):.1f}s', flush=True)
                writer.finish()
            finally:
                writer.close()
            report = dict(header=header, records=records, stream_sha256=file_sha(folder / 'stream.glcv'),
                          seconds=time.perf_counter() - started, encoder_process=True)
        else:
            header, packets, integrity = ref.read_stream(folder / 'stream.glcv', with_integrity=True)
            if header['arithmetic'] != codec.profile or header['table_sha256'] != codec.entropy.table_sha256:
                raise ValueError('Independent decoder arithmetic/table mismatch')
            sent = load_json(folder / 'encode_cache.json')
            if sent['header'] != header or sent['stream_sha256'] != file_sha(folder / 'stream.glcv'):
                raise ValueError('Sender/cache header mismatch')
            for name in ['latent.npy', 'teacher.npy']:
                if (folder / name).exists():
                    raise FileExistsError('Incomplete decode exists; preserve it and use a new pilot directory')
            latent = np.lib.format.open_memmap(folder / 'latent.npy', mode='w+', dtype='float32', shape=(clip['frames'], 256, 20, 32))
            teacher = np.lib.format.open_memmap(folder / 'teacher.npy', mode='w+', dtype='uint8', shape=(clip['frames'], 288, 512, 3))
            captured = {}
            def capture(_module, inputs):
                captured['latent'] = inputs[0].detach().cpu().numpy()[0].copy()
            handles = [m.vqgan.generator.register_forward_pre_hook(capture) for m in [codec.image, codec.video]]
            checksums = ref.SequenceChecksums()
            replay_matches = 0
            try:
                for i, packet in enumerate(packets):
                    check_stop(args.output)
                    rgb, checked = codec.frame(header, i, packet=packet)
                    checksums.add(i, checked)
                    if any(checked[k] != sent['records'][i][k] for k in ('kind', 'latent_sha256', 'rgb_sha256')):
                        raise ValueError(f'Sender/receiver pairing failed at frame {i}')
                    latent[i] = captured['latent']
                    teacher[i] = rgb
                    model = codec.image if checked['kind'] == 0 else codec.video
                    # Replay the *stored* FP32 cache, not the still-live DPB tensor.
                    with torch.inference_mode():
                        replay = ref.rgb_array(model.vqgan.generator(torch.from_numpy(np.array(latent[i]))[None].cuda()), header['width'], header['height'])
                    if sha(replay.tobytes()) != checked['rgb_sha256']:
                        raise ValueError(f'Cached latent does not reproduce teacher RGB at frame {i}')
                    replay_matches += 1
                    records.append(dict(index=i, kind=checked['kind'], latent_input_sha256=sha(latent[i].tobytes()),
                                        reference_sha256=checked['latent_sha256'], teacher_rgb_sha256=checked['rgb_sha256']))
                    if (i + 1) % 16 == 0:
                        print(f'decode+replay {clip["id"]} {i+1}/{clip["frames"]} {(time.perf_counter()-started):.1f}s', flush=True)
                checksums.verify(integrity['sequence_reconstruction_checksums'])
                latent.flush()
                teacher.flush()
            finally:
                for handle in handles:
                    handle.remove()
            report = dict(stream_sha256=file_sha(folder / 'stream.glcv'), latent_array_sha256=file_sha(folder / 'latent.npy'),
                          teacher_array_sha256=file_sha(folder / 'teacher.npy'), records=records,
                          verified_sequence_checksums=checksums.values(), cache_replay_exact_matches=replay_matches,
                          source_reads_forbidden=True, decoder_only=True, seconds=time.perf_counter()-started)
        save_json(done_path, report)
        print(f'completed {args.role} {clip["id"]}: {report["seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
