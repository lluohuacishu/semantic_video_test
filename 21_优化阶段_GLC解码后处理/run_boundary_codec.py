"""Encode fixed q0 boundary cases and decode each in an independent process."""

import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from run_p0 import HERE, REPO, file_sha, load_json, load_range, open_baseline, save_json
from experiment_common import array_sha, input_tensor
from sender_container import SenderWriter, read_sender_stream

CASES = {'g64_r32': (64, 32), 'g72_r32': (72, 32), 'g64_r24': (64, 24)}
RECEIVER = REPO / '17_优化阶段_解码器推理加速/decoder_worker.py'


def reef_inputs(workspace):
    plan = load_json(HERE / 'plan_p0.json')
    clip = next(row for row in plan['clips'] if row['id'] == '01_coral_wall')
    manifest = workspace / '01_项目/GLC后处理/接口与基线.json'
    baseline = next(row for row in load_json(manifest)['baselines'] if row['id'] == clip['baseline'])
    return plan['codec'], clip, baseline


def encode(workspace, assets, output):
    import torch
    from role_codec import RoleCodec
    codec_plan, clip, baseline = reef_inputs(workspace)
    _, source, original = open_baseline(workspace, baseline, codec_plan)
    count = clip['stop']
    frames = load_range(source, 0, count, baseline['shape'][1:])
    original_sender = (workspace / baseline['metadata']).parent.parent / 'glc_sender'
    original_stream = original_sender / 'stream.glcsnd'
    if file_sha(original_stream) != original['stream_sha256']:
        raise ValueError('Original GLC stream differs')
    header, old_packets, integrity = read_sender_stream(original_stream)
    original_result = load_json(original_sender / 'result.json')
    if (original_result['stream_sha256'] != original['stream_sha256']
            or original_result['reference_checksums'] != integrity['expected_reference_checksums']):
        raise ValueError('Original sender receipt differs from the verified stream')
    original_references = {row['index']: row['latent_sha256'] for row in original_result['records']}
    if (header['frames'] != baseline['frames'] or header['fps'] != baseline['fps']
            or any(header[key] != value for key, value in codec_plan.items())):
        raise ValueError('Original header differs from the fixed baseline')
    print(f'Loaded exact source RGB 0–{count - 1}; loading FP32 sender', flush=True)
    codec = RoleCodec(assets, 'sender')
    torch.utils.deterministic.fill_uninitialized_memory = False
    if (codec.profile != header['arithmetic'] or codec.entropy.table_sha256 != header['table_sha256']
            or not torch.are_deterministic_algorithms_enabled()):
        raise ValueError('Sender arithmetic differs from the original')
    with torch.inference_mode():
        for index in range(8):
            codec.frame(header, index, x=input_tensor(frames[index]), render_rgb=False)
        torch.cuda.synchronize()
        for identity, (gop, reset) in CASES.items():
            folder = output / identity
            folder.mkdir()
            stream = folder / 'stream.glcsnd'
            settings = dict(header, frames=count, intra_period=gop, reset_interval=reset)
            codec.dpb = dict(ref_latent=None, ref_y=None)
            writer = SenderWriter(stream, settings)
            records = []
            print(f'{identity}: encode {count} frames, q0/FP32', flush=True)
            try:
                for index, rgb in enumerate(frames):
                    tensor = input_tensor(rgb)
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    _, packet = codec.frame(settings, index, x=tensor, render_rgb=False)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - tick
                    if identity == 'g64_r32':
                        old = old_packets[index]
                        if old['z'] != packet['z'] or old['y'] != packet['y']:
                            raise ValueError(f'Original compressed packet differs at frame {index}')
                        if index in original_references and packet['latent_sha256'] != original_references[index]:
                            raise ValueError(f'Original recorded reference differs at frame {index}')
                    size = writer.add(packet)
                    records.append(dict(index=index, kind=packet['kind'], packet_bytes=size,
                                        seconds=elapsed, latent_sha256=packet['latent_sha256']))
                    if (index + 1) % 64 == 0 or index + 1 == count:
                        print(f'{identity}: sent {index+1}/{count}', flush=True)
                writer.finish()
            finally:
                writer.close()
            save_json(folder / 'sender.json', dict(header=settings, stream_sha256=file_sha(stream),
                                                   bytes=stream.stat().st_size, records=records,
                                                   reference_checksums=writer.references.values(),
                                                   source_rgb_sha256=array_sha(frames),
                                                   old_packets_verified=identity == 'g64_r32',
                                                   old_frame_references_verified=sum(
                                                       index < count for index in original_references
                                                   ) if identity == 'g64_r32' else 0))
    save_json(output / 'inputs.json', dict(source_rgb_sha256=array_sha(frames), original=original,
              sender_code_sha256=file_sha(__file__), role_assets_manifest_sha256=codec.loading['manifest_sha256'],
              frames=count, clip=clip, backend='eager', torch=str(torch.__version__), gpu=torch.cuda.get_device_name()))


def run(workspace, assets, output):
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-B', '-u', '-X', 'utf8']
    subprocess.run([*command, str(Path(__file__).resolve()), '--workspace', str(workspace),
                    '--assets', str(assets), '--output', str(output), '--encode-worker'], check=True)
    codec_plan, clip, baseline = reef_inputs(workspace)
    original_rgb, _, _ = open_baseline(workspace, baseline, codec_plan)
    cases = []
    for identity in CASES:
        folder = output / identity
        print(f'{identity}: independent FP32 receiver', flush=True)
        subprocess.run([*command, str(RECEIVER), '--stream', str(folder / 'stream.glcsnd'),
                        '--output', str(folder / 'receiver'), '--backend', 'eager',
                        '--assets', str(assets)], check=True)
        sender = load_json(folder / 'sender.json')
        receiver = load_json(folder / 'receiver/result.json')
        if (sender['stream_sha256'] != receiver['stream_sha256']
                or sender['header'] != receiver['header']
                or sender['reference_checksums'] != receiver['reference_checksums']
                or not receiver['reference_verified'] or not receiver['source_media_reads_forbidden']):
            raise ValueError(f'Independent receive verification failed: {identity}')
        rgb_path = folder / 'receiver/decoded_rgb.npy'
        frames = np.load(rgb_path, mmap_mode='r', allow_pickle=False)
        if (frames.dtype != np.uint8 or frames.shape != (clip['stop'], *baseline['shape'][1:])
                or file_sha(rgb_path) != receiver['decoded_rgb_sha256']):
            raise ValueError(f'Unexpected lossless receive RGB: {identity}')
        if identity == 'g64_r32':
            for index, frame in enumerate(frames):
                if not np.array_equal(frame, original_rgb[index]):
                    raise ValueError(f'Original decoded RGB differs at frame {index}')
            print(f"g64_r32: {clip['stop']} original packets and decoded RGB match bit-exactly", flush=True)
        header = sender['header']
        seconds = header['frames'] / header['fps']
        cases.append(dict(id=identity, intra_period=header['intra_period'], reset_interval=header['reset_interval'],
                          q=header['q'], fps=header['fps'], frames=header['frames'], bytes=sender['bytes'],
                          bitrate_kbps=sender['bytes'] * 8 / seconds / 1000,
                          stream_sha256=sender['stream_sha256'], decoded_rgb_sha256=receiver['decoded_rgb_sha256'],
                          reference_checksums=sender['reference_checksums'], source_rgb_sha256=sender['source_rgb_sha256'],
                          original_packet_and_rgb_match=identity == 'g64_r32',
                          old_frame_references_verified=sender['old_frame_references_verified'],
                          encode_core_seconds=sum(row['seconds'] for row in sender['records']),
                          decode_core_seconds=receiver['core_seconds']))
    if len({item['source_rgb_sha256'] for item in cases}) != 1:
        raise ValueError('Boundary cases have different source frames')
    save_json(output / 'codec_summary.json', dict(complete=True, cases=cases))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--encode-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    assets = args.assets or args.workspace / '02_实验数据/优化_部署精简/role_assets_20261029_01'
    action = encode if args.encode_worker else run
    action(args.workspace.resolve(), assets.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
