"""Minimal experimental file sender/receiver using verified role-specific assets.

Sender: one video -> GLCSND01 only. Receiver: GLCSND01 -> one watchable video.
No source previews, PNG sequences, source PSNR, or per-frame diagnostic JSON.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import time

from role_codec import RoleCodec, export_assets, ref
from sender_common import file_sha, load_settings, save_json
from sender_container import PROFILE, ReferenceChecksums, SenderWriter, read_sender_stream
from glc_container import MAX_BYTES
from c1_io import MemoryFrames, RawPreview
from c1_runtime import receiver_audit


def encode(args, codec):
    source_args = argparse.Namespace(input=args.input, width=args.width, frames=args.frames,
                                     start_frame=0, frame_stride=args.stride)
    reader = MemoryFrames(source_args, collect_hashes=False)
    packets, buffered = [], 0
    try:
        first = next(reader, None)
        if first is None:
            raise ValueError('No video frames were read')
        metadata = reader.metadata()
        header = {key: metadata[key] for key in ('width', 'height', 'fps')}
        header.update(frames=args.frames, q=args.q, intra_period=64, reset_interval=32,
                      revision=ref.REVISION, entropy=ref.ENTROPY_ID,
                      table_sha256=codec.entropy.table_sha256, arithmetic=codec.profile,
                      weights={name: value['sha256'] for name, value in ref.ASSETS.items()},
                      container_version=3, integrity_profile=PROFILE, sender_rgb_reconstructed=False)
        from sender_container import validate_header
        validate_header(header)
        for index, rgb in enumerate(itertools.chain((first,), reader)):
            tensor = ref.torch.from_numpy(rgb.copy()).to(codec.device).permute(2, 0, 1)[None].float().div(255)
            tensor = ref.F.pad(tensor, (0, (-header['width']) % 64, 0, (-header['height']) % 64), mode='replicate').mul(2).sub(1)
            _, packet = codec.frame(header, index, x=tensor, render_rgb=False)
            buffered += len(packet['z']) + sum(map(len, packet['y'])) + 128
            if buffered > MAX_BYTES:
                raise ValueError('Compressed packet buffer exceeds the file limit')
            packets.append(packet)
        metadata = reader.metadata()
    finally:
        reader.close()
    header['frames'] = len(packets)
    path = args.output / 'stream.glcsnd'
    writer = SenderWriter(path, header)
    try:
        for packet in packets:
            writer.add(packet)
        writer.finish()
    finally:
        writer.close()
    metadata.pop('rgb_sha256', None)
    return dict(frames=len(packets), source=metadata, source_frame_hashes_collected=False,
                stream_sha256=file_sha(path), stream_bytes=path.stat().st_size,
                sender_rgb_reconstructed=False, header=header)


def decode(args, codec):
    with receiver_audit() as reads:
        header, packets, integrity = read_sender_stream(args.input)
        if (header['revision'] != ref.REVISION or header['entropy'] != ref.ENTROPY_ID or
                header['weights'] != {name: value['sha256'] for name, value in ref.ASSETS.items()} or
                header['arithmetic'] != codec.profile or header['table_sha256'] != codec.entropy.table_sha256):
            raise ValueError('Receiver model/table/arithmetic differs; cross-device testing must be explicit')
        video = RawPreview(load_settings()['runtime']['ffmpeg'], args.output / 'video.mp4',
                           header['width'], header['height'], header['fps'])
        references = ReferenceChecksums()
        try:
            for index, packet in enumerate(packets):
                rgb, checked = codec.frame(header, index, packet=packet)
                references.add(index, checked)
                video.add(rgb)
            references.verify(integrity['expected_reference_checksums'])
            # RawPreview publishes only here; a reference failure leaves a partial.
            video.finish()
        except BaseException:
            video.abort()
            raise
    return dict(frames=len(packets), header=header, reference_verified=True,
                sender_rgb_checksum_available=False, sender_rgb_verified=None,
                stream_sha256=file_sha(args.input), source_media_reads=reads,
                video_sha256=file_sha(args.output / 'video.mp4'),
                video_bytes=(args.output / 'video.mp4').stat().st_size,
                video_encoding='H.264 CRF16 viewing output; exact tests use pre-transcode RGB')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    export = commands.add_parser('export')
    export.add_argument('--output', type=Path, required=True)
    for action in ['encode', 'decode']:
        command = commands.add_parser(action)
        command.add_argument('--assets', type=Path, required=True)
        command.add_argument('--input', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
        if action == 'encode':
            command.add_argument('--width', type=int, default=640)
            command.add_argument('--stride', type=int, default=2)
            command.add_argument('--frames', type=int, default=10000)
            command.add_argument('--q', type=int, choices=range(64), default=0)
    args = parser.parse_args()
    if args.action == 'export':
        export_assets(args.output)
        print('Exported exact role-specific FP32 assets')
        return
    if not args.input.is_file():
        raise ValueError('Input must be a saved video or sender file')
    if args.action == 'encode' and (not 1 <= args.frames <= 10000 or not 1 <= args.stride <= 10000 or not 1 <= args.width <= 4096):
        raise ValueError('Invalid frame count, stride, or width')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    codec = RoleCodec(args.assets, 'sender' if args.action == 'encode' else 'receiver', args.device)
    loaded = time.perf_counter()
    try:
        result = encode(args, codec) if args.action == 'encode' else decode(args, codec)
    except BaseException as exc:
        save_json(args.output / 'failure.json', dict(type=type(exc).__name__, message=str(exc)))
        raise
    result.update(action=args.action, model_load_seconds=loaded-started,
                  file_operation_seconds=time.perf_counter()-loaded,
                  elapsed_before_report_seconds=time.perf_counter()-started,
                  deployment_assets=codec.loading, training_steps=0,
                  research_pngs=False, sender_previews=False, per_frame_metric_logs=False)
    save_json(args.output / 'result.json', result)
    print(f'{args.action}: {result["frames"]} frames, operation {result["file_operation_seconds"]:.3f}s; verified output saved')


if __name__ == '__main__':
    main()
