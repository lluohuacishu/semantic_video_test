"""Pack, independently receive, and evaluate actual persistent-reference files."""

import argparse
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

from cache_motion_inputs import REPO, base_frames
from coral_reference_sidecar import file_sha, load_json, save_json
from persistent_structure_stream import (CROP, EVENTS, KEYS, LIMIT, RAMP,
    motion_grid, pack, render, unpack)
from run_structure_memory import CLIP


PROFILES = {'g17q4': ((17, 9), 4, 1.), 'g23q2': ((23, 12), 2, 1.)}
PACKET = 'structure.glcstruct'


def prepare(workspace, long_run, output):
    from run_video1_persistent_structure import load_inputs
    from run_video1_reference_transition import compose
    from run_video1_full_coral import orange_fish
    source, raw, _, back, _, inputs = load_inputs(workspace)
    direct = load_json(long_run/'direct_result/summary.json')
    transition = load_json(long_run/'transition_result/summary.json')
    paths = ((long_run/'direct_result/N2_alpha.npy', direct['outputs_sha256']['N2_alpha.npy']),
        (long_run/'transition_result/transition_weights.npy', transition['outputs_sha256']['transition_weights.npy']),
        (long_run/'bridges/bridge_flow.npy', load_json(long_run/'bridges/flow.json')['flow_sha256']))
    if any(file_sha(path) != digest for path, digest in paths):
        raise ValueError('Offline input changed')
    alpha, weights, bridge = [np.load(path, allow_pickle=False) for path, _ in paths]
    yy, xx = np.mgrid[:288, :512].astype(np.float32)
    old = np.stack([compose(back[key+j], bridge[slot, 0], xx, yy)[0]
        for slot, key in enumerate(KEYS[1:]) for j in range(len(RAMP))])
    masks = np.concatenate((alpha > 0, weights[list(EVENTS)] > 0))
    fish = np.stack([orange_fish(frame) for frame in source])
    output.mkdir(exist_ok=False)
    base = workspace/'02_实验数据/优化_GLC后处理/source_dense_flow_three_clips_20261109_01/video1'
    for name in ('base.glcsnd', 'base_cut.json'):
        shutil.copy2(base/name, output/name)
    save_json(output/'plan.json', dict(frames=150, fps=15, billing_seconds=10,
        maximum_bytes_each_lane=LIMIT, profiles=PROFILES, base_sha256=inputs['base_sha256'],
        source_assisted_sender=True, source_images_transmitted=False,
        receiver_inputs='Independent base decoder RGB and bound auxiliary file only',
        source_uses='Sender RAFT, reliability, hand-selected coral regions, and orange-fish heuristic',
        profile_selection='Development candidates; no independent holdout and no same-total-rate codec claim',
        delay='Whole-clip file compression; not a measured streaming/real-time implementation',
        source_input_receipt=inputs, offline_inputs={str(p): h for p, h in paths}))
    records = {}
    for name, (grid, scale, epsilon) in PROFILES.items():
        folder = output/name; folder.mkdir()
        quant_new = motion_grid(back, alpha, grid, scale)
        quant_old = motion_grid(old, weights[list(EVENTS)], grid, scale)
        packet, header = pack(quant_new, quant_old, masks, fish,
            base_sha=inputs['base_sha256'], grid=grid, scale=scale, epsilon=epsilon)
        with (folder/PACKET).open('xb') as stream:
            stream.write(packet)
        allowed = len(packet) <= LIMIT
        checks = {}
        if allowed:
            decoded_header, decoded = unpack((folder/PACKET).read_bytes(), inputs['base_sha256'])
            checks['quantized_motion_exact'] = bool(np.array_equal(decoded['new_motion'], quant_new)
                and np.array_equal(decoded['old_motion'], quant_old))
            checks['fish_mask_exact'] = bool(np.array_equal(np.unpackbits(decoded['fish'], axis=2), fish))
            expected, _ = render(raw, decoded_header, decoded)
            from experiment_common import array_sha
            checks['expected_rgb_sha256'] = array_sha(expected)
            reject = {}
            broken = bytearray(packet); broken[-10] ^= 1
            for label, data, digest in (('crc_corruption', broken, inputs['base_sha256']),
                    ('wrong_base', packet, '0'*64), ('truncated', packet[:-1], inputs['base_sha256']),
                    ('over_budget', packet+b'\0'*LIMIT, inputs['base_sha256'])):
                try:
                    unpack(data, digest)
                except ValueError:
                    reject[label] = True
                else:
                    raise ValueError(f'Invalid packet was accepted: {label}')
            checks['invalid_packets_rejected'] = reject
            if not checks['quantized_motion_exact'] or not checks['fish_mask_exact']:
                raise ValueError('Wire round-trip failed')
        records[name] = dict(bytes=len(packet), kbps=len(packet)*8/10000, within_budget=allowed,
            file_sha256=file_sha(folder/PACKET), header=header, checks=checks)
        save_json(folder/'send.json', records[name])
        print(name, len(packet), 'bytes', 'PASS' if allowed else 'OVER BUDGET', flush=True)
    save_json(output/'send.json', dict(profiles=records, code_sha256=file_sha(__file__),
        codec_sha256=file_sha(Path(__file__).with_name('persistent_structure_stream.py'))))


def receive(root, profile):
    from experiment_common import array_sha
    folder = root/profile
    output = folder/'receive'; output.mkdir(exist_ok=False)
    allowed = {root/'base.glcsnd', root/'base_receive/decoded_rgb.npy', folder/PACKET}
    reads = set()
    media = {'.npy', '.npz', '.png', '.jpg', '.jpeg', '.mp4', '.mkv', '.avi', '.mov',
        '.raw', '.rgb', '.yuv', '.webm', '.glcsnd', '.glcmot', '.glcsem', '.glcflow', '.glcgate', '.glcstruct'}

    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
            return
        import os
        path = Path(os.fsdecode(args[0])).resolve()
        if path.suffix.lower() in media:
            if path not in allowed and not path.is_relative_to(output):
                raise RuntimeError('Receiver rejected untransmitted media/geometry input')
            if path in allowed:
                reads.add(str(path))

    sys.addaudithook(audit)
    raw, digest, size = base_frames(root, CLIP)
    packet = (folder/PACKET).read_bytes()
    header, values = unpack(packet, digest)
    tick = time.perf_counter()
    result, alpha = render(raw, header, values)
    elapsed = time.perf_counter()-tick
    receipt = load_json(folder/'send.json')
    if array_sha(result) != receipt['checks']['expected_rgb_sha256']:
        raise ValueError('Independent receiver differs from sender reconstruction')
    if reads != {str(path) for path in allowed}:
        raise ValueError('Unexpected input provenance')
    np.save(output/'rgb.npy', result, allow_pickle=False)
    np.save(output/'alpha.npy', alpha, allow_pickle=False)
    save_json(output/'receipt.json', dict(complete=True, source_media_read_guard=True,
        media_inputs=sorted(reads), sender_receiver_bit_exact=True, base_reference_verified=True,
        base_sha256=digest, base_bytes=size, base_kbps=size*8/10000,
        auxiliary_sha256=file_sha(folder/PACKET), auxiliary_bytes=len(packet), auxiliary_kbps=len(packet)*8/10000,
        total_bytes=size+len(packet), total_kbps=(size+len(packet))*8/10000,
        render_cpu_seconds=elapsed, render_ms_per_frame=elapsed*1000/150,
        output_sha256={p.name:file_sha(p) for p in output.glob('*.npy')},
        code_sha256=file_sha(__file__), codec_sha256=file_sha(Path(__file__).with_name('persistent_structure_stream.py'))))
    print(profile, 'receiver PASS', elapsed, 'seconds', flush=True)


def evaluate(workspace, long_run, root, ffmpeg):
    from compare_deflicker import save_csv
    from run_p0 import preview, roi
    from run_video1_persistent_long import grouped
    from run_video1_persistent_structure import load_inputs, score, VIEW_BOX
    source, raw, s, _, _, _ = load_inputs(workspace)
    if not np.array_equal(raw, base_frames(root, CLIP)[0]):
        raise ValueError('Fresh independent base decode differs')
    output = root/'evaluation'; output.mkdir(exist_ok=False)
    sequences = dict(source=source, raw=raw, S=s,
        T=np.load(long_run/'transition_result/T_rgb.npy', allow_pickle=False))
    records = load_json(root/'send.json')['profiles']
    receipts = {}
    for name, record in records.items():
        if record['within_budget']:
            receipt = load_json(root/name/'receive/receipt.json')
            if file_sha(root/name/'receive/rgb.npy') != receipt['output_sha256']['rgb.npy']:
                raise ValueError('Receiver output changed')
            sequences[name] = np.load(root/name/'receive/rgb.npy', allow_pickle=False)
            receipts[name] = receipt
    tracked = np.load(long_run/'direct_result/tracked.npy', allow_pickle=False)
    scores, rows, spatial = score(sequences, tracked, start=0)
    save_csv(output/'temporal.csv', rows); save_csv(output/'spatial.csv', spatial)
    groups = grouped(rows)
    best = min(receipts, key=lambda key: groups['fixed']['all'][key]['coarse_mae'])
    previews = {}
    for name, keys, crop in (
        ('auxiliary_coral_pair_10s.mp4', ('raw', best), roi(VIEW_BOX, raw.shape[1:])),
        ('auxiliary_full_pair_10s.mp4', ('raw', best), None),
        ('offline_vs_auxiliary_10s.mp4', ('T', best), roi(VIEW_BOX, raw.shape[1:]))):
        previews[name] = preview(ffmpeg, output/name, [sequences[key] for key in keys], 15, 0,
            [dict(raw='GLC raw', T='T / offline dense motion').get(key, 'U / actual auxiliary '+key) for key in keys], crop)
    save_json(output/'summary.json', dict(complete=True, same_base_decode_bit_exact=True,
        actual_auxiliary_files=True, frames=150, fps=15, duration_seconds=10,
        selected_development_profile=best, scores=scores, groups=groups, receipts=receipts, previews=previews,
        limits=['One development clip; manual sender ROI and source-assisted motion/masks.',
            'Whole-file compression; streaming latency not validated.',
            'No same-total-bitrate GLC comparison; no full-length/holdout acceptance.'],
        code_sha256=file_sha(__file__)))
    print('BEST', best, flush=True)
    for name in ('all', 'key_refresh', 'I', 'ordinary_P_from6s'):
        print(name, groups['fixed'][name], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'receive', 'evaluate'))
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--long-run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=PROFILES)
    parser.add_argument('--ffmpeg', type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    if args.action in ('prepare', 'evaluate') and args.long_run is None:
        parser.error('--long-run required')
    if args.action == 'prepare':
        prepare(args.workspace.resolve(), args.long_run.resolve(), args.output.resolve())
    elif args.action == 'receive':
        if args.profile is None: parser.error('--profile required')
        receive(args.output.resolve(), args.profile)
    else:
        if args.ffmpeg is None: parser.error('--ffmpeg required')
        evaluate(args.workspace.resolve(), args.long_run.resolve(), args.output.resolve(), args.ffmpeg.resolve())


if __name__ == '__main__':
    main()
