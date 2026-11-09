"""Compare causal GLC texture caches with receiver-side and transmitted motion."""

import argparse
from collections import deque
import os
from pathlib import Path
from struct import Struct
import sys
import time
import zlib

import cv2
import numpy as np

from audit_deflicker_texture import fine_energy, matched_blur
from cache_motion_inputs import (
    CLIPS, FPS, FRAME_SHAPE, HEIGHT, LANE_LIMIT, REPO, WIDTH, base_frames, source_frames,
)
from coral_reference_sidecar import SCALES, detail, file_sha, gray, save_json
from experiment_common import array_sha
from run_p0 import roi


COARSE_SIGMA = 2.4
CACHE_FRAMES = 15
C_FAIR_BUDGET = 2000
PACKET = 'shared.glcmot'
MAGIC = b'GLCMOT01'
HEADER = Struct('<8s32s6H')
CRC = Struct('<I')
RECORD = np.dtype([('valid', 'u1'), ('affine', '<i2', (2, 3))])
CORNERS = np.array([[0, 0], [WIDTH-1, 0], [0, HEIGHT-1], [WIDTH-1, HEIGHT-1]])
ERODE = np.ones((5, 5), np.uint8)


def plausible(affine):
    linear, offset = affine[:, :2], affine[:, 2]
    stretches = np.linalg.svd(linear, compute_uv=False)
    shift = CORNERS @ (linear - np.eye(2)).T + offset
    return (np.linalg.det(linear) > 0 and stretches.min() >= .88 and stretches.max() <= 1.14
            and np.linalg.norm(shift, axis=1).max() <= 45)


def estimate_pair(previous, current):
    features = []
    for row in range(3):
        for col in range(4):
            x, y = col * (WIDTH // 4), row * (HEIGHT // 3)
            block = previous[y:y+HEIGHT//3, x:x+WIDTH//4]
            points = cv2.goodFeaturesToTrack(block, 24, .005, 7, blockSize=7)
            if points is not None:
                features.append(points + np.array([x, y], np.float32))
    if not features:
        return None
    start = np.concatenate(features)
    lk = dict(winSize=(21, 21), maxLevel=3)
    end, forward, error = cv2.calcOpticalFlowPyrLK(previous, current, start, None, **lk)
    if end is None:
        return None
    back, backward, _ = cv2.calcOpticalFlowPyrLK(current, previous, end, None, **lk)
    if back is None:
        return None
    good = ((forward.ravel() == 1) & (backward.ravel() == 1) & (error.ravel() < 25)
            & (np.linalg.norm(start[:, 0] - back[:, 0], axis=1) < 1.2))
    src, dst = start[good, 0], end[good, 0]
    if len(src) < 16:
        return None
    matrix, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=2, maxIters=1500, confidence=.99, refineIters=10)
    if matrix is None:
        return None
    stable = src[inliers.ravel().astype(bool)]
    if (len(stable) < max(16, .4 * len(src)) or np.ptp(stable[:, 0]) < WIDTH / 3
            or np.ptp(stable[:, 1]) < HEIGHT / 4 or not plausible(matrix)):
        return None
    return matrix


def homogeneous(affine):
    return np.vstack((affine, [0, 0, 1]))


def quantize(matrix):
    if matrix is None:
        return None
    quantized = np.rint(matrix * SCALES).astype('<i2')
    return quantized if plausible(quantized / SCALES) else None


def historical_reference(history, steps):
    recent = history[-1]
    if len(history) < 3 or any(step is None for step in steps):
        return recent
    aligned = [recent]
    transform = np.eye(3)
    support = np.ones_like(recent)
    for older, step in zip((history[1], history[0]), reversed(steps), strict=True):
        transform = transform @ step
        affine = transform[:2]
        visible = cv2.warpAffine(support, affine, (WIDTH, HEIGHT))
        projected = cv2.warpAffine(older, affine, (WIDTH, HEIGHT), borderMode=cv2.BORDER_REPLICATE)
        aligned.append(np.where(visible, projected, recent))
    return np.median(np.stack(aligned), axis=0).astype(np.uint8)


def estimate(frames, use_history):
    records = np.zeros(len(frames)-1, RECORD)
    history = deque(maxlen=3)
    steps = deque(maxlen=2)
    for i, frame in enumerate(frames):
        current = cv2.GaussianBlur(gray(frame), (0, 0), COARSE_SIGMA)
        if i:
            reference = historical_reference(history, steps) if use_history else history[-1]
            quantized = quantize(estimate_pair(reference, current))
            if quantized is None:
                steps.append(None)
            else:
                records[i-1] = (1, quantized)
                steps.append(homogeneous(quantized / SCALES))
        history.append(current)
    return records


def packet_header(clip, base_sha):
    return HEADER.pack(MAGIC, bytes.fromhex(base_sha), WIDTH, HEIGHT, FPS,
                       clip.start, clip.start-clip.base_start, clip.count)


def pack(clip, base_sha, records):
    data = packet_header(clip, base_sha) + records.tobytes()
    return data + CRC.pack(zlib.crc32(data))


def unpack(data, clip, base_sha):
    expected = HEADER.size + (clip.count-1)*RECORD.itemsize + CRC.size
    if (len(data) != expected or len(data) > C_FAIR_BUDGET or len(data)*8*FPS > LANE_LIMIT*clip.count
            or data[:HEADER.size] != packet_header(clip, base_sha)
            or CRC.unpack(data[-CRC.size:])[0] != zlib.crc32(data[:-CRC.size])):
        raise ValueError('Invalid motion file, budget, timeline or base stream digest')
    records = np.frombuffer(data, dtype=RECORD, offset=HEADER.size, count=clip.count-1)
    for record in records:
        valid, affine = record
        if ((valid == 1 and not plausible(affine / SCALES))
                or (valid != 1 and (valid != 0 or affine.any()))):
            raise ValueError('Invalid motion record')
    return records


def render(frames, records):
    output = np.empty_like(frames)
    masks = np.zeros(frames.shape[:3], np.uint8)
    support = np.ones(frames.shape[1:3], np.float32)
    for i, rgb in enumerate(frames):
        coarse = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), 3)
        texture = detail(gray(rgb), 3)
        if i == 0 or i % CACHE_FRAMES == 0 or not records[i-1]['valid']:
            anchor = i
            transform = np.eye(3)
            ref_low, ref_texture = coarse, texture
            output[i] = rgb
            continue
        transform = homogeneous(records[i-1]['affine'] / SCALES) @ transform
        affine = transform[:2]
        projected_low = cv2.warpAffine(ref_low, affine, (WIDTH, HEIGHT))
        projected_texture = cv2.warpAffine(ref_texture, affine, (WIDTH, HEIGHT), flags=cv2.INTER_CUBIC)
        visible = cv2.warpAffine(support, affine, (WIDTH, HEIGHT)) >= .999
        difference = np.max(np.abs(projected_low - coarse), axis=2)
        difference = cv2.GaussianBlur(difference, (0, 0), 1.7)
        reliable = cv2.erode((visible & (difference < 16)).astype(np.uint8), ERODE)
        feather = cv2.GaussianBlur(reliable.astype(np.float32), (0, 0), 1.4)
        tone = np.clip((16 - difference) / 9, 0, 1)
        texture_energy = cv2.GaussianBlur(np.abs(projected_texture), (0, 0), 1.5)
        texture_confidence = np.clip((texture_energy - 1) / 3, 0, 1)
        fade = min((i-anchor) / 3, (CACHE_FRAMES - i % CACHE_FRAMES) / 3, 1)
        alpha = .82 * fade * feather * reliable * tone * texture_confidence
        masks[i] = np.rint(alpha * 255).astype(np.uint8)
        change = np.clip(projected_texture - texture, -46, 46) * (masks[i].astype(np.float32) / 255)
        output[i] = np.rint(np.clip(rgb + change[..., None], 0, 255)).astype(np.uint8)
    return output, masks


def send(workspace, folder, clip):
    _, base_sha, _ = base_frames(folder, clip)
    source = source_frames(workspace, clip)
    records = estimate(source, use_history=True)
    payload = pack(clip, base_sha, records)
    if not np.array_equal(unpack(payload, clip, base_sha), records):
        raise ValueError('Motion quantization did not round-trip')
    with (folder/PACKET).open('xb') as stream:
        stream.write(payload)
    size = (folder/PACKET).stat().st_size
    save_json(folder/'motion_send.json', dict(bytes=size, kbps=size*8*FPS/clip.count/1000,
        base_sha256=base_sha, sha256=file_sha(folder/PACKET), source_rgb_sha256=array_sha(source),
        accepted_steps=int(records['valid'].sum()), code_sha256=file_sha(__file__)))
    print(f'{folder.name} C packet: {size}/{C_FAIR_BUDGET} B', flush=True)


def guard_receiver(allowed, output):
    # No source-media or geometry-file input in the autonomous process.
    media = {'.npy', '.npz', '.png', '.jpg', '.jpeg', '.mp4', '.mkv', '.avi', '.mov', '.raw',
             '.rgb', '.yuv', '.webm', '.glcsnd', '.glcmot', '.glcsem'}

    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.fsdecode(args[0]))
        if path.suffix.lower() in media:
            resolved = path.resolve()
            if resolved not in allowed and not resolved.is_relative_to(output):
                raise RuntimeError(f'Unexpected receiver media input: {path.name}')

    sys.addaudithook(audit)


def receive(folder, clip, assisted):
    output = folder/('assisted' if assisted else 'autonomous')
    output.mkdir(exist_ok=False)
    allowed = {folder/'base.glcsnd', folder/'base_receive/decoded_rgb.npy'}
    if assisted:
        allowed.add(folder/PACKET)
    guard_receiver(allowed, output)
    frames, base_sha, _ = base_frames(folder, clip)
    started = time.perf_counter()
    motion_sha = None
    if assisted:
        records = unpack((folder/PACKET).read_bytes(), clip, base_sha)
        variants = {'c': records}
        motion_sha = file_sha(folder/PACKET)
    else:
        variants = {'b1': estimate(frames, use_history=False), 'b2': estimate(frames, use_history=True)}

    measurements = {}
    crop = roi(clip.coral, FRAME_SHAPE)
    for name, records in variants.items():
        restored, alpha = render(frames, records)
        if np.any(restored[alpha == 0] != frames[alpha == 0]):
            raise ValueError('Texture change outside trusted regions')
        np.save(output/f'{name}.npy', restored, allow_pickle=False)
        np.save(output/f'{name}_alpha.npy', alpha, allow_pickle=False)
        data = dict(rgb_sha256=array_sha(restored), alpha_sha256=array_sha(alpha),
            accepted_steps=int(records['valid'].sum()), step_valid=records['valid'].tolist(),
            coverage_by_frame=(alpha > 0).mean(axis=(1, 2)).tolist(),
            mean_alpha_by_frame=(alpha.mean(axis=(1, 2)) / 255).tolist())
        if not assisted:
            sigma, control = matched_blur(frames, restored, crop)
            np.save(output/f'b0_{name}.npy', control, allow_pickle=False)
            target = fine_energy(restored, crop)
            actual = fine_energy(control, crop)
            data['b0'] = dict(sigma=sigma, fine_energy=actual, target_energy=target,
                relative_error=abs(actual-target)/target, rgb_sha256=array_sha(control))
        measurements[name] = data
        print(f'{folder.name} {name}: {data["accepted_steps"]}/{len(records)} accepted motion steps',
              flush=True)
    save_json(output/'result.json', dict(variants=measurements, base_sha256=base_sha,
        motion_sha256=motion_sha, source_media_reads_forbidden=True, frames=clip.count,
        seconds=time.perf_counter()-started, code_sha256=file_sha(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('send', 'receive-b', 'receive-c'))
    parser.add_argument('--clip', choices=CLIPS, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    cv2.setRNGSeed(20261109)
    folder = args.run.resolve()/args.clip
    clip = CLIPS[args.clip]
    if args.action == 'send':
        send(args.workspace.resolve(), folder, clip)
    else:
        receive(folder, clip, assisted=args.action == 'receive-c')


if __name__ == '__main__':
    main()
