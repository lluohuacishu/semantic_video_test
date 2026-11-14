"""Shared fixed-video1 utilities for the November 11 timing experiments."""
from pathlib import Path
import os
import sys
import time

import cv2
import numpy as np

from receiver_structure import KEYS, RAMP, file_sha, load_json, save_json
from receive_guide_texture import texture_gate
from cached_t3 import cached_texture_gate

STAGE = Path(__file__).resolve().parent
REPO = STAGE.parent
WORKSPACE = REPO.parents[2]
DATA = WORKSPACE / '02_实验数据/优化_GLC后处理'
BASE = DATA / 'source_dense_flow_three_clips_20261109_01/video1'
ACCEPTED = DATA / 'anchor_time_20261111_02/CURRENT_I12'
PACKET = DATA / 'guide_compression_20261111_01/video1/S8I/guide.glcguide'
UPSTREAM = DATA / 'deflicker_reef80_20261108_01/upstream'
WEIGHT = UPSTREAM / 'pretrained_weights/raft-things.pth'
FFMPEG = WORKSPACE / '01_项目/运行环境/FFmpeg/ffmpeg.exe'
ASSETS = WORKSPACE / '02_实验数据/优化_部署精简/role_assets_20261029_01'
SOURCE_VIDEO = WORKSPACE / '01_项目/GLC后处理/素材/20261109_三个片段/video1.mp4'
EVENTS = tuple(key + offset for key in KEYS[1:] for offset in range(len(RAMP)))


def fixed_inputs():
    cfg = load_json(STAGE / '当前采用配置.json')
    paths = [BASE/'base_receive/decoded_rgb.npy', BASE/'base_receive/result.json', WEIGHT]
    result = {str(p): file_sha(p) for p in paths}
    for value in cfg['artifacts'].values():
        p = Path(value['path'])
        actual = file_sha(p)
        if p.stat().st_size != value['bytes'] or actual != value['sha256']:
            raise ValueError('Accepted artifact changed: ' + str(p))
        result[str(p)] = actual
    return result


def check_inputs(manifest):
    for path, digest in manifest.items():
        if file_sha(path) != digest:
            raise ValueError('Frozen input changed: ' + path)


def render(raw, images, back, forward, transition, cached=True, on_first=None):
    """Same two-stage rounding and raw-reference formula as the accepted renderer."""
    result = raw.copy()
    alphas = np.zeros(raw.shape[:3], np.float32)
    shared = np.zeros((len(EVENTS), *raw.shape[1:3]), np.float32)
    event_lookup = {frame: index for index, frame in enumerate(EVENTS)}
    gate = cached_texture_gate(images) if cached else texture_gate(images)
    for frame in range(len(raw)):
        anchor_index = int(np.searchsorted(KEYS, frame, side='right')-1)
        key = KEYS[anchor_index]
        alpha, mx, my = gate(images, frame, key, back[frame], forward[frame])
        new = cv2.remap(raw[key].astype(np.float32), mx, my, cv2.INTER_CUBIC)
        current = np.rint(np.clip(raw[frame]*(1-alpha[..., None])+new*alpha[..., None], 0, 255))
        if frame in event_lookup:
            event_index = event_lookup[frame]
            old_key = KEYS[event_index//len(RAMP)]
            old_alpha, ox, oy = gate(images, frame, old_key, *transition[event_index])
            shared[event_index] = np.minimum(alpha, old_alpha)
            old = cv2.remap(raw[old_key].astype(np.float32), ox, oy, cv2.INTER_CUBIC)
            weight = shared[event_index]*RAMP[event_index % len(RAMP)]
            current = np.rint(np.clip(current+weight[..., None]*(old-new), 0, 255))
        result[frame] = current.astype(np.uint8)
        alphas[frame] = alpha
        if frame == 0 and on_first is not None:
            on_first()
    if not np.array_equal(result[alphas == 0], raw[alphas == 0]):
        raise ValueError('Zero-weight pixels changed')
    return dict(rgb=result, alpha=alphas, shared=shared)


def receiver_guard(out, allowed):
    """Reproducibility read boundary; deliberately not an OS security sandbox."""
    permitted = {Path(p).resolve() for p in allowed}
    observed = set()
    out = Path(out).resolve()
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path in permitted:
            observed.add(str(path)); return
        if path.is_relative_to(out):
            return
        runtime = path.is_relative_to(UPSTREAM) and path.suffix.lower() in {'.py','.pyc','.pyd','.dll'}
        if not runtime and (path.is_relative_to(DATA) or path.suffix.lower() in
            {'.npy','.npz','.glcsnd','.glcguide','.glccsp2','.glccsp3','.mp4','.hevc'}):
            raise RuntimeError('Untransmitted receiver input: ' + str(path))
    sys.addaudithook(audit)
    blocked = []
    for path in (BASE/'source.npy', BASE/'base_cut.json', BASE/'flow/to_anchor.npy',
                 BASE/'flow/from_anchor.npy', PACKET.parent/'send.json',
                 PACKET.parent/'sender_expected_guide.npy', SOURCE_VIDEO):
        try:
            path.read_bytes()
        except RuntimeError:
            blocked.append(str(path))
        else:
            raise ValueError('Input guard failed: '+str(path))
    return observed, blocked


def gpu_snapshot():
    import subprocess
    p = subprocess.run(['nvidia-smi','--query-gpu=timestamp,name,utilization.gpu,memory.used,temperature.gpu,power.draw',
                        '--format=csv,noheader'], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else p.stderr.strip()


def stats(values):
    return dict(median=float(np.median(values)), minimum=float(min(values)), maximum=float(max(values)), samples=list(values))
