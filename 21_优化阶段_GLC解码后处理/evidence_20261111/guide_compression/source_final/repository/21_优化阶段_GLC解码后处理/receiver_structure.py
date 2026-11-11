"""Source-free persistent RGB reuse from independently decoded GLC frames.

All flow/mask arrays saved here are receiver-local diagnostics, never a sidecar.
The deployed inputs are decoded RGB and a preinstalled, shared RAFT checkpoint.
The Python audit is a reproducibility guard, not an operating-system sandbox.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np


KEYS = (0, 15, 45, 75, 105, 135)
RAMP = (1., 5/6, 4/6, 3/6, 2/6, 1/6)
EVENTS = tuple(k+j for k in KEYS[1:] for j in range(6))
WEIGHT_SHA = 'fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1'
ITERATIONS = 24


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def reliable(images, current, reference, back, forward):
    """Same fixed correspondence tests as the existing dense source control."""
    h, w = back.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    mx, my = xx+back[..., 0], yy+back[..., 1]
    inside = (mx >= 2) & (mx < w-2) & (my >= 2) & (my < h-2)
    inverse = cv2.remap(forward, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    cycle = np.linalg.norm(back+inverse, axis=2)
    low = cv2.GaussianBlur(images[current].astype(np.float32), (0,0), 3)
    old_low = cv2.GaussianBlur(images[reference].astype(np.float32), (0,0), 3)
    mid = cv2.GaussianBlur(images[current].astype(np.float32), (0,0), 1)
    old_mid = cv2.GaussianBlur(images[reference].astype(np.float32), (0,0), 1)
    dlow = np.max(np.abs(low-cv2.remap(old_low, mx, my, cv2.INTER_LINEAR)), axis=2)
    dmid = np.max(np.abs(mid-cv2.remap(old_mid, mx, my, cv2.INTER_LINEAR)), axis=2)
    grad = np.maximum(np.linalg.norm(np.gradient(back, axis=0), axis=2),
                      np.linalg.norm(np.gradient(back, axis=1), axis=2))
    safe = inside & (cycle < 1.5) & (dlow < 16) & (dmid < 24) & (grad < .75)
    safe = cv2.erode(safe.astype(np.uint8), np.ones((3,3), np.uint8))
    alpha = np.minimum(cv2.distanceTransform(safe, cv2.DIST_L2, 5)/2., 1.)
    return alpha, mx, my


def render(raw, gate_images, back, forward, transition, *, gate_fn=reliable):
    result = raw.copy()
    alphas = np.zeros(raw.shape[:3], np.float32)
    shared = np.zeros((len(EVENTS), *raw.shape[1:3]), np.float32)
    events = {i:j for j,i in enumerate(EVENTS)}
    for i in range(len(raw)):
        key = KEYS[np.searchsorted(KEYS, i, side='right')-1]
        a, mx, my = gate_fn(gate_images, i, key, back[i], forward[i])
        new = cv2.remap(raw[key].astype(np.float32), mx, my, cv2.INTER_CUBIC)
        current = np.rint(np.clip(raw[i]*(1-a[..., None])+new*a[..., None], 0, 255))
        if i in events:
            j = events[i]
            old_key = KEYS[j//6]
            old_a, ox, oy = gate_fn(gate_images, i, old_key, *transition[j])
            shared[j] = np.minimum(a, old_a)
            old = cv2.remap(raw[old_key].astype(np.float32), ox, oy, cv2.INTER_CUBIC)
            weight = shared[j]*RAMP[j % 6]
            current = np.rint(np.clip(current+weight[..., None]*(old-new), 0, 255))
        result[i] = current.astype(np.uint8)
        alphas[i] = a
    if not np.array_equal(result[alphas == 0], raw[alphas == 0]):
        raise ValueError('Zero-weight pixels changed')
    return dict(rgb=result, alpha=alphas, shared=shared)


def receive(base, upstream, out):
    import torch
    sys.path.insert(0, str(upstream))
    from src.models.stage_1.core.raft import RAFT

    out.mkdir(parents=True, exist_ok=False)
    weight = upstream/'pretrained_weights/raft-things.pth'
    allowed = {p.resolve() for p in (base/'base.glcsnd',
        base/'base_receive/decoded_rgb.npy', base/'base_receive/result.json', weight)}
    data_root = base.parents[2]
    observed = set()
    source_extensions = {'.npy', '.npz', '.glcsnd', '.glcfull', '.glccsp2', '.glccsp3', '.mp4', '.mkv'}

    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if p in allowed:
            observed.add(str(p))
            return
        if p.is_relative_to(out):
            return
        # Runtime source and shared libraries may be loaded lazily by PyTorch.
        code = p.is_relative_to(upstream) and p.suffix.lower() in {'.py', '.pyc', '.pyd', '.dll'}
        if not code and (p.is_relative_to(data_root) or p.suffix.lower() in source_extensions):
            raise RuntimeError('Untransmitted receiver input: '+str(p))

    sys.addaudithook(audit)
    blocked = []
    for p in (base/'source.npy', base/'base_cut.json', base/'flow/to_anchor.npy',
              base/'flow/from_anchor.npy', base/'sender_receipt.json', base/'source_mask.npy',
              base.parents[1]/'support_refinement_20261111_01'/base.name/'K/support.glccsp2'):
        try:
            p.read_bytes()
        except RuntimeError:
            blocked.append(str(p))
        else:
            raise ValueError('Input guard accepted an untransmitted input')

    receipt = load_json(base/'base_receive/result.json')
    digest = file_sha(base/'base.glcsnd')
    size = (base/'base.glcsnd').stat().st_size
    rgb_path = base/'base_receive/decoded_rgb.npy'
    if (digest != receipt['stream_sha256'] or size != receipt['stream_bytes']
            or file_sha(rgb_path) != receipt['decoded_rgb_sha256']
            or not receipt['reference_verified'] or not receipt['source_media_reads_forbidden']
            or receipt['header']['fps'] != 15):
        raise ValueError('Independent base receive verification failed')
    raw = np.load(rgb_path, mmap_mode='r', allow_pickle=False)
    if raw.dtype != np.uint8 or raw.shape != (150,288,512,3) or size*8/10 > 40000:
        raise ValueError('Unexpected base geometry/rate')
    if file_sha(weight) != WEIGHT_SHA:
        raise ValueError('RAFT checkpoint mismatch')
    save_json(out/'input_manifest.json', dict(base_sha256=digest, base_bytes=size,
        decoded_rgb_sha256=file_sha(rgb_path), weights_sha256=WEIGHT_SHA,
        keys=KEYS, old_weights=RAMP, auxiliary_bytes=0, parameters_frozen=True))

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = RAFT(argparse.Namespace(small=False, mixed_precision=False, alternate_corr=False))
    state = torch.load(weight, map_location='cpu', weights_only=True)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()}, strict=True)
    model.cuda().eval()
    frames = torch.from_numpy(np.array(raw)).permute(0,3,1,2).to(device='cuda', dtype=torch.float32)
    shape = (150,288,512,2)
    back, forward = [np.lib.format.open_memmap(out/name, mode='w+', dtype=np.float32, shape=shape)
                     for name in ('to_anchor.npy','from_anchor.npy')]
    transition = np.lib.format.open_memmap(out/'transition_flow.npy', mode='w+', dtype=np.float32,
                                          shape=(len(EVENTS),2,288,512,2))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    inferred = 0
    def pair(i, key):
        images = frames[[i,key]]
        _, prediction = model(images, images.flip(0), iters=ITERATIONS, test_mode=True)
        value = prediction.permute(0,2,3,1).contiguous().cpu().numpy()
        if value.shape != (2,288,512,2) or not np.isfinite(value).all():
            raise ValueError('Invalid RAFT flow')
        return value
    with torch.inference_mode():
        for key, stop in zip(KEYS, (*KEYS[1:],150), strict=True):
            back[key] = forward[key] = 0
            for i in range(key+1, stop):
                back[i], forward[i] = pair(i,key)
                inferred += 1
            print(base.name, 'receiver RAFT anchor', key, 'complete', flush=True)
        for j,i in enumerate(EVENTS):
            transition[j] = pair(i,KEYS[j//6])
            inferred += 1
    torch.cuda.synchronize()
    flow_seconds = time.perf_counter()-started
    for value in (back,forward,transition):
        value.flush()
    peak = torch.cuda.max_memory_allocated()/1024**2
    gpu = torch.cuda.get_device_name()
    del model, frames, state
    torch.cuda.empty_cache()
    started = time.perf_counter()
    values = render(raw, raw, back, forward, transition)
    render_seconds = time.perf_counter()-started
    for name,value in values.items():
        np.save(out/(name+'.npy'), value, allow_pickle=False)
    hashes = {p.name:file_sha(p) for p in out.glob('*.npy')}
    save_json(out/'receipt.json', dict(complete=True, method='RR', auxiliary_bytes=0,
        base_sha256=digest, base_bytes=size, total_kbps=size*8/10000,
        input_domain='GLC decoded RGB only; preinstalled shared RAFT weights',
        source_reads_blocked=True, audit_scope='Python open audit; not an OS sandbox',
        reads=sorted(observed), blocked_inputs=blocked,
        no_sender_receipt=True, no_manual_ROI=True, no_source_flow_or_mask=True,
        flow_pairs=inferred, flow_iterations=ITERATIONS, flow_seconds=flow_seconds,
        render_seconds=render_seconds, peak_allocated_mib=peak, gpu=gpu,
        torch=str(torch.__version__), keys=KEYS, old_weights=RAMP,
        all_references='Historical unfiltered GLC RGB',
        lookahead_frames=0, realtime_tested=False,
        zero_weight_unchanged=True, coverage=float((values['alpha'] > 0).mean()),
        outputs_sha256=hashes, code_sha256=file_sha(__file__),
        saved_flow_masks='Receiver-local diagnostics; no transmitted sidecar'))
    print(base.name, 'RR complete; auxiliary=0 B; coverage=', float((values['alpha']>0).mean()),
          'flow_seconds=', flow_seconds, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-dir', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    receive(a.base_dir.resolve(), a.upstream.resolve(), a.output.resolve())
