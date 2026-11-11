"""Receiver and sender stages for the GLC cache motion by reliability ablation."""

import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np

from cache_motion_inputs import FPS, HEIGHT, WIDTH, base_frames
from compare_source_dense_flow import load_flow
from coral_reference_sidecar import detail, file_sha, gray, load_json, save_json
from experiment_common import array_sha
from gate_aux_stream import GATE, MOTION, byte_limit, expand, pack_grid, quantize, unpack_grid
from run_cache_motion import (
    CACHE_FRAMES, blend_cached_texture, cache_anchors, gate_components, guard_receiver, low_frequency_difference,
)
from run_dense_batch import CASES
from run_dense_transfer import source_input
from source_raft_flow import infer
from temporal_filter import correspondence


PILOTS = ('video1', 'video2')
# Receiver-local annotations from its GLC decode, independent of the evaluation masks.
EDIT_WINDOWS = {
    'video1': ((338, 8, 508, 112), (346, 194, 486, 270)),
    'video2': ((406, 170, 508, 284),),
}
WIRES = {'A': (), 'B': (GATE,), 'C': (MOTION,), 'D': (MOTION, GATE)}
GRID = (64, 36)
MOTION_SCALE = 1
GATE_SCALE = 7


def editable(case):
    mask = np.zeros((HEIGHT, WIDTH), bool)
    for x0, y0, x1, y1 in EDIT_WINDOWS[case]:
        mask[y0:y1, x0:x1] = True
    return mask


def source_gate(source, backward, forward, *, anchors=None):
    gates = np.zeros(source.shape[:3], np.float32)
    refresh = set(cache_anchors(len(source), anchors))
    for i, frame in enumerate(source):
        coarse = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), 3)
        if i in refresh:
            anchor = coarse
            continue
        mx, my, inside, error = correspondence(backward[i], forward[i])
        projected = cv2.remap(anchor, mx, my, cv2.INTER_LINEAR)
        difference = low_frequency_difference(projected, coarse)
        reliable, feather, tone = gate_components(inside & (error < 1.5), difference)
        gates[i] = reliable * feather * tone
    return gates


def base_media(base):
    return {base/'base.glcsnd', base/'base_receive/decoded_rgb.npy'}


def fit_receiver(base, run, clip, upstream):
    folder = run/'receiver_flow'
    guard_receiver(base_media(base), folder)
    raw, _, _ = base_frames(base, clip)
    infer(np.array(raw), clip, upstream, folder, receiver_rgb=True)


def receiver_flow(run, raw, clip):
    folder = run/'receiver_flow'
    receipt = load_json(folder/'flow.json')
    path = folder/'to_anchor.npy'
    if (receipt['offline_source_oracle'] or receipt['input_glc_rgb_sha256'] != array_sha(raw)
            or receipt['source_range'] != [clip.start, clip.stop] or receipt['fps'] != FPS
            or receipt['anchor_period'] != CACHE_FRAMES
            or file_sha(path) != receipt['flow_file_sha256'][path.name]):
        raise ValueError('The local motion cache is not bound to the independently decoded GLC RGB')
    flow = np.load(path, mmap_mode='r', allow_pickle=False)
    if flow.shape != (*raw.shape[:3], 2) or flow.dtype != np.float32 or flow[::CACHE_FRAMES].any():
        raise ValueError('Invalid receiver flow')
    return flow


def send(base, run, clip, case):
    _, base_sha, _ = base_frames(base, clip)
    source = source_input(base, clip)
    backward, forward = load_flow(base/'flow', source, clip)
    gate = source_gate(source, backward, forward)
    allowed = editable(case)
    fields = {MOTION: (backward, MOTION_SCALE), GATE: (gate, GATE_SCALE)}
    files = {}
    for name, (frames, scale) in fields.items():
        q = quantize(frames, allowed, GRID, scale, name)
        files[name] = pack_grid(q, name, clip, base_sha, scale)
        if not np.array_equal(unpack_grid(files[name], name, clip, base_sha), q.astype(np.float32)/scale):
            raise ValueError('Auxiliary grid failed round-trip')
    if sum(map(len, files.values())) > byte_limit(clip):
        raise ValueError('D exceeds its independent semantic-lane budget')

    out = run/'sender'
    out.mkdir(exist_ok=False)
    for name, packet in files.items():
        (out/name).write_bytes(packet)
    for variant, names in WIRES.items():
        inbox = run/'delivery'/variant
        inbox.mkdir(parents=True, exist_ok=False)
        for name in names:
            shutil.copyfile(out/name, inbox/name)
    ledger = {name: dict(bytes=len(packet), sha256=file_sha(out/name)) for name, packet in files.items()}
    save_json(out/'result.json', dict(base_sha256=base_sha, source_rgb_sha256=array_sha(source),
        editable_windows=EDIT_WINDOWS[case], grid=GRID, motion_scale=MOTION_SCALE, gate_scale=GATE_SCALE,
        semantic_budget_bytes=byte_limit(clip), files=ledger, code_sha256=file_sha(__file__)))
    print(f'{case}: motion {ledger[MOTION]["bytes"]} B, gate {ledger[GATE]["bytes"]} B', flush=True)


def structure_band(luma, windows):
    image = luma.astype(np.float32)
    band = np.zeros_like(image)
    for x0, y0, x1, y1 in windows:
        patch = image[y0:y1, x0:x1]
        band[y0:y1, x0:x1] = cv2.GaussianBlur(patch, (0, 0), 3) - cv2.GaussianBlur(patch, (0, 0), 12)
    return band


def render(raw, flow, gate, allowed, *, anchors=None, structure_windows=()):
    rgb = raw.copy()
    masks = np.zeros(raw.shape[:3], np.uint8)
    yy, xx = np.mgrid[:HEIGHT, :WIDTH].astype(np.float32)
    reference_support = cv2.erode(allowed.astype(np.uint8), np.ones((5, 5), np.uint8))
    indices = cache_anchors(len(raw), anchors)
    stops = dict(zip(indices, (*indices[1:], len(raw)), strict=True))
    for i, frame in enumerate(raw):
        coarse = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), 3)
        luma = gray(frame)
        texture = detail(luma, 3)
        structure = structure_band(luma, structure_windows) if structure_windows else None
        if i in stops:
            anchor = i
            anchor_low, anchor_texture = coarse, texture
            anchor_structure = structure
            continue
        mx, my = xx+flow[i, ..., 0], yy+flow[i, ..., 1]
        visible = (mx >= 2) & (my >= 2) & (mx < WIDTH-2) & (my < HEIGHT-2)
        protected = ~allowed | (cv2.remap(reference_support, mx, my, cv2.INTER_NEAREST) == 0)
        projected = cv2.remap(anchor_texture, mx, my, cv2.INTER_CUBIC)
        difference = None
        if gate is None:
            aligned_low = cv2.remap(anchor_low, mx, my, cv2.INTER_LINEAR)
            difference = low_frequency_difference(aligned_low, coarse)
        structure_delta = None if structure is None else (
            cv2.remap(anchor_structure, mx, my, cv2.INTER_CUBIC) - structure)
        rgb[i], masks[i] = blend_cached_texture(frame, texture, projected, visible, difference,
            i, anchor, sent_gate=None if gate is None else gate[i], protected=protected,
            anchor_end=None if anchors is None else stops[anchor], structure_delta=structure_delta)
    if masks[:, ~allowed].any() or not np.array_equal(rgb[:, ~allowed], raw[:, ~allowed]):
        raise ValueError('A receiver-local foreground protection window was changed')
    return rgb, masks


def write_render(output, raw, flow, gate, case, **provenance):
    allowed = editable(case)
    rgb, alpha = render(raw, flow, gate, allowed)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output/'rgb.npy', rgb, allow_pickle=False)
    np.save(output/'alpha.npy', alpha, allow_pickle=False)
    save_json(output/'result.json', dict(**provenance, rgb_sha256=array_sha(rgb),
        alpha_sha256=array_sha(alpha), editable_windows=EDIT_WINDOWS[case],
        protected_pixels_bit_exact=True, code_sha256=file_sha(__file__)))
    print(f'{case} {output.name}: protected pixels bit-exact', flush=True)


def receive(base, run, clip, case, variant):
    inbox = run/'delivery'/variant
    names = WIRES[variant]
    if set(names) != {path.name for path in inbox.iterdir()}:
        raise ValueError('Receiver inbox differs from the assigned ablation')
    packets = {name: inbox/name for name in names}
    size = sum(path.stat().st_size for path in packets.values())
    if size > byte_limit(clip):
        raise ValueError('The complete delivered auxiliary files exceed the semantic lane')
    output = run/'receive'/variant
    permitted = base_media(base) | set(packets.values())
    if MOTION not in packets:
        permitted.add(run/'receiver_flow/to_anchor.npy')
    guard_receiver(permitted, output)
    raw, base_sha, _ = base_frames(base, clip)
    grids = {name: unpack_grid(path.read_bytes(), name, clip, base_sha) for name, path in packets.items()}
    flow = expand(grids[MOTION]) if MOTION in grids else receiver_flow(run, raw, clip)
    gate = expand(grids[GATE]) if GATE in grids else None
    write_render(output, raw, flow, gate, case, base_sha256=base_sha,
        offline_upper=False, source_media_reads_forbidden=True, extra_bytes=size,
        files={name: file_sha(path) for name, path in packets.items()})


def upper(base, run, clip, case):
    raw, base_sha, _ = base_frames(base, clip)
    source = source_input(base, clip)
    backward, forward = load_flow(base/'flow', source, clip)
    write_render(run/'upper', raw, backward, source_gate(source, backward, forward), case,
        base_sha256=base_sha, offline_upper=True, extra_bytes=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('fit-receiver', 'send', 'receive', 'upper'))
    parser.add_argument('--case', choices=PILOTS, required=True)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--variant', choices=WIRES)
    args = parser.parse_args()
    base = args.batch.resolve()/args.case
    run = args.run.resolve()/args.case
    clip, _ = CASES[args.case]
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    if args.action == 'fit-receiver':
        if args.upstream is None:
            parser.error('fit-receiver requires --upstream')
        fit_receiver(base, run, clip, args.upstream.resolve())
    elif args.action == 'send':
        send(base, run, clip, args.case)
    elif args.action == 'receive':
        if args.variant is None:
            parser.error('receive requires --variant')
        receive(base, run, clip, args.case, args.variant)
    else:
        upper(base, run, clip, args.case)


if __name__ == '__main__':
    main()
