"""Measure same-frame I/P fidelity and test a short, source-assisted display transition."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

from audit_deflicker_texture import measure
from cache_motion_inputs import FPS, FRAME_SHAPE
from compare_boundary_codec import aligned_metrics
from compare_deflicker import save_csv
from compare_gate_ablation import DETAILS, coarse_scores
from compare_source_dense_flow import load_flow
from compare_structure_memory import strip
from compare_video1_gop import EVENTS, PERIODS, verified_inputs
from coral_reference_sidecar import detail, file_sha, gray, save_json
from experiment_common import array_sha
from run_cache_motion import CACHE_FRAMES, blend_cached_texture, gate_components, low_frequency_difference
from run_gate_ablation import EDIT_WINDOWS, structure_band
from run_p0 import preview, psnr, roi
from run_structure_memory import CASE, CLIP
from temporal_filter import correspondence


UPPER = DETAILS['upper_plate']
EDIT = EDIT_WINDOWS[CASE][:1]
RAMP = (1., .75, .5, .25)
# Fixed for each event after reviewing the source frames from I-1 through I+3.
FOREGROUND_LOCKS = {
    56: ((338, 8, 372, 52),),
    64: ((338, 8, 372, 52),),
    112: ((426, 8, 461, 47), (448, 15, 508, 64)),
    128: ((469, 56, 508, 104), (487, 8, 508, 23)),
}
METRIC_FILES = {
    'torch/hub/checkpoints/alexnet-owt-7be5be79.pth':
        '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02',
    'metric_packages/lpips/weights/v0.1/alex.pth':
        'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0',
}
LABELS = {'source': 'Source RGB', 'raw': 'GLC raw', 'transition': 'GLC display transition'}


def allowed_at(event):
    allowed = np.zeros(FRAME_SHAPE[:2], bool)
    allowed[roi(EDIT[0], FRAME_SHAPE)] = True
    for box in FOREGROUND_LOCKS[event]:
        allowed[roi(box, FRAME_SHAPE)] = False
    return allowed


def compose_flow(to_key, from_key, xx, yy):
    # First sample the displacement from the common RAFT key to the target.
    mx, my = xx + to_key[..., 0], yy + to_key[..., 1]
    h, w = to_key.shape[:2]
    valid = (mx >= 1) & (mx < w - 2) & (my >= 1) & (my < h - 2)
    return to_key + cv2.remap(from_key, mx, my, cv2.INTER_LINEAR), valid


def render_transition(raw, source, flows, events):
    rendered = raw.copy()
    alpha = np.zeros(raw.shape[:3], np.uint8)
    yy, xx = np.mgrid[:FRAME_SHAPE[0], :FRAME_SHAPE[1]].astype(np.float32)
    to_key, from_key = flows
    for event in events:
        anchor = event - 1
        stop = event + len(RAMP)
        if anchor // CACHE_FRAMES != (stop - 1) // CACHE_FRAMES:
            raise ValueError(f'Transition at {event} spans two source RAFT keys')
        allowed = allowed_at(event)
        support = cv2.erode(allowed.astype(np.uint8), np.ones((5, 5), np.uint8))
        ref_luma = gray(raw[anchor])
        ref_detail = detail(ref_luma, 3)
        ref_structure = structure_band(ref_luma, EDIT)
        ref_source = cv2.GaussianBlur(source[anchor].astype(np.float32), (0, 0), 3)
        for i, fade in enumerate(RAMP, start=event):
            backward, current_valid = compose_flow(to_key[i], from_key[anchor], xx, yy)
            forward, anchor_valid = compose_flow(to_key[anchor], from_key[i], xx, yy)
            mx, my, inside, cycle = correspondence(backward, forward)
            supported = cv2.remap(support * anchor_valid, mx, my, cv2.INTER_NEAREST).astype(bool)
            visible = allowed & inside & current_valid & supported & (cycle < 1.5)
            source_low = cv2.GaussianBlur(source[i].astype(np.float32), (0, 0), 3)
            difference = low_frequency_difference(cv2.remap(ref_source, mx, my, cv2.INTER_LINEAR), source_low)
            reliable, feather, tone = gate_components(visible, difference)
            luma = gray(raw[i])
            projected = cv2.remap(ref_detail, mx, my, cv2.INTER_CUBIC)
            structure = cv2.remap(ref_structure, mx, my, cv2.INTER_LINEAR) - structure_band(luma, EDIT)
            rendered[i], alpha[i] = blend_cached_texture(
                raw[i], detail(luma, 3), projected, visible, difference, i, anchor,
                sent_gate=reliable * feather * tone, protected=~allowed, structure_delta=structure, fade=fade)
    return rendered, alpha


def rgb_psnr(source, frame):
    difference = source.astype(np.float32) - frame
    return psnr(float(np.mean(np.square(difference))))


def fidelity_scores(sequences, crop, root):
    hashes = {name: file_sha(root / name) for name in METRIC_FILES}
    if hashes != METRIC_FILES:
        raise ValueError('Unexpected LPIPS weights')
    sys.path.insert(0, str(root / 'metric_packages'))
    import lpips
    import torch

    torch.hub.set_dir(str(root / 'torch/hub'))
    torch.set_num_threads(1)
    model = lpips.LPIPS(net='alex', version='0.1', verbose=False).eval().cpu()
    hashes['metric_packages/lpips/__init__.py'] = file_sha(Path(lpips.__file__))
    scored = {i for event in EVENTS for i in range(event - 2, event + len(RAMP) + 1)}
    methods = {name: video for name, video in sequences.items() if name != 'source'}
    rows = []
    with torch.inference_mode():
        for i, source in enumerate(sequences['source']):
            row = dict(frame=i)
            band = structure_band(gray(source), EDIT)[crop]
            distances = [None] * len(methods)
            if i in scored:
                patches = np.stack([source[crop], *(video[i][crop] for video in methods.values())])
                tensor = torch.from_numpy(patches).permute(0, 3, 1, 2).float() / 255
                distances = model(tensor[:1].expand_as(tensor[1:]), tensor[1:], normalize=True).flatten().tolist()
            for (name, video), distance in zip(methods.items(), distances, strict=True):
                frame = video[i]
                row[f'{name}_upper_psnr_db'] = rgb_psnr(source[crop], frame[crop])
                row[f'{name}_upper_structure_mae'] = float(
                    np.abs(structure_band(gray(frame), EDIT)[crop] - band).mean())
                row[f'{name}_upper_lpips_alex'] = distance
                row[f'{name}_full_psnr_db'] = rgb_psnr(source, frame)
            rows.append(row)
    return rows, dict(device='cpu', model='LPIPS-Alex v0.1', torch=str(torch.__version__),
                      frames=sorted(scored), sha256=hashes)


def same_frame(row, name):
    return {metric: row[f'{name}_{metric}'] for metric in
            ('upper_psnr_db', 'upper_structure_mae', 'upper_lpips_alex', 'full_psnr_db')}


def summarize_temporal(rows):
    methods = ('raw', 'transition')
    coarse = coarse_scores(rows, methods)
    return {name: dict(**aligned_metrics(rows, name), coarse_mae=coarse[name]) for name in methods}


def evaluate(base, gop, output, ffmpeg, metrics_root):
    sequences, streams = verified_inputs(base, gop / 'g56_r32')
    source = sequences['source']
    flows = load_flow(base / 'flow', source, CLIP)
    output.mkdir(exist_ok=False)
    masks = {}
    for name in PERIODS:
        events = [i for i, mode in EVENTS.items() if mode == name]
        result, alpha = render_transition(sequences[name], source, flows, events)
        sequences[f'{name}_transition'] = result
        masks[name] = alpha
        np.save(output / f'{name}_transition_rgb.npy', result, allow_pickle=False)
        np.save(output / f'{name}_alpha.npy', alpha, allow_pickle=False)
        print(f'{name}: rendered boundary windows at {events}', flush=True)

    crop = roi(UPPER, FRAME_SHAPE)
    fidelity, metric = fidelity_scores(sequences, crop, metrics_root)
    save_csv(output / 'fidelity.csv', fidelity)
    print('Same-frame PSNR, structure-band MAE and CPU LPIPS ready', flush=True)

    events = {}
    temporal_rows = []
    strips = {}
    for i, name in EVENTS.items():
        other = next(mode for mode in PERIODS if mode != name)
        stop = i + len(RAMP)
        window = slice(i - 1, stop + 1)
        transition = f'{name}_transition'
        variants = dict(source=source, raw=sequences[name], transition=sequences[transition])
        rows = measure({key: video[window] for key, video in variants.items()},
                       crop, start=i-1, include_coarse=True)
        for row in rows:
            row.update(event=i, gop=PERIODS[name], offset=row['frame']-i)
        temporal_rows.extend(rows)
        events[i] = dict(gop=PERIODS[name], i=same_frame(fidelity[i], name), p=same_frame(fidelity[i], other),
            display_i=same_frame(fidelity[i], transition), anchor=i-1,
            display_frames=list(range(i, stop)), first_untouched_frame=stop,
            upper_alpha_coverage=float((masks[name][i:stop][:, crop[0], crop[1]] > 0).mean()),
            entry=summarize_temporal(rows[:1]), exit=summarize_temporal(rows[-1:]),
            through_exit=summarize_temporal(rows),
            coarse_by_frame={row['frame']: coarse_scores([row], ('raw', 'transition')) for row in rows})
        strips[i] = strip(output / f'frame_{i:03d}_five.png', variants, crop, flows[1], i,
                          labels=LABELS, marked={'raw', 'transition'})
    save_csv(output / 'upper_temporal.csv', temporal_rows)

    names = [mode for name in PERIODS for mode in (name, f'{name}_transition')]
    labels = [f'GOP {period} / {kind}' for period in PERIODS.values() for kind in ('Raw', 'Transition')]
    previews = {name: preview(ffmpeg, output / f'{name}.mp4', [sequences[key] for key in names],
                              FPS, CLIP.start, labels, area)
                for name, area in (('boundary_upper_comparison', crop), ('boundary_full_comparison', None))}
    save_json(output / 'summary.json', dict(fps=FPS, frames=CLIP.count, offline_source_oracle=True,
        base_streams=streams, source_rgb_sha256=array_sha(source), source_flow_receipt_sha256=file_sha(base/'flow/flow.json'),
        fixed_upper_roi=UPPER, edit_windows=EDIT, foreground_locks=FOREGROUND_LOCKS, fade=RAMP,
        metric=metric, events=events, all_events_through_exit=summarize_temporal(temporal_rows),
        display_rgb_sha256={name: array_sha(sequences[f'{name}_transition']) for name in PERIODS},
        alpha_sha256={name: array_sha(alpha) for name, alpha in masks.items()}, previews=previews, strips=strips,
        code_sha256=file_sha(__file__)))
    print('Both original-speed previews and the boundary audit are ready', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('batch', 'gop', 'output', 'ffmpeg', 'metrics-root'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    evaluate(args.batch.resolve()/CASE, args.gop.resolve(), args.output.resolve(), args.ffmpeg.resolve(),
             args.metrics_root.resolve())


if __name__ == '__main__':
    main()
