"""Score the two offline video1 changes against the verified short-cache baseline."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from cache_motion_inputs import FPS, FRAME_SHAPE, base_frames
from compare_cache_motion import read_rgb
from compare_deflicker import save_csv
from compare_gate_ablation import DETAILS, coarse_scores, region_scores
from compare_source_dense_flow import load_flow
from coral_reference_sidecar import file_sha, gray, load_json, save_json
from experiment_common import array_sha
from run_cache_motion import cache_anchors
from run_dense_transfer import source_input
from run_gate_ablation import editable, render, source_gate
from run_p0 import preview, roi
from run_structure_memory import CASE, CLIP, LONG_ANCHORS
from temporal_filter import correspondence


LABELS = dict(source='Source RGB', raw='GLC raw', O='O / short reference',
              S='S / added structure', L='L / longer reference')
I_FRAMES = (64, 128)


def anchor_per_frame(count, anchors):
    starts = np.asarray(anchors)
    return starts[np.searchsorted(starts, np.arange(count), side='right')-1]


def motion_diagnostics(source, flows, crop, allowed, coverage):
    luma = np.stack([gray(frame).astype(np.float32) for frame in source])
    mask = allowed[crop]
    short = cache_anchors(CLIP.count)
    anchors = {'O': anchor_per_frame(CLIP.count, short), 'L': anchor_per_frame(CLIP.count, LONG_ANCHORS)}
    rows = []
    for i, target in enumerate(luma):
        row = dict(frame=CLIP.start+i)
        common = mask.copy()
        errors = {}
        for name, (back, forward) in flows.items():
            anchor = int(anchors[name][i])
            x, y, inside, cycle = correspondence(back[i], forward[i])
            active = mask & inside[crop]
            common &= inside[crop]
            aligned = cv2.remap(luma[anchor], x, y, cv2.INTER_LINEAR)[crop]
            errors[name] = np.abs(target[crop]-aligned)
            row[f'{name}_anchor'] = anchor
            row[f'{name}_out_of_bounds'] = float((mask & ~inside[crop]).sum()/mask.sum())
            row[f'{name}_cycle_rejected'] = float((active & (cycle[crop] >= 1.5)).sum()/mask.sum())
            row[f'{name}_alpha_coverage'] = float((coverage[name][i][crop] > 0).mean())
        row['common_pixels'] = int(common.sum())
        for name, error in errors.items():
            row[f'{name}_source_warp_sum'] = float(error[common].sum())
        rows.append(row)
    return rows


def event_scores(output, clip, methods):
    path = output/'upper_plate_temporal.csv'
    with path.open(encoding='utf-8', newline='') as stream:
        rows = [{key: int(value) if key in ('frame', 'valid_pixels') else float(value)
                 for key, value in row.items()} for row in csv.DictReader(stream)]
    by_frame = {row['frame']: row for row in rows}
    periodic = cache_anchors(clip.count)
    removed = tuple(sorted(set(periodic)-set(LONG_ANCHORS)))
    after_i = tuple(i+1 for i in I_FRAMES)
    groups = dict(I=I_FRAMES, I_next=after_i, removed_refresh=removed,
        other_P=tuple(i for i in by_frame if i not in (1, *I_FRAMES, *after_i) and i not in periodic))
    stats = {}
    for name, frames in groups.items():
        selected = [by_frame[i] for i in frames]
        pixels = sum(row['valid_pixels'] for row in selected)
        stats[name] = dict(pairs=len(selected), coarse_mae=coarse_scores(selected, methods),
            fine_ncc={m: sum(row[f'{m}_fine_ncc']*row['valid_pixels'] for row in selected)/pixels
                      for m in methods})
    events = {i: dict(coarse_mae=coarse_scores([by_frame[i]], methods),
        fine_ncc={m: by_frame[i][f'{m}_fine_ncc'] for m in methods})
              for event in (60, 64, 120, 128) for i in range(event-2, event+3)}
    return dict(groups=stats, around_boundaries=events)


def strip(path, sequences, crop, forward, frame, *, labels=LABELS, marked=None):
    sample = sequences['source'][0][crop]
    height, width = 2*sample.shape[0], 2*sample.shape[1]
    title = 23
    canvas = np.full((len(sequences)*(height+title), 5*width, 3), (24, 35, 48), np.uint8)
    yy, xx = np.mgrid[:FRAME_SHAPE[0], :FRAME_SHAPE[1]].astype(np.float32)
    for col, i in enumerate(range(frame-2, frame+3)):
        mx, my = xx+forward[i, ..., 0], yy+forward[i, ..., 1]
        for row, (name, video) in enumerate(sequences.items()):
            y, x = row*(height+title), col*width
            aligned = cv2.remap(video[i], mx, my, cv2.INTER_LINEAR)[crop]
            canvas[y+title:y+title+height, x:x+width] = cv2.resize(aligned, (width, height),
                interpolation=cv2.INTER_NEAREST)
            is_boundary = i == frame and (marked is None or name in marked)
            text = f'{labels[name]} | f{i}' + (' I boundary' if is_boundary else '')
            cv2.putText(canvas, text, (x+8, y+16), cv2.FONT_HERSHEY_SIMPLEX, .46,
                        (240, 244, 250), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError('Could not encode the aligned frame strip')
    path.write_bytes(encoded.tobytes())
    return dict(frames=list(range(frame-2, frame+3)), sha256=file_sha(path))


def evaluate(base, ablation, run, ffmpeg):
    raw, base_sha, _ = base_frames(base, CLIP)
    source = source_input(base, CLIP)
    prior = load_json(ablation/'upper/result.json')
    structure = load_json(run/'structure/result.json')
    reference = read_rgb(ablation/'upper', 'rgb', prior['rgb_sha256'], CLIP)
    alpha = np.load(ablation/'upper/alpha.npy', mmap_mode='r', allow_pickle=False)
    if not prior['alpha_sha256'] == structure['alpha_sha256'] == array_sha(alpha):
        raise ValueError('Baseline and structure weights differ')
    if not prior['base_sha256'] == structure['base_sha256'] == base_sha:
        raise ValueError('Baseline and structure use different GLC base streams')
    s = read_rgb(run/'structure', 'rgb', structure['rgb_sha256'], CLIP)
    short = load_flow(base/'flow', source, CLIP)
    long = load_flow(run/'long_flow', source, CLIP, anchors=LONG_ANCHORS)
    allowed = editable(CASE)
    gate = source_gate(source, *long, anchors=LONG_ANCHORS)
    l, l_alpha = render(raw, long[0], gate, allowed, anchors=LONG_ANCHORS)
    del gate
    memory = run/'memory'
    memory.mkdir(exist_ok=False)
    np.save(memory/'rgb.npy', l, allow_pickle=False)
    np.save(memory/'alpha.npy', l_alpha, allow_pickle=False)
    save_json(memory/'result.json', dict(offline_source_oracle=True, base_sha256=base_sha,
        anchors=LONG_ANCHORS, rgb_sha256=array_sha(l), alpha_sha256=array_sha(l_alpha),
        flow_receipt_sha256=file_sha(run/'long_flow/flow.json')))
    coverage = dict(O=alpha, S=alpha, L=l_alpha)
    sequences = dict(source=source, raw=raw, O=reference, S=s, L=l)
    boxes = dict(reef=CLIP.coral, **DETAILS)
    crops = {'image': np.s_[:, :]} | {name: roi(box, FRAME_SHAPE) for name, box in boxes.items()}
    output = run/'comparison'
    output.mkdir(exist_ok=False)
    scores = {}
    for name, crop in crops.items():
        scores[name] = region_scores(output, name, sequences, coverage, crop, CLIP)
        print(f'{name}: fixed-region comparison ready', flush=True)
    old = load_json(ablation/'summary.json')['fixed_scores']
    for name in crops:
        for section in ('full', 'last2'):
            if scores[name][section]['scores']['O'] != old[name][section]['scores']['upper']:
                raise ValueError('Baseline measurements differ from the verified O experiment')
    focus = crops['upper_plate']
    geometry = motion_diagnostics(source, dict(O=short, L=long), focus, allowed, coverage)
    save_csv(output/'upper_plate_motion.csv', geometry)
    previews = {}
    for name, keys, crop in (
            ('full_comparison', ('raw','O','S','L'), None),
            ('structure_pair', ('O','S'), focus),
            ('memory_pair', ('O','L'), focus)):
        previews[name] = preview(ffmpeg, output/f'{name}.mp4', [sequences[key] for key in keys],
            FPS, CLIP.start, [LABELS[key] for key in keys], crop)
    strips = {str(i): strip(output/f'frame_{i:03d}_five_aligned.png', sequences, focus, short[1], i)
              for i in I_FRAMES}
    save_json(run/'summary.json', dict(offline_source_oracle=True, base_sha256=base_sha,
        fixed_regions=boxes, fps=FPS, frames=CLIP.count, short_anchors=cache_anchors(CLIP.count),
        long_anchors=LONG_ANCHORS, structure_sigma=[3,12], original_o_measurements_exact=True,
        fixed_scores=scores, upper_plate_events=event_scores(output, CLIP, sequences),
        motion_csv_sha256=file_sha(output/'upper_plate_motion.csv'), previews=previews, strips=strips,
        code_sha256=file_sha(__file__)))
    print('Original-speed comparisons and both five-frame strips verified', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('batch', 'ablation', 'run', 'ffmpeg'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    evaluate(args.batch.resolve()/CASE, args.ablation.resolve()/CASE, args.run.resolve(), args.ffmpeg.resolve())


if __name__ == '__main__':
    main()
