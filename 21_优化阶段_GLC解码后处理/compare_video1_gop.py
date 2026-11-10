"""Compare two raw video1 decodes on one source-aligned evaluation mask."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from audit_deflicker_texture import measure
from cache_motion_inputs import FPS, FRAME_SHAPE, base_frames
from compare_boundary_codec import aligned_metrics, frame_kind
from compare_deflicker import save_csv
from compare_gate_ablation import DETAILS, coarse_scores
from compare_source_dense_flow import load_flow
from compare_structure_memory import strip
from coral_reference_sidecar import file_sha, load_json, save_json
from experiment_common import array_sha
from run_dense_transfer import source_input
from run_p0 import preview, roi
from run_structure_memory import CASE, CLIP
from run_video1_gop import BASELINE_GOP, RESET, TEST_GOP


PERIODS = {'g64_r32': BASELINE_GOP, 'g56_r32': TEST_GOP}
LABELS = {'source': 'Source RGB', 'g64_r32': 'GLC GOP 64', 'g56_r32': 'GLC GOP 56'}
EVENTS = {56: 'g56_r32', 64: 'g64_r32', 112: 'g56_r32', 128: 'g64_r32'}


def verified_inputs(base, candidate):
    source = source_input(base, CLIP)
    raw, old_sha, old_size = base_frames(base, CLIP)
    alternate, new_sha, new_size = base_frames(candidate, CLIP)
    previous = load_json(base/'sender/result.json')
    sender = load_json(candidate/'sender.json')
    before = load_json(base/'base_receive/result.json')
    after = load_json(candidate/'base_receive/result.json')
    expected = dict(previous['header'], intra_period=TEST_GOP)
    if (sender['header'] != expected or after['header'] != expected
            or before['header'] != previous['header'] or sender['bytes'] != new_size
            or sender['source_rgb_sha256'] != array_sha(source)
            or sender['original_base_sha256'] != old_sha or sender['stream_sha256'] != new_sha
            or sender['reference_checksums'] != after['reference_checksums']
            or sender['prefix_packets_and_latents_bit_exact'] != TEST_GOP
            or not np.array_equal(raw[:TEST_GOP], alternate[:TEST_GOP])):
        raise ValueError('Source, settings, stream receipts or pre-change RGB differ')
    streams = {}
    for (key, gop), size, receipt in zip(PERIODS.items(), (old_size, new_size), (before, after), strict=True):
        actual = [(row['index'], row['kind']) for row in receipt['records']]
        expected = [(i, 0 if i % gop == 0 else 1) for i in range(CLIP.count)]
        if actual != expected:
            raise ValueError(f'Incorrect frame types in {key}')
        streams[key] = dict(bytes=size, kbps=size*8*FPS/CLIP.count/1000,
            stream_sha256=receipt['stream_sha256'], rgb_file_sha256=receipt['decoded_rgb_sha256'],
            i_frames=[i for i, kind in actual if kind == 0], independent_references_verified=True)
    return dict(source=source, g64_r32=raw, g56_r32=alternate), streams


def score_rows(rows):
    coarse = coarse_scores(rows, PERIODS)
    return {name: dict(**aligned_metrics(rows, name), coarse_mae=coarse[name]) for name in PERIODS}


def scope_scores(output, sequences, crop):
    rows = measure(sequences, crop, start=CLIP.start, include_coarse=True)
    kinds = {name: [frame_kind(i, period, RESET) for i in range(CLIP.count)]
             for name, period in PERIODS.items()}
    for row in rows:
        i = row['frame']
        for name in PERIODS:
            row[f'{name}_kind'] = kinds[name][i]
        row['common_ordinary_p_pair'] = all(
            seq[i-1] == seq[i] == 'ordinary_p' for seq in kinds.values())
    save_csv(output, rows)
    common = [row for row in rows if row['common_ordinary_p_pair']]
    late = [row for row in common if row['frame'] >= 6*FPS]
    first = [row for row in common if row['frame'] < 2*FPS]
    by_frame = {row['frame']: row for row in rows}
    events = {event: {
        i: dict(frame_types={name: kinds[name][i] for name in PERIODS}, scores=score_rows([by_frame[i]]))
        for i in range(event-2, event+3)} for event in EVENTS}
    by_type = {name: {kind: score_rows([row for row in rows if row[f'{name}_kind'] == kind])[name]
                      for kind in ('i', 'first_p', 'reset_p', 'ordinary_p')} for name in PERIODS}
    return dict(all=score_rows(rows), common_ordinary_p=score_rows(common),
        common_ordinary_p_first2=score_rows(first), common_ordinary_p_from6=score_rows(late),
        grouped_by_current_type=by_type, events=events)


def evaluate(batch, output, ffmpeg):
    base = batch/CASE
    sequences, streams = verified_inputs(base, output/'g56_r32')
    folder = output/'comparison'
    folder.mkdir(exist_ok=False)
    boxes = dict(reef=CLIP.coral, **DETAILS)
    scopes = {}
    for name, box in boxes.items():
        scopes[name] = scope_scores(folder/f'{name}_temporal.csv', sequences, roi(box, FRAME_SHAPE))
        print(f'{name}: boundary and common ordinary-P scores verified', flush=True)
    focus = roi(DETAILS['upper_plate'], FRAME_SHAPE)
    panels = [sequences[name] for name in PERIODS]
    labels = [LABELS[name] for name in PERIODS]
    previews = {name: preview(ffmpeg, folder/f'{name}.mp4', panels, FPS, CLIP.start, labels, crop)
                for name, crop in (('gop_full_pair', None), ('gop_upper_pair', focus))}
    _, forward = load_flow(base/'flow', sequences['source'], CLIP)
    strips = {event: strip(folder/f'frame_{event:03d}_five.png', sequences, focus, forward, event,
                           labels=LABELS, marked={method}) for event, method in EVENTS.items()}
    save_json(output/'summary.json', dict(
        source_rgb_sha256=array_sha(sequences['source']), streams=streams, fps=FPS, frames=CLIP.count,
        same_prefix_rgb_frames=TEST_GOP, fixed_regions=boxes, scopes=scopes, previews=previews, strips=strips,
        code_sha256=file_sha(__file__)))
    print('Both original-speed previews and four five-frame strips verified', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('batch', 'output', 'ffmpeg'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    evaluate(args.batch.resolve(), args.output.resolve(), args.ffmpeg.resolve())


if __name__ == '__main__':
    main()
