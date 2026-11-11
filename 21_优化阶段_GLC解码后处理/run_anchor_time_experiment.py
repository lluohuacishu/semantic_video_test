"""Compare fewer synchronized anchors and fewer RAFT iterations on frozen S8I."""

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

from receiver_structure import (EVENTS, KEYS, RAMP, WEIGHT_SHA, file_sha, load_json,
                               save_json)
from receive_guide_texture import texture_gate
import structure_guide_stream as guide


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[3]
DATA = WORKSPACE/'02_实验数据/优化_GLC后处理'
RUN = DATA/'anchor_time_20261111_01'
BASE = DATA/'source_dense_flow_three_clips_20261109_01/video1'
GUIDE_PACKET = DATA/'guide_compression_20261111_01/video1/S8I/guide.glcguide'
ACCEPTED = DATA/'guide_compression_20261111_01/video1/S8I/receive'
UPSTREAM = DATA/'deflicker_reef80_20261108_01/upstream'
FFMPEG = WORKSPACE/'01_项目/运行环境/FFmpeg/ffmpeg.exe'

PROFILES = {
    'CURRENT_I24': dict(keys=list(KEYS), iterations=24),
    'ISYNC_I24': dict(keys=[0, 64, 128], iterations=24),
    'CURRENT_I12': dict(keys=list(KEYS), iterations=12),
    'ISYNC_I12': dict(keys=[0, 64, 128], iterations=12),
}


def events_for(keys):
    return tuple(key+offset for key in keys[1:] for offset in range(len(RAMP)))


def render_profile(raw, images, back, forward, transition, keys):
    result = raw.copy()
    alphas = np.zeros(raw.shape[:3], np.float32)
    events = events_for(keys)
    shared = np.zeros((len(events), *raw.shape[1:3]), np.float32)
    event_lookup = {frame: index for index, frame in enumerate(events)}
    gate = texture_gate(images)
    for frame in range(len(raw)):
        anchor_index = int(np.searchsorted(keys, frame, side='right')-1)
        key = keys[anchor_index]
        alpha, mx, my = gate(images, frame, key, back[frame], forward[frame])
        new = cv2.remap(raw[key].astype(np.float32), mx, my, cv2.INTER_CUBIC)
        current = np.rint(np.clip(raw[frame]*(1-alpha[..., None])+new*alpha[..., None], 0, 255))
        if frame in event_lookup:
            event_index = event_lookup[frame]
            old_key = keys[event_index//len(RAMP)]
            old_alpha, ox, oy = gate(images, frame, old_key, *transition[event_index])
            shared[event_index] = np.minimum(alpha, old_alpha)
            old = cv2.remap(raw[old_key].astype(np.float32), ox, oy, cv2.INTER_CUBIC)
            weight = shared[event_index]*RAMP[event_index % len(RAMP)]
            current = np.rint(np.clip(current+weight[..., None]*(old-new), 0, 255))
        result[frame] = current.astype(np.uint8)
        alphas[frame] = alpha
    if not np.array_equal(result[alphas == 0], raw[alphas == 0]):
        raise ValueError('Zero-weight pixels changed')
    return dict(rgb=result, alpha=alphas, shared=shared)


def configure(workspace, run_dir):
    global WORKSPACE, DATA, RUN, BASE, GUIDE_PACKET, ACCEPTED, UPSTREAM, FFMPEG
    WORKSPACE = workspace
    DATA = WORKSPACE/'02_实验数据/优化_GLC后处理'
    RUN = run_dir
    BASE = DATA/'source_dense_flow_three_clips_20261109_01/video1'
    GUIDE_PACKET = DATA/'guide_compression_20261111_01/video1/S8I/guide.glcguide'
    ACCEPTED = DATA/'guide_compression_20261111_01/video1/S8I/receive'
    UPSTREAM = DATA/'deflicker_reef80_20261108_01/upstream'
    FFMPEG = WORKSPACE/'01_项目/运行环境/FFmpeg/ffmpeg.exe'


def worker(profile, run_dir):
    configure(run_dir.parents[2], run_dir)
    process_started = time.perf_counter()
    plan = load_json(RUN/'execution_plan.json')
    spec = plan['matrix'][profile]
    keys = tuple(spec['keys'])
    iterations = spec['iterations']
    out = RUN/profile
    out.mkdir(parents=True, exist_ok=False)
    flow_dir = out/'flow'
    receive_dir = out/'receive'
    flow_dir.mkdir()
    receive_dir.mkdir()

    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(UPSTREAM))
    from src.models.stage_1.core.raft import RAFT

    allowed = {p.resolve() for p in (BASE/'base.glcsnd', BASE/'base_receive/decoded_rgb.npy',
        BASE/'base_receive/result.json', GUIDE_PACKET, UPSTREAM/'pretrained_weights/raft-things.pth')}
    data_root = BASE.parents[2]
    observed = set()

    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path in allowed:
            observed.add(str(path))
            return
        if path.is_relative_to(out):
            return
        runtime = path.is_relative_to(UPSTREAM) and path.suffix.lower() in {'.py', '.pyc', '.pyd', '.dll'}
        if not runtime and (path.is_relative_to(data_root) or path.suffix.lower() in
                {'.npy', '.npz', '.glcsnd', '.glccsp2', '.glccsp3', '.glcguide', '.mp4', '.hevc'}):
            raise RuntimeError('Untransmitted guide receiver input: '+str(path))

    sys.addaudithook(audit)
    blocked = []
    for path in (BASE/'source.npy', BASE/'base_cut.json', BASE/'flow/to_anchor.npy',
        BASE/'flow/from_anchor.npy', GUIDE_PACKET.parent/'send.json',
        GUIDE_PACKET.parent/'sender_expected_guide.npy',
        DATA/'support_refinement_20261111_01/video1/K/support.glccsp2'):
        try:
            path.read_bytes()
        except RuntimeError:
            blocked.append(str(path))
        else:
            raise ValueError('Receiver accepted an untransmitted input')

    base_receipt = load_json(BASE/'base_receive/result.json')
    base_digest = file_sha(BASE/'base.glcsnd')
    base_bytes = (BASE/'base.glcsnd').stat().st_size
    raw_path = BASE/'base_receive/decoded_rgb.npy'
    if (base_digest != base_receipt['stream_sha256'] or base_bytes != base_receipt['stream_bytes']
            or file_sha(raw_path) != base_receipt['decoded_rgb_sha256']
            or not base_receipt['reference_verified'] or not base_receipt['source_media_reads_forbidden']):
        raise ValueError('Independent base receive binding changed')
    raw = np.load(raw_path, mmap_mode='r', allow_pickle=False)
    if raw.dtype != np.uint8 or raw.shape != (150, 288, 512, 3):
        raise ValueError('Unexpected base geometry')

    decode_started = time.perf_counter()
    packet = GUIDE_PACKET.read_bytes()
    header, _, images = guide.decode(packet, base_digest, FFMPEG)
    decode_seconds = time.perf_counter()-decode_started
    if header['frames'] != 150 or header['fps'] != 15 or header['guide_size'] != [256, 144]:
        raise ValueError('S8I guide header changed')
    np.save(receive_dir/'guide_rgb.npy', images, allow_pickle=False)

    load_started = time.perf_counter()
    weight = UPSTREAM/'pretrained_weights/raft-things.pth'
    if file_sha(weight) != WEIGHT_SHA:
        raise ValueError('RAFT weight mismatch')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise ValueError('CUDA is required for comparable RAFT timing')
    model = RAFT(argparse.Namespace(small=False, mixed_precision=False, alternate_corr=False))
    state = torch.load(weight, map_location='cpu', weights_only=True)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    model.cuda().eval()
    frames = torch.from_numpy(images).permute(0, 3, 1, 2).to(device='cuda', dtype=torch.float32)
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter()-load_started

    back, forward = [np.lib.format.open_memmap(flow_dir/name, mode='w+', dtype=np.float32,
        shape=(150, 288, 512, 2)) for name in ('to_anchor.npy', 'from_anchor.npy')]
    events = events_for(keys)
    transition = np.lib.format.open_memmap(flow_dir/'transition_flow.npy', mode='w+', dtype=np.float32,
        shape=(len(events), 2, 288, 512, 2))

    def pair(frame, key):
        batch = frames[[frame, key]]
        _, prediction = model(batch, batch.flip(0), iters=iterations, test_mode=True)
        value = prediction.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        if value.shape != (2, 288, 512, 2) or not np.isfinite(value).all():
            raise ValueError('Invalid RAFT output')
        return value

    torch.cuda.reset_peak_memory_stats()
    inferred = 0
    flow_started = time.perf_counter()
    with torch.inference_mode():
        for key, stop in zip(keys, (*keys[1:], 150), strict=True):
            back[key] = forward[key] = 0
            for frame in range(key+1, stop):
                back[frame], forward[frame] = pair(frame, key)
                inferred += 1
            print(profile, 'anchor flow complete', key, flush=True)
        for index, frame in enumerate(events):
            transition[index] = pair(frame, keys[index//len(RAMP)])
            inferred += 1
    torch.cuda.synchronize()
    flow_seconds = time.perf_counter()-flow_started
    for array in (back, forward, transition):
        array.flush()
    peak_memory = torch.cuda.max_memory_allocated()/1024**2
    gpu = torch.cuda.get_device_name()
    del model, frames, state
    torch.cuda.empty_cache()

    render_started = time.perf_counter()
    values = render_profile(raw, images, back, forward, transition, keys)
    render_seconds = time.perf_counter()-render_started
    save_started = time.perf_counter()
    for name, value in values.items():
        np.save(receive_dir/(name+'.npy'), value, allow_pickle=False)
    save_seconds = time.perf_counter()-save_started
    hashes = {path.name: file_sha(path) for path in receive_dir.glob('*.npy')}
    estimate_total = decode_seconds+model_load_seconds+flow_seconds+render_seconds+save_seconds
    save_json(receive_dir/'receipt.json', dict(complete=True, profile=profile, keys=keys,
        iterations=iterations, frames=150, fps=15, seconds=10, base_sha256=base_digest,
        base_bytes=base_bytes, auxiliary_sha256=file_sha(GUIDE_PACKET),
        auxiliary_bytes=len(packet), auxiliary_kbps=len(packet)*8/10000,
        source_reads_blocked=True, reads=sorted(observed), blocked_inputs=blocked,
        flow_pairs=inferred, flow_seconds=flow_seconds, guide_decode_seconds=decode_seconds,
        model_load_and_device_seconds=model_load_seconds, render_seconds=render_seconds,
        output_save_seconds=save_seconds, measured_pipeline_seconds=estimate_total,
        process_wall_seconds=time.perf_counter()-process_started,
        peak_allocated_mib=peak_memory, gpu=gpu, torch=str(torch.__version__),
        zero_weight_unchanged=True, outputs_sha256=hashes,
        receiver_script_sha256=file_sha(__file__), renderer_logic='fixed S8I T3; parameterized anchor keys only',
        timing_note='Single cold worker per profile. Component sum excludes caller/UI startup; includes guide decode, model load, RAFT, render and result arrays.'))
    print(profile, 'complete; flow', round(flow_seconds, 2), 'render', round(render_seconds, 2),
        'pipeline', round(estimate_total, 2), 'pairs', inferred, flush=True)


def aggregate(run, analysis_name):
    from compare_deflicker import save_csv
    from evaluate_guide_compression import motion_change_diagnostic
    import evaluate_fullframe_reference as scoring
    from run_fullframe_factorial import BOX
    from run_p0 import preview
    from verify_video1_persistent_structure import inspect_preview

    base = BASE
    raw = np.load(base/'base_receive/decoded_rgb.npy', mmap_mode='r')
    source = np.load(base/'source.npy', mmap_mode='r')
    accepted = np.load(ACCEPTED/'rgb.npy', mmap_mode='r')
    sequences = dict(source=source, raw=raw, S8I=accepted)
    alphas = {}
    for profile in PROFILES:
        folder = run/profile
        receipt = load_json(folder/'receive/receipt.json')
        assert receipt['complete'] and receipt['source_reads_blocked'] and len(receipt['blocked_inputs']) == 7
        assert receipt['base_sha256'] == file_sha(base/'base.glcsnd')
        assert receipt['auxiliary_sha256'] == file_sha(GUIDE_PACKET)
        for name, digest in receipt['outputs_sha256'].items():
            assert file_sha(folder/'receive'/name) == digest
        sequences[profile] = np.load(folder/'receive/rgb.npy', mmap_mode='r')
        alphas[profile] = np.load(folder/'receive/alpha.npy', mmap_mode='r')
    scoring.SCOPES['coral'] = BOX
    scores, temporal, spatial = scoring.score(sequences)
    archive = load_json(DATA/'guide_compression_20261111_01/evaluation/initial/summary.json')
    for scope in ('full', 'coral', 'top_left', 'top_right', 'bottom_left', 'bottom_right'):
        assert scores[scope]['S8I'] == archive['scores'][scope]['S8I'], (scope, scores[scope]['S8I'])
    analysis = run/analysis_name
    analysis.mkdir(exist_ok=False)
    save_csv(analysis/'temporal.csv', temporal)
    save_csv(analysis/'spatial.csv', spatial)
    dynamic, dynamic_rows = motion_change_diagnostic(sequences)
    save_csv(analysis/'source_change.csv', dynamic_rows)

    rows = {}
    for profile, spec in PROFILES.items():
        receipt = load_json(run/profile/'receive/receipt.json')
        windows = {key+offset for key in spec['keys'][1:] for offset in range(7)}
        intra = {64, 128}
        rows[profile] = dict(keys=spec['keys'], iterations=spec['iterations'],
            flow_pairs=receipt['flow_pairs'], times={k: receipt[k] for k in
                ('guide_decode_seconds', 'model_load_and_device_seconds', 'flow_seconds',
                 'render_seconds', 'output_save_seconds', 'measured_pipeline_seconds',
                 'process_wall_seconds')},
            scores={scope: scores[scope][profile] for scope in ('full', 'coral')},
            quadrant_MAE_change_percent={scope:(scores[scope][profile]['coarse_mae']/scores[scope]['S8I']['coarse_mae']-1)*100
                for scope in ('top_left', 'top_right', 'bottom_left', 'bottom_right')},
            source_change=dynamic[profile],
            coverage_full=float((alphas[profile]>0).mean()),
            coverage_coral=float((alphas[profile][:, BOX[1]:BOX[3], BOX[0]:BOX[2]]>0).mean()),
            timing_groups={})
        for scope in ('full', 'coral'):
            values = [r for r in temporal if r['method'] == profile and r['scope'] == scope]
            for label, mask in (('anchor_transition_exit', lambda i: i in windows),
                                ('outside_anchor_windows', lambda i: i not in windows),
                                ('base_I_frames', lambda i: i in intra)):
                selected = [r for r in values if mask(int(r['frame']))]
                pixels = sum(r['pixels'] for r in selected)
                rows[profile]['timing_groups'].setdefault(scope, {})[label] = dict(
                    pairs=len(selected), pixels=pixels,
                    coarse_MAE=sum(r['coarse_sum'] for r in selected)/pixels)
        peaks = sorted([r for r in temporal if r['method'] == profile and r['scope'] == 'full'],
            key=lambda r: r['coarse_sum']/r['pixels'], reverse=True)[:8]
        rows[profile]['largest_full_error_pairs'] = [dict(frame=int(r['frame']),
            seconds=int(r['frame'])/15, coarse_MAE=r['coarse_sum']/r['pixels'],
            base_I_frame=int(r['frame']) in intra,
            anchor_window=int(r['frame']) in windows) for r in peaks]

    save_json(analysis/'evaluation_summary.json', dict(complete=True, comparison='Same base, same 8865-byte S8I guide and T3 gate; anchor schedule x RAFT iterations.',
        reference_scores_exact=True, profiles=rows, original_S8I_scores=archive['scores'],
        original_S8I_runtime={k:load_json(ACCEPTED.parent/'flow_receive/receipt.json')[k]
            for k in ('flow_seconds', 'render_seconds', 'guide_decode_seconds')},
        original_S8I_full_pipeline='Not measured as one cold process in the prior run.',
        source_change_diagnostic=dynamic, receiver_rule='Only anchor schedule and RAFT iterations vary; gate, blend, ramp, raw RGB and packet fixed.',
        caveats=['Single clip and one cold timing per profile.', 'No realtime or long-video claim.',
            'I-frame temporal association is diagnostic, not causal proof.']))

    preview_dir = analysis/'previews'
    preview_dir.mkdir(exist_ok=False)
    panels_base = dict(S8I=accepted)
    labels_base = dict(S8I='Accepted S8I / aux 7.09 kbps')
    ffmpeg = FFMPEG
    checks = {}
    for profile in ('ISYNC_I24', 'CURRENT_I12', 'ISYNC_I12'):
        candidate = sequences[profile]
        spec = PROFILES[profile]
        labels = dict(labels_base, **{profile:f'{profile}: keys {spec["keys"]}, RAFT {spec["iterations"]}'})
        for suffix, crop in (('full', None), ('coral', (250, 0, 512, 132))):
            name = f'S8I_vs_{profile}_{suffix}.mp4'
            preview_crop = None if crop is None else np.s_[crop[1]:crop[3], crop[0]:crop[2]]
            preview(ffmpeg, preview_dir/name, [panels_base['S8I'], candidate], 15, 0,
                [labels['S8I'], labels[profile]], crop=preview_crop)
            checks[name] = inspect_preview(ffmpeg, preview_dir/name,
                [panels_base['S8I'], candidate], crop=crop)
    save_json(preview_dir/'verification.json', dict(complete=True, profiles=['ISYNC_I24','CURRENT_I12','ISYNC_I12'],
        checks=checks, frames=150, fps=15, seconds=10, lossless_RGB_scored=True,
        MP4_viewing_only=True, panel_order='left: accepted S8I; right: candidate'))


def run(workspace, run_dir):
    configure(workspace, run_dir)
    RUN.mkdir(parents=True, exist_ok=False)
    inputs = [BASE/'base.glcsnd', BASE/'base_receive/decoded_rgb.npy', BASE/'base_receive/result.json',
        GUIDE_PACKET, ACCEPTED/'rgb.npy', ACCEPTED/'alpha.npy', ACCEPTED/'shared.npy',
        ACCEPTED/'receipt.json', ACCEPTED.parent/'flow_receive/receipt.json',
        DATA/'source_dense_flow_three_clips_20261109_01/video1/source.npy',
        UPSTREAM/'pretrained_weights/raft-things.pth']
    code_files = ('run_anchor_time_experiment.py', 'receiver_structure.py', 'receive_guide_texture.py',
        'structure_guide_stream.py', 'evaluate_guide_compression.py', 'evaluate_fullframe_reference.py')
    plan = dict(date='2026-11-11', frozen_before_any_candidate_run=True,
        authorization='User accepted S8I, identified remaining jumps, and asked to start the next round with more attention to processing time.',
        base='Immutable independently decoded GLC base; no re-encoding.',
        auxiliary='Accepted S8I packet fixed at 8865 B / 7.0920 kbps; no new transmission or sender signal.',
        scene='video1, 150 frames, 15 FPS, 512x288, 10 seconds.',
        receiver_fixed='T3 texture std >=3, six-frame overlap ramp, 2px feather, two rounding stages, historical raw GLC RGB texture.',
        matrix=PROFILES, anchor_logic=dict(current=list(KEYS), I_synchronized=[0,64,128],
            note='I-sync schedule tests fewer refreshes at GLC intra frames; it is not a globally fixed single anchor.'),
        iterations='24 accepted baseline vs 12 lower-compute candidate; no quality-based rate selection.',
        measures=['exact replay of CURRENT_I24', 'lossless source-matched full/coral/quadrant metrics',
            'separate anchor-window and GLC I-frame score groups', 'cold per-worker decode/model/RAFT/render/save/wall times',
            'original-speed full-frame and coral comparisons'],
        limitations=['One 10-second clip; no realtime claim.', 'A fixed single-anchor candidate is not repeated because the earlier fixed-frame-0 control aged out; current question is fewer synchronized refreshes.',
            'Temporal grouping is association and cannot prove event causality.'],
        inputs_sha256={str(p):file_sha(p) for p in inputs},
        code_sha256={n:file_sha(ROOT/n) for n in code_files})
    save_json(RUN/'execution_plan.json', plan)
    save_json(RUN/'profile_order.json', dict(order=['CURRENT_I24','ISYNC_I24','CURRENT_I12','ISYNC_I12'],
        frozen_before_runs=True, quality_not_used_for_order=True))
    for profile in PROFILES:
        command = [sys.executable, '-B', '-u', '-X', 'utf8', str(ROOT/'run_anchor_time_experiment.py'),
            '--worker', profile, '--run-dir', str(RUN)]
        subprocess.run(command, cwd=ROOT, check=True)
    aggregate(RUN, 'analysis')
    print('ANCHOR/TIME EXPERIMENT COMPLETE', RUN, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--worker', choices=tuple(PROFILES))
    parser.add_argument('--aggregate-existing', action='store_true')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    if args.worker:
        if not args.run_dir:
            parser.error('--run-dir is required for a worker')
        worker(args.worker, args.run_dir.resolve())
    elif args.aggregate_existing:
        if not args.run_dir:
            parser.error('--run-dir is required for aggregation')
        aggregate(args.run_dir.resolve(), 'analysis_rerun01')
    else:
        if not args.workspace or not args.run_dir:
            parser.error('--workspace and --run-dir are required for the parent run')
        run(args.workspace.resolve(), args.run_dir.resolve())
