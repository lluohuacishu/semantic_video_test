"""Verify and archive the completed S8I anchor/receiver-time experiment."""

import argparse
import importlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from receiver_structure import file_sha, load_json, save_json


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
WORKSPACE = REPO.parents[2]
RUN = WORKSPACE/'02_实验数据/优化_GLC后处理/anchor_time_20261111_02'
EVIDENCE = ROOT/'evidence_20261111/anchor_time'
VIEW = WORKSPACE/'05_成果与观看/GLC_接收时间与锚点实验_20261111'
PROFILES = ('CURRENT_I24', 'ISYNC_I24', 'CURRENT_I12', 'ISYNC_I12')
PREVIEWS = (
    'S8I_vs_ISYNC_I24_full.mp4', 'S8I_vs_ISYNC_I24_coral.mp4',
    'S8I_vs_CURRENT_I12_full.mp4', 'S8I_vs_CURRENT_I12_coral.mp4',
    'S8I_vs_ISYNC_I12_full.mp4', 'S8I_vs_ISYNC_I12_coral.mp4',
)


def copy_verified(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = file_sha(source)
    if destination.exists():
        assert file_sha(destination) == digest
    else:
        shutil.copy2(source, destination)
        assert file_sha(destination) == digest
    return dict(source=str(source), bytes=destination.stat().st_size, sha256=digest)


def release(workspace, run):
    global WORKSPACE, RUN, EVIDENCE, VIEW
    WORKSPACE = workspace
    RUN = run
    ROOT_PATH = WORKSPACE/'01_项目/代码/semantic_video_test/21_优化阶段_GLC解码后处理'
    EVIDENCE = ROOT_PATH/'evidence_20261111/anchor_time'
    VIEW = WORKSPACE/'05_成果与观看/GLC_接收时间与锚点实验_20261111'
    if any(path.exists() for path in (EVIDENCE, VIEW, RUN/'source_final')):
        assert EVIDENCE.exists() and VIEW.exists() and (RUN/'source_final').exists()

    plan = load_json(RUN/'execution_plan.json')
    assert all(file_sha(Path(path)) == digest for path, digest in plan['inputs_sha256'].items())
    result = load_json(RUN/'analysis_rerun01/evaluation_summary.json')
    assert result['complete'] and result['reference_scores_exact']
    accepted = RUN.parent/'guide_compression_20261111_01/video1/S8I/receive'
    exact = {}
    for name in ('rgb', 'alpha', 'shared'):
        import numpy as np
        left = np.load(RUN/'CURRENT_I24/receive'/f'{name}.npy', mmap_mode='r')
        right = np.load(accepted/f'{name}.npy', mmap_mode='r')
        exact[name] = bool(np.array_equal(left, right))
    assert all(exact.values())
    for profile in PROFILES:
        receipt = load_json(RUN/profile/'receive/receipt.json')
        assert receipt['complete'] and receipt['source_reads_blocked']
        assert len(receipt['blocked_inputs']) == 7
        assert receipt['auxiliary_bytes'] == 8865
        assert receipt['receiver_script_sha256'] == plan['code_sha256']['run_anchor_time_experiment.py']
        for filename, digest in receipt['outputs_sha256'].items():
            assert file_sha(RUN/profile/'receive'/filename) == digest
    checks = load_json(RUN/'analysis_rerun01/previews/verification.json')
    assert checks['complete'] and len(checks['checks']) == 6
    copies = {}
    VIEW.mkdir(parents=True, exist_ok=True)
    labels = {
        'S8I_vs_ISYNC_I24_full.mp4': '01_S8I对I帧稀疏锚_I24_全画面.mp4',
        'S8I_vs_ISYNC_I24_coral.mp4': '02_S8I对I帧稀疏锚_I24_珊瑚放大.mp4',
        'S8I_vs_CURRENT_I12_full.mp4': '03_S8I对当前锚I12_全画面.mp4',
        'S8I_vs_CURRENT_I12_coral.mp4': '04_S8I对当前锚I12_珊瑚放大.mp4',
        'S8I_vs_ISYNC_I12_full.mp4': '05_S8I对I帧稀疏锚_I12_全画面.mp4',
        'S8I_vs_ISYNC_I12_coral.mp4': '06_S8I对I帧稀疏锚_I12_珊瑚放大.mp4',
    }
    for source_name in PREVIEWS:
        source = RUN/'analysis_rerun01/previews'/source_name
        assert file_sha(source) == checks['checks'][source_name]['sha256']
        destination = VIEW/labels[source_name]
        copies[destination.name] = copy_verified(source, destination)
    save_json(VIEW/'来源核验.json', dict(complete=True, files=copies,
        panel_order='left: accepted S8I; right: candidate', frames=150, fps=15, seconds=10))

    save_json(RUN/'replay_verification.json', dict(complete=True,
        current_anchor_24_rgb_alpha_shared_bit_exact_to_accepted_S8I=exact,
        archived_S8I_scores_exact=result['reference_scores_exact'],
        same_base_and_same_8865_byte_packet=True, frozen_inputs_unchanged=True,
        all_receivers_source_free=True, preview_frame_rate_duration_and_panels_verified=True))
    save_json(RUN/'decision.json', dict(complete=True,
        accepted_reference='S8I, unchanged 8865-byte sidecar',
        primary_timing_candidate='CURRENT_I12, current anchor schedule, RAFT 12 iterations',
        primary_candidate_visual_acceptance='pending user original-speed review',
        sparse_I_synchronized_anchor='rejected by numeric regression; not promoted',
        bitrate_changed=False, receiver_default_changed=False,
        next='Review CURRENT_I12 at original speed; if accepted, focus subsequent runtime work on RAFT and T3 CPU gate.'))
    save_json(RUN/'visual_review.json', dict(date='2026-11-11',
        accepted='S8I, previously reviewed by user',
        pending=['CURRENT_I12', 'ISYNC_I24', 'ISYNC_I12'],
        primary_question='Does CURRENT_I12 retain the accepted S8I appearance while reducing processing time?',
        no_candidate_promoted=True))
    failed = WORKSPACE/'02_实验数据/优化_GLC后处理/anchor_time_20261111_01'
    if failed.exists():
        assert not (failed/'CURRENT_I24').exists()
        save_json(failed/'incomplete_attempt.json', dict(complete=False,
            stage='Before any candidate receiver started',
            cause='Worker read the frozen profile matrix using the wrong key; no optical flow/render output was created.',
            superseded_by=str(RUN), contents_preserved=True))

    source_snapshot = RUN/'source_final'
    source_snapshot.mkdir(exist_ok=True)
    for name in ('run_anchor_time_experiment.py', 'receiver_structure.py', 'receive_guide_texture.py',
        'structure_guide_stream.py', 'evaluate_guide_compression.py', 'evaluate_fullframe_reference.py',
        'run_fullframe_factorial.py', 'run_p0.py', 'verify_video1_persistent_structure.py',
        'compare_deflicker.py', 'audit_deflicker_texture.py', 'run_fullframe_reference.py',
        'probe_reef.py', 'temporal_filter.py'):
        source = ROOT/name if (ROOT/name).exists() else ROOT.parent/'11_优化阶段_纹理稳定性'/name
        if source.exists():
            copy_verified(source, source_snapshot/'repository'/source.relative_to(REPO))
    upstream = RUN.parent/'deflicker_reef80_20261108_01/upstream'
    for source in sorted((upstream/'src/models/stage_1/core').rglob('*.py')):
        copy_verified(source, source_snapshot/'upstream'/source.relative_to(upstream))
    for rel in ('src/__init__.py', 'src/models/__init__.py', 'src/models/stage_1/__init__.py'):
        if (upstream/rel).exists():
            copy_verified(upstream/rel, source_snapshot/'upstream'/rel)
    import cv2
    import numpy
    import torch
    ffmpeg = WORKSPACE/'01_项目/运行环境/FFmpeg/ffmpeg.exe'
    ffmpeg_version = subprocess.run([str(ffmpeg), '-version'], capture_output=True,
        text=True, check=True).stdout.splitlines()[0]
    save_json(source_snapshot/'environment.json', dict(python=sys.version,
        executable=sys.executable, platform=platform.platform(), numpy=numpy.__version__,
        opencv=cv2.__version__, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), ffmpeg=ffmpeg_version, ffmpeg_sha256=file_sha(ffmpeg),
        note='GLC base assets, RAFT weights and runtime environment are external workspace dependencies.'))
    frozen_script_sha = plan['code_sha256']['run_anchor_time_experiment.py']
    current_script_sha = file_sha(ROOT/'run_anchor_time_experiment.py')
    assert all(load_json(RUN/p/'receive/receipt.json')['receiver_script_sha256'] == frozen_script_sha
        for p in PROFILES)
    save_json(source_snapshot/'executed_code_note.json', dict(
        worker_code_sha256_at_inference=frozen_script_sha,
        current_archived_script_sha256=current_script_sha,
        worker_hash_matches_every_receipt=True,
        post_inference_script_changes=['Separate aggregation output folder for retry',
            'Correct preview crop tuple passed to preview inspection',
            'Read-only aggregate-existing CLI option'],
        inference_worker_logic_changed_after_run=False))
    manifest = {p.relative_to(source_snapshot).as_posix(): file_sha(p)
        for p in source_snapshot.rglob('*') if p.is_file()}
    save_json(source_snapshot/'manifest.json', manifest)

    flow_outputs = {}
    for profile in PROFILES:
        folder = RUN/profile/'flow'
        flow_outputs[profile] = {p.name: dict(bytes=p.stat().st_size, sha256=file_sha(p))
            for p in sorted(folder.glob('*.npy'))}
        assert set(flow_outputs[profile]) == {'to_anchor.npy', 'from_anchor.npy', 'transition_flow.npy'}
    save_json(RUN/'flow_outputs.json', dict(complete=True,
        note='Receiver-local flow caches; hashes were calculated after render without modifying the arrays.',
        profiles=flow_outputs))

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    small_files = [RUN/'execution_plan.json', RUN/'profile_order.json', RUN/'analysis_rerun01/evaluation_summary.json',
        RUN/'analysis_rerun01/temporal.csv', RUN/'analysis_rerun01/spatial.csv',
        RUN/'analysis_rerun01/source_change.csv', RUN/'analysis_rerun01/previews/verification.json',
        RUN/'decision.json', RUN/'visual_review.json', RUN/'replay_verification.json', RUN/'flow_outputs.json']
    for profile in PROFILES:
        small_files.append(RUN/profile/'receive/receipt.json')
    for source in small_files:
        copy_verified(source, EVIDENCE/source.relative_to(RUN))
    copy_verified(ROOT/'run_anchor_time_experiment.py',
        EVIDENCE/'source/run_anchor_time_experiment.py')
    copy_verified(ROOT/'release_anchor_time_experiment.py',
        EVIDENCE/'source/release_anchor_time_experiment.py')
    copy_verified(ROOT/'锚点与接收时间联合实验_20261111.md',
        EVIDENCE/'锚点与接收时间联合实验_20261111.md')
    copy_verified(ROOT/'evidence_20261111/anchor_time/README.md', EVIDENCE/'README.md')
    copy_verified(source_snapshot/'manifest.json', EVIDENCE/'source/source_final_manifest.json')
    copy_verified(source_snapshot/'executed_code_note.json', EVIDENCE/'source/executed_code_note.json')
    copy_verified(source_snapshot/'environment.json', EVIDENCE/'source/environment.json')
    save_json(RUN/'release_verification.json', dict(complete=True,
        frozen_input_count=len(plan['inputs_sha256']), frozen_inputs_unchanged=True,
        baseline_RGB_alpha_shared_exact=True, receiver_output_hashes_valid=True,
        source_free_receivers=True, profile_count=len(PROFILES), copied_preview_count=len(copies),
        viewer_copies_exact=True, manifests_created=True, realtime_claim=False,
        current_I12_visual_acceptance='pending'))
    # Include the just-written release check in the compact evidence tree.
    copy_verified(RUN/'release_verification.json', EVIDENCE/'release_verification.json')
    evidence_manifest = {p.relative_to(EVIDENCE).as_posix(): file_sha(p)
        for p in EVIDENCE.rglob('*') if p.is_file()}
    save_json(EVIDENCE/'manifest.json', evidence_manifest)
    save_json(RUN/'viewing_copies.json', dict(complete=True, directory=str(VIEW), files=copies))
    print('ANCHOR/TIME RELEASE COMPLETE', VIEW, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    release(args.workspace.resolve(), args.run_dir.resolve())
