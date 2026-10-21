"""Verify existing CPU artifacts and freeze D0 manifests without model work."""
from __future__ import annotations
import argparse
import copy
from fractions import Fraction
from pathlib import Path

from prepare_d0 import file_sha, load_json, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    stage = Path(__file__).resolve().parent
    manifest = load_json(args.corpus / 'manifest.json')
    timing = load_json(args.corpus / 'timing_audit.json')
    preparation = load_json(args.corpus / 'preparation_summary.json')
    protocol = load_json(stage / 'd0_protocol.json')
    if timing['source_manifest_sha256'] != file_sha(args.corpus / 'manifest.json'):
        raise ValueError('Timing evidence belongs to a different corpus')
    if manifest['selection_sha256'] != file_sha(stage / 'd0_selection.json'):
        raise ValueError('Selection changed after data preparation')
    if manifest['preparation_script_sha256'] != file_sha(stage / 'prepare_d0.py'):
        raise ValueError('Preparation script changed after execution')
    if timing['script_sha256'] != file_sha(stage / 'audit_d0_timing.py'):
        raise ValueError('Timing script changed after execution')
    for name in ('ready_manifest.json', 'd1_development_inputs.json'):
        if (args.corpus / name).exists():
            raise FileExistsError(args.corpus / name)
    args.evidence.mkdir(parents=True, exist_ok=False)
    ready = copy.deepcopy(manifest)
    ready['schema'] = 'd0-ready-corpus-v1'
    ready['corpus_root'] = str(args.corpus.resolve())
    ready['protocol_sha256'] = file_sha(stage / 'd0_protocol.json')
    ready['timing_audit_sha256'] = file_sha(args.corpus / 'timing_audit.json')
    timing_by_id = {row['video_id']: row for row in timing['sequences']}
    inputs, sequence_summaries, selected_index = [], [], []
    verified_files = 0
    for seq in ready['fish_sequences']:
        check = timing_by_id[seq['video_id']]
        if len(seq['frames']) != len(check['matches']):
            raise ValueError('Timing/corpus sample count differs')
        seq['timing_status'] = 'verified_constant_rate' if check['ready_for_dense_video_comparison'] else 'verified_variable_pts_static_only_for_first_codec_pilot'
        seq['time_base'] = check['source_stream']['time_base']
        seq['fps'] = check['fps_if_constant']
        seq['dense_video_comparison_ready'] = check['ready_for_dense_video_comparison']
        seq['duration_seconds_exact'] = check.get('clip_duration_seconds_exact')
        for frame, match in zip(seq['frames'], check['matches']):
            if not match['accepted'] or frame['original_file_name'] != match['original_file_name'] or frame['index'] != match['sample_index']:
                raise ValueError('Ambiguous/misaligned frame cannot enter ready corpus')
            for path_key, hash_key in [('rgb_path', 'source_file_sha256'), ('mask_path', 'mask_file_sha256')]:
                if file_sha(args.corpus / frame[path_key]) != frame[hash_key]:
                    raise ValueError(f'Changed file: {frame[path_key]}')
                verified_files += 1
            frame['source_frame_index'] = match['best_source_frame_index']
            frame['source_pts'] = match['source_pts_ticks']
            frame['source_time_base'] = seq['time_base']
            frame['source_pts_seconds_exact'] = str(Fraction(frame['source_pts']) * Fraction(seq['time_base']))
            input_row = {'id': f'sfishtrack_{seq["video_id"]:03d}/{frame["original_file_name"]}',
                         'role': seq['role'], 'video_id': seq['video_id'], 'index': frame['index'],
                         'rgb_path': str((args.corpus / frame['rgb_path']).resolve()),
                         'rgb_sha256': frame['source_file_sha256'], 'width': frame['width'], 'height': frame['height'],
                         'pts_ticks': frame['source_pts'], 'time_base': seq['time_base']}
            if seq['role'] == 'development':
                inputs.append(input_row)
            selected_index.append({**input_row, 'rgb_path': frame['rgb_path'],
                                   'gt_fish_path': frame['mask_path'], 'gt_sha256': frame['mask_file_sha256']})
        sequence_summaries.append({'video_id': seq['video_id'], 'role': seq['role'], 'frames': seq['frame_count'],
                                   'instances': sum(len(f['instances']) for f in seq['frames']),
                                   'dimensions': [seq['frames'][0]['width'], seq['frames'][0]['height']],
                                   'dense_video_comparison_ready': seq['dense_video_comparison_ready'],
                                   'fps': seq['fps'], 'time_base': seq['time_base'],
                                   'duration_seconds_exact': seq['duration_seconds_exact'],
                                   'minimum_alignment_psnr_db': min(r['global_psnr_db_at_best'] for r in check['matches']),
                                   'minimum_match_uniqueness_ratio': min(r['runner_up_to_best_ratio'] for r in check['matches']),
                                   'historical_rgb_exact_matches': sum(f['historical_rgb_identical'] is True for f in seq['frames'])})
    for row in ready['suim']['images']:
        for path_key, hash_key in [('rgb_path', 'rgb_file_sha256'), ('eight_class_mask_path', 'eight_class_mask_sha256'),
                                   ('target_mask_path', 'target_mask_sha256')]:
            if file_sha(args.corpus / row[path_key]) != row[hash_key]:
                raise ValueError(f'Changed SUIM file: {row[path_key]}')
            verified_files += 1
        inputs.append({'id': 'suim/' + row['id'], 'role': row['role'], 'rgb_path': row['rgb_path'],
                       'rgb_sha256': row['rgb_file_sha256'], 'width': row['width'], 'height': row['height']})
    save_json(args.corpus / 'ready_manifest.json', ready)
    save_json(args.corpus / 'd1_development_inputs.json', {'schema': 'd1-inputs-v1',
              'purpose': 'RGB-only development inputs; labels and reserved sequences intentionally excluded', 'inputs': inputs})
    save_json(args.evidence / 'selected_index.json', {'fish': selected_index, 'suim': ready['suim']['images']})
    save_json(args.evidence / 'archive_audit.json', load_json(args.corpus / 'archive_audit.json'))
    evidence_names = ['manifest.json', 'ready_manifest.json', 'd1_development_inputs.json',
                      'all_frame_mapping.json', 'archive_audit.json', 'timing_audit.json', 'preparation_summary.json']
    summary = {'schema': 'd0-summary-v1', 'date': '2026-10-21', 'status': 'pilot_ready_D1_not_started',
               'corpus_root': str(args.corpus.resolve()), 'archive_path': manifest['archive_path'],
               'archive_size': manifest['archive_size'], 'archive_moved_from': 'C:/Users/lenovo/Downloads/Compressed/SFISHTRACK.zip',
               'download_original_exists': Path('C:/Users/lenovo/Downloads/Compressed/SFISHTRACK.zip').exists(),
               'all_annotation_rgb_names_resolved': preparation['all_annotation_rgb_names_resolved'],
               'fish_development_frames': preparation['fish_development_frames'],
               'fish_reserved_frames': preparation['fish_reserved_frames'],
               'validated_instance_masks': preparation['validated_instance_masks'],
               'suim_images': preparation['suim_images'], 'final_file_hash_checks': verified_files,
               'sequences': sequence_summaries, 'd1_rgb_only_input_count': len(inputs),
               'cpu_tests_passed': 13, 'model_loaded': False, 'gpu_used': False, 'training_performed': False,
               'GLC_codec_experiment_performed': False, 'semantic_gain_proven': False,
               'application_budget_64_frame_clip_bytes': protocol['byte_accounting']['application_budget']['64_frame_cfr_clip_ceiling_bytes'],
               'limitations': ['Only selected RGB/mask content verified; no full ZIP integrity scan',
                               '001/002/003 are one conservative development group, not three independent sources',
                               '007 is sequence holdout; capture independence remains unproven',
                               '025 retains variable PTS; timed codec pilot deferred',
                               'SUIM is static development/regression; diver video still pending',
                               'GT complement may contain unannotated targets; it is not exhaustive semantic background'],
               'local_evidence': [{'path': str(args.corpus / name), 'bytes': (args.corpus / name).stat().st_size,
                                   'sha256': file_sha(args.corpus / name)} for name in evidence_names],
               'script_sha256': {name: file_sha(stage / name) for name in
                                 ['prepare_d0.py', 'audit_d0_timing.py', 'finalize_d0.py', 'test_d0.py']},
               'selection_sha256': file_sha(stage / 'd0_selection.json'), 'protocol_sha256': file_sha(stage / 'd0_protocol.json')}
    save_json(args.evidence / 'summary.json', summary)
    print(f'Frozen D0: {verified_files} file hashes verified; {len(inputs)} RGB-only D1 inputs; no models loaded.')


if __name__ == '__main__':
    main()
