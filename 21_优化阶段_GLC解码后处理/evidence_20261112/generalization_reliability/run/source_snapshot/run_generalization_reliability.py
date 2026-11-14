"""Second fixed generalization round: modest gains without broad feather loss."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np

from runtime_speed_common import STAGE, REPO, DATA, FFMPEG, WEIGHT, file_sha, load_json, save_json, check_inputs
from receiver_structure import render, KEYS, RAMP
from cached_t3 import cached_texture_gate
from mild_reliability_gate import mild_gate
import structure_guide_stream as guide


PREVIOUS = DATA / 'generalization_factorial_20261112_01'
PROFILES = ('Q6_T3', 'Q6_R25', 'Q6_D25', 'Q9_T3', 'Q9_R25', 'Q9_D25')
CODE = ('run_generalization_reliability.py', 'mild_reliability_gate.py',
        'receiver_structure.py', 'cached_t3.py', 'receive_guide_texture.py',
        'runtime_speed_common.py', 'runtime_chain_worker.py', 'raft_inference_runtime.py',
        'structure_guide_stream.py', 'run_generalization_extra.py', 'run_complete_runtime_benchmark.py',
        'evaluate_fullframe_reference.py', 'evaluate_structure_guide.py')


def event(run, stage, **data):
    value = dict(stage=stage, **data)
    save_json(run / 'progress.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def prepare(run):
    if (run / 'execution_plan.json').exists():
        raise FileExistsError(run / 'execution_plan.json')
    assert load_json(run / 'user_feedback_20261112.json')['further_experiments_authorized']
    previous_plan = load_json(PREVIOUS / 'execution_plan.json')
    assert load_json(PREVIOUS / 'final_verification.json')['complete']
    records = {}
    for case, spec in previous_plan['cases'].items():
        records[case] = dict(spec, prior_root=str(PREVIOUS / case))
    extra = load_json(PREVIOUS / 'extra_scene/receiver_input_registry.json')
    for case, spec in extra.items():
        records[case] = dict(spec, prior_root=str(PREVIOUS / 'extra_scene' / case),
                             current=str(PREVIOUS / 'extra_scene' / case / 'Q6/receive/T3'))
    frozen = {str(STAGE / '当前采用配置.json'): file_sha(STAGE / '当前采用配置.json'),
              str(PREVIOUS / 'execution_plan.json'): file_sha(PREVIOUS / 'execution_plan.json'),
              str(PREVIOUS / 'final_verification.json'): file_sha(PREVIOUS / 'final_verification.json'),
              str(WEIGHT): file_sha(WEIGHT)}
    for case, spec in records.items():
        for name in ('source', 'base', 'raw', 'base_receipt'):
            frozen[spec[name]] = file_sha(spec[name])
        prior = Path(spec['prior_root'])
        for quality in ('Q6', 'Q12'):
            for name in ('guide.glcguide', 'send.json', 'receive/receipt.json'):
                path = prior / quality / name
                frozen[str(path)] = file_sha(path)
            for name in ('rgb', 'alpha', 'shared'):
                path = prior / quality / 'receive/T3' / (name + '.npy')
                frozen[str(path)] = file_sha(path)
        for name in ('guide_rgb', 'to_anchor', 'from_anchor', 'transition_flow'):
            path = prior / 'Q6/receive' / (name + '.npy')
            frozen[str(path)] = file_sha(path)
    plan = dict(date='2026-11-12', frozen_before_trials=True,
        authorization=load_json(run / 'user_feedback_20261112.json'),
        cases=records, new_profiles=PROFILES, reference_only='Q12_T3 from the prior completed round',
        newly_rendered_outputs=24, existing_reference_conditions=4,
        guide=dict(Q6='Existing internal 6kbps packet and hash-bound receiver flow',
                   Q9='New internal 9kbps packet, same x265 slower two-pass and 256x144/15fps; no search'),
        gates=dict(T3='Existing gate, 2px feather unchanged',
                   R25='Smooth risk near existing cycle/guide-photometric/flow-gradient cutoffs; reduce alpha by at most 25%',
                   D25='R25 plus raw decoded low-frequency RGB mismatch risk (12 to 32); combined maximum, not compounded attenuation'),
        thresholds=dict(cycle=[.75, 1.5], guide_low=[8, 16], guide_mid=[12, 24],
                        flow_gradient=[.375, .75], decoded_low=[12, 32], attenuation=.25),
        fixed=dict(anchors=list(KEYS), transition=list(RAMP), iterations=12, FP32=True, TF32=False,
                   no_per_clip_tuning=True, raw_references_only=True, no_source_masks=True,
                   fixed_support=True, two_stage_rounding=True),
        numeric_screen=dict(purpose='Prioritization, not a quality guarantee or user acceptance',
                            video2_full_mae_ratio_max=.98, other_full_and_reef_mae_ratio_max=1.05,
                            all_psnr_drop_max_db=.1, all_dynamic_error_ratio_max=1.03,
                            boundary='diagnostic only; no mandatory improvement',
                            selection='Passing profiles: prefer unchanged auxiliary bytes, then lowest video2 MAE'),
        hypotheses=['Moderate guide bitrate may trade off video2 gains against video3 regression.',
                    'Small confidence attenuation can soften unreliable reuse while preserving strong stable texture.',
                    'Raw decoded mismatch may identify large texture/shape replacement; it may also suppress useful corrections.'],
        scope='Four historical 10-second clips; development experiment, not independent holdout or long-video/realtime validation',
        input_sha256=frozen, code_sha256={n: file_sha(STAGE / n) for n in CODE},
        timing='Q6 reuses verified local flow; Q9 regenerates flow; no end-to-end speed comparison')
    save_json(run / 'execution_plan.json', plan)
    snap = run / 'source_snapshot'
    snap.mkdir()
    for name in CODE:
        shutil.copy2(STAGE / name, snap / name)
    event(run, 'prepared', cases=list(records), profiles=PROFILES, newly_rendered_outputs=24)


def encode(run):
    from run_generalization_extra import encode_packet
    plan = load_json(run / 'execution_plan.json')
    check_inputs(plan['input_sha256'])
    for case, spec in plan['cases'].items():
        root = run / case
        root.mkdir(exist_ok=False)
        low = root / 'Q6'
        low.mkdir()
        shutil.copy2(Path(spec['prior_root']) / 'Q6/guide.glcguide', low / 'guide.glcguide')
        previous_send = load_json(Path(spec['prior_root']) / 'Q6/send.json')
        save_json(low / 'send.json', dict(previous_send, reused=True, reused_from=str(Path(spec['prior_root']) / 'Q6'),
                                         timing_note='Historical encoding time, not rerun'))
        encode_packet(np.load(spec['source'], mmap_mode='r'), Path(spec['base']), root / 'Q9', 9)
        event(run, 'encoded', case=case, Q6_kbps=previous_send['total_kbps'],
              Q9_kbps=load_json(root / 'Q9/send.json')['total_kbps'])


def worker(run, case, quality):
    # Read metadata before the guard; source pixels, source-flow and prior output
    # files stay forbidden throughout receiver execution.
    plan = load_json(run / 'execution_plan.json')
    spec = plan['cases'][case]
    cache = Path(spec['prior_root']) / 'Q6/receive'
    root = run / case / quality
    out = root / 'receive'
    import torch
    from runtime_chain_worker import ReadGuard
    allowed = [spec['base'], spec['raw'], spec['base_receipt'], root / 'guide.glcguide']
    if quality == 'Q6':
        allowed += [cache / 'receipt.json'] + [cache / (n + '.npy') for n in ('guide_rgb', 'to_anchor', 'from_anchor', 'transition_flow')]
    guard = ReadGuard('mild-reliability')
    guard.authorize(out, allowed)
    blocked = []
    for path in [Path(spec['source']), root / 'send.json', Path(spec['prior_root']) / 'Q6/receive/T3/rgb.npy',
                 Path(spec['source']).parent / 'flow/to_anchor.npy', Path(spec['source']).parent / 'source_mask.npy']:
        try:
            path.read_bytes()
        except RuntimeError:
            blocked.append(str(path))
        else:
            raise ValueError('Receiver read boundary failed: ' + str(path))
    out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base_receipt = load_json(spec['base_receipt'])
    digest = file_sha(spec['base'])
    assert base_receipt['stream_sha256'] == digest and base_receipt['decoded_rgb_sha256'] == file_sha(spec['raw'])
    assert base_receipt['reference_verified'] and base_receipt['source_media_reads_forbidden']
    raw = np.load(spec['raw'], mmap_mode='r', allow_pickle=False)
    header, _, images = guide.decode((root / 'guide.glcguide').read_bytes(), digest, FFMPEG)
    if quality == 'Q6':
        assert file_sha(cache / 'receipt.json') == plan['input_sha256'][str(cache / 'receipt.json')]
        previous = load_json(cache / 'receipt.json')
        assert previous['base_sha256'] == digest and previous['auxiliary_sha256'] == file_sha(root / 'guide.glcguide')
        assert previous['source_reads_blocked'] and previous['no_source_pixels'] and previous['flow']['pairs'] == 174
        for name, h in previous['flow_sha256'].items():
            assert file_sha(cache / (name + '.npy')) == h
        assert np.array_equal(images, np.load(cache / 'guide_rgb.npy', mmap_mode='r'))
        flows = [np.load(cache / (name + '.npy'), mmap_mode='r') for name in ('to_anchor', 'from_anchor', 'transition_flow')]
        flow_paths = {n: str(cache / (n + '.npy')) for n in ('to_anchor', 'from_anchor', 'transition_flow')}
        flow_record = dict(reused_local_cache=True, provenance_receipt=str(cache / 'receipt.json'),
                           provenance_sha256=file_sha(cache / 'receipt.json'), pairs=174, seconds=None)
    else:
        from raft_inference_runtime import load_model, estimate
        model = load_model()
        flows, flow_record = estimate(model, images, 'feature_cache')
        flow_record['reused_local_cache'] = False
        flow_paths = {}
        for name, array in zip(('to_anchor', 'from_anchor', 'transition_flow'), flows):
            path = out / (name + '.npy')
            np.save(path, array, allow_pickle=False)
            flow_paths[name] = str(path)
        del model
        torch.cuda.empty_cache()
    np.save(out / 'guide_rgb.npy', images, allow_pickle=False)
    variants = {}
    for mode in ('T3', 'R25', 'D25'):
        folder = out / mode
        folder.mkdir()
        tick = time.perf_counter()
        gate = cached_texture_gate(images) if mode == 'T3' else mild_gate(images, raw, mode)
        values = render(raw, images, *flows, gate_fn=gate)
        elapsed = time.perf_counter() - tick
        for name, array in values.items():
            np.save(folder / (name + '.npy'), array, allow_pickle=False)
        variants[mode] = dict(outputs_sha256={n: file_sha(folder / (n + '.npy')) for n in values},
                              render_seconds=elapsed, active_fraction=float((values['alpha'] > 0).mean()),
                              mean_alpha=float(values['alpha'].mean()), diagnostics=getattr(gate, 'diagnostics', None))
        print('RENDERED', case, quality, mode, round(elapsed, 2), flush=True)
    save_json(out / 'receipt.json', dict(complete=True, case=case, quality=quality, base_sha256=digest,
        base_decode_verified=True, auxiliary_sha256=file_sha(root / 'guide.glcguide'), guide_header=header,
        source_reads_blocked=True, blocked=blocked, reads=sorted(guard.reads), allowed=[str(Path(p).resolve()) for p in allowed],
        flow=flow_record, flow_paths=flow_paths, flow_sha256={n: file_sha(p) for n, p in flow_paths.items()},
        guide_rgb_sha256=file_sha(out / 'guide_rgb.npy'), variants=variants,
        no_per_clip_parameters=True, no_source_pixels=True, code_sha256=file_sha(__file__),
        gate_code_sha256=file_sha(STAGE / 'mild_reliability_gate.py')))


def subprocess_action(run, action, case=None, quality=None):
    command = [sys.executable, '-B', '-u', '-X', 'utf8', str(Path(__file__).resolve()), action, '--run', str(run)]
    if case:
        command += ['--case', case]
    if quality:
        command += ['--quality', quality]
    log = run / (f'{action}_{case}_{quality}.log')
    with log.open('x', encoding='utf-8') as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True, cwd=REPO)


def receivers(run):
    plan = load_json(run / 'execution_plan.json')
    assert all(file_sha(STAGE / name) == digest for name, digest in plan['code_sha256'].items())
    for case, spec in plan['cases'].items():
        for quality in ('Q6', 'Q9'):
            subprocess_action(run, 'worker', case, quality)
            if quality == 'Q6':
                hashes = load_json(run / case / quality / 'receive/receipt.json')['variants']['T3']['outputs_sha256']
                for name, digest in hashes.items():
                    assert file_sha(Path(spec['prior_root']) / 'Q6/receive/T3' / (name + '.npy')) == digest
            event(run, 'received', case=case, quality=quality, baseline_exact=quality=='Q6')


def evaluate_case(run, case):
    import evaluate_fullframe_reference as scoring
    from evaluate_structure_guide import motion_change_diagnostic
    from compare_deflicker import save_csv
    plan = load_json(run / 'execution_plan.json')
    spec = plan['cases'][case]
    root = run / case
    out = root / 'evaluation'
    out.mkdir(exist_ok=False)
    seq = dict(source=np.load(spec['source'], mmap_mode='r'), raw=np.load(spec['raw'], mmap_mode='r'))
    rates, coverage = {}, {}
    for profile in PROFILES:
        q, mode = profile.split('_')
        folder = root / q / 'receive' / mode
        seq[profile] = np.load(folder / 'rgb.npy', mmap_mode='r')
        a = np.load(folder / 'alpha.npy', mmap_mode='r')
        rates[profile] = load_json(root / q / 'send.json')['total_kbps']
        coverage[profile] = dict(active_fraction=float((a > 0).mean()), mean_alpha=float(a.mean()))
    prior = Path(spec['prior_root'])
    seq['Q12_T3'] = np.load(prior / 'Q12/receive/T3/rgb.npy', mmap_mode='r')
    rates['Q12_T3'] = load_json(prior / 'Q12/send.json')['total_kbps']
    if 'reef' in spec:
        scoring.SCOPES['reef'] = tuple(spec['reef'])
    else:
        scoring.SCOPES.pop('reef', None)
    scores, temporal, spatial = scoring.score(seq)
    save_csv(out / 'temporal.csv', temporal)
    save_csv(out / 'spatial.csv', spatial)
    dynamic, rows = motion_change_diagnostic(seq)
    save_csv(out / 'dynamic.csv', rows)
    baseline_alpha = np.load(root / 'Q6/receive/T3/alpha.npy', mmap_mode='r')
    border = {name: 0. for name in seq}
    count, rows = 0, []
    kernel = np.ones((5, 5), np.uint8)
    for i in range(150):
        support = (baseline_alpha[i] > .5).astype(np.uint8)
        mask = (cv2.dilate(support, kernel) - cv2.erode(support, kernel)).astype(bool)
        pixels = int(mask.sum())
        count += pixels
        for name, values in seq.items():
            residual = values[i].astype(np.float32) - seq['source'][i].astype(np.float32)
            gradient = np.mean(np.abs(np.gradient(residual, axis=0)) + np.abs(np.gradient(residual, axis=1)), axis=2)
            total = float(gradient[mask].sum())
            border[name] += total
            rows.append(dict(frame=i, method=name, pixels=pixels, residual_gradient_sum=total))
    save_csv(out / 'boundary.csv', rows)
    p95, worst = {}, {}
    for name in seq:
        frames = [dict(frame=x['frame'], mae=x['coarse_sum']/x['pixels']) for x in temporal
                  if x['scope']=='full' and x['method']==name]
        p95[name] = float(np.percentile([x['mae'] for x in frames], 95))
        worst[name] = sorted(frames, key=lambda x:x['mae'], reverse=True)[:5]
    save_json(out / 'summary.json', dict(complete=True, case=case, scores=scores, rates=rates, coverage=coverage,
        dynamic=dynamic, boundary_residual_gradient={name: value/count for name, value in border.items()},
        boundary_pixels=count, full_frame_p95=p95, worst_frames=worst,
        boundary_note='Fixed baseline support residual-gradient proxy; diagnostic only'))
    print('EVALUATED', case, {n: round(s['coarse_mae'], 6) for n, s in scores['full'].items()}, flush=True)


def aggregate(run):
    plan = load_json(run / 'execution_plan.json')
    cases = {case:load_json(run / case / 'evaluation/summary.json') for case in plan['cases']}
    limits = plan['numeric_screen']
    screens = {}
    for name in PROFILES[1:]:
        checks = {}
        for case, data in cases.items():
            a, b = data['scores']['full']['Q6_T3'], data['scores']['full'][name]
            ratios = dict(full_mae_ratio=b['coarse_mae']/a['coarse_mae'],
                          psnr_change_db=b['psnr_db']-a['psnr_db'],
                          dynamic_error_ratio=data['dynamic'][name]['RGB_delta_error']/data['dynamic']['Q6_T3']['RGB_delta_error'],
                          boundary_ratio=data['boundary_residual_gradient'][name]/data['boundary_residual_gradient']['Q6_T3'])
            if 'reef' in data['scores']:
                ratios['reef_mae_ratio'] = data['scores']['reef'][name]['coarse_mae']/data['scores']['reef']['Q6_T3']['coarse_mae']
            full_limit = limits['video2_full_mae_ratio_max'] if case == 'video2' else limits['other_full_and_reef_mae_ratio_max']
            passed = ratios['full_mae_ratio'] <= full_limit
            if case != 'video2' and 'reef_mae_ratio' in ratios:
                passed &= ratios['reef_mae_ratio'] <= limits['other_full_and_reef_mae_ratio_max']
            passed &= ratios['psnr_change_db'] >= -limits['all_psnr_drop_max_db']
            passed &= ratios['dynamic_error_ratio'] <= limits['all_dynamic_error_ratio_max']
            ratios['pass'] = bool(passed)
            checks[case] = ratios
        screens[name] = dict(passed=all(x['pass'] for x in checks.values()), cases=checks)
    passing = [name for name, record in screens.items() if record['passed']]
    selected = min(passing, key=lambda n:(cases['video2']['rates'][n], cases['video2']['scores']['full'][n]['coarse_mae'])) if passing else None
    diagnostic = min(PROFILES[1:], key=lambda n:cases['video2']['scores']['full'][n]['coarse_mae'])
    check_inputs(plan['input_sha256'])
    save_json(run / 'summary.json', dict(complete=True, cases=cases, numeric_screen=screens,
        selected=selected, best_video2_diagnostic=diagnostic, automatic_adoption=False,
        user_visual_review='pending', profiles=PROFILES, source_free_receivers=True,
        scope=plan['scope'], objective='Modest video2 improvement; do not exaggerate perceptual severity'))
    event(run, 'evaluated_all', selected=selected, passing=passing, diagnostic=diagnostic)


def evaluate(run):
    plan = load_json(run / 'execution_plan.json')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(subprocess_action, run, 'evaluate-case', case) for case in plan['cases']]
        for case, future in zip(plan['cases'], futures):
            future.result()
            print('SCORED', case, flush=True)
    aggregate(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'encode', 'run', 'worker', 'evaluate', 'evaluate-case'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--case')
    parser.add_argument('--quality', choices=('Q6', 'Q9'))
    a = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    try:
        if a.action == 'prepare': prepare(a.run)
        elif a.action == 'encode': encode(a.run)
        elif a.action == 'run': receivers(a.run)
        elif a.action == 'worker': worker(a.run, a.case, a.quality)
        elif a.action == 'evaluate-case': evaluate_case(a.run, a.case)
        else: evaluate(a.run)
    except BaseException as error:
        save_json(a.run / f'failure_{a.action}_{a.case}_{a.quality}.json',
                  dict(error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
