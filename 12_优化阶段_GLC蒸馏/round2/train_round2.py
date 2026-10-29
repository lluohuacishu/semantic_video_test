"""Four finite paired cases isolate initialization, then RGB supervision loss."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot_common import configure_training, file_sha, load_json, model_sha, save_json, sha, check_stop
from train_pilot import atomic_checkpoint, open_cache, image_tensor, validate
from model_transfer import architecture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    plan = load_json(args.output / 'plan.json')
    plan_hash = file_sha(args.output / 'plan.json')
    if not plan['complete'] or plan_hash != load_json(args.output / 'plan_identity.json')['sha256']:
        raise ValueError('Incomplete or changed data/protocol plan')
    protocol = plan['training']
    case = protocol['cases'][args.case]
    initialization = case['initialization']
    init_path = args.output / 'initialization' / f'{initialization}.pt'
    audit = load_json(args.output / 'initialization/transfer_audit.json')
    if file_sha(init_path) != audit['initialization_files'][initialization] or not audit['disposable_step_probe']['candidate_initialization_files_unchanged']:
        raise ValueError('Initialization differs from inspected snapshot')
    configure_training(protocol['seed'])
    folder = args.output / f'train_{args.case}'
    if folder.exists() and not args.resume:
        raise FileExistsError('Existing case; use explicit --resume')
    folder.mkdir(exist_ok=args.resume)
    trains = [c for c in plan['clips'] if c['role'] == 'train']
    vals = [c for c in plan['clips'] if c['role'] == 'validation']
    cache = open_cache(args.output, trains + vals)
    sampling_path = args.output / 'sampling_pairs.json'
    sampling = load_json(sampling_path)
    if sampling['plan_sha256'] != plan_hash:
        raise ValueError('Sampling belongs to another data plan')
    pairs = sampling['pairs']
    allowed = {c['id']: c['frames'] for c in trains}
    if any(cid not in allowed or not 0 <= i < allowed[cid] or i % 64 == 0 for cid, i in pairs):
        raise ValueError('Sampling includes a non-training or I frame')
    sampling_manifest_sha = file_sha(sampling_path)
    rng = np.random.default_rng(protocol['seed'])
    samples = rng.integers(0, len(pairs), size=protocol['steps_each'], dtype=np.int64)
    sample_sha = sha(samples.tobytes())
    model = architecture().cuda().train()
    init = torch.load(init_path, map_location='cpu', weights_only=False)
    model.load_state_dict(init['model'], strict=True)
    del init
    initial_sha = model_sha(model)
    if initial_sha != audit[f'{initialization}_model_sha256']:
        raise ValueError('Loaded model differs from initialization audit')
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'], weight_decay=0)
    start_step, best_mse, previous_wall = 0, float('inf'), 0.0
    validations, durations = [], []
    if args.resume:
        state = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        if any(state[k] != v for k, v in [('plan_sha256', plan_hash), ('case', args.case), ('sampling_sha256', sample_sha), ('initial_model_sha256', initial_sha), ('sampling_manifest_sha256', sampling_manifest_sha)]):
            raise ValueError('Resume checkpoint protocol/case/init mismatch')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        start_step, best_mse, previous_wall = state['step'], state['best_mse'], state['wall_seconds']
        validations, durations = state['validations'], state['step_seconds']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    report = dict(case=args.case, initialization=initialization, loss=case['loss'], status='running',
                  plan_sha256=plan_hash, parameters=10985155, source_train_clips=len(trains), training_p_frames=len(pairs),
                  contributing_clips_after_dedup=len({cid for cid, i in pairs}),
                  initial_model_sha256=initial_sha, initialization_file_sha256=file_sha(init_path), sampling_sha256=sample_sha,
                  sampling_manifest_sha256=sampling_manifest_sha, candidate_p_frames_before_dedup=sampling['candidate_p_frames'],
                  validation_ids=[c['id'] for c in vals], final_holdout_used_for_selection=False,
                  requested_steps=protocol['steps_each'], completed_steps=start_step, precision=protocol['precision'])
    save_json(folder / 'summary.json', report)
    begin = time.perf_counter()
    def checkpoint(step, path):
        atomic_checkpoint(path, dict(case=args.case, step=step, plan_sha256=plan_hash, sampling_sha256=sample_sha,
                                    sampling_manifest_sha256=sampling_manifest_sha,
                                    initial_model_sha256=initial_sha, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                                    best_mse=best_mse, validations=validations, step_seconds=durations,
                                    wall_seconds=previous_wall+time.perf_counter()-begin))
    torch.cuda.reset_peak_memory_stats()
    if start_step == 0:
        validations.append(dict(step=0, **validate(model, cache, vals)))
        best_mse = validations[0]['mse_rgb']
        checkpoint(0, folder / 'initial.pt')
        checkpoint(0, folder / 'best.pt')
        print(f'{args.case} initial validation PSNR {validations[0]["psnr_db"]:.3f}dB', flush=True)
    steps = protocol['steps_each']
    log_path = folder / (f'loss_resume_{start_step}.jsonl' if args.resume else 'loss.jsonl')
    try:
        with log_path.open('x', encoding='utf-8', buffering=1) as stream:
            for step in range(start_step+1, steps+1):
                check_stop(args.output)
                tick = time.perf_counter()
                cid, index = pairs[int(samples[step-1])]
                arrays = cache[cid]
                latent = torch.from_numpy(np.array(arrays['latent'][index]))[None].cuda()
                original, teacher = image_tensor(arrays['source'][index]), image_tensor(arrays['teacher'][index])
                warmup = protocol['warmup_steps']
                if step <= warmup:
                    lr = protocol['learning_rate']*step/warmup
                else:
                    fraction = (step-warmup)/(steps-warmup)
                    lr = protocol['final_learning_rate'] + .5*(protocol['learning_rate']-protocol['final_learning_rate'])*(1+math.cos(math.pi*fraction))
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = model(latent)[:, :, :288, :512].float()
                    gt_l1, teacher_l1 = F.l1_loss(prediction, original), F.l1_loss(prediction, teacher)
                    loss = .5*(gt_l1+teacher_l1) if case['loss'] == 'mixed' else teacher_l1 if case['loss'] == 'teacher' else gt_l1
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                seconds = time.perf_counter()-tick
                durations.append(seconds)
                row = dict(step=step, clip=cid, frame_index=index, loss=float(loss), l1_original=float(gt_l1),
                           l1_teacher=float(teacher_l1), grad_norm_before_clip=float(norm), learning_rate=lr, seconds=seconds)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                if step % 100 == 0:
                    report.update(completed_steps=step, last_loss=float(loss), elapsed_wall_seconds=previous_wall+time.perf_counter()-begin)
                    save_json(folder / 'summary.json', report)
                    print(f'{args.case} {step}/{steps} loss={float(loss):.5f} wall={report["elapsed_wall_seconds"]:.1f}s', flush=True)
                if step % 500 == 0:
                    metric = dict(step=step, **validate(model, cache, vals))
                    validations.append(metric)
                    if metric['mse_rgb'] < best_mse:
                        best_mse = metric['mse_rgb']
                        checkpoint(step, folder / 'best.pt')
                    checkpoint(step, folder / 'last.pt')
                    save_json(folder / 'validation.json', validations)
                    print(f'{args.case} validation {step}: {metric["psnr_db"]:.3f}dB', flush=True)
        final_sha = model_sha(model)
        if final_sha == initial_sha:
            raise AssertionError('No parameter change')
        report.update(status='complete', completed_steps=steps, final_model_sha256=final_sha, parameters_changed=True,
                      best_validation=min(validations, key=lambda r: r['mse_rgb']),
                      total_wall_seconds=previous_wall+time.perf_counter()-begin,
                      mean_training_step_seconds=float(np.mean(durations)), median_training_step_seconds=float(np.median(durations)),
                      maximum_training_step_seconds=max(durations), steps_over_half_second=sum(t>.5 for t in durations),
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                      best_checkpoint_file_sha256=file_sha(folder / 'best.pt'))
        save_json(folder / 'summary.json', report)
        print(json.dumps(report), flush=True)
    except (KeyboardInterrupt, InterruptedError):
        checkpoint(len(durations), folder / 'last.pt')
        report.update(status='stopped', completed_steps=len(durations))
        save_json(folder / 'summary.json', report)
        raise


if __name__ == '__main__':
    main()
