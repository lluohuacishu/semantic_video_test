"""Finite real training: paired 5000-step KD / GT-only RGB generator trials."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from pilot_common import (check_stop, configure_training, file_sha, load_json,
                          model_sha, save_json, sha, student)


def atomic_checkpoint(path, value):
    part = path.with_name(path.name + '.tmp')
    torch.save(value, part)
    part.replace(path)


def open_cache(output, clips, include_teacher=True):
    result = {}
    for clip in clips:
        folder = output / clip['id']
        evidence = load_json(folder / 'decode_cache.json')
        if evidence['cache_replay_exact_matches'] != clip['frames']:
            raise ValueError('Cache teacher replay incomplete')
        names = ['latent', 'source'] + (['teacher'] if include_teacher else [])
        arrays = {}
        for name in names:
            path = folder / f'{name}.npy'
            digest = (clip['source_array_sha256'] if name == 'source' else evidence[f'{name}_array_sha256'])
            if file_sha(path) != digest:
                raise ValueError(f'Cache changed: {path}')
            arrays[name] = np.load(path, mmap_mode='r')
        result[clip['id']] = arrays
    return result


def image_tensor(value):
    return torch.from_numpy(np.array(value)).cuda().permute(2, 0, 1)[None].float().div(127.5).sub(1)


@torch.inference_mode()
def validate(model, cache, clips):
    was_training = model.training
    model.eval()
    errors = []
    l1 = []
    for clip in clips:
        arrays = cache[clip['id']]
        for i in range(clip['frames']):
            if i % 64 == 0:
                continue
            latent = torch.from_numpy(np.array(arrays['latent'][i]))[None].cuda()
            raw = model(latent)[0, :, :288, :512].float()
            rgb = raw.clamp(-1, 1).add(1).mul(127.5).round().permute(1, 2, 0).cpu().numpy()
            delta = rgb - arrays['source'][i].astype('float32')
            errors.append(float(np.mean(delta ** 2)))
            l1.append(float(np.mean(np.abs(delta))))
    if was_training:
        model.train()
    mse = float(np.mean(errors))
    return dict(p_frames=len(errors), mse_rgb=mse, psnr_db=10 * math.log10(255 ** 2 / max(mse, 1e-12)),
                mean_frame_psnr_db=float(np.mean(10 * np.log10(255 ** 2 / np.maximum(errors, 1e-12)))),
                l1_rgb=float(np.mean(l1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['kd', 'gt'], required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    plan_path = args.output / 'plan.json'
    plan = load_json(plan_path)
    plan_hash = file_sha(plan_path)
    if plan_hash != load_json(args.output / 'plan_identity.json')['sha256']:
        raise ValueError('Frozen plan changed')
    configure_training(plan['training']['seed'])
    folder = args.output / f'train_{args.mode}'
    if folder.exists() and not args.resume:
        raise FileExistsError('Training output exists; use explicit --resume')
    folder.mkdir(exist_ok=args.resume)
    trains = [c for c in plan['clips'] if c['role'] == 'train']
    vals = [c for c in plan['clips'] if c['role'] == 'validation']
    # No test-static arrays opened during training or checkpoint selection.
    cache = open_cache(args.output, trains + vals)
    pairs = [(c['id'], i) for c in trains for i in range(c['frames']) if i % 64]
    steps = plan['training']['steps_each']
    rng = np.random.default_rng(plan['training']['seed'])
    samples = rng.integers(0, len(pairs), size=steps, dtype=np.int64)
    sample_sha = sha(samples.tobytes())
    model = student().cuda().train()
    initial_sha = model_sha(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['training']['learning_rate'], weight_decay=0)
    start_step, best_mse = 0, float('inf')
    validations, times = [], []
    previous_wall = 0.0
    if args.resume:
        state = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        if state['plan_sha256'] != plan_hash or state['mode'] != args.mode or state['sampling_sha256'] != sample_sha:
            raise ValueError('Checkpoint does not match this training protocol')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        start_step, best_mse = state['step'], state['best_mse']
        validations, times = state['validations'], state['step_seconds']
        previous_wall = state['wall_seconds']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        # A crash may have logged steps beyond last.pt; preserve that log,
        # resume into a separate file rather than duplicating step records.
    report = dict(mode=args.mode, plan_sha256=plan_hash, parameters=8800707, initial_model_sha256=initial_sha,
                  sampling_sha256=sample_sha, distinct_training_p_frames=len(pairs), seed=plan['training']['seed'],
                  source='verified real byte-stream receiver cache', precision=plan['training']['precision'],
                  test_used_for_selection=False, perceptual_loss=False, temporal_loss=False,
                  status='running', requested_steps=steps, completed_steps=start_step)
    if args.mode == 'gt' and (args.output / 'train_kd/summary.json').is_file():
        other = load_json(args.output / 'train_kd/summary.json')
        if (other['initial_model_sha256'], other['sampling_sha256']) != (initial_sha, sample_sha):
            raise ValueError('KD and GT control do not share initialization/sampling')
    save_json(folder / 'summary.json', report)
    begin = time.perf_counter()

    def checkpoint(step, path):
        atomic_checkpoint(path, dict(step=step, mode=args.mode, plan_sha256=plan_hash,
                                    sampling_sha256=sample_sha, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                                    best_mse=best_mse, validations=validations, step_seconds=times,
                                    wall_seconds=previous_wall + time.perf_counter() - begin))

    torch.cuda.reset_peak_memory_stats()
    if start_step == 0:
        validations.append(dict(step=0, **validate(model, cache, vals)))
        checkpoint(0, folder / 'initial.pt')
    log_path = folder / (f'loss_resume_{start_step}.jsonl' if args.resume else 'loss.jsonl')
    step = start_step
    try:
        with log_path.open('x', encoding='utf-8', buffering=1) as log:
            for step in range(start_step + 1, steps + 1):
                check_stop(args.output)
                tick = time.perf_counter()
                cid, index = pairs[int(samples[step - 1])]
                arrays = cache[cid]
                x = torch.from_numpy(np.array(arrays['latent'][index]))[None].cuda()
                original = image_tensor(arrays['source'][index])
                teacher = image_tensor(arrays['teacher'][index]) if args.mode == 'kd' else None
                warmup = plan['training']['warmup_steps']
                if step <= warmup:
                    lr = plan['training']['learning_rate'] * step / warmup
                else:
                    fraction = (step - warmup) / (steps - warmup)
                    lr = plan['training']['final_learning_rate'] + 0.5 * (plan['training']['learning_rate'] - plan['training']['final_learning_rate']) * (1 + math.cos(math.pi * fraction))
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = model(x)[:, :, :288, :512].float()
                    gt_loss = F.l1_loss(prediction, original)
                    kd_loss = F.l1_loss(prediction, teacher) if teacher is not None else gt_loss.new_zeros(())
                    loss = 0.5 * (gt_loss + kd_loss) if teacher is not None else gt_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss at step {step}')
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                duration = time.perf_counter() - tick
                times.append(duration)
                row = dict(step=step, clip=cid, frame_index=index, loss=float(loss), l1_original=float(gt_loss),
                           l1_teacher=float(kd_loss) if teacher is not None else None,
                           grad_norm_before_clip=float(grad_norm), learning_rate=lr, seconds=duration)
                log.write(__import__('json').dumps(row, allow_nan=False) + '\n')
                if step % 100 == 0:
                    report.update(completed_steps=step, last_loss=row['loss'], elapsed_wall_seconds=previous_wall + time.perf_counter() - begin)
                    save_json(folder / 'summary.json', report)
                    print(f'{args.mode} step {step}/{steps} loss={float(loss):.5f} lr={lr:.7f} wall={report["elapsed_wall_seconds"]:.1f}s', flush=True)
                if step % 500 == 0 or step == steps:
                    metric = dict(step=step, **validate(model, cache, vals))
                    validations.append(metric)
                    if metric['mse_rgb'] < best_mse:
                        best_mse = metric['mse_rgb']
                        checkpoint(step, folder / 'best.pt')
                    checkpoint(step, folder / 'last.pt')
                    save_json(folder / 'validation.json', validations)
                    print(f'{args.mode} validation {step}: P-frame PSNR={metric["psnr_db"]:.3f}dB', flush=True)
        final_sha = model_sha(model)
        if final_sha == initial_sha:
            raise AssertionError('Parameters did not change during real training')
        report.update(status='complete', completed_steps=step, final_model_sha256=final_sha, parameters_changed=True,
                      best_validation=min(validations[1:], key=lambda r: r['mse_rgb']),
                      total_wall_seconds=previous_wall + time.perf_counter()-begin,
                      mean_training_step_seconds=float(np.mean(times)), median_training_step_seconds=float(np.median(times)),
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                      best_checkpoint_file_sha256=file_sha(folder / 'best.pt'))
        save_json(folder / 'summary.json', report)
        print(__import__('json').dumps(report, ensure_ascii=False), flush=True)
    except (InterruptedError, KeyboardInterrupt):
        completed = len(times)
        checkpoint(completed, folder / 'last.pt')
        report.update(status='stopped', completed_steps=completed, total_wall_seconds=previous_wall + time.perf_counter()-begin)
        save_json(folder / 'summary.json', report)
        raise


if __name__ == '__main__':
    main()
