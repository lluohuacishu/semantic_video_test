"""Depth-preserving student and explicit, approximate teacher channel mapping.

This is an initialization baseline, not function-preserving pruning. Channel
removal changes convolution sums, GroupNorm statistics and attention scaling.
"""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot_common import configure_training, file_sha, model_sha, save_json, settings, sha


def architecture(width=64):
    path = settings()['runtime']['repo'] / 'src/models/vqgan_arch.py'
    spec = importlib.util.spec_from_file_location('round2_vqgan', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Generator(width, 256, [1, 1, 2, 2, 4], 3, 256, [16],
                             swish_last=True, quant_conv=True, patch=32)
    if width == 64:
        assert sum(p.numel() for p in model.parameters()) == 10985155
    return model


def teacher_cpu():
    path = settings()['runtime']['weights'] / 'GLC_Video.pth.tar'
    payload = torch.load(path, map_location='cpu', weights_only=False)
    while isinstance(payload, dict) and any(k in payload for k in ['state_dict', 'net_g']):
        payload = payload['state_dict'] if 'state_dict' in payload else payload['net_g']
    state = {k.removeprefix('module.').removeprefix('vqgan.generator.'): v
             for k, v in payload.items() if k.removeprefix('module.').startswith('vqgan.generator.')}
    teacher = architecture(128)
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    return teacher


def balanced_indices(teacher):
    scores = {}
    for module in teacher.modules():
        if isinstance(module, torch.nn.GroupNorm):
            scores.setdefault(module.num_channels, []).append(module.weight.detach().abs())
    selected, description = {}, {}
    for channels, values in scores.items():
        score = torch.stack(values).mean(0)
        assert channels % 64 == 0 and torch.isfinite(score).all()
        per_group = channels // 32
        keep = per_group // 2
        indices = []
        for group in range(32):
            start = group * per_group
            local = torch.argsort(score[start:start+per_group], descending=True, stable=True)[:keep]
            indices.extend((local.sort().values + start).tolist())
        selected[channels] = torch.tensor(indices, dtype=torch.long)
        description[str(channels)] = dict(teacher_channels=channels, student_channels=channels//2,
                                         norm_groups=32, kept_per_original_group=keep, indices=indices,
                                         ranking='mean absolute pretrained GN gamma over all layers of this width; deterministic stable ties')
    return selected, description


def map_teacher(teacher, width=64):
    target = architecture(width)
    source_modules = dict(teacher.named_modules())
    maps, description = balanced_indices(teacher)
    copied = set()
    def indices(original, narrowed):
        if original == narrowed:
            return torch.arange(original)
        if original != 2 * narrowed:
            raise ValueError('Only width128 identity or width64 half-width mapping supported')
        return maps[original]
    with torch.no_grad():
        for name, module in target.named_modules():
            original = source_modules[name]
            if type(module).__name__ != type(original).__name__:
                raise ValueError('Depth/module order differs; mapping would be semantically ambiguous')
            if isinstance(module, torch.nn.Conv2d):
                rows = indices(original.out_channels, module.out_channels)
                cols = indices(original.in_channels, module.in_channels)
                module.weight.copy_(original.weight.index_select(0, rows).index_select(1, cols))
                copied.add(name + '.weight')
                if module.bias is not None:
                    module.bias.copy_(original.bias.index_select(0, rows))
                    copied.add(name + '.bias')
            elif isinstance(module, torch.nn.GroupNorm):
                keep = indices(original.num_channels, module.num_channels)
                if original.num_groups != module.num_groups:
                    raise ValueError('Unexpected GroupNorm group policy')
                module.weight.copy_(original.weight.index_select(0, keep))
                module.bias.copy_(original.bias.index_select(0, keep))
                copied.update([name + '.weight', name + '.bias'])
    if copied != set(target.state_dict()):
        raise ValueError('Some student state tensors were left random or not mapped')
    return target, description


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination = args.output / 'initialization'
    destination.mkdir(exist_ok=False)
    configure_training(29)
    teacher = teacher_cpu()
    teacher_hash = model_sha(teacher)
    # Teacher state tensor names/values match the previous pinned checkpoint audit.
    identity, _ = map_teacher(teacher, width=128)
    if model_sha(identity) != teacher_hash:
        raise AssertionError('Full-width transfer failed identity test')
    del identity
    mapped, channels = map_teacher(teacher)
    mapped_sha = model_sha(mapped)
    torch.save(dict(schema='round2-generator-init-v1', model=mapped.state_dict(), initialization='mapped',
                    source_teacher_state_sha256=teacher_hash), destination / 'mapped.pt')
    torch.manual_seed(29)
    random = architecture()
    random_sha = model_sha(random)
    torch.save(dict(schema='round2-generator-init-v1', model=random.state_dict(), initialization='random'), destination / 'random.pt')
    report = dict(student_parameters=10985155, teacher_parameters=42515587, depth_preserved=True,
                  block_modules=len(mapped.blocks), full_width_identity_state_match=True,
                  teacher_model_sha256=teacher_hash, mapped_model_sha256=mapped_sha, random_model_sha256=random_sha,
                  channel_maps=channels, all_state_tensors_mapped=True, channel_mapping_is_approximate=True,
                  validation_or_test_used_for_mapping=False, weight_rescaling=False,
                  caveat='Not function-preserving: dropped convolution inputs, changed GN statistics and attention scale require finetuning',
                  initialization_files={name: file_sha(destination / f'{name}.pt') for name in ['random', 'mapped']},
                  official_video_checkpoint_sha256=file_sha(settings()['runtime']['weights'] / 'GLC_Video.pth.tar'))
    old = settings()['workspace'] / '02_实验数据/优化_GLC蒸馏/pilot_20261029_01'
    teacher = teacher.cuda().eval()
    random = random.cuda().eval()
    mapped = mapped.cuda().eval()
    previews = []
    for cid in ['v001', 'v007', 'v025']:
        latent = torch.from_numpy(np.array(np.load(old / cid / 'latent.npy', mmap_mode='r')[32]))[None].cuda()
        source = np.load(old / cid / 'source.npy', mmap_mode='r')[32]
        reference = np.load(old / cid / 'teacher.npy', mmap_mode='r')[32]
        row = dict(clip=cid, frame_index=32, scope='one historical P frame, diagnostic only; not round2 final holdout')
        with torch.inference_mode():
            for name, model in [('teacher', teacher), ('random', random), ('mapped', mapped)]:
                output = model(latent)[0, :, :288, :512].clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                mse = float(np.mean((output.astype('float32') - source.astype('float32')) ** 2))
                row[name + '_psnr_db'] = float(10 * np.log10(255 ** 2 / max(mse, 1e-12)))
                if name == 'teacher' and not np.array_equal(output, reference):
                    raise AssertionError('Pinned P teacher does not reproduce cached RGB')
        previews.append(row)
    report['historical_frame_preflight'] = previews
    del teacher, random
    torch.cuda.empty_cache()
    # One disposable real-data AdamW update probes memory and finite gradients.
    # Saved initialization tensors remain untouched and are rehashed afterwards.
    mapped.train()
    optimizer = torch.optim.AdamW(mapped.parameters(), lr=1e-4, weight_decay=0)
    x = torch.from_numpy(np.array(np.load(old / 'v001/latent.npy', mmap_mode='r')[32]))[None].cuda()
    y = torch.from_numpy(np.array(np.load(old / 'v001/source.npy', mmap_mode='r')[32])).cuda().permute(2, 0, 1)[None].float().div(127.5).sub(1)
    torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = torch.nn.functional.l1_loss(mapped(x)[:, :, :288, :512].float(), y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mapped.parameters(), 1, error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize()
    report['disposable_step_probe'] = dict(loss=float(loss), seconds=time.perf_counter()-tick,
                                        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                                        peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                                        candidate_initialization_files_unchanged=all(file_sha(destination/f'{n}.pt') == v for n, v in report['initialization_files'].items()))
    if not report['disposable_step_probe']['candidate_initialization_files_unchanged']:
        raise AssertionError('Preflight modified a saved initialization')
    save_json(destination / 'transfer_audit.json', report)
    print('Initialization preflight', previews, report['disposable_step_probe'], flush=True)


if __name__ == '__main__':
    main()
