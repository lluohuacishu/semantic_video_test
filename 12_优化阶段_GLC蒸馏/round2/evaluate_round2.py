"""Full-frame quality with sparse GT gates and six-panel frame-order previews."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot_common import file_sha, load_json, save_json, settings
from evaluate_pilot import ssim_rgb, edge_energy, psnr


def quality_metrics(source, teacher, output, masks, available):
    rows = []
    for i in range(len(source)):
        delta = output[i].astype('float32') - source[i].astype('float32')
        roi = masks[i] > 0 if available[i] else np.zeros(source[i].shape[:2], dtype='bool')
        row = dict(index=i, kind='I' if i % 64 == 0 else 'P', mse_rgb=float(np.mean(delta**2)),
                   l1_rgb=float(np.mean(np.abs(delta))), ssim_rgb=ssim_rgb(source[i], output[i]),
                   teacher_mse_rgb=float(np.mean((output[i].astype('float32')-teacher[i].astype('float32'))**2)),
                   gt_available=bool(available[i]), fish_pixels=int(roi.sum()),
                   fish_squared_error_sum=float(np.sum(delta[roi].astype('float64')**2)),
                   edge_energy=edge_energy(output[i]), original_edge_energy=edge_energy(source[i]))
        row['psnr_db'] = psnr(row['mse_rgb'])
        if i:
            previous = output[i-1].astype('float32')-source[i-1].astype('float32')
            row['index_error_delta_mae_rgb'] = float(np.mean(np.abs(delta-previous)))
        rows.append(row)
    groups = {}
    for name, subset in [('all', rows), ('p_only', [r for r in rows if r['kind']=='P'])]:
        fish = sum(r['fish_pixels'] for r in subset)
        mse = float(np.mean([r['mse_rgb'] for r in subset]))
        teacher_mse = float(np.mean([r['teacher_mse_rgb'] for r in subset]))
        original_energy = float(np.mean([r['original_edge_energy'] for r in subset]))
        groups[name] = dict(frames=len(subset), mse_rgb=mse, psnr_db=psnr(mse),
                            l1_rgb=float(np.mean([r['l1_rgb'] for r in subset])),
                            ssim_rgb=float(np.mean([r['ssim_rgb'] for r in subset])),
                            annotated_frames=sum(r['gt_available'] for r in subset),
                            annotated_positive_fish_frames=sum(r['fish_pixels']>0 for r in subset), fish_pixels=fish,
                            fish_psnr_db=psnr(sum(r['fish_squared_error_sum'] for r in subset)/(3*fish)) if fish else None,
                            teacher_mse_rgb=teacher_mse, psnr_to_teacher_db=psnr(teacher_mse) if teacher_mse else None,
                            edge_energy_ratio_to_original=float(np.mean([r['edge_energy'] for r in subset])/original_energy) if original_energy else None)
    return dict(groups=groups, frames=rows)


def panel(arrays, index, font, selected):
    canvas = Image.new('RGB', (1536, 640), '#101820')
    names = ['Original / 原图', 'GLC teacher / 教师', 'Random + mixed / 随机混合',
             'Mapped + mixed / 迁移混合', 'Mapped + teacher / 迁移教师', 'Mapped + GT / 迁移原图']
    order = ['source', 'teacher', 'random_mixed', 'mapped_mixed', 'mapped_teacher', 'mapped_gt']
    draw = ImageDraw.Draw(canvas)
    for k, (name, key) in enumerate(zip(names, order)):
        x, y = (k%3)*512, (k//3)*320
        if key == selected:
            name += ' *'
        draw.text((x+6, y+4), name, font=font, fill='white')
        canvas.paste(Image.fromarray(arrays[key][index]), (x, y+32))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluation = args.output / 'evaluation'
    if (evaluation / 'quality.json').exists():
        raise FileExistsError('Existing quality evidence')
    plan = load_json(args.output / 'plan.json')
    selection = load_json(evaluation / 'selection_frozen.json')
    cases = list(plan['training']['cases'])
    report = dict(plan_sha256=file_sha(args.output / 'plan.json'), selected_case=selection['selected_case'], clips=[],
                  sparse_gt_policy='Only available published labels contribute fish pixels; absent masks are not background',
                  psnr_policy='RGB aggregate MSE; all and P-only reported separately',
                  temporal_scope='frame-index diagnostic proxy and frame-order preview only; no motion compensation or verified perceptual flicker claim',
                  source_independence_established=False, source_caveats=plan['source_caveats'])
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    for clip in [c for c in plan['clips'] if c['role']!='train']:
        folder = args.output / clip['id']
        cached = load_json(folder / 'decode_cache.json')
        if file_sha(folder/'source.npy') != clip['source_array_sha256'] or file_sha(folder/'teacher.npy') != cached['teacher_array_sha256']:
            raise ValueError('Evaluation source/teacher changed')
        arrays = {name: np.load(folder/f'{name}.npy', mmap_mode='r') for name in ['source', 'teacher']}
        for name, key in [('fish_mask', 'fish_mask_array_sha256'), ('gt_available', 'gt_available_array_sha256')]:
            if file_sha(folder/f'{name}.npy') != clip[key]:
                raise ValueError('Evaluation GT changed')
        masks = np.load(folder/'fish_mask.npy', mmap_mode='r')
        available = np.load(folder/'gt_available.npy', mmap_mode='r')
        entry = dict(id=clip['id'], role=clip['role'], dataset=clip['dataset'], frames=clip['frames'],
                     stream_bytes=(folder/'stream.glcv').stat().st_size, models={})
        entry['models']['teacher'] = quality_metrics(arrays['source'], arrays['teacher'], arrays['teacher'], masks, available)
        for case in cases:
            decoded = evaluation/'full_decode'/f'{clip["id"]}_{case}_0'
            evidence = load_json(decoded/'decode.json')
            if not evidence['original_reference_chain_verified'] or file_sha(decoded/'rgb.npy') != evidence['rgb_array_sha256']:
                raise ValueError('Independent decoded evidence changed')
            if evidence['model_identity']['checkpoint_sha256'] != selection['cases'][case]['checkpoint_sha256']:
                raise ValueError('Wrong selected checkpoint used')
            if evidence['stream_sha256'] != cached['stream_sha256']:
                raise ValueError('Wrong original stream used')
            arrays[case] = np.load(decoded/'rgb.npy', mmap_mode='r')
            if not np.array_equal(arrays[case][::64], arrays['teacher'][::64]):
                raise ValueError('I frames changed')
            entry['models'][case] = quality_metrics(arrays['source'], arrays['teacher'], arrays[case], masks, available)
            if clip['role']=='validation':
                selected = load_json(args.output/f'train_{case}/summary.json')['best_validation']
                if abs(entry['models'][case]['groups']['p_only']['mse_rgb']-selected['mse_rgb']) > 1e-3:
                    raise ValueError('Independent decoder differs from cached validation MSE')
                entry['models'][case]['independent_decode_matches_cached_validation'] = True
        if clip['id'] in ['v007', 'v025']:
            old = Path(clip['reused_original_run'])/'evaluation/full_decode'/f'{clip["id"]}_kd_0'
            previous = load_json(old/'decode.json')
            if file_sha(old/'rgb.npy') != previous['rgb_array_sha256'] or previous['stream_sha256'] != cached['stream_sha256']:
                raise ValueError('Previous pilot comparison changed')
            old_rgb = np.load(old/'rgb.npy', mmap_mode='r')
            entry['models']['previous_8m8_kd'] = quality_metrics(arrays['source'], arrays['teacher'], old_rgb, masks, available)
        for index in sorted({min(32, clip['frames']-1), min(63, clip['frames']-1), min(64, clip['frames']-1), min(65, clip['frames']-1)}):
            panel(arrays, index, font, selection['selected_case']).save(evaluation/f'comparison_{clip["id"]}_frame{index+1:03d}.png')
        if clip['id'] in ['uv096', 'v007']:
            path = evaluation/f'comparison_{clip["id"]}_frame_order_30fps.mp4'
            command = [str(settings()['runtime']['ffmpeg']), '-hide_banner', '-loglevel', 'error', '-n',
                       '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1536x640', '-r', '30', '-i', 'pipe:0',
                       '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
            with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
                for i in range(clip['frames']):
                    canvas = panel(arrays, i, font, selection['selected_case'])
                    ImageDraw.Draw(canvas).text((1250, 612), f'index {i+1:03d} / 30fps preview', font=font, fill='white', stroke_width=1, stroke_fill='black')
                    process.stdin.write(canvas.tobytes())
                process.stdin.close()
                if process.wait()!=0:
                    raise RuntimeError('Preview encoding failed')
        report['clips'].append(entry)
        print(f'quality {clip["id"]} complete', flush=True)
    exports = args.output/'student_exports'
    exports.mkdir(exist_ok=False)
    report['exports'] = []
    for case in cases:
        checkpoint = torch.load(args.output/f'train_{case}/best.pt', map_location='cpu', weights_only=False)
        target = exports/f'{case}_p_rgb_q12.pt'
        torch.save(dict(schema='glc-p-rgb-round2-v1', case=case, step=checkpoint['step'], model=checkpoint['model'],
                        architecture=plan['architecture'], plan_sha256=checkpoint['plan_sha256'],
                        scope='experimental P RGB only; original latent chain and I generator required'), target)
        report['exports'].append(dict(case=case, step=checkpoint['step'], bytes=target.stat().st_size, sha256=file_sha(target)))
    save_json(evaluation/'quality.json', report)


if __name__ == '__main__':
    main()
