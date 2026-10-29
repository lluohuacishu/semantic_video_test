"""Quality and media from independently decoded outputs; no checkpoint selection."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from pilot_common import file_sha, load_json, save_json, settings, sha


def psnr(mse):
    return 10 * math.log10(255 ** 2 / max(float(mse), 1e-12))


def ssim_rgb(a, b):
    a, b = a.astype('float64'), b.astype('float64')
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)[5:-5, 5:-5]
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
    value = ((2 * ma * mb + 6.5025) * (2 * cov + 58.5225)) / ((ma * ma + mb * mb + 6.5025) * (va + vb + 58.5225))
    return float(value.mean())


def edge_energy(rgb):
    a = rgb.astype('float32')
    return float((np.mean(np.abs(np.diff(a, axis=0))) + np.mean(np.abs(np.diff(a, axis=1)))) / 2)


def metrics(source, rendered, teacher, masks):
    rows = []
    for i in range(len(source)):
        original = source[i].astype('float32')
        image = rendered[i].astype('float32')
        delta = image - original
        roi = masks[i] > 0
        row = dict(index=i, kind='I' if i % 64 == 0 else 'P', mse_rgb=float(np.mean(delta ** 2)),
                   psnr_db=psnr(np.mean(delta ** 2)), l1_rgb=float(np.mean(np.abs(delta))),
                   ssim_rgb=ssim_rgb(source[i], rendered[i]), fish_pixels=int(roi.sum()),
                   fish_squared_error_sum=float(np.sum(delta[roi].astype('float64') ** 2)),
                   fish_psnr_db=psnr(np.mean(delta[roi] ** 2)) if roi.any() else None,
                   teacher_mse_rgb=float(np.mean((image - teacher[i].astype('float32')) ** 2)),
                   edge_energy=edge_energy(rendered[i]), original_edge_energy=edge_energy(source[i]))
        if i > 0:
            # Frame-index proxy only: no motion compensation, no perceptual
            # interpretation and no unverified timing assumption.
            previous_error = rendered[i-1].astype('float32') - source[i-1].astype('float32')
            row['frame_index_error_delta_mae_rgb'] = float(np.mean(np.abs(delta - previous_error)))
        rows.append(row)
    groups = {}
    for key, selected in [('all', rows), ('p_only', [r for r in rows if r['kind'] == 'P'])]:
        fish_count = sum(r['fish_pixels'] for r in selected)
        mse = float(np.mean([r['mse_rgb'] for r in selected]))
        groups[key] = dict(frames=len(selected), mse_rgb=mse, psnr_db=psnr(mse),
                           mean_frame_psnr_db=float(np.mean([r['psnr_db'] for r in selected])),
                           ssim_rgb=float(np.mean([r['ssim_rgb'] for r in selected])),
                           l1_rgb=float(np.mean([r['l1_rgb'] for r in selected])),
                           fish_psnr_db=psnr(sum(r['fish_squared_error_sum'] for r in selected) / max(3 * fish_count, 1)),
                           fish_pixels=fish_count,
                           mean_teacher_mse_rgb=float(np.mean([r['teacher_mse_rgb'] for r in selected])),
                           edge_energy_ratio_to_original=float(np.mean([r['edge_energy'] for r in selected]) / np.mean([r['original_edge_energy'] for r in selected])))
    return dict(groups=groups, frames=rows)


def panel(arrays, index, font):
    result = Image.new('RGB', (1024, 640), '#101820')
    draw = ImageDraw.Draw(result)
    labels = ['Original / 原图', 'GLC teacher / 原GLC', 'KD student / 蒸馏学生', 'GT-only student / 原图监督对照']
    for k, (name, values) in enumerate(zip(labels, arrays)):
        x, y = (k % 2) * 512, (k // 2) * 320
        draw.text((x + 8, y + 4), name, font=font, fill='white')
        result.paste(Image.fromarray(values[index]), (x, y + 32))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = load_json(args.output / 'plan.json')
    evaluation = args.output / 'evaluation'
    frozen_selection = load_json(evaluation / 'selection_frozen.json')
    report_path = evaluation / 'quality.json'
    if report_path.exists():
        raise FileExistsError(report_path)
    report = dict(plan_sha256=file_sha(args.output / 'plan.json'), selection_complete_before_test=True,
                  metric_definition='RGB uint8; PSNR from aggregate MSE; SSIM 11x11 Gaussian sigma1.5 valid border, RGB mean; fish ROI is verified GT union resized nearest',
                  limitations=['Small source-narrow pilot, source independence not established',
                               'Only q12 and 512x288 tested', 'No LPIPS/DISTS or detection task score measured',
                               'No subjective video pass recorded; no validated flicker metric',
                               'Full 96-frame validation timing not uniquely verified; 30fps previews use frame order only',
                               'v025 VFR is evaluated as static frames only'], clips=[])
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    font = ImageFont.truetype(str(font_path), 18) if font_path.is_file() else ImageFont.load_default()
    for clip in [c for c in plan['clips'] if c['role'] != 'train']:
        folder = args.output / clip['id']
        source = np.load(folder / 'source.npy', mmap_mode='r')
        teacher = np.load(folder / 'teacher.npy', mmap_mode='r')
        masks = np.load(folder / 'fish_mask.npy', mmap_mode='r')
        source_manifest = load_json(folder / 'source_manifest.json')
        if any(sha(masks[i].tobytes()) != row['fish_mask_sha256'] for i, row in enumerate(source_manifest['records'])):
            raise ValueError('Fish GT mask cache changed')
        cache = load_json(folder / 'decode_cache.json')
        if file_sha(folder / 'teacher.npy') != cache['teacher_array_sha256'] or file_sha(folder / 'source.npy') != clip['source_array_sha256']:
            raise ValueError('Evaluation source or teacher changed')
        outputs = dict(teacher=teacher)
        for mode in ['kd', 'gt']:
            decoded_folder = evaluation / 'full_decode' / f'{clip["id"]}_{mode}_0'
            decoded = load_json(decoded_folder / 'decode.json')
            if file_sha(decoded_folder / 'rgb.npy') != decoded['rgb_array_sha256']:
                raise ValueError('Independent decoded output changed')
            if decoded['stream_sha256'] != cache['stream_sha256'] or not decoded['original_ordered_latent_sha256_verified']:
                raise ValueError('Student output not paired to exact original stream')
            if decoded['model_identity']['checkpoint_sha256'] != frozen_selection['modes'][mode]['checkpoint_sha256']:
                raise ValueError('Decoded student differs from frozen selected checkpoint')
            outputs[mode] = np.load(decoded_folder / 'rgb.npy', mmap_mode='r')
        entry = dict(id=clip['id'], role=clip['role'], frames=clip['frames'], timing_status=clip['timing_status'],
                     stream_bytes=(folder / 'stream.glcv').stat().st_size, models={})
        for mode, rendered in outputs.items():
            entry['models'][mode] = metrics(source, rendered, teacher, masks)
            if mode != 'teacher' and not np.array_equal(rendered[::64], teacher[::64]):
                raise ValueError('Student path unexpectedly changed I-frame RGB')
        if clip['id'] == 'v007':
            for mode in ['kd', 'gt']:
                selected = load_json(args.output / f'train_{mode}/summary.json')['best_validation']
                if abs(entry['models'][mode]['groups']['p_only']['mse_rgb'] - selected['mse_rgb']) > 1e-3:
                    raise ValueError('Independent full decode does not reproduce selected cached-validation MSE')
                entry['models'][mode]['independent_decode_matches_selected_validation_mse'] = True
            for mode in outputs:
                first64 = entry['models'][mode]['frames'][1:64]
                entry['models'][mode]['verified_first64_error_delta_proxy'] = float(np.mean([r['frame_index_error_delta_mae_rgb'] for r in first64]))
            # This is an index-order preview, not a validated timed reconstruction.
            path = evaluation / 'comparison_v007_frame_order_30fps.mp4'
            ffmpeg = settings()['runtime']['ffmpeg']
            command = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                       '-s', '1024x640', '-r', '30', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'medium',
                       '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
            with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
                for i in range(clip['frames']):
                    frame = panel([source, teacher, outputs['kd'], outputs['gt']], i, font)
                    ImageDraw.Draw(frame).text((760, 607), f'frame {i+1:03d} / index preview', font=font, fill='white', stroke_width=1, stroke_fill='black')
                    process.stdin.write(frame.tobytes())
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError('Comparison preview encoding failed')
            for index in [32, 63, 64, 65]:
                panel([source, teacher, outputs['kd'], outputs['gt']], index, font).save(evaluation / f'comparison_v007_frame{index+1:03d}.png')
        else:
            panel([source, teacher, outputs['kd'], outputs['gt']], 32, font).save(evaluation / 'comparison_v025_frame033.png')
        report['clips'].append(entry)
        print(f'quality completed {clip["id"]}', flush=True)
    # Export inference-only weights; retain optimizer checkpoints separately.
    exports = args.output / 'student_exports'
    exports.mkdir(exist_ok=False)
    export_records = []
    for mode in ['kd', 'gt']:
        path = args.output / f'train_{mode}' / 'best.pt'
        state = torch.load(path, map_location='cpu', weights_only=False)
        export_path = exports / f'p_generator_{mode}_q12_pilot.pt'
        torch.save(dict(schema='glc-p-rgb-student-pilot-v1', model=state['model'], mode=mode, step=state['step'],
                        plan_sha256=state['plan_sha256'], parameters=8800707,
                        warning='Pilot only; P RGB renderer; original I generator and original latent chain required'), export_path)
        export_records.append(dict(mode=mode, selected_step=state['step'], bytes=export_path.stat().st_size,
                                   file_sha256=file_sha(export_path), path=str(export_path)))
    report['exports'] = export_records
    save_json(report_path, report)


if __name__ == '__main__':
    main()
