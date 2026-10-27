"""Evaluate lossless RGB, then create synchronized viewing copies."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from run_trial import HERE, REPO, read, write, sha
from temporal_filter import TemporalFilter, lowpass, gray, motion, correspondence, warp


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def high(rgb):
    return gray(rgb).astype(np.float32) - gray(lowpass(rgb))


def pooled_psnr(sse, count):
    return 10 * math.log10(255 ** 2 / (sse / count)) if sse else None


def metrics(source, methods, masks):
    accum = {name: dict(sse=0., count=0, fish_sse=0., fish_count=0,
        hf_error=0., hf_count=0, hf_energy=0., reference_hf_energy=0.,
        temporal_error=0., temporal_count=0, boundary_error=0., boundary_count=0)
        for name in methods}
    frames = []
    for i, original in enumerate(source):
        src = original.astype(np.float32)
        original_high = high(original)
        mask = masks[i]
        boundary = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) != cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
        spatial = {}
        for name, video in methods.items():
            output = video[i].astype(np.float32)
            squared = (output - src) ** 2
            hf = high(video[i])
            a = accum[name]
            a['sse'] += float(squared.sum(dtype=np.float64)); a['count'] += squared.size
            a['fish_sse'] += float(squared[mask].sum(dtype=np.float64)); a['fish_count'] += int(mask.sum()) * 3
            a['hf_error'] += float(np.abs(hf - original_high).sum()); a['hf_count'] += hf.size
            a['hf_energy'] += float(np.abs(hf).sum()); a['reference_hf_energy'] += float(np.abs(original_high).sum())
            a['boundary_error'] += float(np.abs(output - src)[boundary].sum()); a['boundary_count'] += int(boundary.sum()) * 3
            spatial[name] = dict(psnr=pooled_psnr(float(squared.sum()), squared.size))
        if i:
            back, forward = motion(source[i - 1], original)
            mx, my, valid, fb = correspondence(back, forward)
            photo = np.abs(gray(lowpass(original)) - warp(gray(lowpass(source[i - 1])), mx, my))
            valid &= (fb < 1.5) & (photo < 12)
            valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            texture = cv2.GaussianBlur(np.abs(original_high), (0, 0), 2) > 2.0
            area = valid & texture
            ref_delta = original_high - warp(high(source[i - 1]), mx, my)
            for name, video in methods.items():
                delta = high(video[i]) - warp(high(video[i - 1]), mx, my)
                error = np.abs(delta - ref_delta)
                accum[name]['temporal_error'] += float(error[area].sum())
                accum[name]['temporal_count'] += int(area.sum())
                spatial[name]['temporal_hf_error'] = float(error[area].mean()) if area.any() else None
            coverage = float(area.mean())
        else:
            coverage = None
        frames.append(dict(index=i, source_motion_texture_coverage=coverage, methods=spatial))
    result = {}
    for name, a in accum.items():
        result[name] = dict(psnr=pooled_psnr(a['sse'], a['count']),
            fish_psnr=pooled_psnr(a['fish_sse'], a['fish_count']),
            spatial_hf_mae=a['hf_error'] / a['hf_count'],
            hf_energy_vs_source=a['hf_energy'] / a['reference_hf_energy'],
            temporal_hf_mae=a['temporal_error'] / a['temporal_count'],
            fish_boundary_mae=a['boundary_error'] / a['boundary_count'],
            temporal_pixels=a['temporal_count'])
    return result, frames


def render_video(ffmpeg, output, frames, fps):
    if output.exists(): raise FileExistsError(output)
    h, w = frames[0].shape[:2]
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', f'{w}x{h}', '-framerate', fps,
        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '12',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-n', str(output)]
    with output.with_suffix('.ffmpeg.log').open('wb') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            for frame in frames: process.stdin.write(frame.tobytes())
            process.stdin.close()
            if process.wait(timeout=60): raise RuntimeError(f'Preview failed: {output}')
        finally:
            if process.poll() is None:
                process.kill(); process.wait()


def viewing(root, clip, source, methods, ffmpeg, modes=None):
    folder = root / f'v{clip["video_id"]:03d}'
    views = [('原图（共同缩小）', source),
        (f'普通 GLC · q{clip["baseline_q"]}', methods['baseline']),
        ('普通 GLC + 时域滤波 · 同一码流', methods['filtered']),
        ('提高码率 · q32', methods['higher_rate'])]
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    font = ImageFont.truetype(str(font_path), 18) if font_path.is_file() else ImageFont.load_default()
    if modes is None:
        modes = ('full', 'crop', 'texture') if clip['video_id'] == 11 else ('full', 'crop')
    for mode in modes:
        width, height = {'full': (512, 288), 'crop': (576, 192), 'texture': (624, 384)}[mode]
        rendered = []
        for index in range(64):
            canvas = Image.new('RGB', (width * 2, (height + 36) * 2), '#0e1720')
            draw = ImageDraw.Draw(canvas)
            for j, (label, video) in enumerate(views):
                image = Image.fromarray(video[index])
                if mode == 'crop': image = image.crop(tuple(clip['crop'])).resize((width, height), Image.Resampling.NEAREST)
                if mode == 'texture': image = image.crop((280, 112, 488, 240)).resize((width, height), Image.Resampling.NEAREST)
                x, y = (j % 2) * width, (j // 2) * (height + 36)
                canvas.paste(image, (x, y + 36))
                draw.text((x + 10, y + 6), label, font=font, fill='#e6edf3')
            rendered.append(np.asarray(canvas))
        render_video(ffmpeg, folder / f'{mode}.mp4', rendered, clip['fps'])
        Image.fromarray(rendered[32]).save(folder / f'{mode}_frame32.png')


def build_page(root, summary):
    template = (HERE / 'viewer.html').read_text('utf-8')
    (root / 'index.html').write_text(template.replace('__SUMMARY__', json.dumps(summary, ensure_ascii=False)), 'utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(); root = args.run.resolve()
    if (root / 'summary.json').exists(): raise FileExistsError('Analysis already complete')
    plan = read(root / 'plan.json')
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    all_results = []
    corpus_path = Path(plan['clips'][0]['frames'][0]).parents[4] / 'inputs.json'
    corpus = {c['video_id']: c for c in read(corpus_path)['clips']}
    for clip in plan['clips']:
        number = clip['video_id']; folder = root / f'v{number:03d}'
        source = [load_rgb(p) for p in clip['frames']]
        assert all(sha(p) == h for p, h in zip(clip['frames'], clip['source_sha256']))
        methods = {}
        audits = {}
        for variant, key in (('baseline', 'baseline'), ('high', 'higher_rate')):
            decoder = folder / f'{variant}_decoder'
            result = read(decoder / 'decode.json')
            assert result['independent_decoder'] and result['encoder_modules_disabled']
            assert result['all_frame_checksums_match'] and result['source_media_reads'] == []
            video = [load_rgb(decoder / f'receiver/frames/im{i + 1:05d}.png') for i in range(64)]
            assert all(hashlib.sha256(image.tobytes()).hexdigest() == row['rgb_sha256']
                       for image, row in zip(video, result['frames']))
            enc = read(folder / 'baseline_encode.json' if variant == 'baseline' else folder / 'high_encoder/encode.json')
            assert enc['file_sha256'] == result['input_sha256']
            assert all(a['rgb_sha256'] == b['rgb_sha256'] and a['latent_sha256'] == b['latent_sha256']
                       for a, b in zip(enc['frames'], result['frames']))
            methods[key] = video
            audits[key] = dict(archive_sha256=result['input_sha256'], file_bytes=enc['file_bytes'],
                decoder_core_seconds=result['frame_processing_seconds'], source_media_reads=[],
                all_rgb_and_latents_match=True, q=result['header']['q'])
        assert audits['higher_rate']['file_bytes'] > audits['baseline']['file_bytes']
        filt = TemporalFilter(); times, stats, output = [], [], []
        for frame in methods['baseline']:
            start = time.perf_counter(); image, state = filt.apply(frame)
            times.append((time.perf_counter() - start) * 1000); output.append(image); stats.append(state)
        methods['filtered'] = output
        filtered_folder = folder / 'filtered_frames'; filtered_folder.mkdir()
        for i, frame in enumerate(output): Image.fromarray(frame).save(filtered_folder / f'{i:04d}.png')
        masks = [np.asarray(Image.open(f['resized']['512x288']['gt'])) != 0 for f in corpus[number]['frames']]
        measured, per_frame = metrics(source, methods, masks)
        durations = np.array(times[1:])
        for key in ('baseline', 'filtered', 'higher_rate'):
            audit = audits['baseline' if key == 'filtered' else key]
            measured[key].update(file_bytes=audit['file_bytes'], q=audit['q'],
                kbps=audit['file_bytes'] * 8 / (64 / float(Fraction(clip['fps']))) / 1000)
        row = dict(video_id=number, methods=measured, verification=audits,
            filter_timing=dict(mean_ms=float(durations.mean()), median_ms=float(np.median(durations)),
                p95_ms=float(np.percentile(durations, 95)), total_seconds=sum(times)/1000,
                frames_including_first=64, timed_nonfirst_frames=63, cpu_threads=1,
                includes='forward/backward optical flow, confidence, filtering; excludes PNG/preview/evaluation'),
            mean_blend_weight=float(np.mean([s['mean_weight'] for s in stats[1:]])),
            mean_active_fraction=float(np.mean([s['active_fraction'] for s in stats[1:]])),
            detected_cuts=sum(s['reset'] for s in stats[1:]),
            original_raw_unchanged=sha(folder / 'baseline.glcv') == clip['baseline_sha256'],
            subjective_assessment='pending viewing; numerical changes are not a pass verdict')
        write(folder / 'metrics.json', row)
        write(folder / 'frame_metrics.json', per_frame)
        write(folder / 'filter_stats.json', [dict(index=i, ms=t, **s) for i, (t, s) in enumerate(zip(times, stats))])
        viewing(root, clip, source, methods, plan['runtime']['ffmpeg'])
        all_results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = dict(date=plan['date'], complete=True, clips=all_results, fps=plan['clips'][0]['fps'],
        filter=plan['filter'], metric_note='Common source-derived motion/texture masks; temporal high-frequency residual error against original, lower better, not a perceptual flicker score. PSNR pooled RGB; energy is not detail fidelity.',
        limitations=['two short development clips, same conservative source group', 'higher rate is not equal bitrate',
            'CPU i9-14900HX only; Orin untested', 'one fixed filter, no tuning after results',
            'browser videos are lossy H264 CRF12 viewing copies; measurements use lossless PNG',
            'causal filter uses reconstructed frames only; original and fish labels used only offline for metrics'])
    write(root / 'summary.json', summary)
    build_page(root, summary)
    print('ANALYSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
