"""Compare three-frame GLC postprocessing on fixed, lossless RGB clips."""

import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import cv2
import numpy as np

from window_filter import FAR_PAIR_WEIGHT, FLOW_EDGE_LIMIT, KERNEL, STAGE11, blend_center
from temporal_filter import Settings, TemporalFilter, correspondence, gray, lowpass, motion, warp

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
STAGE16 = REPO / '16_优化阶段_输入编码器推理加速'
sys.path.insert(0, str(STAGE16))
from web_frames import Frames
from sender_common import file_sha, load_json, save_json

def chunks(root, directory, metadata):
    folder = root / directory
    # Archived metadata can still have the old absolute paths. Resolve the
    # unchanged filenames under the canonical directory in the baseline list.
    return {**metadata, 'chunks': [
        {**chunk, 'path': str(folder / Path(chunk['path']).name)}
        for chunk in metadata['chunks']
    ]}


def open_baseline(root, baseline, codec):
    is_chunked = 'rgb_chunks' in baseline
    result_path = (root / baseline['metadata'] if is_chunked
                   else (root / baseline['rgb']).with_name('result.json'))
    result = load_json(result_path)
    if result['frames'] != baseline['frames'] or not result['reference_verified']:
        raise ValueError(f'Unverified GLC input: {result_path}')
    if 'header' in result and any(result['header'][key] != value for key, value in codec.items()):
        raise ValueError(f'Unexpected GLC settings: {result_path}')
    if is_chunked:
        meta = baseline['decoded_storage_metadata']
        decoded_meta = load_json(root / meta['file'])[meta['key']]
        decoded_meta = chunks(root, baseline['rgb_chunks'], decoded_meta)
    else:
        decoded_meta = dict(frames=baseline['frames'], source=root / baseline['rgb'],
                            source_sha256=result['decoded_rgb_sha256'])
    source_path = root / baseline['source_storage_metadata']
    source_meta = chunks(root, baseline['source_chunks'], load_json(source_path))
    for meta in (decoded_meta, source_meta):
        if meta['frames'] != baseline['frames']:
            raise ValueError(f'Frame count differs for {baseline["id"]}')
    receipt = dict(stream_sha256=result['stream_sha256'], result_sha256=file_sha(result_path),
                   source_metadata_sha256=file_sha(source_path),
                   decoded_storage='chunked-rgb-v1' if is_chunked else 'npy')
    return Frames(decoded_meta), Frames(source_meta), receipt


def load_range(sequence, start, stop, shape):
    frames = np.stack([sequence[index] for index in range(start, stop)])
    if frames.dtype != np.uint8 or frames.shape != (stop - start, *shape):
        raise ValueError(f'Expected lossless uint8 RGB {shape}, got {frames.dtype} {frames.shape}')
    return frames


def filter_centered(decoded, first, stop, output, stats, method, fusion, radius):
    # Each estimate can be reused in the opposite direction when the later
    # frame becomes the center. Pop consumed pairs to bound memory.
    flows = {}

    def past_pair(earlier, later):
        pair = flows.pop((earlier, later), None)
        return pair if pair is not None else motion(decoded[earlier], decoded[later])

    def future_pair(earlier, later):
        pair = motion(decoded[earlier], decoded[later])
        if later < stop:
            flows[(earlier, later)] = pair
        return pair

    for index, offset in enumerate(range(first, stop)):
        tick = time.perf_counter()
        confidence = 0.0
        if 0 < offset < len(decoded) - 1:
            before = past_pair(offset - 1, offset)
            after = future_pair(offset, offset + 1)
            distant = None
            if radius == 2 and 1 < offset < len(decoded) - 2:
                distant = (decoded[offset - 2], decoded[offset + 2],
                           past_pair(offset - 2, offset), future_pair(offset, offset + 2))
            output[index], confidence = blend_center(
                decoded[offset - 1], decoded[offset], decoded[offset + 1], before, after,
                fusion=fusion, distant=distant)
        else:
            output[index] = decoded[offset]
        stats[index][f'{method}_ms'] = (time.perf_counter() - tick) * 1000
        stats[index][f'{method}_valid'] = confidence


def apply_filters(decoded, start, stop, context_start, fusion, compare_five=False):
    first = start - context_start
    last = first + stop - start
    raw = decoded[first:last]
    filtered = {'raw': raw}
    stats = [{} for _ in raw]
    if not compare_five:
        causal = TemporalFilter()
        if first:
            causal.apply(decoded[first - 1])
        filtered['stage11'] = np.empty_like(raw)
        for i, frame in enumerate(raw):
            tick = time.perf_counter()
            filtered['stage11'][i], info = causal.apply(frame)
            stats[i]['stage11_ms'] = (time.perf_counter() - tick) * 1000
            stats[i]['stage11_valid'] = info['active_fraction']
    windows = (('center3', 1), ('center5', 2)) if compare_five else (('center3', 1),)
    for method, radius in windows:
        filtered[method] = np.empty_like(raw)
        filter_centered(decoded, first, last, filtered[method], stats, method, fusion, radius)
    return filtered, stats


def roi(box, shape):
    x0, y0, x1, y1 = box
    height, width = shape[:2]
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError(f'Invalid crop {box} for {shape}')
    return np.s_[y0:y1, x0:x1]


def high(rgb):
    return gray(rgb).astype(np.float32) - gray(lowpass(rgb))


def psnr(mse):
    return float(10 * np.log10(255 ** 2 / mse)) if mse else None


def evaluate(source, decoded, stats, clip, codec):
    coral = roi(clip['coral_roi'], source.shape[1:])
    moving = roi(clip['motion_roi'], source.shape[1:])
    coral_pixels = source[0][coral].shape[0] * source[0][coral].shape[1]
    rows = []
    previous_source = previous_hf = previous_low = None
    for i, original in enumerate(source):
        index = clip['start'] + i
        is_i = index % codec['intra_period'] == 0
        reset = not is_i and index % codec['reset_interval'] == 1
        row = dict(frame=index, source_frame=index * clip['source_stride'],
                   is_i_frame=is_i, ref_y_reset=reset, **stats[i])
        original_low = gray(lowpass(original))
        original_hf = gray(original).astype(np.float32) - original_low
        row['source_coral_hf_energy'] = float(np.abs(original_hf[coral]).mean())
        if previous_source is None:
            valid = np.zeros((source.shape[1], source.shape[2]), dtype=bool)
        else:
            backward, forward = motion(previous_source, original)
            x, y, inside, fb = correspondence(backward, forward)
            photo = np.abs(original_low - warp(previous_low, x, y))
            safe = inside & (fb < 1.5) & (photo < 12)
            safe = cv2.erode(safe.astype(np.uint8), KERNEL).astype(bool)
            valid = safe & (cv2.GaussianBlur(np.abs(original_hf), (0, 0), 2) > 2.0)
            ref_delta = original_hf - warp(previous_hf['source'], x, y)
        valid_coral = valid[coral]
        count = int(np.count_nonzero(valid_coral))
        row['coral_temporal_valid_pixels'] = count
        current_hf = {'source': original_hf}
        for method, video in decoded.items():
            frame = video[i]
            frame_hf = high(frame)
            current_hf[method] = frame_hf
            squared = np.square(frame.astype(np.float32) - original)
            rgb_change = np.abs(frame[moving].astype(np.int16) - decoded['raw'][i][moving])
            row[f'{method}_rgb_mse'] = float(squared.mean())
            row[f'{method}_motion_roi_mse'] = float(squared[moving].mean())
            row[f'{method}_motion_roi_change'] = float(rgb_change.mean())
            row[f'{method}_coral_hf_mae'] = float(np.abs(frame_hf[coral] - original_hf[coral]).mean())
            row[f'{method}_coral_hf_energy'] = float(np.abs(frame_hf[coral]).mean())
            row[f'{method}_coral_temporal_sum'] = 0.0
            if count:
                delta = frame_hf - warp(previous_hf[method], x, y)
                row[f'{method}_coral_temporal_sum'] = float(np.abs(delta - ref_delta)[coral][valid_coral].sum())
        rows.append(row)
        previous_source, previous_hf, previous_low = original, current_hf, original_low

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    total = sum(r['coral_temporal_valid_pixels'] for r in rows)
    summary = dict(coral_temporal_mask_coverage=total / max(1, (len(rows) - 1) * coral_pixels),
                   coral_temporal_mask_pixels=total, methods={})
    for method in decoded:
        temporal_key = f'{method}_coral_temporal_sum'
        temporal_means = [r[temporal_key] / r['coral_temporal_valid_pixels'] for r in rows
                          if r['coral_temporal_valid_pixels']]
        metrics = dict(
            rgb_psnr_db=psnr(mean(f'{method}_rgb_mse')),
            motion_roi_psnr_db=psnr(mean(f'{method}_motion_roi_mse')),
            motion_roi_mean_abs_change=mean(f'{method}_motion_roi_change'),
            coral_spatial_hf_mae=mean(f'{method}_coral_hf_mae'),
            coral_hf_energy_ratio=mean(f'{method}_coral_hf_energy') / mean('source_coral_hf_energy')
                                  if mean('source_coral_hf_energy') else None,
            coral_temporal_hf_mae=sum(r[temporal_key] for r in rows) / total if total else None,
            coral_temporal_hf_frame_p95=float(np.percentile(temporal_means, 95)) if temporal_means else None,
        )
        if method != 'raw':
            metrics.update(filter_mean_ms=mean(f'{method}_ms'),
                           filter_p95_ms=float(np.percentile([r[f'{method}_ms'] for r in rows], 95)),
                           mean_valid_fraction=mean(f'{method}_valid'))
        summary['methods'][method] = metrics
    return summary, rows


def preview(ffmpeg, output, panels, fps, start, labels, crop=None):
    height, width = (panels[0][0] if crop is None else panels[0][0][crop]).shape[:2]
    if crop is not None:
        height, width = height * 2, width * 2
    title = 26
    row_height = height + title + height % 2
    frame_width = 2 * width
    frame_height = ((len(panels) + 1) // 2) * row_height
    command = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin', '-f', 'rawvideo',
               '-pixel_format', 'rgb24', '-video_size', f'{frame_width}x{frame_height}',
               '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast',
               '-crf', '14', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-n', str(output)]
    canvas = np.full((frame_height, frame_width, 3), (24, 35, 48), np.uint8)
    with output.with_suffix('.ffmpeg.log').open('wb') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            for i, frames in enumerate(zip(*panels, strict=True)):
                for j, frame in enumerate(frames):
                    y = (j // 2) * row_height
                    x = (j % 2) * width
                    canvas[y:y + title, x:x + width] = (24, 35, 48)
                    label = f'{labels[j]} | {(start + i) / float(Fraction(fps)):.2f}s'
                    cv2.putText(canvas, label, (x + 8, y + 18), cv2.FONT_HERSHEY_SIMPLEX,
                                0.47, (235, 244, 252), 1, cv2.LINE_AA)
                    if crop is not None:
                        frame = cv2.resize(frame[crop], (width, height), interpolation=cv2.INTER_NEAREST)
                    canvas[y + title:y + title + height, x:x + width] = frame
                process.stdin.write(canvas.tobytes())
            process.stdin.close()
            if process.wait(timeout=90):
                raise RuntimeError(f'FFmpeg failed for {output}')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    ffprobe = ffmpeg.with_name('ffprobe' + ffmpeg.suffix)
    probe = subprocess.check_output([str(ffprobe), '-v', 'error', '-select_streams', 'v:0',
                                     '-count_frames', '-show_entries',
                                     'stream=width,height,avg_frame_rate,nb_read_frames,duration',
                                     '-of', 'json', str(output)])
    stream = json.loads(probe)['streams'][0]
    if (int(stream['nb_read_frames']) != len(panels[0]) or Fraction(stream['avg_frame_rate']) != Fraction(fps)
            or (stream['width'], stream['height']) != (frame_width, frame_height)
            or abs(float(stream['duration']) - len(panels[0]) / float(Fraction(fps))) > 0.001):
        raise ValueError(f'Preview differs in frame count, cadence or dimensions: {output}')
    return {**stream, 'sha256': file_sha(output)}


def run_clip(baseline, inputs, clip, codec, ffmpeg, output, fusion, compare_five):
    decoded_frames, source_frames, receipt = inputs
    start, stop = clip['start'], clip['stop']
    count, *shape = baseline['shape']
    if not 0 <= start < stop <= count:
        raise ValueError(f'Bad clip range: {clip}')
    radius = 2 if compare_five else 1
    context_start = max(0, start - radius)
    window = load_range(decoded_frames, context_start, min(stop + radius, count), shape)
    filtered, stats = apply_filters(window, start, stop, context_start, fusion, compare_five)
    output.mkdir()
    generated = list(filtered)[1:]
    for name in generated:
        np.save(output / f'{name}_rgb.npy', filtered[name], allow_pickle=False)
    print(f'{clip["id"]}: filtered {stop - start} RGB frames, measuring against source', flush=True)
    source = load_range(source_frames, start, stop, shape)
    summary, rows = evaluate(source, filtered, stats, clip, codec)
    with (output / 'per_frame.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    panels = (source, *filtered.values())
    labels = ('Source RGB', 'GLC raw', '3 frames / mean', '5 frames / mean') if compare_five else (
        'Source RGB', 'GLC raw', 'Stage 11 / 2 frames', f'3 frames / {fusion}')
    videos = {}
    fps = baseline['fps']
    for name, crop in [('compare.mp4', None), ('zoom.mp4', roi(clip[clip['zoom']], shape))]:
        videos[name] = preview(ffmpeg, output / name, panels, fps, start, labels, crop)
    summary.update(clip=clip, codec=codec, fps=fps, frames=stop - start, shape=shape, input=receipt,
                   previews=videos, output_sha256={
                       name: file_sha(output / f'{name}_rgb.npy') for name in generated
                   })
    save_json(output / 'result.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--plan', type=Path, default=HERE / 'plan_p0.json')
    parser.add_argument('--fusion', choices=('consensus', 'mean'))
    parser.add_argument('--compare-five', action='store_true', help='Compare the mean three- and five-frame filters')
    parser.add_argument('--ffmpeg', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.compare_five and args.fusion == 'consensus':
        parser.error('--compare-five requires mean fusion')
    fusion = args.fusion or ('mean' if args.compare_five else 'consensus')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.workspace / '01_项目/GLC后处理/接口与基线.json'
    plan = load_json(args.plan)
    baselines = {b['id']: b for b in load_json(manifest)['baselines']}
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    summary = dict(schema='glc-temporal-p0-v1', complete=False, plan_sha256=file_sha(args.plan),
                   baseline_manifest_sha256=file_sha(manifest), cv2=cv2.__version__, numpy=np.__version__,
                   python=platform.python_version(), processor=platform.processor(), cv2_threads=cv2.getNumThreads(),
                   opencl=cv2.ocl.useOpenCL(), settings=vars(Settings()), flow_edge_limit=FLOW_EDGE_LIMIT,
                   center3_fusion=fusion, compare_five=args.compare_five,
                   code_sha256={p.name: file_sha(p) for p in (
                       Path(__file__), HERE / 'window_filter.py', STAGE11 / 'temporal_filter.py', STAGE16 / 'web_frames.py'
                   )}, clips=[])
    if args.compare_five:
        summary['far_pair_weight'] = FAR_PAIR_WEIGHT
    save_json(args.output / 'summary.json', summary)
    inputs = {}
    for clip in plan['clips']:
        baseline = baselines[clip['baseline']]
        if baseline['id'] not in inputs:
            print(f'{baseline["id"]}: verifying existing lossless inputs', flush=True)
            inputs[baseline['id']] = open_baseline(args.workspace, baseline, plan['codec'])
        tick = time.perf_counter()
        result = run_clip(baseline, inputs[baseline['id']], clip, plan['codec'], args.ffmpeg,
                          args.output / clip['id'], fusion, args.compare_five)
        summary['clips'].append(result)
        save_json(args.output / 'summary.json', summary)
        print(f'{clip["id"]}: saved and verified two comparisons ({time.perf_counter() - tick:.1f}s)', flush=True)
    summary['complete'] = True
    save_json(args.output / 'summary.json', summary)


if __name__ == '__main__':
    main()
