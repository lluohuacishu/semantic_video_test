"""Source / matched HEVC / original GLC / current low-rate GLC, in that order."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from coral_reference_sidecar import file_sha, load_json, save_json
from experiment_common import array_sha
from cache_hevc_baseline import X265_PARAMS


def match_hevc(ffmpeg, source, prior, budget, folder):
    """Refine CRF by bytes, without adding padding or changing the prior run."""
    folder.mkdir(exist_ok=False)
    record = load_json(prior/'result.json')
    assert array_sha(source) == record['source_rgb_sha256']
    original = record['controls']['h265_E']
    old_file = prior/'h265_E.mp4'
    assert file_sha(old_file) == original['sha256']
    scale = 4096
    upper = round(original['crf']*scale)
    lower = max(round(float(q)*scale) for q, size in record['candidates'].items()
                if size > budget and float(q) < original['crf'])
    bracket = (lower, upper)
    cache = {upper: (original['bytes'], old_file)}
    pixels = source.tobytes()
    common = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin']

    def encode(tick):
        if tick not in cache:
            path = folder/f'crf_{tick}.mp4'
            subprocess.run(common+['-f', 'rawvideo', '-pixel_format', 'rgb24',
                '-video_size', '512x288', '-framerate', '15', '-i', 'pipe:0', '-an',
                '-vf', 'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p',
                '-color_range', 'tv', '-colorspace', 'bt709', '-color_primaries', 'bt709',
                '-color_trc', 'bt709', '-c:v', 'libx265', '-preset', 'medium',
                '-crf', str(tick/scale), '-x265-params', X265_PARAMS, '-tag:v', 'hvc1',
                '-movflags', '+faststart', '-n', str(path)], input=pixels, check=True)
            cache[tick] = (path.stat().st_size, path)
            print('H.265', tick/scale, cache[tick][0], '/', budget, 'bytes', flush=True)
        return cache[tick][0]

    while upper-lower > 1:
        tick = (lower+upper)//2
        if encode(tick) <= budget:
            upper = tick
        else:
            lower = tick
        best_size = max(size for size, _ in cache.values() if size <= budget)
        if budget-best_size <= 24:
            break
    # Quantized encoder decisions are not perfectly monotonic near the crossing.
    if budget-max(size for size, _ in cache.values() if size <= budget) > 24:
        for tick in range(max(0, upper-6), upper+7):
            encode(tick)
            if budget-max(size for size, _ in cache.values() if size <= budget) <= 24:
                break
    # A wider scan covers non-monotonic pockets missed by binary refinement.
    if budget-max(size for size, _ in cache.values() if size <= budget) > 24:
        for tick in np.linspace(*bracket, 33).round().astype(int):
            encode(int(tick))
            if budget-max(size for size, _ in cache.values() if size <= budget) <= 24:
                break
    tick, (size, candidate) = max(((q, v) for q, v in cache.items() if v[0] <= budget),
                                  key=lambda item: item[1][0])
    if (budget-size)/budget > .0025:
        raise ValueError('HEVC did not match the total byte budget within 0.25%')
    destination = folder/'h265_matched.mp4'
    shutil.copyfile(candidate, destination)
    decoded = subprocess.check_output(common+['-i', str(destination), '-map', '0:v:0',
        '-vf', 'scale=in_color_matrix=bt709:in_range=tv,format=rgb24', '-f', 'rawvideo', 'pipe:1'])
    rgb = np.frombuffer(decoded, 'u1').reshape(150, 288, 512, 3)
    np.save(folder/'rgb.npy', rgb, allow_pickle=False)
    result = dict(target_bytes=budget, bytes=size, kbps=size*8/10000, crf=tick/scale,
                  relative_shortfall=(budget-size)/budget, sha256=file_sha(destination),
                  rgb_array_sha256=array_sha(rgb), preset='medium', x265_params=X265_PARAMS,
                  candidates={str(q/scale): size for q,(size,_) in sorted(cache.items())})
    save_json(folder/'receipt.json', result)
    return rgb, result


def run(workspace, output, ffmpeg, case):
    output.mkdir(parents=True, exist_ok=False)
    root = workspace/'02_实验数据/优化_GLC后处理'
    batch = root/'source_dense_flow_three_clips_20261109_01'/case
    run_root = root/'compact_support_20261110_01'
    selected = load_json(run_root/'selected.json')['profiles'][case]['E']
    folder = run_root/case/selected['name']
    receipt = load_json(folder/'receive/receipt.json')
    original_receipt = load_json(batch/'base_receive/result.json')
    source_path = batch/'source.npy'
    raw_path = batch/'base_receive/decoded_rgb.npy'
    optimized_path = folder/'receive/rgb.npy'
    assert file_sha(raw_path) == original_receipt['decoded_rgb_sha256']
    assert file_sha(optimized_path) == receipt['outputs_sha256']['rgb.npy']
    assert file_sha(batch/'base.glcsnd') == receipt['base_sha256']
    assert file_sha(folder/'support.glccsp') == receipt['aux_sha256']
    source = np.load(source_path, mmap_mode='r')
    raw = np.load(raw_path, mmap_mode='r')
    optimized = np.load(optimized_path, mmap_mode='r')
    base_bytes = (batch/'base.glcsnd').stat().st_size
    aux_bytes = (folder/'support.glccsp').stat().st_size
    hevc, matched = match_hevc(ffmpeg, source, run_root/case/'hevc', base_bytes+aux_bytes,
                               output/'matched_hevc')
    videos = [source, hevc, raw, optimized]
    assert all(v.shape == (150, 288, 512, 3) and v.dtype == np.uint8 for v in videos)
    base_rate, aux_rate = base_bytes*8/10000, aux_bytes*8/10000
    total_rate = base_rate+aux_rate
    header_height, width, height = 56, 1024, 688
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 21)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    headings = ['原视频', f'H.265   {matched["kbps"]:.2f} kbps',
                f'原始 GLC   {base_rate:.2f} kbps', f'优化 GLC   {total_rate:.2f} kbps']
    notes = ['', '', '', f'本体 {base_rate:.2f} + 辅助 {aux_rate:.2f} kbps']
    canvas = Image.new('RGB', (width, height), (20, 29, 40))
    draw = ImageDraw.Draw(canvas)
    positions = []
    for i, (title, note) in enumerate(zip(headings, notes, strict=True)):
        x, y = (i % 2)*512, (i//2)*(288+header_height)
        assert draw.textbbox((0, 0), title, font=font)[2] < 492
        assert draw.textbbox((0, 0), note, font=small)[2] < 492
        draw.text((x+10, y+4), title, fill=(239, 246, 252), font=font)
        if note:
            draw.text((x+10, y+31), note, fill=(169, 198, 218), font=small)
        positions.append((x, y+header_height))
    path = output/f'{case}_四宫格_原片_H265_原GLC_优化GLC.mp4'
    command = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin',
        '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
        '-framerate', '15', '-i', 'pipe:0', '-an',
        '-vf', 'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p',
        '-color_range', 'tv', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '12', '-threads', '4',
        '-movflags', '+faststart', '-n', str(path)]
    with (output/'preview.ffmpeg.log').open('wb') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            for frame in range(150):
                for video, position in zip(videos, positions, strict=True):
                    canvas.paste(Image.fromarray(video[frame]), position)
                process.stdin.write(canvas.tobytes())
            process.stdin.close()
            if process.wait(timeout=60):
                raise RuntimeError('Four-panel encoding failed')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    probe = json.loads(subprocess.check_output([str(ffmpeg.with_name('ffprobe.exe')), '-v', 'error',
        '-select_streams', 'v:0', '-count_frames', '-show_entries',
        'stream=width,height,avg_frame_rate,nb_read_frames,duration', '-of', 'json', str(path)]))['streams'][0]
    assert int(probe['nb_read_frames']) == 150 and Fraction(probe['avg_frame_rate']) == 15
    assert (probe['width'], probe['height']) == (width, height) and float(probe['duration']) == 10.
    encoded = subprocess.check_output([str(ffmpeg), '-v', 'error', '-i', str(path),
        '-vf', 'scale=in_color_matrix=bt709:in_range=tv,format=rgb24', '-f', 'rawvideo', 'pipe:1'])
    frames = np.frombuffer(encoded, 'u1').reshape(150, height, width, 3)
    errors = []
    for video, (x, y) in zip(videos, positions, strict=True):
        differences = [float(np.abs(frames[i, y:y+288, x:x+512].astype(np.float32)-video[i]).mean())
                       for i in (0, 75, 149)]
        assert max(differences) < 6
        errors.append(differences)
    Image.fromarray(frames[75]).save(output/'verified_frame75.png')
    save_json(output/'verification.json', dict(complete=True, case=case, frames=150, fps=15,
              duration=10, layout=['source', 'matched_hevc', 'original_glc', 'optimized_L'],
              headings=headings, notes=notes, base_bytes=base_bytes, aux_bytes=aux_bytes,
              optimized_total_kbps=total_rate, matched_hevc=matched, probe=probe,
              pixel_panel_origins=positions, check_frames=[0,75,149], panel_mae=errors,
              output_sha256=file_sha(path), output=str(path),
              inputs_sha256={str(p):file_sha(p) for p in (source_path,raw_path,optimized_path)},
              labels_are_complete_transmission_file_rates_not_preview_rate=True,
              code_sha256=file_sha(__file__)))
    shutil.copyfile(__file__, output/Path(__file__).name)
    print('PASS', str(path), 'H265', matched['kbps'], 'optimized', total_rate, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    parser.add_argument('--case', choices=('video1','video2','video3'), default='video1')
    args = parser.parse_args()
    run(args.workspace.resolve(), args.output.resolve(), args.ffmpeg.resolve(), args.case)
