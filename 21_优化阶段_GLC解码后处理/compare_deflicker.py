"""Compare the restored pilot to the same source, GLC and three-frame RGB."""

import argparse
import csv
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from run_deflicker import FPS, SHAPE, START, STOP
from run_p0 import HERE, REPO, file_sha, load_json, load_range, open_baseline, preview, psnr, roi, save_json
from experiment_common import array_sha
from probe_reef import probe

METHODS = ('raw', 'center3', 'deflicker')
LABELS = ('Source RGB', 'GLC raw', '3 frames / mean', 'All-In-One-Deflicker')


def save_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verified_pilot(workspace, previous, model_run):
    inputs = load_json(model_run / 'inputs.json')
    result = load_json(model_run / 'run.json')
    restored_path = model_run / 'deflicker_rgb.npy'
    if (not result['complete'] or result['frames'] != STOP - START or result['fps'] != FPS
            or inputs['source_start'] != START or inputs['source_stop'] != STOP
            or result['inputs_sha256'] != file_sha(model_run / 'inputs.json')
            or result['output_rgb_sha256'] != file_sha(restored_path)):
        raise ValueError('Missing, incomplete or changed model result')
    plan_path = HERE / 'plan_p0.json'
    manifest = workspace / '01_项目/GLC后处理/接口与基线.json'
    if (file_sha(plan_path) != inputs['plan_sha256']
            or file_sha(manifest) != inputs['baseline_manifest_sha256']):
        raise ValueError('The comparison baseline changed')
    plan = load_json(plan_path)
    clip = dict(inputs['clip'], start=START, stop=STOP)
    baseline = next(row for row in load_json(manifest)['baselines'] if row['id'] == clip['baseline'])
    decoded, reference, receipt = open_baseline(workspace, baseline, plan['codec'])
    if receipt != inputs['glc_baseline']:
        raise ValueError('The decoded RGB differs from model input')
    source = load_range(reference, START - 1, STOP + 1, SHAPE)
    raw = load_range(decoded, START - 1, STOP + 1, SHAPE)
    if array_sha(raw[1:-1]) != inputs['raw_rgb_sha256']:
        raise ValueError('The comparison range differs from model input')
    prior = load_json(previous / 'summary.json')
    old = next(item for item in prior['clips'] if item['clip']['id'] == clip['id'])
    if not prior['complete'] or prior['center3_fusion'] != 'mean' or old['clip'] != inputs['clip']:
        raise ValueError('Expected the original mean three-frame control')
    mean_path = previous / clip['id'] / 'center3_rgb.npy'
    if file_sha(mean_path) != old['output_sha256']['center3']:
        raise ValueError('The three-frame control changed')
    offset = START - old['clip']['start']
    mean3 = np.array(np.load(mean_path, mmap_mode='r', allow_pickle=False)[offset:offset + STOP - START])
    restored = np.load(restored_path, mmap_mode='r', allow_pickle=False)
    if any(frames.shape != raw[1:-1].shape or frames.dtype != np.uint8 for frames in (mean3, restored)):
        raise ValueError('A comparison lost geometry, frames or RGB precision')
    # The prior run had neighbors outside this pilot. Apply the same
    # no-neighbor fallback as the three-frame filter at the new endpoints.
    mean3[0], mean3[-1] = raw[1], raw[-2]
    return clip, plan['codec'], source, raw, mean3, restored, result


def perceptual(source, methods, root):
    os.environ['TORCH_HOME'] = str(root / 'torch')
    sys.path.insert(0, str(root / 'metric_packages'))
    import torch
    import lpips
    from DISTS_pytorch import DISTS

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = root / 'torch/hub/checkpoints'
    backbone_hashes = {}
    for filename, prefix in [('alexnet-owt-7be5be79.pth', '7be5be79'), ('vgg16-397923af.pth', '397923af')]:
        digest = file_sha(checkpoint / filename)
        if not digest.startswith(prefix):
            raise ValueError(f'Unexpected perceptual backbone: {filename}')
        backbone_hashes[filename] = digest
    cal_path = root / 'metric_packages/DISTS_pytorch/weights.pt'

    def tensor(frame):
        return torch.from_numpy(np.array(frame)).permute(2, 0, 1)[None].cuda().float() / 255

    rows = []
    with torch.inference_mode():
        alex = lpips.LPIPS(net='alex', version='0.1', verbose=False).eval().cuda().float()
        dists = DISTS(load_weights=False)
        calibration = torch.load(cal_path, map_location='cpu', weights_only=True)
        dists.alpha.copy_(calibration['alpha'])
        dists.beta.copy_(calibration['beta'])
        dists = dists.eval().cuda().float()
        for i, original in enumerate(source):
            x = tensor(original)
            row = dict(frame=START + i)
            for method, video in methods.items():
                y = tensor(video[i])
                row[f'{method}_lpips'] = float(alex(x, y, normalize=True).item())
                row[f'{method}_dists'] = float(dists(x, y).item())
            rows.append(row)
    return rows, dict(backbone_hashes=backbone_hashes, dists_calibration_sha256=file_sha(cal_path),
                      lpips_code_sha256=file_sha(lpips.__file__), torch=str(torch.__version__))


def temporal_plot(path, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, layout='constrained')
    valid = [row for row in rows if row['valid_pixels']]
    x = [row['frame'] / FPS for row in valid]
    colors = ('#444950', '#bb6520', '#087f88')
    for axis, band, title in zip(axes, ('all', 'fine'), ('Full-band luma', 'Fine luma'), strict=True):
        for method, label, color in zip(METHODS, LABELS[1:], colors, strict=True):
            y = [row[f'{method}_{band}_sum'] / row['valid_pixels'] for row in valid]
            axis.plot(x, y, label=label, color=color, linewidth=1.5)
        axis.axvline(192 / FPS, color='#8b4c93', linewidth=1, linestyle=':', label='I frame at 6.40 s')
        axis.set_ylabel('Temporal MAE')
        axis.set_title(title, loc='left')
        axis.grid(alpha=.18)
    axes[0].legend(loc='upper left', ncol=2, fontsize=8)
    axes[-1].set_xlabel('Time in source video (seconds)')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--model-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    parser.add_argument('--metrics-root', type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    clip, codec, source, raw, mean3, restored, result = verified_pilot(
        args.workspace, args.previous, args.model_run)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    methods = dict(zip(METHODS, (raw[1:-1], mean3, restored), strict=True))
    panels = (source[1:-1], *methods.values())
    previews = {}
    for name, crop in [('compare.mp4', None), ('zoom.mp4', roi(clip['coral_roi'], SHAPE))]:
        previews[name] = preview(args.ffmpeg, output / name, panels, FPS, START, LABELS, crop)

    # Use exactly the frozen source-motion and frequency-band protocol.
    # Its generic fourth candidate was called "repair" in the previous trial.
    raw_rows, probe_stats = probe(source, raw, mean3, clip, codec, repair=restored)
    temporal_rows = [
        {key.replace('repair_', 'deflicker_'): value for key, value in row.items()
         if not key.startswith('source_flow_control_')}
        for row in raw_rows
    ]
    for row in temporal_rows:
        if row['frame'] % codec['intra_period'] == 1:
            row['kind'] = 'first_p_after_i'
    save_csv(output / 'per_frame_temporal.csv', temporal_rows)
    temporal_plot(output / 'temporal_curve.png', temporal_rows)
    key = {'raw': 'raw', 'center3': 'center3', 'deflicker': 'repair'}
    stats = {}
    for method, video in methods.items():
        mse = np.mean(np.square(video.astype(np.float32) - source[1:-1]), dtype=np.float64)
        stats[method] = dict(rgb_psnr_db=psnr(float(mse)), **probe_stats['spatial'][key[method]],
                             temporal_mae=probe_stats['error'][key[method]])
    metric_provenance = None
    if args.metrics_root is not None:
        scores, metric_provenance = perceptual(source[1:-1], methods, args.metrics_root.resolve())
        save_csv(output / 'per_frame_perceptual.csv', scores)
        for method in methods:
            for name in ('lpips', 'dists'):
                stats[method][name] = float(np.mean([row[f'{method}_{name}'] for row in scores]))
    boundary = {}
    for i in (192, 193):
        row = next(row for row in temporal_rows if row['frame'] == i)
        boundary[str(i)] = {m: row[f'{m}_all_sum'] / row['valid_pixels'] for m in METHODS}
    save_json(output / 'summary.json', dict(
        clip=clip, fps=FPS, shape=SHAPE, frames=STOP - START,
        paired_temporal_frames=len(temporal_rows), valid_temporal_pixels=probe_stats['valid_pixels'],
        center3_endpoint_rule='current raw RGB at both pilot endpoints',
        methods=stats, boundary_full_band_mae=boundary, previews=previews,
        model_run_sha256=file_sha(args.model_run / 'run.json'),
        model_rgb_sha256=result['output_rgb_sha256'], metric_provenance=metric_provenance))
    print('PASS: verified both 30-FPS comparison previews and lossless evaluation', flush=True)


if __name__ == '__main__':
    main()
