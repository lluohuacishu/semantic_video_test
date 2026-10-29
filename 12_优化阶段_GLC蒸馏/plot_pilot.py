"""Standalone scientific plots using the existing CPU matplotlib environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pilot_common import load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    quality = load_json(args.output / 'evaluation/quality.json')
    val = next(c for c in quality['clips'] if c['role'] == 'validation')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    colors = dict(kd='#00798c', gt='#d1495b')
    for mode in ['kd', 'gt']:
        folder = args.output / f'train_{mode}'
        rows = [json.loads(line) for line in (folder / 'loss.jsonl').read_text('utf-8').splitlines()]
        # Plot a common observable, GT L1, since optimized KD/GT losses differ.
        a = np.array([r['l1_original'] for r in rows])
        block = 100
        mean = [a[i:i+block].mean() for i in range(0, len(a), block)]
        x = [min(i+block, len(a)) for i in range(0, len(a), block)]
        axes[0].plot(x, mean, label=mode.upper() + ' (100-step mean)', color=colors[mode])
        metrics = load_json(folder / 'validation.json')
        axes[1].plot([r['step'] for r in metrics], [r['psnr_db'] for r in metrics], 'o-',
                     label=mode.upper(), color=colors[mode], markersize=3)
    axes[0].set(xlabel='Training updates', ylabel='L1 to original in [-1, 1]', title='Training: identical frame sampling')
    teacher_psnr = val['models']['teacher']['groups']['p_only']['psnr_db']
    axes[1].axhline(teacher_psnr, color='#52616b', ls='--', label=f'Teacher ({teacher_psnr:.2f} dB)')
    axes[1].set(xlabel='Training updates', ylabel='PSNR to original (dB)', title='Held-out v007: P frames, FP32')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(args.output / 'evaluation/learning_curves.png', dpi=170)
    fig.savefig(args.output / 'evaluation/learning_curves.svg')
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True, sharex=True)
    for mode, color in [('teacher', '#52616b'), ('kd', colors['kd']), ('gt', colors['gt'])]:
        rows = val['models'][mode]['frames']
        axes[0].plot([r['index']+1 for r in rows], [r['psnr_db'] for r in rows], label=mode.upper(), color=color)
        axes[1].plot([r['index']+1 for r in rows[1:]], [r['frame_index_error_delta_mae_rgb'] for r in rows[1:]], color=color)
    for ax in axes:
        ax.axvline(65, color='black', ls=':', alpha=.6)
        ax.grid(alpha=.2)
    axes[0].set(ylabel='Frame PSNR (dB)', title='v007 full frame order; dotted line = unchanged I frame')
    axes[0].legend()
    axes[1].set(xlabel='Annotation frame index (1-based)', ylabel='Error-change MAE (0..255)',
                title='Uncompensated proxy only: no perceptual flicker claim; 96-frame timing not fully verified')
    fig.savefig(args.output / 'evaluation/frame_metrics_v007.png', dpi=170)
    fig.savefig(args.output / 'evaluation/frame_metrics_v007.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
