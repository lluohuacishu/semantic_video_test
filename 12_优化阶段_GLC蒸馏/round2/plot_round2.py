"""Scientific plots; CPU matplotlib environment, no model loading."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot_common import load_json


def save(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=170)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = load_json(args.output/'plan.json')
    quality = load_json(args.output/'evaluation/quality.json')
    cases = list(plan['training']['cases'])
    labels = dict(random_mixed='Random / mixed L1', mapped_mixed='Mapped / mixed L1',
                  mapped_teacher='Mapped / teacher L1', mapped_gt='Mapped / original L1')
    colors = dict(zip(cases, ['#d1495b', '#00798c', '#3066be', '#c88c1d']))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for case in cases:
        folder = args.output/f'train_{case}'
        rows = [json.loads(x) for x in (folder/'loss.jsonl').read_text('utf-8').splitlines()]
        blocks = [rows[i:i+100] for i in range(0, len(rows), 100)]
        axes[0].plot([b[-1]['step'] for b in blocks], [np.mean([r['l1_original'] for r in b]) for b in blocks], label=labels[case], color=colors[case])
        vals = load_json(folder/'validation.json')
        axes[1].plot([r['step'] for r in vals], [r['psnr_db'] for r in vals], 'o-', label=labels[case], color=colors[case], markersize=3)
    val = next(c for c in quality['clips'] if c['id']=='uv096')
    teacher = val['models']['teacher']['groups']['p_only']['psnr_db']
    axes[1].axhline(teacher, color='#52616b', ls='--', label=f'Teacher {teacher:.2f} dB')
    axes[0].set(xlabel='Actual optimizer updates', ylabel='L1 to original in [-1,1]', title='Common observable; 100-step means')
    axes[1].set(xlabel='Actual optimizer updates', ylabel='PSNR to original (dB)', title='UV96 validation P frames; FP32')
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    save(fig, args.output/'evaluation/learning_curves_round2')
    fig, axis = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    clip_order = ['uv096', 'uv108', 'sf026', 'v007', 'v025']
    x = np.arange(len(clip_order))
    for j, mode in enumerate(['teacher']+cases):
        values = []
        for cid in clip_order:
            clip = next(c for c in quality['clips'] if c['id']==cid)
            values.append(clip['models'][mode]['groups']['p_only']['psnr_db'])
        axis.bar(x+(j-2)*.15, values, .145, label='Teacher' if mode=='teacher' else labels[mode], color='#52616b' if mode=='teacher' else colors[mode])
    axis.set(xticks=x, xticklabels=['UV96\nvalidation', 'UV108\nsequence holdout', 'SF26\nrelated capture holdout', 'SF7\nregression', 'SF25\nregression'], ylabel='P-frame PSNR to original (dB)', title='Fixed-data comparison; capture independence remains unverified')
    axis.legend(fontsize=8, ncol=3)
    axis.grid(axis='y', alpha=.2)
    save(fig, args.output/'evaluation/quality_comparison_round2')


if __name__ == '__main__':
    main()
