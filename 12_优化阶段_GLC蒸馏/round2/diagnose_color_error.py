"""Post-hoc MSE decomposition; never alters a model, output RGB or selection."""
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot_common import load_json, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    quality = load_json(args.output/'evaluation/quality.json')
    case, rows = quality['selected_case'], []
    for clip in quality['clips']:
        original = np.load(args.output/clip['id']/'source.npy', mmap_mode='r')
        rendered = np.load(args.output/'evaluation/full_decode'/f'{clip["id"]}_{case}_0/rgb.npy', mmap_mode='r')
        mses, bias, means = [], [], []
        for i in range(len(original)):
            if i % 64 == 0:
                continue
            error = rendered[i].astype('float64')-original[i].astype('float64')
            mean = error.mean(axis=(0, 1))
            mses.append(float(np.mean(error**2)))
            bias.append(float(np.mean(mean**2)))
            means.append(mean)
        rows.append(dict(clip=clip['id'], case=case, p_mse=float(np.mean(mses)),
                         spatially_constant_channel_bias_mse=float(np.mean(bias)),
                         bias_fraction=float(np.mean(bias)/np.mean(mses)),
                         mean_signed_error_rgb=np.mean(means, axis=0).tolist()))
    target = args.output/'evaluation/posthoc_color_error.json'
    if args.verify:
        previous = load_json(target)['rows']
        for actual, expected in zip(rows, previous):
            assert actual['clip']==expected['clip'] and actual['case']==expected['case']
            for key in ['p_mse', 'spatially_constant_channel_bias_mse', 'bias_fraction', 'mean_signed_error_rgb']:
                assert np.allclose(actual[key], expected[key], atol=1e-9, rtol=1e-12)
        assert len(rows)==len(previous)
        print('Post-hoc color decomposition reproduced; original output files unchanged', flush=True)
    else:
        if target.exists():
            raise FileExistsError(target)
        save_json(target, dict(scope='Post-hoc original-RGB-based error diagnosis only; not deployable correction or new selection',
                  formula='MSE = mean_c(mean_HW(error_c)^2) + mean_c(var_HW(error_c)) per frame', rows=rows))


if __name__=='__main__':
    main()
