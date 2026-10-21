"""Prepare the fixed two-clip D2 controls from existing source RGB and masks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from d1_protection import hard_protection, alpha_protection


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, data):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def blend(original, blurred, alpha):
    weight = alpha.astype(np.float32)[..., None]/255
    return np.rint(weight*original + (1-weight)*blurred).clip(0,255).astype(np.uint8)


def run(args):
    plan = read_json(args.plan)
    manifest = read_json(args.corpus/'ready_manifest.json')
    predicted = {row['id']: row for row in read_json(args.predictions/'selected_outputs.json')['outputs']}
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output/'plan.json', plan)
    started = time.perf_counter()
    clips = []
    count = 0
    cfg = plan['preprocessing']['protection']
    for video in plan['scope']['video_ids']:
        sequence = next(s for s in manifest['fish_sequences'] if s['video_id'] == video)
        if sequence['role'] != 'development' or sequence['fps'] != '30000/1001' or len(sequence['frames']) != 64:
            raise ValueError('Prepared source differs from the registered D2 scope')
        for method in ('B','C','O'):
            (args.output/'inputs'/f'video_{video:03d}'/method).mkdir(parents=True)
        rows = []
        previous_gt = None
        for frame in sequence['frames']:
            index = frame['index']
            source_path = args.corpus/frame['rgb_path']
            rgb = np.asarray(Image.open(source_path).convert('RGB'))
            gt = np.asarray(Image.open(args.corpus/frame['mask_path'])) != 0
            if rgb.shape != (1080,1920,3) or gt.shape != rgb.shape[:2]:
                raise ValueError('D2 does not silently resize or reinterpret frame alignment')
            sigma = plan['preprocessing']['sigma']
            blurred = np.rint(gaussian_filter(rgb.astype(np.float32), sigma=(sigma,sigma,0),
                                            mode='reflect', truncate=4)).clip(0,255).astype(np.uint8)
            p = predicted[f'sfishtrack_{video:03d}/{frame["original_file_name"]}']
            alpha = np.asarray(Image.open(args.predictions/p['alpha_path']))
            hard_gt = hard_protection(gt, previous_gt, cfg)
            oracle_alpha, fallback = alpha_protection(gt, hard_gt)
            images = {'B':blurred, 'C':blend(rgb,blurred,alpha), 'O':blend(rgb,blurred,oracle_alpha)}
            # Target-core preservation is a preprocessing requirement, not a codec quality claim.
            if not np.array_equal(images['C'][alpha==255],rgb[alpha==255]):
                raise ValueError('Prediction-protected core changed in preprocessing')
            if not np.array_equal(images['O'][gt],rgb[gt]):
                raise ValueError('GT core changed in preprocessing')
            paths = {'A':str(source_path.resolve())}
            for method, value in images.items():
                target = args.output/'inputs'/f'video_{video:03d}'/method/f'{index:04d}.png'
                Image.fromarray(value).save(target, compress_level=1)
                paths[method] = str(target.resolve())
            rows.append({'index':index,'source_index':frame['source_frame_index'],
                         'source_pts':frame['source_pts'],'gt_path':str((args.corpus/frame['mask_path']).resolve()),
                         'paths':paths,'prediction_fallback':p['fallback'],'oracle_fallback':fallback,
                         'prediction_alpha_mean':float(alpha.mean()/255),
                         'oracle_alpha_mean':float(oracle_alpha.mean()/255)})
            previous_gt = gt
            count += 1
            if count%16==0:
                print(f'Prepared {count}/128 source frames and B/C/O controls',flush=True)
                save_json(args.output/'preparation_progress.json',{'state':'running','frames':count,'total':128})
        clips.append({'video_id':video,'width':1920,'height':1080,'frames':64,
                      'fps':sequence['fps'],'duration_exact':sequence['duration_seconds_exact'],'records':rows})
    save_json(args.output/'inputs.json',{'clips':clips})
    save_json(args.output/'preparation.json',{'state':'completed','frames':count,'new_lossless_images':3*count,
              'seconds':time.perf_counter()-started,'source_corpus':str(args.corpus.resolve()),
              'prediction_source':str(args.predictions.resolve()),'integrity_rescan':False,
              'protected_cores_preserved':True,'training':False,'new_semantic_inference':False})
    save_json(args.output/'preparation_progress.json',{'state':'completed','frames':count,'total':128})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=Path(__file__).with_name('d2_run_plan_v1.json'))
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
