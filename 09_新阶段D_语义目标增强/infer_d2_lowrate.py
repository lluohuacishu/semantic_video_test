"""Frozen RGB-only protection for the two registered larger-fish clips."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
from scipy.ndimage import gaussian_filter
from infer_d1 import load_model, tiled_fish, letterbox_diver, torch, np, Image
from d1_protection import hard_protection, alpha_protection
from prepare_d2 import read_json, save_json, blend


def run(args):
    root=args.run.resolve(); plan=read_json(root/'plan.json'); gate=read_json(root/'rate_gate.json')
    if gate['state']!='passed': raise ValueError('Hard baseline rate gate must pass before semantic inference')
    clips=read_json(root/'inputs.json')['clips']; width,height=gate['size']
    if [c['video_id'] for c in clips]!=[c['video_id'] for c in plan['clips']]:
        raise ValueError('Input scope differs from registered development windows')
    out=root/'controls';out.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(20261013)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    start=time.perf_counter();model=load_model(plan['semantics']['checkpoint'])
    cfg={'margin_at_1920':16,'previous_seed_frames':1}; records=[]
    with torch.inference_mode():
        for clip in clips:
            folder=out/f'v{clip["video_id"]:03d}'
            for name in ('B','C','alpha','seed_original'): (folder/name).mkdir(parents=True)
            previous=None
            for f in clip['frames']:
                if (root/'STOP').exists(): raise RuntimeError('Stopped at semantic frame boundary')
                original=Image.open(f['original_rgb']).convert('RGB'); tick=time.perf_counter()
                fish=tiled_fish(model,np.asarray(original).copy())>=.5
                labels,confidence=letterbox_diver(model,original)
                diver=(labels==1)&(confidence>=.5); seed=fish|diver
                hard=hard_protection(seed,previous,cfg);alpha,fallback=alpha_protection(seed,hard)
                previous=seed
                alpha_small=np.asarray(Image.fromarray(alpha).resize((width,height),Image.Resampling.NEAREST))
                source=np.asarray(Image.open(f['resized'][f'{width}x{height}']['rgb']).convert('RGB'))
                blurred=np.rint(gaussian_filter(source.astype(np.float32),sigma=(1.2,1.2,0),mode='reflect',truncate=4)).clip(0,255).astype(np.uint8)
                protected=blend(source,blurred,alpha_small)
                if not np.array_equal(protected[alpha_small==255],source[alpha_small==255]):
                    raise ValueError('Protected source core changed before encoding')
                name=f'{f["index"]:04d}.png'
                for label,image in [('B',blurred),('C',protected),('alpha',alpha_small),('seed_original',fish.astype(np.uint8)+2*diver.astype(np.uint8))]:
                    Image.fromarray(image).save(folder/label/name,compress_level=1)
                row={'video_id':clip['video_id'],'index':f['index'],'seconds':time.perf_counter()-tick,
                     'fallback':fallback,'raw_seed_fraction':float(seed.mean()),'protected_alpha_mean':float(alpha_small.mean()/255)}
                records.append(row)
                save_json(root/'semantic_progress.json',{'state':'running','completed':len(records),'total':sum(len(c['frames']) for c in clips),'last_clip':clip['video_id']})
                if len(records)%16==0: print(f'Semantic controls {len(records)}/128',flush=True)
    save_json(root/'controls.json',{'state':'completed','frames':len(records),'size':[width,height],
             'seconds':time.perf_counter()-start,'records':records,'GT_pixels_loaded':False,'training':False,
             'integrity_rescan':False,'frozen_thresholds':True,'masks_transmitted':False})
    save_json(root/'semantic_progress.json',{'state':'completed','completed':len(records)})
    print('Frozen semantic controls completed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    try: run(a)
    except BaseException as exc:
        save_json(a.run/'semantic_failure.json',{'type':type(exc).__name__,'message':str(exc)})
        raise
