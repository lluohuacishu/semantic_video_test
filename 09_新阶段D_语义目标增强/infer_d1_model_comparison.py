"""Three new frozen profiles for a 2x2 D1 checkpoint/preprocessing comparison."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import time

from infer_d1 import read_json, save_json, load_model, tiled_fish, np, Image, torch, F


def letterbox_fish(model, image):
    width, height = image.size
    ratio = 256/max(width,height)
    sw, sh = max(1,round(width*ratio)),max(1,round(height*ratio))
    left,top = (256-sw)//2,(256-sh)//2
    canvas = Image.new('RGB',(256,256))
    canvas.paste(image.resize((sw,sh),Image.Resampling.BILINEAR),(left,top))
    tensor=torch.from_numpy(np.asarray(canvas).copy()).permute(2,0,1)[None].float().to('cuda')/255
    logits=model(tensor)[:,:,top:top+sh,left:left+sw]
    return F.interpolate(logits,size=(height,width),mode='bilinear',align_corners=False).softmax(1)[0,6].cpu().numpy()


def run(args):
    comparison=read_json(args.plan)
    old_plan=read_json(args.base_run/'plan.json')
    inputs=read_json(args.base_run/'inputs.json')['inputs']
    frozen=read_json(args.base_run/'frozen_protection.json')
    if len(inputs)!=216 or any(r.get('video_id') not in (None,1,2,3) for r in inputs):
        raise ValueError('Only the 216 original D1 development inputs are allowed')
    if frozen['selected_name']!='recall_hold1':
        raise ValueError('This comparison requires the already frozen D1 diagnostic rule')
    args.output.mkdir(parents=True,exist_ok=False)
    args.output_created=True
    save_json(args.output/'comparison_plan.json',comparison)
    save_json(args.output/'inputs.json',{'inputs':inputs})
    torch.set_num_threads(4)
    torch.manual_seed(20261013)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    started=time.perf_counter()
    models={
        'suim_base':load_model(old_plan['models']['diver']['checkpoint']),
        'fish_adapted':load_model(old_plan['models']['fish']['checkpoint']),
    }
    profile_runs={'adapted_tiles512':str(args.base_run.resolve())}
    summaries={}
    for profile in comparison['profiles'][1:]:
        name=profile['name']
        output=args.output/name
        output.mkdir()
        (output/'packed').mkdir()
        plan=copy.deepcopy(old_plan)
        source_key='diver' if profile['weights']=='suim_base' else 'fish'
        plan['models']['fish']['checkpoint']=old_plan['models'][source_key]['checkpoint']
        plan['models']['fish']['previously_recorded_sha256']=old_plan['models'][source_key]['previously_recorded_sha256']
        plan['models']['fish']['inference']=profile['preprocessing']
        plan['model_policy']='Registered D1 checkpoint/preprocessing comparison, fixed protection and cached diver branch'
        plan['candidates']=[frozen['config']]
        plan['evaluation']['selection']='Only the frozen recall_hold1 configuration is evaluated for this fish profile; no additional threshold/margin search.'
        plan['comparison_profile']=profile
        plan['comparison_plan']=str(args.plan.resolve())
        plan['cached_diver_source']=str(args.base_run.resolve())
        plan['SUIM_base_regression_reused_not_profile_specific']=True
        save_json(output/'plan.json',plan)
        save_json(output/'inputs.json',{'inputs':inputs})
        rows=[]
        profile_started=time.perf_counter()
        print('Running '+name,flush=True)
        with (output/'predictions.jsonl').open('x',encoding='utf-8') as records:
            import json
            with torch.inference_mode():
                for idx,source in enumerate(inputs):
                    with Image.open(source['rgb_path']) as original:
                        image=original.convert('RGB')
                    torch.cuda.synchronize()
                    t0=time.perf_counter()
                    if profile['preprocessing']=='tiles512_stride384':
                        probability=tiled_fish(models[profile['weights']],np.asarray(image).copy())
                    else:
                        probability=letterbox_fish(models[profile['weights']],image)
                    seconds=time.perf_counter()-t0
                    filename=f'{idx:04d}.png'
                    # Copy only the already computed diver bits; no GT is accessed.
                    packed=np.asarray(Image.open(args.base_run/'packed'/filename)).copy() & 24
                    for threshold in comparison['fish_thresholds']:
                        bit=plan['packed_bits'][f'fish_{threshold}']
                        packed |= (probability>=threshold).astype(np.uint8)<<bit
                    Image.fromarray(packed).save(output/'packed'/filename)
                    row={'input_index':idx,'id':source['id'],'packed_path':'packed/'+filename,
                         'base_labels_path':str((args.base_run/'base_labels'/filename).resolve()) if 'video_id' not in source else None,
                         'fish_seconds':seconds,'diver_seconds':0.0,'diver_result_reused':True}
                    rows.append(row)
                    records.write(json.dumps(row,ensure_ascii=False)+'\n');records.flush()
                    if (idx+1)%32==0 or idx+1==len(inputs):
                        print(f'{name}: {idx+1}/216, {time.perf_counter()-profile_started:.1f}s',flush=True)
                        save_json(args.output/'progress.json',{'status':'running','profile':name,'completed':idx+1,'total':216})
        summary={'status':'completed','profile':profile,'input_count':len(rows),
                 'process_work_seconds':time.perf_counter()-profile_started,
                 'fish_inference_seconds':sum(r['fish_seconds'] for r in rows),
                 'diver_inference_seconds':0,'diver_reused_from':str(args.base_run.resolve()),
                 'SUIM_base_labels_reused':True,'gt_loaded':False,'training':False,'codec_run':False,
                 'integrity_rescan_performed':False,'torch':torch.__version__,'device':torch.cuda.get_device_name(0)}
        save_json(output/'inference_summary.json',summary)
        summaries[name]=summary
        profile_runs[name]=str(output.resolve())
    save_json(args.output/'profile_runs.json',profile_runs)
    save_json(args.output/'inference_summary.json',{'status':'completed','new_profile_count':3,
              'reused_profile_count':1,'new_frame_profile_inferences':648,
              'process_work_seconds':time.perf_counter()-started,'profiles':summaries,
              'gt_loaded':False,'training':False,'codec_run':False,'integrity_rescan_performed':False})
    save_json(args.output/'progress.json',{'status':'completed','new_profiles':3,'inputs_each':216})
    print(f'All new profiles complete in {time.perf_counter()-started:.1f}s',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=Path(__file__).with_name('d1_model_comparison_plan_v1.json'))
    parser.add_argument('--base-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        if getattr(args,'output_created',False):
            save_json(args.output/'failure.json',{'type':type(exc).__name__,'message':str(exc)})
        raise
