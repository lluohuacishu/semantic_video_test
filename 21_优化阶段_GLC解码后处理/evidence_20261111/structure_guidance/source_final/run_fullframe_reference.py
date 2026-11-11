"""Full-frame persistent RGB reference controls on three frozen source clips.

One common schedule and matching rule. No hand-drawn polygons, crop-only edit,
color-specific fish detection, or source RGB replacement anchors.
"""

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from cache_motion_inputs import REPO, base_frames
from compare_source_dense_flow import load_flow
from coral_reference_sidecar import file_sha, load_json, save_json
from experiment_common import array_sha
from run_dense_batch import CASES
from run_dense_transfer import source_input
from run_p0 import preview
from source_raft_flow import WEIGHT_SHA, ITERATIONS, infer
from temporal_filter import correspondence


KEYS = (0, 30, 60, 90, 120)
RAMP = (1., 5/6, 4/6, 3/6, 2/6, 1/6)
EVENTS = tuple(k+j for k in KEYS[1:] for j in range(len(RAMP)))


def get_inputs(batch, case):
    clip = CASES[case][0]
    base = batch/case
    source = source_input(base, clip)
    raw, digest, size = base_frames(base, clip)
    return source, raw, digest, size


def flow(batch, case, root, upstream, keys=KEYS):
    import torch
    source, _, digest, _ = get_inputs(batch, case)
    clip = CASES[case][0]
    root.mkdir(parents=True, exist_ok=False)
    events=tuple(k+j for k in keys[1:] for j in range(len(RAMP)))
    save_json(root/'plan.json', dict(case=case,frames=150,fps=15,keys=keys,
        old_reference_weights=RAMP,full_frame_candidate_support=True,
        manual_roi=False,fish_color_rule=False,source_RGB_reference=False,
        common_parameters_all_cases=True,base_sha256=digest,
        criterion='Inside; bidirectional RAFT <1.5 px; sigma3 RGB diff <16; sigma1 RGB diff <24; motion gradient <0.75; 3x3 erosion; 2px feather.',
        current_scope='Full-frame offline diagnostic, followed separately by actual auxiliary encoding.',
        limitations='Correspondence reliability is not semantic object segmentation. Independently moving surfaces and thin structures require visual review.'))
    existing = load_flow(batch/case/'flow', source, clip)
    infer(source, clip, upstream, root/'flow', anchors=keys, periodic_flows=existing)
    sys.path.insert(0, str(upstream))
    from src.models.stage_1.core.raft import RAFT
    weight = upstream/'pretrained_weights/raft-things.pth'
    if file_sha(weight) != WEIGHT_SHA: raise ValueError('RAFT checkpoint changed')
    model = RAFT(argparse.Namespace(small=False,mixed_precision=False,alternate_corr=False))
    state = torch.load(weight,map_location='cpu',weights_only=True)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    model.cuda().eval()
    values = np.zeros((len(events),2,288,512,2),np.float32)
    start=time.perf_counter()
    with torch.inference_mode():
        for j,i in enumerate(events):
            old=keys[j//len(RAMP)]
            images=torch.from_numpy(source[[i,old]]).permute(0,3,1,2).cuda().float()
            _,result=model(images,images.flip(0),iters=ITERATIONS,test_mode=True)
            values[j]=result.permute(0,2,3,1).contiguous().cpu().numpy()
            if not np.isfinite(values[j]).all(): raise ValueError('Nonfinite transition flow')
    np.save(root/'transition_flow.npy',values,allow_pickle=False)
    save_json(root/'transition_flow.json',dict(events=events,keys=keys,
        order=['current_to_old','old_to_current'],source_rgb_sha256=array_sha(source),
        weights_sha256=WEIGHT_SHA,iterations=ITERATIONS,seconds=time.perf_counter()-start,
        flow_sha256=file_sha(root/'transition_flow.npy')))
    print(case,'direct old-reference flow complete',flush=True)


def reliable(source, current, reference, back, forward):
    mx,my,inside,cycle=correspondence(back,forward)
    low=cv2.GaussianBlur(source[current].astype(np.float32),(0,0),3)
    old_low=cv2.GaussianBlur(source[reference].astype(np.float32),(0,0),3)
    mid=cv2.GaussianBlur(source[current].astype(np.float32),(0,0),1)
    old_mid=cv2.GaussianBlur(source[reference].astype(np.float32),(0,0),1)
    dlow=np.max(np.abs(low-cv2.remap(old_low,mx,my,cv2.INTER_LINEAR)),axis=2)
    dmid=np.max(np.abs(mid-cv2.remap(old_mid,mx,my,cv2.INTER_LINEAR)),axis=2)
    grad=np.maximum(np.linalg.norm(np.gradient(back,axis=0),axis=2),
        np.linalg.norm(np.gradient(back,axis=1),axis=2))
    safe=inside&(cycle<1.5)&(dlow<16)&(dmid<24)&(grad<.75)
    safe=cv2.erode(safe.astype(np.uint8),np.ones((3,3),np.uint8))
    alpha=np.minimum(cv2.distanceTransform(safe,cv2.DIST_L2,5)/2.,1.)
    return alpha,mx,my


def render(batch,case,root,ffmpeg):
    source,raw,digest,size=get_inputs(batch,case)
    keys=tuple(load_json(root/'plan.json')['keys'])
    events=tuple(k+j for k in keys[1:] for j in range(len(RAMP)))
    back,forward=load_flow(root/'flow',source,CASES[case][0],anchors=keys)
    record=load_json(root/'transition_flow.json')
    if record['source_rgb_sha256']!=array_sha(source) or file_sha(root/'transition_flow.npy')!=record['flow_sha256']:
        raise ValueError('Transition input changed')
    transition=np.load(root/'transition_flow.npy',mmap_mode='r')
    out=root/'offline';out.mkdir(exist_ok=False)
    result=raw.copy();alphas=np.zeros(raw.shape[:3],np.float32)
    shared=np.zeros((len(events),288,512),np.float32)
    event_index={i:j for j,i in enumerate(events)}
    for i in range(len(raw)):
        key=keys[np.searchsorted(keys,i,side='right')-1]
        a,mx,my=reliable(source,i,key,back[i],forward[i])
        new=cv2.remap(raw[key].astype(np.float32),mx,my,cv2.INTER_CUBIC)
        current=np.rint(np.clip(raw[i]*(1-a[...,None])+new*a[...,None],0,255))
        if i in event_index:
            j=event_index[i]
            old_key=keys[j//len(RAMP)]
            old_a,old_x,old_y=reliable(source,i,old_key,*transition[j])
            shared[j]=np.minimum(a,old_a)
            old=cv2.remap(raw[old_key].astype(np.float32),old_x,old_y,cv2.INTER_CUBIC)
            weight=shared[j]*RAMP[j%6]
            current=np.rint(np.clip(current+weight[...,None]*(old-new),0,255))
        result[i]=current.astype(np.uint8);alphas[i]=a
    if not np.array_equal(result[alphas==0],raw[alphas==0]):raise ValueError('Zero-weight pixels changed')
    for name,value in dict(rgb=result,alpha=alphas,shared=shared).items():
        np.save(out/(name+'.npy'),value,allow_pickle=False)
    previews={}
    previews['full_pair.mp4']=preview(ffmpeg,out/'full_pair.mp4',[raw,result],15,0,
        ['Original GLC', 'F / full-frame persistent structure'])
    canvas=Image.new('RGB',(512*3,312*3),(24,35,48));draw=ImageDraw.Draw(canvas)
    for row,i in enumerate((0,75,149)):
        for col,(label,rgb) in enumerate((('Source',source[i]),('Full-frame result',result[i]),
                ('Green = applied support',np.rint(source[i]*(1-alphas[i,...,None]*.45)+np.array([0,255,170])*alphas[i,...,None]*.45).astype(np.uint8)))):
            canvas.paste(Image.fromarray(rgb),(col*512,row*312+24))
            draw.text((col*512+6,row*312+5),f'{case} {label} f{i}',fill='white')
    canvas.save(out/'support.png')
    quadrants={name:float((alphas[:,ys,xs]>0).mean()) for name,ys,xs in (
        ('top_left',slice(0,144),slice(0,256)),('top_right',slice(0,144),slice(256,512)),
        ('bottom_left',slice(144,288),slice(0,256)),('bottom_right',slice(144,288),slice(256,512)))}
    save_json(out/'summary.json',dict(complete=True,offline_source_assisted=True,
        mean_coverage=float((alphas>0).mean()),quadrant_coverage=quadrants,
        zero_alpha_unchanged=True,base_sha256=digest,base_bytes=size,previews=previews,
        outputs_sha256={p.name:file_sha(p) for p in out.glob('*.npy')},code_sha256=file_sha(__file__)))
    print(case,'full-frame render coverage',quadrants,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('flow','render'))
    parser.add_argument('--case',choices=CASES,required=True)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--upstream',type=Path)
    parser.add_argument('--ffmpeg',type=Path)
    parser.add_argument('--schedule',choices=('regular','warmup'),default='regular')
    args=parser.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    root=args.output.resolve()/args.case
    if args.action=='flow':
        if args.upstream is None:parser.error('--upstream required')
        flow(args.batch.resolve(),args.case,root,args.upstream.resolve(),
            KEYS if args.schedule=='regular' else (0,15,45,75,105,135))
    else:
        if args.ffmpeg is None:parser.error('--ffmpeg required')
        render(args.batch.resolve(),args.case,root,args.ffmpeg.resolve())


if __name__=='__main__':main()
