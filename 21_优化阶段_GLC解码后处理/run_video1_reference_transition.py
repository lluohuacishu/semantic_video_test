"""Test six-frame, motion-aligned transitions between full RGB references."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from cache_motion_inputs import FPS, FRAME_SHAPE, REPO
from compare_deflicker import save_csv
from coral_reference_sidecar import file_sha, load_json, save_json
from run_p0 import preview, roi
from run_structure_memory import LONG_ANCHORS
from run_video1_full_coral import orange_fish
from run_video1_persistent_long import mask_for, grouped
from run_video1_persistent_structure import VIEW_BOX, load_inputs, score
from temporal_filter import correspondence


RAMP=(1.,5/6,4/6,3/6,2/6,1/6)


def compose(first, second, xx, yy):
    mx,my=xx+first[...,0],yy+first[...,1]
    h,w=first.shape[:2]
    valid=(mx>=2)&(mx<w-2)&(my>=2)&(my<h-2)
    return first+cv2.remap(second,mx,my,cv2.INTER_LINEAR),valid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=REPO.parents[2])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--ffmpeg',type=Path,required=True)
    args=parser.parse_args()
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    root=args.run.resolve();out=root/'transition_result';out.mkdir(exist_ok=False)
    save_json(out/'plan.json',dict(frames=150,fps=FPS,source_assisted_offline=True,
        events=LONG_ANCHORS[1:],old_reference_weights=RAMP,transition_seconds=len(RAMP)/FPS,
        method='Crossfade complete motion-aligned old and new RGB references only on shared reliable support.',
        no_recursive_image_warp=True,shared_support='min(new reference alpha, old reference alpha)',
        unchanged_frames=150-len(RAMP)*len(LONG_ANCHORS[1:]),
        limit='Transitions may trade abrupt shape change for transient softness or morphing; original-speed review required.'))
    source,raw,s,back,forward,inputs=load_inputs(args.workspace.resolve())
    record=load_json(root/'direct_result/summary.json')
    names=('N2_rgb.npy','N2_alpha.npy','tracked.npy')
    for name in names:
        if file_sha(root/'direct_result'/name)!=record['outputs_sha256'][name]:raise ValueError('N2 changed')
    n2=np.load(root/'direct_result/N2_rgb.npy',allow_pickle=False)
    a2=np.load(root/'direct_result/N2_alpha.npy',allow_pickle=False)
    tracked=np.load(root/'direct_result/tracked.npy',allow_pickle=False)
    bridge_record=load_json(root/'bridges/flow.json')
    if file_sha(root/'bridges/bridge_flow.npy')!=bridge_record['flow_sha256']:raise ValueError('Bridge changed')
    bridge=np.load(root/'bridges/bridge_flow.npy',allow_pickle=False)
    result=n2.copy();weights=np.zeros(raw.shape[:3],np.float32)
    yy,xx=np.mgrid[:FRAME_SHAPE[0],:FRAME_SHAPE[1]].astype(np.float32)
    coverage=[]
    for slot,(old,new) in enumerate(zip(LONG_ANCHORS[:-1],LONG_ANCHORS[1:],strict=True)):
        old_low=cv2.GaussianBlur(source[old].astype(np.float32),(0,0),3)
        old_support=mask_for(old)*(1-orange_fish(source[old]))
        for offset,fade in enumerate(RAMP):
            i=new+offset
            to_old,valid_a=compose(back[i],bridge[slot,0],xx,yy)
            from_old,valid_b=compose(bridge[slot,1],forward[i],xx,yy)
            mx,my,inside,cycle=correspondence(to_old,from_old)
            mapped_support=cv2.remap(old_support*valid_b,mx,my,cv2.INTER_NEAREST).astype(bool)
            current_low=cv2.GaussianBlur(source[i].astype(np.float32),(0,0),3)
            difference=np.max(np.abs(cv2.remap(old_low,mx,my,cv2.INTER_LINEAR)-current_low),axis=2)
            safe=inside&valid_a&mapped_support&(cycle<1.5)&(difference<16)&~orange_fish(source[i]).astype(bool)
            safe=cv2.erode(safe.astype(np.uint8),np.ones((3,3),np.uint8))
            old_alpha=np.minimum(cv2.distanceTransform(safe,cv2.DIST_L2,5)/2.,1.)
            weight=np.minimum(a2[i],old_alpha)*fade
            old_rgb=cv2.remap(raw[old].astype(np.float32),mx,my,cv2.INTER_CUBIC)
            new_rgb=cv2.remap(raw[new].astype(np.float32),xx+back[i,...,0],yy+back[i,...,1],cv2.INTER_CUBIC)
            result[i]=np.rint(np.clip(n2[i].astype(np.float32)+weight[...,None]*(old_rgb-new_rgb),0,255)).astype(np.uint8)
            weights[i]=weight
            area=tracked[i].astype(bool)
            coverage.append(dict(frame=i,event=new,fade=fade,shared_pixels=int((weight>0).sum()),
                mean_weight_on_target=float(weight[area].mean())))
    sequences=dict(source=source,raw=raw,S=s,N2=n2,T=result)
    scores,rows,spatial=score(sequences,tracked,start=0)
    save_csv(out/'temporal.csv',rows);save_csv(out/'spatial.csv',spatial);save_csv(out/'coverage.csv',coverage)
    np.save(out/'T_rgb.npy',result,allow_pickle=False);np.save(out/'transition_weights.npy',weights,allow_pickle=False)
    groups=grouped(rows)
    event_windows={i for key in LONG_ANCHORS[1:] for i in range(key,key+len(RAMP)+1)}
    window_scores={}
    for scope in ('fixed','tracked'):
        window_scores[scope]={}
        for method in sequences:
            pairs=[r for r in rows if r['scope']==scope and r['method']==method and r['frame'] in event_windows]
            count=sum(r['valid_pixels'] for r in pairs)
            images=[r for r in spatial if r['scope']==scope and r['method']==method and r['frame'] in event_windows]
            pixels=sum(r['pixels'] for r in images)
            window_scores[scope][method]=dict(pairs=len(pairs),coarse_mae=sum(r['coarse_sum'] for r in pairs)/count,
                max_coarse_mae=max(r['coarse_sum']/r['valid_pixels'] for r in pairs),
                fine_amplitude=sum(r['fine_amplitude_sum'] for r in images)/pixels,
                fine_ncc=sum(r['fine_ncc']*r['valid_pixels'] for r in pairs)/count)
    labels=dict(raw='GLC raw',S='Previous S',N2='N2 / abrupt reference refresh',T='T / aligned reference transition')
    previews={}
    for filename,keys,crop in (
        ('transition_coral_pair_10s.mp4',('raw','T'),roi(VIEW_BOX,FRAME_SHAPE)),
        ('transition_full_pair_10s.mp4',('raw','T'),None),
        ('transition_ablation_10s.mp4',('raw','S','N2','T'),roi(VIEW_BOX,FRAME_SHAPE))):
        previews[filename]=preview(args.ffmpeg,out/filename,[sequences[k] for k in keys],FPS,0,[labels[k] for k in keys],crop)
    for event in (45,105):
        indices=(event-1,event,event+2,event+5,event+6)
        canvas=Image.new('RGB',(524*5,288*3),(24,35,48));draw=ImageDraw.Draw(canvas)
        for row,name in enumerate(('source','N2','T')):
            for col,i in enumerate(indices):
                x,y=524*col,288*row
                canvas.paste(Image.fromarray(sequences[name][i][roi(VIEW_BOX,FRAME_SHAPE)]).resize((524,264)),(x,y+24))
                draw.text((x+6,y+4),f'{name} | f{i}',fill='white')
        canvas.save(out/f'transition_{event:03d}.png')
    changed={i for key in LONG_ANCHORS[1:] for i in range(key,key+len(RAMP))}
    untouched=[i for i in range(150) if i not in changed]
    checks=dict(other120_frames_exact=bool(np.array_equal(result[untouched],n2[untouched])),
        zero_weight_pixels_exact=bool(np.array_equal(result[weights==0],n2[weights==0])),
        frozen_inputs_unchanged=all(file_sha(Path(p))==h for p,h in inputs['files'].items()))
    if not all(checks.values()):raise ValueError('Transition preservation check failed')
    save_json(out/'summary.json',dict(complete=True,source_assisted_offline=True,scores=scores,groups=groups,
        event_windows=window_scores,exit_rows=[r for r in rows if r['scope']=='fixed' and r['frame'] in [i+len(RAMP) for i in LONG_ANCHORS[1:]]],
        verification=checks,previews=previews,code_sha256=file_sha(__file__),
        outputs_sha256={p.name:file_sha(p) for p in out.glob('*.npy')}))
    print('PASS',flush=True)
    print('all',groups['fixed']['all'],flush=True)
    print('refresh',groups['fixed']['key_refresh'],flush=True)
    print('windows including exits',window_scores['fixed'],flush=True)


if __name__=='__main__':main()
