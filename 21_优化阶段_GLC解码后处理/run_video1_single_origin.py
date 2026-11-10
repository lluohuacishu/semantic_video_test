"""Separate direct long-reference motion from composed, fragmented memory."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from cache_motion_inputs import FPS, FRAME_SHAPE, REPO
from compare_deflicker import save_csv
from compare_source_dense_flow import load_flow
from coral_reference_sidecar import file_sha, save_json
from run_p0 import preview, roi
from run_structure_memory import CLIP, LONG_ANCHORS
from run_video1_full_coral import orange_fish
from run_video1_persistent_long import mask_for, grouped
from run_video1_persistent_structure import VIEW_BOX, load_inputs, score
from source_raft_flow import infer
from temporal_filter import correspondence


def render_full(source, raw, backward, forward, anchors):
    result=raw.copy()
    alpha=np.zeros(raw.shape[:3],np.float32)
    tracked=np.zeros(raw.shape[:3],np.uint8)
    locks=np.stack([orange_fish(f) for f in source])
    rows=[]
    for i in range(len(raw)):
        if i in anchors:
            anchor=i
            anchor_mask=mask_for(i)
            support=anchor_mask*(1-locks[i])
            anchor_low=cv2.GaussianBlur(source[i].astype(np.float32),(0,0),3)
        mx,my,inside,cycle=correspondence(backward[i],forward[i])
        area=cv2.remap(anchor_mask,mx,my,cv2.INTER_NEAREST).astype(bool)&inside
        allowed=cv2.remap(support,mx,my,cv2.INTER_NEAREST).astype(bool)
        low=cv2.GaussianBlur(source[i].astype(np.float32),(0,0),3)
        reference_low=cv2.remap(anchor_low,mx,my,cv2.INTER_LINEAR)
        photo=np.max(np.abs(reference_low-low),axis=2)
        safe=allowed&inside&(cycle<1.5)&(photo<16)&~locks[i].astype(bool)
        safe=cv2.erode(safe.astype(np.uint8),np.ones((3,3),np.uint8))
        alpha[i]=np.minimum(cv2.distanceTransform(safe,cv2.DIST_L2,5)/2.,1.)
        ref=cv2.remap(raw[anchor].astype(np.float32),mx,my,cv2.INTER_CUBIC)
        a=alpha[i,...,None]
        result[i]=np.rint(np.clip(raw[i]*(1-a)+ref*a,0,255)).astype(np.uint8)
        tracked[i]=area
        rows.append(dict(frame=i,anchor=anchor,tracked_pixels=int(area.sum()),
            active_pixels=int((alpha[i]>0).sum()),full_weight_pixels=int((alpha[i]==1).sum()),
            source_low_warp_mae=float(np.abs(reference_low-low)[area].mean()) if area.any() else 0.))
    return result,alpha,tracked,rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('flow','render'))
    parser.add_argument('--workspace',type=Path,default=REPO.parents[2])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--upstream',type=Path)
    parser.add_argument('--ffmpeg',type=Path)
    args=parser.parse_args()
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    root=args.output.resolve()
    source,raw,s,back,forward,inputs=load_inputs(args.workspace.resolve())
    if args.action=='flow':
        if args.upstream is None:parser.error('flow requires --upstream')
        infer(source,CLIP,args.upstream.resolve(),root/'single_anchor0_flow',anchors=(0,),periodic_flows=(back,forward))
        return
    if args.ffmpeg is None:parser.error('render requires --ffmpeg')
    out=root/'direct_result';out.mkdir(exist_ok=False)
    save_json(out/'plan.json',dict(frames=150,fps=15,offline_source_assisted=True,
        F='one original GLC RGB reference at frame0; direct source RAFT for every frame',
        N2='full RGB reset at long keys with independent current-key reliability',
        purpose='Separate long-reference loss from accumulated coordinate drift and fragmented RGB origins',
        gate='same full-coral C2 binary photo/cycle checks and 2px feather',
        limits='Newly visible, occluded or unreliable areas remain raw GLC. No real auxiliary stream.'))
    zero=load_flow(root/'single_anchor0_flow',source,CLIP,anchors=(0,))
    f,f_alpha,f_track,f_rows=render_full(source,raw,*zero,(0,))
    n,n_alpha,n_track,n_rows=render_full(source,raw,back,forward,LONG_ANCHORS)
    pilot=args.workspace/'02_实验数据/优化_GLC后处理/video1_persistent_structure_20261110_01/expanded'
    old_c2=np.load(pilot/'C2_rgb.npy',allow_pickle=False)
    if not np.array_equal(n[105:128],old_c2):raise ValueError('Expanded C2 interval must reproduce bit exactly')
    m=np.load(root/'result/M_rgb.npy',allow_pickle=False)
    sequences=dict(source=source,raw=raw,S=s,M=m,N2=n,F=f)
    scores,rows,spatial=score(sequences,n_track,start=0)
    save_csv(out/'temporal.csv',rows);save_csv(out/'spatial.csv',spatial)
    save_csv(out/'F_coverage.csv',f_rows);save_csv(out/'N2_coverage.csv',n_rows)
    for name,value in dict(F_rgb=f,N2_rgb=n,F_alpha=f_alpha,N2_alpha=n_alpha,F_tracked=f_track,tracked=n_track).items():
        np.save(out/f'{name}.npy',value,allow_pickle=False)
    labels=dict(raw='GLC raw',S='Previous S',M='M / composed memory',N2='N2 / full reference refresh',F='F / direct frame0 reference')
    previews={}
    for filename,names,crop in (
        ('direct_reference_pair_10s.mp4',('raw','F'),roi(VIEW_BOX,FRAME_SHAPE)),
        ('fresh_reference_pair_10s.mp4',('raw','N2'),roi(VIEW_BOX,FRAME_SHAPE)),
        ('direct_reference_ablation_10s.mp4',('raw','N2','M','F'),roi(VIEW_BOX,FRAME_SHAPE))):
        previews[filename]=preview(args.ffmpeg,out/filename,[sequences[k] for k in names],FPS,0,[labels[k] for k in names],crop)
    canvas=Image.new('RGB',(512*3,312*2),(24,35,48));draw=ImageDraw.Draw(canvas)
    for j,i in enumerate((0,30,60,90,120,149)):
        a=f_alpha[i,...,None]*.45
        overlay=np.rint(source[i]*(1-a)+np.array([0,255,170])*a).astype(np.uint8)
        x,y=j%3*512,j//3*312
        canvas.paste(Image.fromarray(overlay),(x,y+24));draw.text((x+6,y+4),f'f{i} / direct frame0 support',fill='white')
    canvas.save(out/'direct_support.png')
    checks=dict(C2_interval_bit_exact=True,F_zero_mask_exact=bool(np.array_equal(f[f_alpha==0],raw[f_alpha==0])),
        N2_zero_mask_exact=bool(np.array_equal(n[n_alpha==0],raw[n_alpha==0])),
        inputs_unchanged=all(file_sha(Path(p))==h for p,h in inputs['files'].items()))
    if not all(checks.values()):raise ValueError('Preservation failed')
    groups=grouped(rows)
    save_json(out/'summary.json',dict(complete=True,source_assisted_offline=True,scores=scores,groups=groups,
        previews=previews,verification=checks,code_sha256=file_sha(__file__),
        mean_F_active_pixels=float(np.mean([r['active_pixels'] for r in f_rows])),
        mean_N2_active_pixels=float(np.mean([r['active_pixels'] for r in n_rows])),
        outputs_sha256={p.name:file_sha(p) for p in out.glob('*.npy')}))
    print('PASS',flush=True)
    for group in ('all','key_refresh','I','ordinary_P_from6s','last2s'):
        print(group,groups['fixed'][group],flush=True)


if __name__=='__main__':main()
