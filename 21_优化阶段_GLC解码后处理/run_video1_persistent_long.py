"""Ten-second source-assisted coral memory with coordinate inheritance at keys.

Carry origin coordinates and reference IDs across keys, then sample each
original GLC anchor RGB once. Never recursively resample displayed RGB.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from cache_motion_inputs import FPS, FRAME_SHAPE, REPO
from compare_boundary_codec import frame_kind
from compare_deflicker import save_csv
from coral_reference_sidecar import file_sha, load_json, save_json
from experiment_common import array_sha
from run_p0 import preview, roi
from run_structure_memory import LONG_ANCHORS
from run_video1_full_coral import OUTLINE, orange_fish
from run_video1_persistent_structure import VIEW_BOX, load_inputs, score
from source_raft_flow import ITERATIONS, WEIGHT_SHA
from temporal_filter import correspondence


OUTLINES = {
    0: ((369,0),(460,0),(489,22),(488,52),(473,74),(435,98),(394,113),
        (351,126),(314,137),(298,130),(300,112),(311,90),(330,62),(349,35)),
    15: ((366,0),(503,0),(506,34),(490,61),(452,82),(404,98),(359,113),
         (326,121),(306,115),(308,100),(319,78),(333,53),(349,26)),
    45: ((377,0),(511,0),(511,45),(492,60),(450,77),(407,91),(364,104),
         (333,113),(318,105),(322,87),(338,64),(355,39)),
    75: ((367,0),(511,0),(511,53),(485,70),(441,85),(393,99),(350,111),
         (323,121),(308,112),(309,96),(323,72),(340,47),(355,24)),
    105: OUTLINE,
    135: ((317,0),(511,0),(511,13),(486,29),(450,43),(406,58),(360,72),
          (315,87),(279,98),(258,96),(249,84),(253,69),(266,48),(284,28),(302,11)),
}


def mask_for(anchor):
    mask = np.zeros(FRAME_SHAPE[:2], np.uint8)
    cv2.fillPoly(mask, [np.asarray(OUTLINES[anchor], np.int32)], 1)
    return mask


def prepare(source, upstream, output):
    import torch
    sys.path.insert(0, str(upstream))
    from src.models.stage_1.core.raft import RAFT
    weight = upstream/'pretrained_weights/raft-things.pth'
    if file_sha(weight) != WEIGHT_SHA:
        raise ValueError('Unexpected RAFT weights')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = RAFT(argparse.Namespace(small=False, mixed_precision=False, alternate_corr=False))
    state = torch.load(weight, map_location='cpu', weights_only=True)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    model.cuda().eval()
    pairs = list(zip(LONG_ANCHORS[:-1], LONG_ANCHORS[1:], strict=True))
    flows = np.empty((len(pairs), 2, *FRAME_SHAPE[:2], 2), np.float32)
    tick = time.perf_counter()
    with torch.inference_mode():
        for j, (old, new) in enumerate(pairs):
            frames = torch.from_numpy(source[[new, old]]).permute(0,3,1,2).cuda().float()
            _, flow = model(frames, frames.flip(0), iters=ITERATIONS, test_mode=True)
            flows[j] = flow.permute(0,2,3,1).contiguous().cpu().numpy()
            if not np.isfinite(flows[j]).all():
                raise ValueError('Nonfinite bridge flow')
            print(f'Key bridge {new} <-> {old}', flush=True)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output/'bridge_flow.npy', flows, allow_pickle=False)
    save_json(output/'flow.json', dict(pairs=pairs, order=['new_to_old','old_to_new'],
        source_rgb_sha256=array_sha(source), weights_sha256=WEIGHT_SHA, iterations=ITERATIONS,
        offline_source_assisted=True, elapsed_seconds=time.perf_counter()-tick,
        upstream_commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'], text=True).strip(),
        flow_sha256=file_sha(output/'bridge_flow.npy'), code_sha256=file_sha(__file__)))


def project_memory(ids, coords, mx, my):
    """Interpolate coordinates only among pixels sharing the chosen RGB origin."""
    chosen = cv2.remap(ids.astype(np.float32), mx, my, cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=-1).astype(np.int16)
    mapped = np.zeros_like(coords)
    for origin in np.unique(chosen):
        if origin < 0:
            continue
        weight = (ids == origin).astype(np.float32)
        denominator = cv2.remap(weight, mx, my, cv2.INTER_LINEAR)
        numerator = cv2.remap(coords*weight[..., None], mx, my, cv2.INTER_LINEAR)
        take = chosen == origin
        mapped[take] = numerator[take]/np.maximum(denominator[take, None], 1e-6)
    return chosen, mapped


def sample_origins(images, ids, coords):
    result = np.zeros((*FRAME_SHAPE[:2], 3), np.float32)
    for origin in np.unique(ids):
        if origin < 0:
            continue
        ref = cv2.remap(images[LONG_ANCHORS[origin]].astype(np.float32), coords[...,0], coords[...,1],
                        cv2.INTER_CUBIC)
        take = ids == origin
        result[take] = ref[take]
    return result


def memory_render(source, raw, back, forward, bridge):
    h, w = FRAME_SHAPE[:2]
    yy, xx = np.mgrid[:h,:w].astype(np.float32)
    identity = np.stack((xx, yy), axis=-1)
    low = np.stack([cv2.GaussianBlur(f.astype(np.float32),(0,0),3) for f in source])
    n, m = raw.copy(), raw.copy()
    alpha = np.zeros(raw.shape[:3], np.uint8)
    tracked = np.zeros_like(alpha)
    locks = np.stack([orange_fish(f) for f in source])
    key_ids, key_coords, key_rows, rows = [], [], [], []
    previous_key = None
    for i in range(len(raw)):
        if i in LONG_ANCHORS:
            slot = LONG_ANCHORS.index(i)
            allowed = mask_for(i).astype(bool) & ~locks[i].astype(bool)
            if previous_key is None:
                ids = np.full((h,w), slot, np.int16)
                coords = identity.copy()
                inherited = np.zeros((h,w), bool)
            else:
                b, f = bridge[slot-1]
                mx, my, inside, cycle = correspondence(b, f)
                inherited_ids, inherited_coords = project_memory(ids, coords, mx, my)
                previous_low = cv2.remap(low[previous_key], mx, my, cv2.INTER_LINEAR)
                bridge_photo = np.max(np.abs(previous_low-low[i]),axis=2)
                original_low = sample_origins(low, inherited_ids, inherited_coords)
                origin_photo = np.max(np.abs(original_low-low[i]),axis=2)
                inherited = allowed & inside & (cycle<1.5) & (bridge_photo<16) & (origin_photo<16) & (inherited_ids>=0)
                ids = np.where(inherited, inherited_ids, slot).astype(np.int16)
                coords = np.where(inherited[...,None], inherited_coords, identity)
            ids[~allowed] = -1
            key_ids.append(ids.copy()); key_coords.append(coords.copy())
            key_rows.append(dict(frame=i, allowed=int(allowed.sum()), inherited=int(inherited.sum()),
                newly_seeded=int((allowed & ~inherited).sum()), inherited_fraction=float(inherited.sum()/max(allowed.sum(),1))))
            previous_key = i
        anchor = previous_key
        mx, my, inside, cycle = correspondence(back[i], forward[i])
        target_ids, target_coords = project_memory(ids, coords, mx, my)
        area = cv2.remap(mask_for(anchor), mx, my, cv2.INTER_NEAREST).astype(bool) & inside
        tracked[i] = area
        anchor_low = cv2.remap(low[anchor], mx, my, cv2.INTER_LINEAR)
        anchor_photo = np.max(np.abs(anchor_low-low[i]),axis=2)
        old_low = sample_origins(low, target_ids, target_coords)
        origin_photo = np.max(np.abs(old_low-low[i]),axis=2)
        safe = area & (cycle<1.5) & (anchor_photo<16) & (origin_photo<16) & (target_ids>=0) & ~locks[i].astype(bool)
        safe = cv2.erode(safe.astype(np.uint8), np.ones((3,3),np.uint8))
        weight = np.minimum(cv2.distanceTransform(safe,cv2.DIST_L2,5)/2.,1.)
        alpha[i] = np.rint(weight*255).astype(np.uint8)
        a = (alpha[i].astype(np.float32)/255)[...,None]
        fresh = cv2.remap(raw[anchor].astype(np.float32),mx,my,cv2.INTER_CUBIC)
        remembered = sample_origins(raw,target_ids,target_coords)
        n[i] = np.rint(np.clip(raw[i]*(1-a)+fresh*a,0,255)).astype(np.uint8)
        m[i] = np.rint(np.clip(raw[i]*(1-a)+remembered*a,0,255)).astype(np.uint8)
        valid = safe.astype(bool)
        age = np.zeros((h,w),np.float32)
        for origin in np.unique(target_ids):
            if origin>=0:
                age[target_ids==origin] = i-LONG_ANCHORS[origin]
        rows.append(dict(frame=i, anchor=anchor, tracked_pixels=int(area.sum()),
            active_pixels=int((alpha[i]>0).sum()), full_weight_pixels=int((alpha[i]==255).sum()),
            mean_reference_age_frames=float(age[valid].mean()) if valid.any() else 0.,
            original_source_low_mae=float(np.mean(np.abs(old_low[valid]-low[i][valid]))) if valid.any() else 0.,
            mean_weight=float(alpha[i][area].mean()/255) if area.any() else 0.))
    return n, m, alpha, tracked, locks, np.stack(key_ids), np.stack(key_coords), key_rows, rows


def grouped(rows):
    groups = {
        'all': lambda i: True,
        'key_refresh': lambda i: i in LONG_ANCHORS[1:],
        'I': lambda i: i in (64,128),
        'ordinary_P_from6s': lambda i: i>=90 and frame_kind(i,64,32)==frame_kind(i-1,64,32)=='ordinary_p' and i not in LONG_ANCHORS,
        'first2s': lambda i: i<30,
        'last2s': lambda i: i>=120,
    }
    result = {}
    for scope in ('fixed','tracked'):
        result[scope] = {}
        for label, predicate in groups.items():
            result[scope][label] = {}
            for method in sorted({r['method'] for r in rows}):
                group = [r for r in rows if r['scope']==scope and r['method']==method and predicate(r['frame'])]
                count=sum(r['valid_pixels'] for r in group)
                result[scope][label][method]=dict(pairs=len(group), coarse_mae=sum(r['coarse_sum'] for r in group)/count,
                    fine_ncc=sum(r['fine_ncc']*r['valid_pixels'] for r in group)/count)
    return result


def evaluate(workspace, output, ffmpeg):
    source, raw, s, back, forward, inputs = load_inputs(workspace)
    receipt = load_json(output/'bridges/flow.json')
    path=output/'bridges/bridge_flow.npy'
    if receipt['source_rgb_sha256'] != array_sha(source) or file_sha(path)!=receipt['flow_sha256']:
        raise ValueError('Bridge source or flow changed')
    bridge=np.load(path,allow_pickle=False)
    result=output/'result'
    result.mkdir(exist_ok=False)
    save_json(result/'plan.json',dict(offline_source_assisted=True,frames=150,fps=15,keys=LONG_ANCHORS,
        source_outlines=OUTLINES,reference_update='inherit RGB origin coordinates and IDs; seed only failed/new pixels at keys',
        RGB_resampling='one cubic sample of original GLC RGB per displayed frame; coordinate fields may be composed',
        comparison='N resets raw RGB at each key; M inherits origin maps; both use identical final masks',
        limits='Source RAFT, source reliability and source masks not transmitted. Current-frame fallback outside reliable memory.',
        no_quality_parameter_sweep=True))
    tick=time.perf_counter()
    n,m,alpha,tracked,locks,ids,coords,key_rows,coverage=memory_render(source,raw,back,forward,bridge)
    for name,value in dict(N_rgb=n,M_rgb=m,alpha=alpha,tracked=tracked,fish_locks=locks,key_origin_ids=ids,key_origin_coords=coords).items():
        np.save(result/f'{name}.npy',value,allow_pickle=False)
    save_csv(result/'refresh.csv',key_rows); save_csv(result/'coverage.csv',coverage)
    sequences=dict(source=source,raw=raw,S=s,N=n,M=m)
    scores,temporal,spatial=score(sequences,tracked,start=0)
    save_csv(result/'temporal.csv',temporal);save_csv(result/'spatial.csv',spatial)
    groups=grouped(temporal)
    labels=dict(source='Source RGB',raw='GLC raw',S='Previous S',N='N / reset RGB at keys',M='M / inherited structure')
    previews={}
    for filename,names,crop in (
        ('memory_coral_pair_10s.mp4',('raw','M'),roi(VIEW_BOX,FRAME_SHAPE)),
        ('memory_full_pair_10s.mp4',('raw','M'),None),
        ('memory_ablation_10s.mp4',('raw','S','N','M'),roi(VIEW_BOX,FRAME_SHAPE))):
        previews[filename]=preview(ffmpeg,result/filename,[sequences[k] for k in names],FPS,0,[labels[k] for k in names],crop)
    samples=(0,30,60,90,120,149)
    canvas=Image.new('RGB',(512*3,312*2),(24,35,48));draw=ImageDraw.Draw(canvas)
    for j,i in enumerate(samples):
        a=alpha[i,...,None].astype(np.float32)/255*.45
        rgb=np.rint(source[i]*(1-a)+np.array([0,255,170])*a).astype(np.uint8)
        rgb[locks[i].astype(bool)]=[255,70,50]
        x,y=j%3*512,j//3*312
        canvas.paste(Image.fromarray(rgb),(x,y+24));draw.text((x+6,y+5),f'f{i} / green memory support',fill='white')
    canvas.save(result/'memory_support.png')
    for event in (15,45,75,105,135):
        indices=list(range(event-1,event+2))
        canvas=Image.new('RGB',(524*3,288*3),(24,35,48));draw=ImageDraw.Draw(canvas)
        for row,name in enumerate(('source','N','M')):
            for col,i in enumerate(indices):
                x,y=col*524,row*288
                canvas.paste(Image.fromarray(sequences[name][i][roi(VIEW_BOX,FRAME_SHAPE)]).resize((524,264)),(x,y+24))
                draw.text((x+6,y+5),f'{labels[name]} | f{i}',fill='white')
        canvas.save(result/f'refresh_{event:03d}.png')
    checks=dict(raw_frame0_exact=bool(np.array_equal(m[0],raw[0])),
        zero_alpha_exact=bool(np.array_equal(m[alpha==0],raw[alpha==0])),
        fish_locks_exact=bool(np.array_equal(m[locks.astype(bool)],raw[locks.astype(bool)])),
        inputs_unchanged=all(file_sha(Path(p))==h for p,h in inputs['files'].items()))
    if not all(checks.values()):raise ValueError('Memory preservation check failed')
    save_json(result/'summary.json',dict(complete=True,source_assisted_offline=True,input_receipt=inputs,
        scores=scores,groups=groups,key_refresh=key_rows,previews=previews,verification=checks,
        mean_active_pixels=float(np.mean([r['active_pixels'] for r in coverage])),
        mean_reference_age_frames=float(np.mean([r['mean_reference_age_frames'] for r in coverage])),
        cpu_seconds=time.perf_counter()-tick,code_sha256=file_sha(__file__),
        outputs_sha256={p.name:file_sha(p) for p in result.glob('*.npy')}))
    print('PASS',flush=True)
    for group in ('all','key_refresh','I','ordinary_P_from6s','last2s'):
        print(group,groups['fixed'][group],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('bridges','render'))
    parser.add_argument('--workspace',type=Path,default=REPO.parents[2])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--upstream',type=Path)
    parser.add_argument('--ffmpeg',type=Path)
    args=parser.parse_args()
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    if args.action=='bridges':
        if args.upstream is None:parser.error('bridges requires --upstream')
        source,*_=load_inputs(args.workspace.resolve())
        prepare(source,args.upstream.resolve(),args.output.resolve()/'bridges')
    else:
        if args.ffmpeg is None:parser.error('render requires --ffmpeg')
        evaluate(args.workspace.resolve(),args.output.resolve(),args.ffmpeg.resolve())


if __name__=='__main__':main()
