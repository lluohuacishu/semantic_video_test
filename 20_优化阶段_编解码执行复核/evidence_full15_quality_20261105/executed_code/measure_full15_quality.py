"""Full-frame GT metrics and FP32 reconstruction differences for both 15FPS clips."""
from __future__ import annotations
import argparse
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parents[2]
METRICS=ROOT/'02_实验数据/GVC1D/perceptual_20261104_01'
os.environ['TORCH_HOME']=str(METRICS/'torch')
sys.path.insert(0,str(METRICS/'metric_packages'))


def load(p):
    return json.loads(Path(p).read_text('utf-8-sig'))


def save(p,d):
    Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    plan=load(args.plan)
    if not plan['complete'] or len(plan['clips'])!=2:raise ValueError('FP32 controls incomplete')
    import numpy as np
    import torch
    import torchvision
    import lpips
    from DISTS_pytorch import DISTS
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    weights=load(METRICS/'weight_sources.json')
    for key in ('alexnet','vgg16'):
        if sha(weights[key]['cache'])!=weights[key]['sha256']:raise ValueError('Metric backbone differs')
    alex=lpips.LPIPS(net='alex',version='0.1',verbose=False).eval().float().cuda()
    dists=DISTS(load_weights=False)
    calibration_path=METRICS/'metric_packages/DISTS_pytorch/weights.pt'
    calibration=torch.load(calibration_path,map_location='cpu',weights_only=True)
    if set(calibration)!={'alpha','beta'}:raise ValueError('DISTS calibration differs')
    dists.alpha.copy_(calibration['alpha']);dists.beta.copy_(calibration['beta'])
    dists=dists.eval().float().cuda()

    class RGB:
        def __init__(self,path):
            self.path=Path(path)
            self.meta=None
            self.receipts=[]
            self.array=None
            self.key=None
            if self.path.suffix=='.json':
                record=load(self.path)
                self.meta=record.get('decoded_storage',record)
                total=0
                for c in self.meta['chunks']:
                    if c['start']!=total or not 0<c['frames']<=self.meta['chunk_frames']:
                        raise ValueError('Invalid chunk ordering')
                    if sha(c['path'])!=c['sha256']:raise ValueError('RGB source chunk changed')
                    self.receipts.append(dict(path=c['path'],sha256=c['sha256']))
                    total+=c['frames']
                self.count=total
                if total!=self.meta['frames']:raise ValueError('Incomplete source')
            else:
                self.array=np.load(self.path,mmap_mode='r')
                self.count=len(self.array)
            self.file_sha256=sha(self.path)

        def __getitem__(self,index):
            if not 0<=index<self.count:raise IndexError(index)
            if self.meta is None:
                a=self.array[index]
            else:
                key=next(i for i,c in enumerate(self.meta['chunks']) if c['start']<=index<c['start']+c['frames'])
                c=self.meta['chunks'][key]
                if key!=self.key:
                    self.array=np.load(c['path'],mmap_mode='r');self.key=key
                a=self.array[index-c['start']]
            if a.dtype!=np.uint8:raise ValueError('Expected pre-MP4 uint8 RGB')
            return np.array(a)

    def tensor(a):
        return torch.from_numpy(a).permute(2,0,1)[None].cuda().float().div(255)

    def metrics(x,y):
        a=float(alex(x,y,normalize=True).item());b=float(dists(x,y).item())
        if not all(math.isfinite(v) and v>=-1e-5 for v in (a,b)):raise ValueError('Invalid perceptual score')
        return a,b

    def psnr(mse):
        return None if mse==0 else 10*math.log10(255**2/mse)

    def distribution(a):
        a=np.asarray(a,dtype=np.float64)
        return dict(mean=float(a.mean()),median=float(np.median(a)),p05=float(np.percentile(a,5)),
            p95=float(np.percentile(a,95)),minimum=float(a.min()),maximum=float(a.max()))

    started=time.perf_counter()
    overall=dict(complete=False,plan_sha256=sha(args.plan),producer_sha256=sha(Path(__file__)),
        protocol=dict(reference='same-resolution lossless source RGB before encoding',
            lpips='Alex v0.1, normalize=True',dists='official alpha/beta; native RGB [0,1]',
            dtype='float32',batch_size=1,resize=False,padding=False,frame_sampling=False,
            mp4_used=False,psnr='global RGB MSE, data_range=255',temporal_quality=False),
        environment=dict(torch=str(torch.__version__),torchvision=torchvision.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),tf32=False,lpips_code_sha256=sha(inspect.getfile(lpips.LPIPS)),
            dists_code_sha256=sha(inspect.getfile(DISTS)),dists_calibration_sha256=sha(calibration_path)),clips=[])
    for clip in plan['clips']:
        folder=args.output/f'width{clip["width"]}';folder.mkdir()
        source=RGB(clip['source'])
        seq={name:RGB(path) for name,path in clip['rgb'].items()}
        if source.count!=1239 or any(s.count!=1239 for s in seq.values()):raise ValueError('Expected all 1239 frames')
        shape=(clip['height'],clip['width'],3)
        x0=tensor(source[0]);a0=tensor(seq['fp32'][0])
        identity=metrics(x0,x0);negative=metrics(x0,1-x0)
        pair=metrics(x0,a0);reverse=metrics(a0,x0);repeat=metrics(x0,a0)
        if (max(map(abs,identity))>=1e-5 or min(negative)<=1e-3 or
            max(abs(a-b) for a,b in zip(pair,reverse))>=1e-5 or
            max(abs(a-b) for a,b in zip(pair,repeat))>=1e-6):
            raise ValueError('Metric identity/negative/symmetry/repeat check failed')
        rows=[];pixels=[]
        with (folder/'frames.partial.jsonl').open('w',encoding='utf-8') as log:
            for i in range(1239):
                gt=source[i]
                if gt.shape!=shape:raise ValueError('Source geometry differs')
                x=tensor(gt)
                arrays={name:s[i] for name,s in seq.items()}
                for name,a in arrays.items():
                    if a.shape!=shape:raise ValueError('Reconstruction geometry differs')
                    lp,di=metrics(x,tensor(a))
                    mse=float(np.mean((a.astype(np.float64)-gt)**2))
                    row=dict(name=name,index=i,source_index=2*i,seconds=i/15,kind=0 if i%64==0 else 1,
                        lpips_alex=lp,dists=di,mse=mse,psnr_db=psnr(mse))
                    rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n')
                for label,left,right in [('encoder_only_vs_fp32','fp32','encoder_only'),
                                          ('decoder_only','encoder_only','accelerated'),
                                          ('total_vs_fp32','fp32','accelerated')]:
                    diff=arrays[right].astype(np.int16)-arrays[left].astype(np.int16)
                    mse=float(np.mean(diff.astype(np.float64)**2))
                    pixels.append(dict(comparison=label,index=i,kind=0 if i%64==0 else 1,
                        mse=mse,mae=float(np.abs(diff).mean()),maximum=int(np.abs(diff).max()),
                        changed_values=int(np.count_nonzero(diff)),changed_pixels=int(np.count_nonzero(np.any(diff!=0,axis=2)))))
                if (i+1)%64==0 or i==1238:
                    log.flush();print(f'QUALITY {clip["width"]}: {i+1}/1239, three complete paths',flush=True)
        for path,rr in [(folder/'per_frame.csv',rows),(folder/'pixel_differences.csv',pixels)]:
            with path.open('w',newline='',encoding='utf-8-sig') as f:
                writer=csv.DictWriter(f,fieldnames=list(rr[0]));writer.writeheader();writer.writerows(rr)
        summaries={}
        for name in seq:
            rr=[r for r in rows if r['name']==name]
            m=float(np.mean([r['mse'] for r in rr]))
            summaries[name]=dict(lpips_alex=distribution([r['lpips_alex'] for r in rr]),
                dists=distribution([r['dists'] for r in rr]),mse=m,psnr_db=psnr(m),
                per_kind={str(k):dict(frames=sum(r['kind']==k for r in rr),
                    lpips_alex=float(np.mean([r['lpips_alex'] for r in rr if r['kind']==k])),
                    dists=float(np.mean([r['dists'] for r in rr if r['kind']==k])),
                    psnr_db=psnr(float(np.mean([r['mse'] for r in rr if r['kind']==k])))) for k in (0,1)})
        indexed={name:{r['index']:r for r in rows if r['name']==name} for name in seq}
        deltas={};pixel_summary={}
        for label,left,right in [('encoder_only_vs_fp32','fp32','encoder_only'),
                                 ('decoder_only','encoder_only','accelerated'),
                                 ('total_vs_fp32','fp32','accelerated')]:
            deltas[label]={}
            for key in ('lpips_alex','dists'):
                values=[indexed[right][i][key]-indexed[left][i][key] for i in range(1239)]
                worst=max(range(1239),key=values.__getitem__)
                deltas[label][key]=dict(**distribution(values),maximum_index=worst,maximum_seconds=worst/15)
            deltas[label]['psnr_db']=summaries[right]['psnr_db']-summaries[left]['psnr_db']
            rr=[r for r in pixels if r['comparison']==label]
            m=float(np.mean([r['mse'] for r in rr]))
            worst=max(rr,key=lambda r:r['maximum'])
            pixel_summary[label]=dict(mse=m,psnr_between_reconstructions_db=psnr(m),
                mae=float(np.mean([r['mae'] for r in rr])),maximum_channel_difference=worst['maximum'],
                maximum_channel_difference_index=worst['index'],
                changed_channel_fraction=sum(r['changed_values'] for r in rr)/(1239*math.prod(shape)),
                changed_pixel_fraction=sum(r['changed_pixels'] for r in rr)/(1239*shape[0]*shape[1]),
                per_frame_mse=distribution([r['mse'] for r in rr]))
        result=dict(width=clip['width'],height=clip['height'],frames=1239,fps=15,
            summaries=summaries,deltas=deltas,pixel_differences=pixel_summary,
            metric_validation=dict(identity=identity,negative=negative,pair=pair,reverse=reverse,repeat=repeat,passed=True),
            provenance=dict(source=dict(path=str(source.path),sha256=source.file_sha256,chunks=source.receipts),
                reconstructions={name:dict(path=str(s.path),sha256=s.file_sha256) for name,s in seq.items()}),
            stream_rates_kbps=clip['stream_rates_kbps'])
        save(folder/'summary.json',result)
        overall['clips'].append(result);save(args.output/'summary.partial.json',overall)
        print('RESULT '+json.dumps(dict(width=clip['width'],metrics={n:dict(lpips=v['lpips_alex']['mean'],
            dists=v['dists']['mean'],psnr=v['psnr_db']) for n,v in summaries.items()})),flush=True)
    overall.update(complete=True,quality_records=1239*3*2,pixel_comparison_records=1239*3*2,
                   elapsed_seconds=time.perf_counter()-started)
    save(args.output/'summary.json',overall)
    print('COMPLETE 7434 full-frame quality records and 7434 pixel-difference records',flush=True)


if __name__=='__main__':
    main()
