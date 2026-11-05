"""Pinned full-frame LPIPS/DISTS on saved uint8 RGB; never part of receiver input."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parents[2]
METRICS = ROOT/'02_实验数据/GVC1D/perceptual_20261104_01'
os.environ['TORCH_HOME'] = str(METRICS/'torch')
sys.path.insert(0, str(METRICS/'metric_packages'))


def load(p):
    return json.loads(Path(p).read_text('utf-8-sig'))


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser()
    gt = ap.add_mutually_exclusive_group(required=True)
    gt.add_argument('--gt-npy', type=Path)
    gt.add_argument('--gt-plan', type=Path)
    ap.add_argument('--indices', type=Path)
    ap.add_argument('--rgb', nargs=2, action='append', required=True, metavar=('NAME','PATH'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    import lpips
    from DISTS_pytorch import DISTS
    from PIL import Image
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    for name, expected in [('alexnet-owt-7be5be79.pth','7be5be79'),('vgg16-397923af.pth','397923af')]:
        if not sha(METRICS/'torch/hub/checkpoints'/name).startswith(expected):
            raise ValueError('Metric backbone checksum mismatch')
    alex = lpips.LPIPS(net='alex', version='0.1', verbose=False).eval().cuda().float()
    dists = DISTS(load_weights=False)
    weights = METRICS/'metric_packages/DISTS_pytorch/weights.pt'
    calibration = torch.load(weights, map_location='cpu', weights_only=True)
    dists.alpha.copy_(calibration['alpha'])
    dists.beta.copy_(calibration['beta'])
    dists = dists.eval().cuda().float()
    class ChunkRGB:
        def __init__(self,path):
            record=load(path)
            self.meta=record.get('decoded_storage',record)
            self.dtype=np.dtype('uint8')
            self.cached=None
            self.number=None
            count=0
            for item in self.meta['chunks']:
                if item['start']!=count or not 0<item['frames']<=self.meta['chunk_frames']:
                    raise ValueError('Non-contiguous RGB chunks')
                count+=item['frames']
            if count!=self.meta['frames']:
                raise ValueError('Incomplete RGB chunks')

        def __len__(self):
            return self.meta['frames']

        def __getitem__(self,index):
            number=next(j for j,c in enumerate(self.meta['chunks']) if c['start']<=index<c['start']+c['frames'])
            item=self.meta['chunks'][number]
            if self.number!=number:
                if sha(item['path'])!=item['sha256']:
                    raise ValueError('Stored RGB chunk changed')
                self.cached=np.load(item['path'],mmap_mode='r')
                if self.cached.dtype!=self.dtype:
                    raise ValueError('Expected stored uint8 RGB')
                self.number=number
            return self.cached[index-item['start']]

    candidates = {name:(ChunkRGB(path) if Path(path).suffix=='.json' else np.load(path,mmap_mode='r')) for name,path in args.rgb}
    if len(candidates) != len(args.rgb):
        raise ValueError('Duplicate candidate name')
    count = len(next(iter(candidates.values())))
    indices = load(args.indices)['selected_indices'] if args.indices else list(range(count))
    if len(indices) != count or any(len(a)!=count or a.dtype!=np.uint8 for a in candidates.values()):
        raise ValueError('Unaligned RGB inputs')
    if args.gt_npy:
        reference = np.load(args.gt_npy, mmap_mode='r')
        def read_gt(i):
            return np.array(reference[indices[i]])
        gt_sha = sha(args.gt_npy)
    else:
        clip = load(args.gt_plan)['clips'][0]
        def read_gt(i):
            index = indices[i]
            p = Path(clip['frames'][index])
            if sha(p) != clip['source_sha256'][index]:
                raise ValueError('Source PNG changed')
            a = np.array(Image.open(p).convert('RGB'))
            if hashlib.sha256(a.tobytes()).hexdigest() != clip['raw_rgb_sha256'][index]:
                raise ValueError('Source pixels changed')
            return a
        gt_sha = sha(args.gt_plan)

    def tensor(a):
        return torch.from_numpy(np.array(a)).permute(2,0,1)[None].cuda().float().div(255)

    def scores(a,b):
        return float(alex(a,b,normalize=True).item()), float(dists(a,b).item())

    first = tensor(read_gt(0))
    identity = scores(first, first)
    if max(map(abs,identity)) >= 1e-5:
        raise ValueError('Metric identity check failed')
    rows = []
    for i in range(count):
        gt_array = read_gt(i)
        x = tensor(gt_array)
        for name, array in candidates.items():
            a = np.array(array[i])
            if a.shape != gt_array.shape:
                raise ValueError('RGB geometry differs')
            lp, di = scores(x, tensor(a))
            mse = float(np.mean((a.astype(np.float64)-gt_array)**2))
            if not all(math.isfinite(v) for v in (lp,di,mse)):
                raise ValueError('Non-finite quality score')
            rows.append(dict(name=name,index=indices[i],kind=0 if indices[i]%64==0 else 1,lpips_alex=lp,dists=di,mse=mse))
        if (i+1)%64 == 0 or i==count-1:
            print(f'Quality {i+1}/{count}', flush=True)
    with (args.output/'per_frame.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summaries = {}
    for name in candidates:
        selected = [r for r in rows if r['name']==name]
        summaries[name] = {key:float(np.mean([r[key] for r in selected])) for key in ('lpips_alex','dists','mse')}
        summaries[name]['psnr_db'] = 10*math.log10(255**2/max(summaries[name]['mse'],1e-12))
    by_kind={}
    for name in candidates:
        by_kind[name]={}
        for kind in (0,1):
            selected=[r for r in rows if r['name']==name and r['kind']==kind]
            if selected:
                by_kind[name][str(kind)]=dict(frames=len(selected),**{key:float(np.mean([r[key] for r in selected])) for key in ('lpips_alex','dists','mse')})
    result = dict(frames=count, indices=indices, summaries=summaries, summaries_by_kind=by_kind,metric_identity=identity,
        rgb_receipts={name:dict(path=path,sha256=sha(path)) for name,path in args.rgb},
        gt_sha256=gt_sha, producer_sha256=sha(__file__), dists_weights_sha256=sha(weights),
        environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),tf32=False,
                         lpips_source_sha256=sha(Path(lpips.__file__)),dtype='float32',batch_size=1),
        metric_scope='LPIPS Alex v0.1 normalize=True; DISTS native [0,1] full RGB FP32. No resize/padding/MP4 decode.',
        temporal_quality_measured=False)
    (args.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summaries,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
