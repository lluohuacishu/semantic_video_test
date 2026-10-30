"""Stage 16 paths, frozen inputs and numerical/timing summaries."""
from __future__ import annotations
import hashlib
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WORKSPACE = REPO.parents[2]
for name in ['13_优化阶段_发送端免RGB重建', '14_优化阶段_部署精简', '15_优化阶段_算子精确排查']:
    sys.path.insert(0, str(REPO / name))
from sender_common import file_sha, load_json, save_json, load_settings

PACKAGES = WORKSPACE / '02_实验数据/优化_输入编码器推理/engine_packages_1050'


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def difference(expected, actual):
    a, b = np.asarray(expected, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(b).all():
        raise ValueError('Nonfinite output or incompatible shape')
    d = b - a
    return dict(bit_exact=array_sha(expected) == array_sha(actual),
                max_abs=float(np.abs(d).max()), mean_abs=float(np.abs(d).mean()),
                rmse=float(np.sqrt(np.mean(d*d))))


def input_tensor(rgb, device='cuda'):
    import torch
    from torch.nn import functional as F
    h, w = rgb.shape[:2]
    x = torch.from_numpy(np.array(rgb)).to(device).permute(2,0,1)[None].float().div(255)
    return F.pad(x, (0,(-w)%64,0,(-h)%64), mode='replicate').mul(2).sub(1)


def frozen(plan):
    if file_sha(Path(plan['assets'])/'manifest.json') != plan['assets_manifest_sha256']:
        raise ValueError('Role assets changed')
    for name, digest in plan['code_sha256'].items():
        if file_sha(REPO/name) != digest:
            raise ValueError('Frozen code changed: '+name)
    for c in plan['clips']:
        if file_sha(c['source']) != c['source_sha256']:
            raise ValueError('Input changed: '+c['id'])
