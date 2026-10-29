"""Shared helpers for the bounded, output-generator-only distillation pilot."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / '08_新阶段C_编解码加速与网页展示'))
sys.path.insert(0, str(REPO / '04_外部依赖快照/semantic-video-training/scripts'))
from c4_config import load_settings


def settings():
    return load_settings()


def load_json(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def save_json(path, value):
    path = Path(path)
    part = path.with_name(path.name + '.tmp')
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf-8')
    part.replace(path)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def model_sha(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def student():
    runtime = settings()['runtime']
    path = runtime['repo'] / 'src/models/vqgan_arch.py'
    spec = importlib.util.spec_from_file_location('pilot_vqgan_arch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Generator(64, 256, [1, 1, 2, 2, 4], 2, 256, [16],
                             swish_last=True, quant_conv=True, patch=32)
    assert sum(p.numel() for p in model.parameters()) == 8800707
    return model


def configure_training(seed=29):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This pilot requires the existing BF16-capable CUDA runtime')


def check_stop(output):
    if (Path(output) / 'STOP').exists():
        raise InterruptedError('Pilot STOP file requested a checkpointed stop')


def new_codec(decoder_only=False):
    import glc_video_codec as ref
    from c2_native import NativeGaussianRans
    runtime = settings()['runtime']
    codec = ref.NeuralCodec(runtime['repo'], runtime['weights'], 'cuda', decoder_only=decoder_only)
    codec.entropy = NativeGaussianRans(runtime['native_library'], codec.entropy)
    return codec, ref
