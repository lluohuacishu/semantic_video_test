"""Local paths and JSON helpers; no model imports or runtime installation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCRIPTS = REPO / '04_外部依赖快照/semantic-video-training/scripts'
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO / '08_新阶段C_编解码加速与网页展示'))
from c4_config import load_settings


def file_sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def save_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8', newline='\n')
    temp.replace(path)


def new_codec(decoder_only=False):
    import glc_video_codec as ref
    from c2_native import NativeGaussianRans
    runtime = load_settings()['runtime']
    codec = ref.NeuralCodec(runtime['repo'], runtime['weights'], 'cuda', decoder_only=decoder_only)
    codec.entropy = NativeGaussianRans(runtime['native_library'], codec.entropy)
    return codec, ref
