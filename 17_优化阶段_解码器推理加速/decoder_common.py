"""Paths and immutable provenance for decoder-only inference experiments."""
from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WORKSPACE = REPO.parents[2]
STAGE16 = REPO / '16_优化阶段_输入编码器推理加速'
sys.path.insert(0, str(STAGE16))
from experiment_common import PACKAGES, file_sha, load_json, save_json, load_settings

ASSETS = WORKSPACE / '02_实验数据/优化_部署精简/role_assets_20261029_01'
ARCHIVE = WORKSPACE / '02_实验数据/优化_输入编码器推理/inference_20261030_01'
TRIAL = WORKSPACE / '02_实验数据/优化_解码器推理/decoder_20261031_05'
BACKENDS = ('eager', 'trt_fp32', 'trt_fp16_safe')


def code_hashes():
    files = [HERE / name for name in ('decoder_common.py', 'generator_backend.py', 'decoder_worker.py')] + [
        REPO / '14_优化阶段_部署精简/role_codec.py',
        REPO / '13_优化阶段_发送端免RGB重建/sender_container.py',
        REPO / '04_外部依赖快照/semantic-video-training/scripts/glc_video_codec.py',
    ]
    return {str(p.relative_to(REPO)): file_sha(p) for p in sorted(files)}
