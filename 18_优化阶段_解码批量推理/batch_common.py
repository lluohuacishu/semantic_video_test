"""Small P-generator batch experiment; the stage 17 codec and web remain frozen."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import urllib.request

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WORKSPACE = REPO.parents[2]
STAGE17 = REPO / '17_优化阶段_解码器推理加速'
sys.path.insert(0, str(STAGE17))
from decoder_common import (ASSETS, ARCHIVE, TRIAL as DECODER_TRIAL, file_sha,
                            load_json, save_json, load_settings, code_hashes as decoder_hashes)

TRIAL = WORKSPACE / '02_实验数据/优化_解码批量/batch_pilot_20261031_01'
STREAM = ARCHIVE / 'compact/long108_q0_640/eager/stream.glcsnd'
SELECTED = list(range(1, 9)) + list(range(29, 37)) + [61, 62, 63, 65]


def code_hashes():
    result = decoder_hashes()
    for name in ('batch_common.py', 'capture_latents.py', 'benchmark_batch.py'):
        result[str((HERE / name).relative_to(REPO))] = file_sha(HERE / name)
    return result


def idle():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for url in ('http://127.0.0.1:8773/api/jobs', 'http://127.0.0.1:8774/api/jobs',
                'http://127.0.0.1:8773/inference/api/jobs'):
        with opener.open(url, timeout=5) as response:
            jobs = json.load(response)
        if any(j['status'] not in ('done', 'error', 'failed', 'cancelled', 'interrupted', 'staged') for j in jobs):
            raise RuntimeError('A user web job is active/queued; do not benchmark concurrently')


def groups(count, batch):
    if type(count) is not int or count < 1 or batch not in (1, 2, 4):
        raise ValueError('Invalid batch partition')
    # No padding or fake frames. A real integration would use batch-1 for a tail.
    return [(i, min(i + batch, count)) for i in range(0, count, batch)]
