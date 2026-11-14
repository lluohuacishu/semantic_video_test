"""Explicit fixed-video1 I12 receiver with verified exact computation caches.

This is a 150-frame research entry point, not a general/streaming deployment CLI.
The base RGB must come from an independently verified GLCSND receiver.
"""
import time
STARTED=time.perf_counter()

import argparse
from pathlib import Path

from runtime_speed_common import load_json, save_json
from runtime_chain_worker import Worker


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','base-receipt','raw','guide','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--raft-mode',choices=('original','feature_cache'),default='feature_cache',
        help='feature_cache is faster on the tested clip but increases GPU memory; original keeps only the T3 optimization')
    args=p.parse_args()
    receipt=load_json(args.base_receipt)
    if not receipt.get('reference_verified') or not receipt.get('source_media_reads_forbidden'):
        raise ValueError('Base receipt does not certify independent, source-free decode')
    if args.output.exists():raise FileExistsError('Choose a new output directory')
    worker=Worker('post')
    result=worker.job(dict(base=str(args.base.resolve()),base_receipt=str(args.base_receipt.resolve()),
        raw=str(args.raw.resolve()),guide=str(args.guide.resolve()),output=str(args.output.resolve()),
        raft_mode=args.raft_mode,cached_t3=True))
    save_json(args.output/'entry_receipt.json',dict(complete=True,
        internal_from_first_python_statement_seconds=time.perf_counter()-STARTED,
        process_startup_note='Excludes interpreter startup before first statement; benchmark driver measures external child wall time',
        research_scope='video1-compatible 150-frame 512x288/15FPS guide format; no generalization claim',
        outputs_sha256=result['outputs_sha256']))
    print('CURRENT I12 EXACT-CACHE RECEIVE COMPLETE',args.output,flush=True)


if __name__=='__main__':main()
