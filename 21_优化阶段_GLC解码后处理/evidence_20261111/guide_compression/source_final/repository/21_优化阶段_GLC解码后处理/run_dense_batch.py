"""Apply the frozen dense-flow cache comparison to three supplied underwater clips."""

import argparse
from pathlib import Path

import cv2

from cache_motion_inputs import Clip
from dense_transfer_inputs import prepare
from run_dense_transfer import evaluate, source_input
from source_raft_flow import infer


CASES = {
    'video1': (Clip(0, 0, 150, (330, 16, 506, 262)),
               '39090c7c26edfaad45fc98cd88f72dbffcd404bdcd483c5113eddb5de62b6802'),
    'video2': (Clip(0, 0, 150, (380, 156, 507, 281)),
               'b7a05eaa1b52f5d45461a1948087505ae7729ecc98a5b39b7e61f33862d7fb5c'),
    'video3': (Clip(0, 0, 150, (145, 190, 510, 286)),
               'd1b29b1eb9ba8048e31fb2ad394e67268b5fede6008db6e58916f07dcb062a5d'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'flow', 'compare'))
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--batch', type=Path, required=True)
    for name in ('input-dir', 'ffmpeg', 'assets', 'upstream'):
        parser.add_argument(f'--{name}', type=Path)
    args = parser.parse_args()
    clip, digest = CASES[args.case]
    run = args.batch.resolve()/args.case

    if args.action == 'prepare':
        if any(path is None for path in (args.input_dir, args.ffmpeg, args.assets)):
            parser.error('prepare requires --input-dir, --ffmpeg and --assets')
        video = args.input_dir.resolve()/f'{args.case}.mp4'
        prepare(video, args.ffmpeg.resolve(), args.assets.resolve(), run,
                clip=clip, source_sha=digest, clip_id=args.case)
        return

    source = source_input(run, clip)
    if args.action == 'flow':
        if args.upstream is None:
            parser.error('flow requires --upstream')
        infer(source, clip, args.upstream.resolve(), run/'flow')
        return

    if args.ffmpeg is None:
        parser.error('compare requires --ffmpeg')
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    cv2.setRNGSeed(20261109)
    evaluate(run, args.ffmpeg.resolve(), source, clip=clip, region='reef', pair=True)


if __name__ == '__main__':
    main()
