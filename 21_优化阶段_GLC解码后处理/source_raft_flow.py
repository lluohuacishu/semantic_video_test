"""Compute full-grid bidirectional RAFT flow between source RGB and each cache anchor."""

import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from cache_motion_inputs import CLIPS, FPS, HEIGHT, REPO, WIDTH, source_frames
from coral_reference_sidecar import file_sha, save_json
from experiment_common import array_sha
from run_cache_motion import CACHE_FRAMES


CLIP = CLIPS['moving_reef']
ITERATIONS = 24
WEIGHT_SHA = 'fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1'


def infer(workspace, upstream, output):
    import torch

    sys.path.insert(0, str(upstream))
    from src.models.stage_1.core.raft import RAFT

    weight = upstream/'pretrained_weights/raft-things.pth'
    if file_sha(weight) != WEIGHT_SHA:
        raise ValueError('Expected the verified RAFT-Things checkpoint from the Deflicker experiment')
    if not torch.cuda.is_available() or HEIGHT % 8 or WIDTH % 8:
        raise ValueError('This pilot expects CUDA and dimensions divisible by eight')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = RAFT(argparse.Namespace(small=False, mixed_precision=False, alternate_corr=False))
    state = torch.load(weight, map_location='cpu', weights_only=True)
    model.load_state_dict({name.removeprefix('module.'): value for name, value in state.items()}, strict=True)
    model.cuda().eval()

    source = source_frames(workspace, CLIP)
    frames = torch.from_numpy(source).permute(0, 3, 1, 2).to(device='cuda', dtype=torch.float32)
    output.mkdir(parents=True, exist_ok=False)
    files = [output/'to_anchor.npy', output/'from_anchor.npy']
    shape = (CLIP.count, HEIGHT, WIDTH, 2)
    backward, forward = [np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=shape)
                         for path in files]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        for first in range(0, CLIP.count, CACHE_FRAMES):
            backward[first] = forward[first] = 0
            for i in range(first+1, min(first+CACHE_FRAMES, CLIP.count)):
                images = frames[[i, first]]
                _, flow = model(images, images.flip(0), iters=ITERATIONS, test_mode=True)
                pair = flow.permute(0, 2, 3, 1).contiguous().cpu().numpy()
                if pair.shape != (2, HEIGHT, WIDTH, 2) or not np.isfinite(pair).all():
                    raise ValueError(f'Invalid flow at source frame {CLIP.start+i}')
                backward[i], forward[i] = pair
            print(f'RAFT source anchor {CLIP.start+first}: complete', flush=True)
    backward.flush()
    forward.flush()
    save_json(output/'flow.json', dict(
        offline_source_oracle=True, input_source_rgb_sha256=array_sha(source),
        source_range=[CLIP.start, CLIP.stop], fps=FPS, flow_shape=shape,
        dtype='float32', iterations=ITERATIONS, anchor_period=CACHE_FRAMES,
        anchors=list(range(CLIP.start, CLIP.stop, CACHE_FRAMES)),
        model='RAFT-Things (large)', weights_sha256=WEIGHT_SHA,
        upstream_commit=subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'],
                                                text=True).strip(),
        gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        elapsed_seconds=time.perf_counter()-started,
        flow_file_sha256={path.name: file_sha(path) for path in files},
        code_sha256=file_sha(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    infer(args.workspace.resolve(), args.upstream.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
