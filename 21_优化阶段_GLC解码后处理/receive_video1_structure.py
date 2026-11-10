"""Receive the video1 structure file without any sender-side receipt or data.

First run the existing independent GLC decoder on --base. Its --decoded folder
contains local decoded_rgb.npy and result.json; neither is transmitted side data.
"""

import argparse
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from coral_reference_sidecar import file_sha, load_json, read_sender_stream, save_json
from persistent_structure_stream import LIMIT, render, unpack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--decoded', type=Path, required=True)
    parser.add_argument('--auxiliary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    base, decoded, auxiliary, output = [p.resolve() for p in
        (args.base, args.decoded, args.auxiliary, args.output)]
    output.mkdir(exist_ok=False)
    allowed = {base, auxiliary, decoded/'decoded_rgb.npy', decoded/'result.json'}
    observed = set()
    suffixes = {'.npy', '.npz', '.json', '.png', '.jpg', '.jpeg', '.mp4', '.mkv',
        '.avi', '.glcsnd', '.glcstruct', '.glcflow', '.glcgate', '.glcsem', '.glcmot'}

    def audit(event, values):
        if event != 'open' or not values or not isinstance(values[0], (str, bytes)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        if path.suffix.lower() in suffixes:
            if path not in allowed and not path.is_relative_to(output):
                raise RuntimeError('Receiver denied non-transmitted input: '+str(path))
            if path in allowed:
                observed.add(str(path))

    sys.addaudithook(audit)
    # Verify that even a sender receipt would be blocked before reading its bytes.
    try:
        (auxiliary.parent/'send.json').read_bytes()
    except RuntimeError:
        blocked_sender_receipt = True
    else:
        raise ValueError('Read guard failed')
    digest = file_sha(base)
    header, _, integrity = read_sender_stream(base)
    receipt = load_json(decoded/'result.json')
    if (base.stat().st_size > LIMIT or (header['frames'], header['fps'], header['width'], header['height']) != (150, 15, 512, 288)
            or digest != receipt['stream_sha256'] or not receipt['reference_verified']
            or not receipt['source_media_reads_forbidden']
            or receipt['reference_checksums'] != integrity['expected_reference_checksums']
            or file_sha(decoded/'decoded_rgb.npy') != receipt['decoded_rgb_sha256']):
        raise ValueError('Independent base decode integrity failed')
    raw = np.load(decoded/'decoded_rgb.npy', allow_pickle=False)
    profile, arrays = unpack(auxiliary.read_bytes(), digest)
    start = time.perf_counter()
    result, alpha = render(raw, profile, arrays)
    elapsed = time.perf_counter()-start
    np.save(output/'rgb.npy', result, allow_pickle=False)
    np.save(output/'alpha.npy', alpha, allow_pickle=False)
    save_json(output/'receipt.json', dict(complete=True, sender_receipt_read_blocked=blocked_sender_receipt,
        only_base_aux_and_locally_decoded_inputs=observed == {str(p) for p in allowed}, input_paths=sorted(observed),
        base_sha256=digest, auxiliary_sha256=file_sha(auxiliary), render_cpu_seconds=elapsed,
        base_bytes=base.stat().st_size, auxiliary_bytes=auxiliary.stat().st_size, billing_seconds=10,
        output_sha256={p.name:file_sha(p) for p in output.glob('*.npy')},
        code_sha256=file_sha(__file__), codec_sha256=file_sha(Path(__file__).with_name('persistent_structure_stream.py'))))
    print('PASS: independent render without sender receipt, source RGB, flow or mask files', flush=True)


if __name__ == '__main__':
    main()
