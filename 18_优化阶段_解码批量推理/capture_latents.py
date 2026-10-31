"""Decode one old stream in order, verify every reference, retain 20 distinct P latents."""
from __future__ import annotations

import sys
import time
from batch_common import *
import numpy as np
import torch
from role_codec import RoleCodec
from sender_container import ReferenceChecksums, read_sender_stream
from decoder_worker import audit_media_reads, validate_stream


@torch.inference_mode()
def main():
    idle()
    output = TRIAL / 'capture'
    output.mkdir(parents=True, exist_ok=False)
    hashes = code_hashes()
    audit_media_reads(output)
    header, packets, integrity = read_sender_stream(STREAM)
    old = load_json(DECODER_TRIAL / 'short/eager1/result.json')
    if file_sha(STREAM) != old['stream_sha256']:
        raise ValueError('The captured bitstream is not the stage 17 baseline')
    codec = RoleCodec(ASSETS, 'receiver')
    torch.utils.deterministic.fill_uninitialized_memory = False
    validate_stream(header, codec)
    current = -1
    selected, latents, images, references = [], [], [], []
    def capture(_model, args):
        if current in SELECTED:
            latents.append(args[0].detach().cpu().numpy().copy())
    hook = codec.video.vqgan.generator.register_forward_pre_hook(capture)
    checks = ReferenceChecksums()
    started = time.perf_counter()
    try:
        for i, packet in enumerate(packets):
            if i % 8 == 0:
                idle()
            current = i
            rgb, row = codec.frame(header, i, packet=packet)
            if row['latent_sha256'] != old['records'][i]['latent_sha256']:
                raise ValueError('Captured reference differs from stage 17')
            checks.add(i, row)
            if i in SELECTED:
                if row['kind'] != 1:
                    raise ValueError('An I frame was incorrectly assigned to the P generator')
                selected.append(i);images.append(rgb);references.append(row['latent_sha256'])
        checks.verify(integrity['expected_reference_checksums'])
    finally:
        hook.remove()
    if selected != SELECTED or len(latents) != len(selected):
        raise ValueError('Real P-latent capture is incomplete or out of order')
    x = np.concatenate(latents, axis=0)
    if x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError('Captured latent is not finite FP32')
    np.save(output / 'latents.npy', x)
    np.save(output / 'teacher_rgb.npy', np.stack(images))
    if hashes != code_hashes():
        raise ValueError('Producer changed during capture')
    save_json(output / 'result.json', dict(frames_decoded=len(packets), selected_indices=selected,
        distinct_source_frames=True, model='video', input_shape=list(x.shape), header=header,
        input_stream_sha256=file_sha(STREAM), reference_checksums=checks.values(), reference_verified=True,
        per_frame_reference_matches_stage17=True, source_media_reads_forbidden=True,
        selected_reference_sha256=references, latents_sha256=file_sha(output / 'latents.npy'),
        teacher_rgb_sha256=file_sha(output / 'teacher_rgb.npy'), code_sha256=hashes,
        capture_seconds=time.perf_counter() - started, training_steps=0))
    print(f'PASS capture: {len(packets)} references verified, {len(selected)} distinct P latents retained', flush=True)


if __name__ == '__main__':main()
