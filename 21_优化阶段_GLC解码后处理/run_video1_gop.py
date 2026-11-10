"""Encode the GOP 56 control from the frozen 15 FPS video1 input."""

import argparse
from pathlib import Path

from cache_motion_inputs import base_frames
from coral_reference_sidecar import file_sha, load_json, save_json
from experiment_common import array_sha, input_tensor
from run_boundary_codec import encode_case, read_sender_stream
from run_dense_transfer import source_input
from run_structure_memory import CASE, CLIP


BASELINE_GOP = 64
TEST_GOP = 56
RESET = 32


def encode(base, output, assets):
    import torch
    from role_codec import RoleCodec

    _, baseline_sha, _ = base_frames(base, CLIP)
    source = source_input(base, CLIP)
    header, packets, integrity = read_sender_stream(base/'base.glcsnd')
    original = load_json(base/'sender/result.json')
    plan = load_json(base/'encoder_plan.json')
    asset_sha = file_sha(assets/'manifest.json')
    if (header != original['header'] or original['stream_sha256'] != baseline_sha
            or original['reference_checksums'] != integrity['expected_reference_checksums']
            or (header['intra_period'], header['reset_interval'], header['q']) != (BASELINE_GOP, RESET, 0)
            or original['variant'] != 'eager' or original['fill'] != 'on'
            or asset_sha != plan['assets_manifest_sha256']
            or file_sha(base/'source.npy') != plan['clips'][0]['source_sha256']):
        raise ValueError('GOP 64 source, settings or weights differ from the verified baseline')
    settings = dict(header, intra_period=TEST_GOP)

    def verify_prefix(index, packet):
        if index >= TEST_GOP:
            return
        old, recorded = packets[index], original['records'][index]
        if (any(old[key] != packet[key] for key in ('kind', 'z', 'y')) or recorded['index'] != index
                or recorded['latent_sha256'] != packet['latent_sha256']):
            raise ValueError(f'Unexpected compressed/reference difference before the new I frame at {index}')

    output.mkdir(parents=True, exist_ok=False)
    codec = RoleCodec(assets, 'sender')
    torch.utils.deterministic.fill_uninitialized_memory = True
    if (codec.profile != header['arithmetic'] or codec.entropy.table_sha256 != header['table_sha256']
            or not torch.are_deterministic_algorithms_enabled()):
        raise ValueError('The GOP 56 sender does not match the baseline arithmetic')
    stream = output/'base.glcsnd'
    with torch.inference_mode():
        for index in range(plan['warmup_frames']):
            codec.frame(settings, index, x=input_tensor(source[index]), render_rgb=False)
        torch.cuda.synchronize()
        records, references = encode_case(codec, source, settings, stream, verify_packet=verify_prefix)
    digest, size = file_sha(stream), stream.stat().st_size
    save_json(output/'base_cut.json', dict(source_start=CLIP.base_start, source_stop=CLIP.stop,
        display_start=CLIP.start, display_stop=CLIP.stop, bytes=size, fps=header['fps'],
        base_stream_sha256=digest, reference_checksums=references))
    save_json(output/'sender.json', dict(header=settings, bytes=size, stream_sha256=digest,
        reference_checksums=references, source_rgb_sha256=array_sha(source), original_base_sha256=baseline_sha,
        prefix_packets_and_latents_bit_exact=TEST_GOP, role_assets_manifest_sha256=asset_sha,
        records=records, code_sha256=file_sha(__file__),
        encoder_loop_sha256=file_sha(Path(__file__).with_name('run_boundary_codec.py'))))
    print(f'GOP {TEST_GOP}: {size} bytes, {TEST_GOP} identical prefix packets and references', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('batch', 'output', 'assets'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    encode(args.batch.resolve()/CASE, args.output.resolve(), args.assets.resolve())


if __name__ == '__main__':
    main()
