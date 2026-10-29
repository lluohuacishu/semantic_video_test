"""Real-video validation of isolated bundles and failure-safe video publication."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from role_codec import REPO
from sender_common import file_sha, load_json, load_settings, save_json
from sender_container import SenderWriter, read_sender_stream

HERE = Path(__file__).resolve().parent


def guarded(role, remaining):
    output = Path(remaining[remaining.index('--output') + 1]).resolve()
    original = load_settings()['runtime']['weights'].resolve()
    blocked = []
    def audit(event, values):
        if event != 'open' or not values or not isinstance(values[0], (str, bytes)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        mode, flags = values[1:]
        reading = ((isinstance(mode, str) and 'r' in mode) or
                   (mode is None and not flags & (os.O_WRONLY | os.O_RDWR)))
        if not reading:
            return
        if (path.is_relative_to(original) or
                path.name.startswith('receiver_' if role == 'encode' else 'sender_') and path.suffix == '.pt' or
                role == 'decode' and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.npy', '.npz', '.mkv', '.avi', '.mp4'}
                and path != output / 'video.mp4'):
            blocked.append(str(path))
            raise RuntimeError('Isolated deployment attempted an unavailable/forbidden read: ' + str(path))
    sys.addaudithook(audit)
    import deploy_codec
    sys.argv = [str(HERE / 'deploy_codec.py'), *remaining]
    try:
        deploy_codec.main()
    finally:
        if output.is_dir():
            save_json(output / 'access_audit.json', dict(
                original_full_checkpoints_forbidden=True, other_role_weights_forbidden=True,
                source_media_forbidden=role == 'decode', blocked_reads=blocked))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guard-role', choices=['encode', 'decode'])
    args, rest = parser.parse_known_args()
    if args.guard_role:
        guarded(args.guard_role, rest)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(rest)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    assets = args.assets.resolve()
    manifest = load_json(assets / 'manifest.json')
    for role in ['sender', 'receiver']:
        bundle = output / (role + '_bundle')
        bundle.mkdir()
        shutil.copyfile(assets / 'manifest.json', bundle / 'manifest.json')
        for info in manifest['roles'][role]['files'].values():
            os.link(assets / info['file'], bundle / info['file'])
    runtime = load_settings()['runtime']
    plan = load_json(args.previous / 'plan.json')
    clip = plan['clips'][0]
    source = Path(clip['original_source'])
    def run(action, name, source_path, expected_returncode=0):
        role = 'sender' if action == 'encode' else 'receiver'
        command = [str(runtime['python']), '-X', 'utf8', str(HERE / 'validate_minimal.py'),
                   '--guard-role', action, action, '--assets', str(output / (role + '_bundle')),
                   '--input', str(source_path), '--output', str(output / name)]
        if action == 'encode':
            command += ['--frames', '66', '--width', '640', '--stride', '2', '--q', '0']
        started = time.perf_counter()
        with (output / (name + '.log')).open('w', encoding='utf-8') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if (process.returncode == 0) != (expected_returncode == 0):
            raise ValueError('Unexpected minimal CLI exit status: ' + name)
        return dict(returncode=process.returncode, process_wall_seconds=time.perf_counter()-started, command=command)
    encoded = run('encode', 'sender', source)
    stream = output / 'sender/stream.glcsnd'
    expected_stream = args.previous / clip['id'] / 'pair1_no_rgb/stream.glcsnd'
    if stream.read_bytes() != expected_stream.read_bytes():
        raise ValueError('Real-video minimal sender differs from verified array-source encoding')
    decoded = run('decode', 'receiver', stream)
    video = output / 'receiver/video.mp4'
    probe = subprocess.run([str(runtime['ffmpeg'].with_name('ffprobe.exe' if os.name == 'nt' else 'ffprobe')),
                            '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                            '-show_entries', 'stream=width,height,nb_read_frames,avg_frame_rate,duration',
                            '-of', 'json', str(video)], capture_output=True, check=True)
    media = json.loads(probe.stdout)['streams'][0]
    if (int(media['nb_read_frames']), int(media['width']), int(media['height'])) != (66, 640, 360):
        raise ValueError('Receiver output video geometry/count differs')
    header, packets, _ = read_sender_stream(stream)
    header = dict(header, frames=2)
    known = load_json(args.previous / clip['id'] / 'pair1_no_rgb/result.json')['records']
    bad = output / 'bad_reference.glcsnd'
    writer = SenderWriter(bad, header)
    try:
        for index in range(2):
            packet = {key: packets[index][key] for key in ('kind', 'z', 'y')}
            packet['latent_sha256'] = known[index]['latent_sha256'] if index == 0 else '0' * 64
            writer.add(packet)
        writer.finish()
    finally:
        writer.close()
    failed = run('decode', 'failed_receiver', bad, expected_returncode=1)
    failure = load_json(output / 'failed_receiver/failure.json')
    if 'reference states differ' not in failure['message']:
        raise ValueError('Negative test failed for an unrelated reason')
    if (output / 'failed_receiver/video.mp4').exists() or (output / 'failed_receiver/result.json').exists():
        raise ValueError('Failed reference validation published a successful result')
    for folder in ['sender', 'receiver', 'failed_receiver']:
        if load_json(output / folder / 'access_audit.json')['blocked_reads']:
            raise ValueError('Deployment read guard tripped')
    old_preview = args.previous / clip['id'] / 'receiver/verified_receiver_preview.mp4'
    save_json(output / 'validation.json', dict(
        success=True, real_video_stream_byte_exact=True, separate_role_bundles=True,
        original_full_weight_reads_forbidden=True, source_blind_receiver=True,
        video_probe=media, sender_process=encoded, receiver_process=decoded, failed_process=failed,
        video_sha256=file_sha(video), video_identical_to_previous_viewing_copy=file_sha(video)==file_sha(old_preview),
        failed_reference_never_published=True, failure=failure,
        source_sha256=file_sha(source), manifest_sha256=file_sha(assets / 'manifest.json'),
        validation_code_sha256={p.relative_to(REPO).as_posix():file_sha(p)
                                for p in [HERE/'validate_minimal.py',HERE/'deploy_codec.py',HERE/'role_codec.py',
                                          REPO/'08_新阶段C_编解码加速与网页展示/c1_io.py']}))
    print('Minimal sender/receiver, isolated bundles, video and negative reference test passed')


if __name__ == '__main__':
    main()
