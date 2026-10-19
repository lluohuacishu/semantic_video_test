"""Reference validation for a second clip; never labels it as the B4 main clip."""
from pathlib import Path

from run_c0 import INPUT_SHA, SOURCE, read, sha
from run_c1 import preview_info
from glc_container import read_stream
from c2_watch import write


def require_distinct_source(path, digest):
    if Path(path).resolve() == SOURCE.resolve() or digest == INPUT_SHA:
        raise ValueError('Second clip must differ from the main input by content, not only path')


def compare_frames(actual, expected, count):
    if len(actual) != count or len(expected) != count:
        raise ValueError('Unexpected reference or candidate frame count')
    for index, (a, b) in enumerate(zip(actual, expected), 1):
        if a['frame'] != index or b['frame'] != index:
            raise ValueError('Out-of-order reference or candidate frame')
        for key in ('rgb_sha256', 'latent_sha256'):
            if a[key] != b[key]:
                raise ValueError(f'{key} mismatch at frame {index}')


def validate(root, task, role):
    root, folder = Path(root), Path(task['output'])
    config = read(root / 'paired_config.json')
    frozen = read(root / 'manifest.json')['input']
    count = 8 if task['group'] == 'warmup' else config['frames']
    reference_folder = root / ('encode_warmup_python' if task['group'] == 'warmup' else 'encode_1_python')
    reference = read(reference_folder / 'encoder/encode.json')
    path = folder / ('encoder/encode.json' if role == 'encode' else 'receiver/decode.json')
    result = read(path)
    compare_frames(result['frames'], reference['frames'], count)
    if result['header'] != reference['header']:
        raise ValueError('Reference and candidate headers differ')
    header = result['header']
    if (header['frames'], header['width'], header['height'], header['fps']) != (count, 640, 360, 12.5):
        raise ValueError('Second-clip compact geometry/timing changed')
    expected_stream = reference_folder / 'stream.glcv'
    record = {'frames_checked': count, 'all_rgb_and_latent_hashes_match_reference': True,
              'reference_kind': 'C1 Python on second existing clip; not B4 main-clip evidence',
              'reference_result': str(reference_folder / 'encoder/encode.json'),
              'reference_result_sha256': sha(reference_folder / 'encoder/encode.json'),
              'source_file_sha256': frozen['sha256'], 'result_sha256': sha(path)}
    if role == 'encode':
        if Path(result['input']['input_path']).resolve() != Path(frozen['path']).resolve():
            raise ValueError('Encoder used a different source file')
        if result['input']['sampled_source_frame_indices'] != list(range(0, 2 * count, 2)):
            raise ValueError('Source sampling differs from compact preset')
        hashes = result['input']['rgb_sha256']
        if len(hashes) != count or hashes != reference['input']['rgb_sha256']:
            raise ValueError('Source RGB differs from the Python reference')
        stream = folder / 'stream.glcv'
        parsed, packets, integrity = read_stream(stream, with_integrity=True)
        if parsed != header or len(packets) != count or integrity['container_version'] != 2:
            raise ValueError('Container structure differs')
        if sha(stream) != sha(expected_stream):
            raise ValueError('Entire Python/native stream differs')
        record.update(whole_stream_matches_reference=True, stream_sha256=sha(stream),
                      stream_bytes=stream.stat().st_size, source_rgb_matches_reference=True,
                      is_python_reference=(folder == reference_folder))
    else:
        for key in ('independent_decoder', 'encoder_modules_disabled', 'whole_file_checksum_verified', 'all_frame_checksums_match'):
            if result.get(key) is not True:
                raise ValueError('Independent decoder check failed: ' + key)
        if result['source_media_reads'] != [] or result['input_sha256'] != sha(expected_stream):
            raise ValueError('Receiver read source media or decoded another stream')
        record.update(independent_decoder=True, source_media_reads=[])
    previews = list(path.parent.glob('*_preview.mp4'))
    if len(previews) != (2 if role == 'encode' else 1):
        raise ValueError('Preview count differs')
    record['previews'] = {}
    for preview in previews:
        info = preview_info(preview)
        a, b = map(int, info['avg_frame_rate'].split('/'))
        if (int(info['nb_read_frames']), info['width'], info['height'], a / b) != (count, 640, 360, 12.5):
            raise ValueError('Preview shape, count or fps differs')
        record['previews'][preview.name] = info
    if list(path.parent.rglob('*.png')):
        raise ValueError('Display task unexpectedly saved PNG intermediates')
    record['png_count'] = 0
    write(folder / 'validation.json', record)
    return record
