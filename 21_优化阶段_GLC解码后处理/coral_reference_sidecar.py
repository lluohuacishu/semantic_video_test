"""Send one source coral tile with sparse motion and reuse it after GLC decode."""

import argparse
import io
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import cv2
import numpy as np

from run_p0 import REPO, file_sha, load_json, load_range, open_baseline, save_json
from experiment_common import array_sha
from sender_container import SenderWriter, read_sender_stream
from temporal_filter import gray

BASE_START, START, STOP = 576, 585, 675
FPS, HEIGHT, WIDTH = 15, 288, 512
BOX = (144, 128, 240, 224)
GRID = 6
SCALES = np.array([4096, 4096, 32], dtype=np.float32)
RECORD = np.dtype([('affine', '<i2', (2, 3)), ('mask', 'u1', (5,))])
LIMIT = 40000 * (STOP - START) // (8 * FPS)


def baseline(workspace):
    manifest = load_json(workspace / '01_项目/GLC后处理/接口与基线.json')
    return next(row for row in manifest['baselines'] if row['id'] == 'glc_q0_full15_512_fp32')


def cut(workspace, folder):
    spec = baseline(workspace)
    receipt_path = (workspace / spec['rgb']).with_name('result.json')
    receipt = load_json(receipt_path)
    original = receipt_path.parent.parent / 'fp32_sender/stream.glcsnd'
    if file_sha(original) != receipt['stream_sha256']:
        raise ValueError('The frozen GLC stream changed')
    header, packets, integrity = read_sender_stream(original)
    if (not receipt['reference_verified'] or receipt['reference_checksums'] != integrity['expected_reference_checksums']
            or (header['fps'], header['width'], header['height'], header['q']) != (FPS, WIDTH, HEIGHT, 0)
            or any(BASE_START % interval for interval in (header['intra_period'], header['reset_interval'], 8))):
        raise ValueError('Expected the original GLC stream and an independent I-frame cut')
    path = folder / 'base.glcsnd'
    writer = SenderWriter(path, dict(header, frames=STOP - BASE_START))
    try:
        for i in range(BASE_START, STOP):
            record = receipt['records'][i]
            packet = packets[i].copy()
            if record['index'] != i or record['kind'] != packet['kind'] or packet.pop('rgb_sha256') is not None:
                raise ValueError(f'Original packet layout changed at frame {i}')
            packet['latent_sha256'] = record['latent_sha256']
            writer.add(packet)
        writer.finish()
    finally:
        writer.close()
    save_json(folder / 'base_cut.json', dict(
        source_start=BASE_START, source_stop=STOP, display_start=START, display_stop=STOP,
        bytes=path.stat().st_size, fps=header['fps'], source_stream_sha256=file_sha(original),
        base_stream_sha256=file_sha(path), reference_checksums=writer.references.values(),
        input_receipt_sha256=file_sha(receipt_path)))


def track(source):
    x0, y0, x1, y1 = BOX
    images = [gray(frame) for frame in source]
    mask = np.zeros(images[0].shape, np.uint8)
    mask[y0:y1, x0:x1] = 255
    points = cv2.goodFeaturesToTrack(images[0], 150, .006, 3, mask=mask).reshape(-1, 2)
    anchor = points.copy()
    matrices = [np.array([[1., 0., x0], [0., 1., y0]])]
    counts = [len(points)]
    for previous, current in zip(images, images[1:]):
        found, status, error = cv2.calcOpticalFlowPyrLK(previous, current, points, None,
                                                      winSize=(17, 17), maxLevel=2)
        back, status_back, _ = cv2.calcOpticalFlowPyrLK(current, previous, found, None,
                                                     winSize=(17, 17), maxLevel=2)
        good = ((status.ravel() > 0) & (status_back.ravel() > 0)
                & (np.linalg.norm(back - points, axis=1) < .85) & (error.ravel() < 28))
        anchor, points = anchor[good], found[good]
        if len(points) < 7:
            break
        affine, inliers = cv2.estimateAffine2D(anchor, points, method=cv2.RANSAC, ransacReprojThreshold=1.8)
        if affine is None or int(inliers.sum()) < 7:
            break
        affine[:, 2] += affine[:, :2] @ [x0, y0]
        matrices.append(affine)
        counts.append(int(inliers.sum()))
    matrices = np.array(matrices, np.float32)
    stable = matrices.copy()
    for i in range(1, len(matrices) - 1):
        stable[i] = np.median(matrices[i - 1:i + 2], axis=0)
    return stable, counts


def detail(luma, sigma):
    value = luma.astype(np.float32)
    return value - cv2.GaussianBlur(value, (0, 0), sigma)


def visibility(source, tile, matrices):
    h, w = tile.shape[:2]
    original = gray(tile)
    ref_low = cv2.GaussianBlur(original.astype(np.float32), (0, 0), 1.2)
    ref_high = original - ref_low
    result = np.zeros((len(source), GRID, GRID), bool)
    full = np.ones(source.shape[1:3], np.float32)
    for i, affine in enumerate(matrices):
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        moved = cv2.warpAffine(gray(source[i]), affine, (w, h), flags=flags)
        visible = cv2.warpAffine(full, affine, (w, h), flags=flags) > .99
        low = cv2.GaussianBlur(moved.astype(np.float32), (0, 0), 1.2)
        high = moved - low
        offset = np.median((low - ref_low)[visible]) if visible.any() else 0.
        for row in range(GRID):
            for col in range(GRID):
                cell = np.s_[row*h//GRID:(row+1)*h//GRID, col*w//GRID:(col+1)*w//GRID]
                a, b = ref_high[cell], high[cell]
                norm = float(np.linalg.norm(a) * np.linalg.norm(b))
                result[i, row, col] = (visible[cell].mean() > .97 and norm > 1.
                    and np.mean(np.abs((low - ref_low)[cell] - offset)) < 17.
                    and float((a*b).sum()) / norm > .44)
    stable = result.copy()
    stable[1:] &= result[:-1]
    stable[:-1] &= result[1:]
    return stable


def send(workspace, folder):
    spec = baseline(workspace)
    _, ref, _ = open_baseline(workspace, spec, dict(q=0, intra_period=64, reset_interval=32))
    source = load_range(ref, START, STOP, (HEIGHT, WIDTH, 3))
    x0, y0, x1, y1 = BOX
    tile = source[0, y0:y1, x0:x1]
    matrices, counts = track(source)
    quantized = np.rint(matrices * SCALES)
    if np.max(np.abs(quantized)) > 32767:
        raise ValueError('Affine transform exceeds the wire precision')
    records = np.zeros(len(source), dtype=RECORD)
    records['affine'][:len(matrices)] = quantized.astype('<i2')
    masks = visibility(source, tile, quantized.astype(np.float32) / SCALES)
    records['mask'] = np.packbits(masks.reshape(len(source), -1), axis=1, bitorder='little')
    ok, png = cv2.imencode('.png', tile[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError('Could not encode the source texture tile')
    base = folder / 'base.glcsnd'
    manifest = dict(profile='glc-coral-tile-v1', base_sha256=file_sha(base), anchor_box=BOX,
                    width=WIDTH, height=HEIGHT, fps=FPS, source_frame=START,
                    base_offset=START-BASE_START, frames=len(source))
    package = io.BytesIO()
    with ZipFile(package, 'w') as archive:
        archive.writestr('manifest.json', json.dumps(manifest, separators=(',', ':')), compress_type=ZIP_DEFLATED)
        archive.writestr('anchor.png', png.tobytes(), compress_type=ZIP_STORED)
        archive.writestr('track.bin', records.tobytes(), compress_type=ZIP_DEFLATED)
    payload = package.getvalue()
    if len(payload) > LIMIT:
        raise ValueError(f'Semantic file uses {len(payload)} bytes for a {LIMIT}-byte budget')
    path = folder / 'coral.glcsem'
    with path.open('xb') as output:
        output.write(payload)
    with ZipFile(path) as archive:
        ledger = {item.filename: item.compress_size for item in archive.infolist()}
    ledger['container'] = path.stat().st_size - sum(ledger.values())
    save_json(folder / 'send.json', dict(
        manifest=manifest, semantic_sha256=file_sha(path), semantic_bytes=len(payload),
        semantic_limit_bytes=LIMIT, ledger=ledger, tracked_frames=len(matrices), inliers=counts,
        source_rgb_sha256=array_sha(source), active_cells=masks.sum(axis=(1, 2)).tolist(),
        code_sha256=file_sha(__file__)))
    print(f'Semantic file: {len(payload)}/{LIMIT} bytes, tracks on {len(matrices)} frames', flush=True)


def receive(folder):
    base, semantic = folder / 'base.glcsnd', folder / 'coral.glcsem'
    directory = (folder / 'base_receive').resolve()
    output = (folder / 'received').resolve()
    output.mkdir(exist_ok=False)

    def audit(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes)):
            path = Path(str(args[0]))
            if path.suffix.lower() in ('.npy', '.npz', '.png', '.jpg', '.mp4', '.mkv'):
                if not any(path.resolve().is_relative_to(root) for root in (directory, output)):
                    raise RuntimeError('Receiver is isolated from source media')

    sys.addaudithook(audit)
    receipt = load_json(directory / 'result.json')
    base_hash = file_sha(base)
    base_rgb_path = directory / 'decoded_rgb.npy'
    if (not receipt['reference_verified'] or not receipt['source_media_reads_forbidden']
            or receipt['stream_sha256'] != base_hash or file_sha(base_rgb_path) != receipt['decoded_rgb_sha256']):
        raise ValueError('Expected the independent GLC decode of the received base file')
    if semantic.stat().st_size > LIMIT:
        raise ValueError('Semantic file exceeds its independent budget')
    with ZipFile(semantic) as archive:
        if sorted(archive.namelist()) != ['anchor.png', 'manifest.json', 'track.bin']:
            raise ValueError('Unexpected semantic fields')
        meta = json.loads(archive.read('manifest.json'))
        if (meta['profile'], meta['base_sha256'], meta['width'], meta['height'], meta['fps']) != (
                'glc-coral-tile-v1', base_hash, WIDTH, HEIGHT, FPS):
            raise ValueError('The semantic file belongs to a different base stream')
        if (meta['frames'], meta['base_offset'], meta['source_frame']) != (STOP-START, START-BASE_START, START):
            raise ValueError('Unexpected semantic timeline')
        records = np.frombuffer(archive.read('track.bin'), dtype=RECORD)
        if len(records) != meta['frames']:
            raise ValueError('Incorrect semantic frame count')
        tile_bgr = cv2.imdecode(np.frombuffer(archive.read('anchor.png'), np.uint8), cv2.IMREAD_COLOR)
    x0, y0, x1, y1 = meta['anchor_box']
    if tile_bgr is None or not 0 <= x0 < x1 <= WIDTH or not 0 <= y0 < y1 <= HEIGHT:
        raise ValueError('Invalid source tile or coordinates')
    tile = tile_bgr[..., ::-1]
    h, w = tile.shape[:2]
    if (w, h) != (x1-x0, y1-y0):
        raise ValueError('Source tile dimensions differ from the transmitted coordinates')
    start = meta['base_offset']
    raw = np.load(base_rgb_path, mmap_mode='r', allow_pickle=False)[start:start+meta['frames']]
    if raw.shape != (meta['frames'], HEIGHT, WIDTH, 3) or raw.dtype != np.uint8:
        raise ValueError('Invalid independently decoded RGB')
    detail_planes = [detail(gray(image), 2.5) for image in (raw[0, y0:y1, x0:x1], tile)]
    masks = np.unpackbits(records['mask'], axis=1, count=GRID*GRID, bitorder='little').reshape(-1, GRID, GRID)
    affine = records['affine'].astype(np.float32) / SCALES
    gy, gx = np.mgrid[:h, :w]
    border = np.minimum.reduce([gx, gy, w-1-gx, h-1-gy]).astype(np.float32)
    feather = np.minimum(border / 5., 1.)
    glc_anchor, source_anchor = np.array(raw), np.array(raw)
    active = np.zeros(raw.shape[:3], np.uint8)
    for i, frame in enumerate(raw):
        if not masks[i].any():
            continue
        support = cv2.resize(masks[i].astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        alpha = support * cv2.GaussianBlur(support, (0, 0), 1.4) * feather
        alpha = cv2.warpAffine(alpha, affine[i], (WIDTH, HEIGHT))
        active[i] = np.rint(alpha * 255).astype(np.uint8)
        base_detail = detail(gray(frame), 2.5)
        for plane, target in zip(detail_planes, (glc_anchor, source_anchor), strict=True):
            warped = cv2.warpAffine(plane, affine[i], (WIDTH, HEIGHT))
            delta = np.clip(warped - base_detail, -48., 48.) * (alpha * .9)
            target[i] = np.rint(frame.astype(np.float32) + delta[..., None]).clip(0, 255).astype(np.uint8)
    files = {}
    for name, array in dict(glc_anchor=glc_anchor, source_anchor=source_anchor, alpha=active).items():
        path = output / f'{name}.npy'
        np.save(path, array, allow_pickle=False)
        files[name] = file_sha(path)
    save_json(output / 'result.json', dict(
        complete=True, base_sha256=base_hash, semantic_sha256=file_sha(semantic), frames=len(raw),
        fps=FPS, source_start=START, source_media_reads_forbidden=True, files=files,
        active_frames=int(np.count_nonzero((active >= 128).any(axis=(1, 2)))),
        code_sha256=file_sha(__file__)))
    print(f'Receiver: {len(raw)} frames from the two local streams', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('cut', 'send', 'receive'))
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    if args.phase == 'receive':
        receive(args.run)
    else:
        (cut if args.phase == 'cut' else send)(args.workspace, args.run)


if __name__ == '__main__':
    main()
