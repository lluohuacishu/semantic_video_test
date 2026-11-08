"""Budget and decode multiple local coral references on the same 15 FPS clip."""

import argparse
from dataclasses import dataclass
import io
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import cv2
import numpy as np

from coral_reference_sidecar import (
    BASE_START, START, STOP, FPS, HEIGHT, WIDTH, GRID, SCALES, RECORD, LIMIT,
    REPO, baseline, detail, file_sha, gray, load_json, load_range, open_baseline, save_json, track, visibility,
)

COUNT = STOP - START
JPEG_QUALITY = 79
SIGMA, MIX = 4.5, .94
FISH = {
    0: (196, 5, 404, 130), 10: (198, 5, 415, 130), 20: (202, 10, 423, 136),
    30: (210, 36, 446, 143), 40: (209, 65, 463, 155), 50: (211, 60, 454, 178),
    60: (212, 64, 437, 178), 75: (225, 57, 423, 178), 89: (245, 50, 389, 148),
}
ANCHORS = {
    0: [(80, 156), (120, 90), (142, 134), (185, 147), (247, 129), (308, 139),
        (390, 124), (390, 12), (145, 12)],
    30: [(105, 153), (135, 105), (178, 144), (214, 160), (316, 154), (389, 143),
         (381, 5), (164, 4)],
    44: [(143, 154), (187, 141), (218, 159), (299, 162), (374, 164), (391, 4)],
    54: [(118, 140), (152, 115), (193, 159), (250, 157), (309, 159), (387, 156),
         (386, 10), (164, 1), (283, 1)],
    69: [(147, 127), (191, 162), (243, 152), (290, 158), (373, 164), (391, 128), (388, 4)],
}
SIDE = 120
COVERAGE_TARGET = 14000


@dataclass
class Tile:
    frame: int
    box: tuple
    image: bytes
    track: bytes
    support: np.ndarray | None = None


def fish_boxes():
    indices = np.arange(COUNT)
    anchors = np.array(list(FISH))
    values = np.array(list(FISH.values()))
    return np.rint(np.stack([np.interp(indices, anchors, values[:, col]) for col in range(4)], axis=1)).astype('<u2')


def guards(boxes):
    for x0, y0, x1, y1 in boxes:
        if not 0 <= x0 < x1 <= WIDTH or not 0 <= y0 < y1 <= HEIGHT:
            raise ValueError('Invalid foreground exclusion box')
        mask = np.ones((HEIGHT, WIDTH), np.float32)
        mask[y0:y1, x0:x1] = 0
        yield mask * cv2.GaussianBlur(mask, (0, 0), 2.)


def tile_frames(tile, guard):
    records = np.frombuffer(tile.track, RECORD)
    masks = np.unpackbits(records['mask'], axis=1, count=GRID*GRID, bitorder='little').reshape(-1, GRID, GRID)
    matrices = records['affine'].astype(np.float32) / SCALES
    y, x = np.mgrid[:SIDE, :SIDE]
    feather = np.minimum(np.minimum.reduce([x, y, SIDE-1-x, SIDE-1-y]).astype(np.float32) / 6., 1.)
    for j, (cells, affine) in enumerate(zip(masks, matrices, strict=True)):
        if not cells.any():
            continue
        i = tile.frame + j
        support = cv2.resize(cells.astype(np.float32), (SIDE, SIDE), interpolation=cv2.INTER_NEAREST)
        alpha = support * cv2.GaussianBlur(support, (0, 0), 1.4) * feather
        ramp = min(1., (j+1)/4.) if tile.frame else 1.
        if tile.frame + len(records) < COUNT:
            ramp = min(ramp, (len(records)-j)/4.)
        alpha = cv2.warpAffine(alpha, affine, (WIDTH, HEIGHT)) * guard[i] * ramp
        yield i, affine, alpha


def candidate(source, boxes, guard, start, x, y):
    box = (x, y, x+SIDE, y+SIDE)
    video = source[start:]
    tile = video[0, y:y+SIDE, x:x+SIDE]
    matrices, _ = track(video, box)
    if len(matrices) < 8:
        return None
    quantized = np.rint(matrices * SCALES)
    if np.max(np.abs(quantized)) > 32767:
        return None
    records = np.zeros(len(matrices), dtype=RECORD)
    records['affine'] = quantized.astype('<i2')
    mask = visibility(video[:len(matrices)], tile, quantized.astype(np.float32) / SCALES)
    x0, y0, x1, y1 = boxes[start]
    original = np.ones((HEIGHT, WIDTH), np.uint8)
    original[y0:y1, x0:x1] = 0
    original = original[y:y+SIDE, x:x+SIDE]
    safe_anchor = original.reshape(GRID, SIDE//GRID, GRID, SIDE//GRID).min(axis=(1, 3)).astype(bool)
    mask &= safe_anchor
    records['mask'] = np.packbits(mask.reshape(len(records), -1), axis=1, bitorder='little')
    ok, data = cv2.imencode('.jpg', gray(tile), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError('Could not encode the luminance reference')
    item = Tile(start, box, data.tobytes(), records.tobytes())
    item.support = np.zeros((COUNT, HEIGHT, WIDTH), bool)
    for i, _, alpha in tile_frames(item, guard):
        item.support[i] = alpha >= .5
    return item if item.support.any() else None


def pack(base_sha, boxes, tiles):
    meta = dict(profile='glc-multi-coral-v1', base_sha256=base_sha, start=START, offset=START-BASE_START,
                frames=COUNT, fps=FPS, width=WIDTH, height=HEIGHT,
                tiles=[dict(frame=tile.frame, box=tile.box) for tile in tiles])
    payload = io.BytesIO()
    with ZipFile(payload, 'w') as archive:
        archive.writestr('manifest.json', json.dumps(meta, separators=(',', ':')), ZIP_DEFLATED)
        archive.writestr('fish.bin', boxes.tobytes(), ZIP_DEFLATED)
        for i, tile in enumerate(tiles):
            archive.writestr(f'{i:02d}.jpg', tile.image, ZIP_STORED)
            archive.writestr(f'{i:02d}.bin', tile.track, ZIP_DEFLATED)
    return payload.getvalue()


def send(workspace, base_run, run):
    spec = baseline(workspace)
    _, reference, _ = open_baseline(workspace, spec, dict(q=0, intra_period=64, reset_interval=32))
    source = load_range(reference, START, STOP, (HEIGHT, WIDTH, 3))
    base_sha = file_sha(base_run / 'base.glcsnd')
    boxes = fish_boxes()
    guard = list(guards(boxes))
    options = [item for start, positions in ANCHORS.items() for x, y in positions
               if (item := candidate(source, boxes, guard, start, x, y)) is not None]
    chosen = []
    coverage = np.zeros((COUNT, HEIGHT, WIDTH), bool)
    payload = pack(base_sha, boxes, chosen)
    gains = []
    while options:
        best = None
        remaining = np.maximum(COVERAGE_TARGET - coverage.sum(axis=(1, 2)), 0)
        for item in options:
            data = pack(base_sha, boxes, [*chosen, item])
            if len(data) > LIMIT:
                continue
            added = np.count_nonzero(item.support & ~coverage, axis=(1, 2))
            gain = int(added.sum())
            score = (np.minimum(added, remaining).sum() + .15*gain) / (len(data) - len(payload))
            if gain >= 48*COUNT and (best is None or score > best[0]):
                best = score, item, data, gain
        if best is None:
            break
        _, item, payload, gain = best
        coverage |= item.support
        chosen.append(item)
        options = [option for option in options if option is not item]
        gains.append(dict(anchor_frame=START+item.frame, box=item.box,
                          jpeg_bytes=len(item.image), mean_added_pixels=gain/COUNT))
    if not chosen:
        raise RuntimeError('No reliable reference tile fits the budget')
    path = run / 'coral_multi.glcsem'
    with path.open('xb') as out:
        out.write(payload)
    with ZipFile(path) as archive:
        ledger = {entry.filename: entry.compress_size for entry in archive.infolist()}
    ledger['container'] = len(payload) - sum(ledger.values())
    save_json(run/'send.json', dict(tiles=gains, semantic_bytes=len(payload), byte_limit=LIMIT, ledger=ledger,
        semantic_sha256=file_sha(path), base_sha256=base_sha, jpeg_quality=JPEG_QUALITY,
        mean_effective_pixels=float(coverage.sum()/COUNT), coverage_target_pixels=COVERAGE_TARGET,
        code_sha256=file_sha(__file__)))
    print(f'Selected {len(chosen)} tiles, {len(payload)} bytes, {coverage.sum()/COUNT:.0f} effective pixels per frame', flush=True)


def read_package(path, base_sha):
    if path.stat().st_size > LIMIT:
        raise ValueError('Semantic stream exceeds its independent byte budget')
    with ZipFile(path) as archive:
        meta = json.loads(archive.read('manifest.json'))
        expected = ('glc-multi-coral-v1', base_sha, START, START-BASE_START, COUNT, FPS, WIDTH, HEIGHT)
        if tuple(meta[key] for key in ('profile','base_sha256','start','offset','frames','fps','width','height')) != expected:
            raise ValueError('Semantic stream does not match the GLC base')
        boxes = np.frombuffer(archive.read('fish.bin'), '<u2').reshape(COUNT, 4)
        tiles = []
        for i, item in enumerate(meta['tiles']):
            frame = item['frame']
            x0, y0, x1, y1 = item['box']
            payload = archive.read(f'{i:02d}.bin')
            records = np.frombuffer(payload, RECORD)
            if not (0 <= x0 < x1 <= WIDTH and 0 <= y0 < y1 <= HEIGHT and
                    x1-x0 == y1-y0 == SIDE and 0 <= frame < frame+len(records) <= COUNT):
                raise ValueError('Invalid reference tile extent')
            tiles.append(Tile(frame, tuple(item['box']), archive.read(f'{i:02d}.jpg'), payload))
    if not tiles:
        raise ValueError('Empty semantic stream')
    return tiles, boxes


def receive(base_run, run):
    directory = (base_run / 'base_receive').resolve()
    output = (run / 'received').resolve()
    output.mkdir(exist_ok=False)

    def audit(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes)):
            path = Path(str(args[0]))
            if path.suffix.lower() in ('.npy','.npz','.png','.jpg','.mp4','.mkv'):
                if not any(path.resolve().is_relative_to(root) for root in (directory, output)):
                    raise RuntimeError('Receiver is isolated from source media')

    sys.addaudithook(audit)
    receipt = load_json(directory/'result.json')
    base_sha = file_sha(base_run/'base.glcsnd')
    path = directory/'decoded_rgb.npy'
    if (not receipt['reference_verified'] or not receipt['source_media_reads_forbidden']
            or receipt['stream_sha256'] != base_sha or receipt['decoded_rgb_sha256'] != file_sha(path)):
        raise ValueError('Expected the independently decoded GLC base')
    raw = np.load(path, mmap_mode='r', allow_pickle=False)[START-BASE_START:STOP-BASE_START]
    if raw.shape != (COUNT, HEIGHT, WIDTH, 3) or raw.dtype != np.uint8:
        raise ValueError('Invalid base RGB geometry')
    package = run/'coral_multi.glcsem'
    tiles, boxes = read_package(package, base_sha)
    guard = list(guards(boxes))
    weight, support = (np.zeros(raw.shape[:3], np.float32) for _ in range(2))
    totals = [np.zeros(raw.shape[:3], np.float32) for _ in range(2)]
    for tile in tiles:
        x0, y0, x1, y1 = tile.box
        luma = cv2.imdecode(np.frombuffer(tile.image, np.uint8), cv2.IMREAD_GRAYSCALE)
        if luma is None or luma.shape != (SIDE, SIDE):
            raise ValueError('Incorrect compressed luma shape')
        planes = [detail(image, SIGMA) for image in (gray(raw[tile.frame, y0:y1, x0:x1]), luma)]
        for i, affine, alpha in tile_frames(tile, guard):
            weight[i] += alpha
            np.maximum(support[i], alpha, out=support[i])
            for total, plane in zip(totals, planes, strict=True):
                warped = cv2.warpAffine(plane, affine, (WIDTH, HEIGHT), flags=cv2.INTER_CUBIC)
                total[i] += alpha * warped
    alpha = np.rint(support * 255.).astype(np.uint8)
    active = alpha >= 128
    files = {}
    for name, total in zip(('glc_multi', 'source_multi'), totals, strict=True):
        video = np.empty_like(raw)
        for i, frame in enumerate(raw):
            projected = total[i] / np.maximum(weight[i], 1e-6)
            correction = np.clip(projected-detail(gray(frame), SIGMA), -54., 54.) * (support[i]*MIX)
            video[i] = np.rint(frame.astype(np.float32) + correction[..., None]).clip(0,255).astype(np.uint8)
            x0, y0, x1, y1 = boxes[i]
            if not np.array_equal(video[i, y0:y1, x0:x1], frame[y0:y1, x0:x1]):
                raise ValueError('Composition changed the excluded fish region')
        path = output/f'{name}.npy'
        np.save(path, video, allow_pickle=False)
        files[name] = file_sha(path)
    np.save(output/'alpha.npy', alpha, allow_pickle=False)
    files['alpha'] = file_sha(output/'alpha.npy')
    save_json(output/'result.json', dict(complete=True, files=files, base_sha256=base_sha,
        semantic_sha256=file_sha(package), source_media_reads_forbidden=True, excluded_fish_region_bit_exact=True,
        frames=COUNT, fps=FPS, active_frames=int(active.any(axis=(1,2)).sum()),
        mean_effective_pixels=float(active.sum()/COUNT), code_sha256=file_sha(__file__)))
    print(f'Source-free composition complete on {COUNT} frames', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('send', 'receive'))
    parser.add_argument('--workspace', type=Path, default=REPO.parents[2])
    parser.add_argument('--base-run', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    if args.phase == 'send':
        send(args.workspace, args.base_run, args.run)
    else:
        receive(args.base_run, args.run)


if __name__ == '__main__':
    main()
