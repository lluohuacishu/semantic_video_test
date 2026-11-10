"""Bounded auxiliary-file pilot for persistent GLC RGB references.

No source images or source geometry files are needed by unpack/render. The
wire file includes quantized motion, region contours, foreground locks, keys,
transition weights, and the identity of the independently decoded base.
"""

import json
import lzma
from struct import Struct
import zlib

import cv2
import numpy as np


MAGIC = b'GLCSTR01'
PREFIX = Struct('<8sI')
CRC = Struct('<I')
LIMIT = 50000  # 40,000 bit/s * 10 seconds / 8
PREDICTORS = ((), (0,), (2,), (0, 2), (1, 2), (0, 1, 2))
KEYS = (0, 15, 45, 75, 105, 135)
RAMP = (1., 5/6, 4/6, 3/6, 2/6, 1/6)
CROP = (240, 0, 512, 144)
EVENTS = tuple(key+j for key in KEYS[1:] for j in range(len(RAMP)))


def compressed(values, predict):
    choices = []
    for axes in PREDICTORS if predict else ((),):
        residual = values
        for axis in axes:
            residual = np.diff(residual, axis=axis,
                prepend=np.zeros_like(np.take(residual, [0], axis=axis)))
        choices.append((lzma.compress(residual.tobytes()), axes))
    payload, axes = min(choices, key=lambda x: len(x[0]))
    return payload, dict(dtype=values.dtype.str, shape=list(values.shape),
        axes=list(axes), raw_bytes=values.nbytes, bytes=len(payload))


def contour_values(masks, epsilon):
    values = []
    x0, y0, x1, y1 = CROP
    for mask in masks:
        contours, _ = cv2.findContours(mask[y0:y1, x0:x1].astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        values.append(len(contours))
        for contour in contours:
            points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            values.extend((len(points), *points.reshape(-1).tolist()))
    return np.asarray(values, dtype='<u2')


def motion_grid(flows, reliability, grid, scale):
    """Average motion on reliable coral; affine-fill unused grid locations."""
    x0, y0, x1, y1 = CROP
    gy, gx = np.mgrid[:grid[1], :grid[0]]
    basis = np.stack((gx, gy, np.ones_like(gx)), axis=-1)
    result = []
    for flow, alpha in zip(flows, reliability, strict=True):
        a = alpha[y0:y1, x0:x1]
        denominator = cv2.resize(a, grid, interpolation=cv2.INTER_AREA)
        numerator = cv2.resize(flow[y0:y1, x0:x1]*a[..., None], grid,
            interpolation=cv2.INTER_AREA)
        value = numerator / np.maximum(denominator[..., None], 1e-6)
        valid = denominator > .01
        if valid.sum() < 3:
            raise ValueError('Insufficient reliable motion support')
        fit = np.linalg.lstsq(basis[valid], value[valid], rcond=None)[0]
        value[~valid] = (basis @ fit)[~valid]
        result.append(value)
    quantized = np.rint(np.stack(result)*scale)
    if not np.isfinite(quantized).all() or np.abs(quantized).max() > 4096:
        raise ValueError('Unexpected motion magnitude')
    return quantized.astype('<i2')


def pack(new_motion, old_motion, masks, fish, *, base_sha, grid, scale, epsilon):
    fields = dict(new_motion=new_motion, old_motion=old_motion,
        contours=contour_values(masks, epsilon),
        fish=np.packbits(fish.astype(np.uint8), axis=2))
    payloads, descriptors = [], {}
    for name, value in fields.items():
        payload, descriptor = compressed(value, name.endswith('motion'))
        payloads.append(payload)
        descriptors[name] = descriptor
    header = dict(base_sha256=base_sha, width=512, height=288, fps=15, start=0,
        frames=150, keys=KEYS, old_weights=RAMP, crop=CROP, grid=grid,
        motion_scale=scale, contour_epsilon=epsilon, inward_feather_px=2.,
        sections=descriptors)
    encoded = json.dumps(header, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode('ascii')
    body = PREFIX.pack(MAGIC, len(encoded)) + encoded + b''.join(payloads)
    return body + CRC.pack(zlib.crc32(body)), header


def unpack(data, base_sha):
    if not PREFIX.size+CRC.size < len(data) <= LIMIT:
        raise ValueError('Auxiliary file is empty, truncated, or over budget')
    body, checksum = data[:-CRC.size], data[-CRC.size:]
    if zlib.crc32(body) != CRC.unpack(checksum)[0]:
        raise ValueError('Auxiliary file CRC mismatch')
    magic, size = PREFIX.unpack_from(body)
    if magic != MAGIC or not 0 < size <= 8192 or PREFIX.size+size >= len(body):
        raise ValueError('Unknown auxiliary header')
    header = json.loads(body[PREFIX.size:PREFIX.size+size])
    expected = dict(base_sha256=base_sha, width=512, height=288, fps=15,
        frames=150, start=0, keys=list(KEYS), old_weights=list(RAMP), crop=list(CROP), inward_feather_px=2.)
    if any(header.get(k) != value for k, value in expected.items()):
        raise ValueError('Auxiliary file does not match the base/profile')
    grid = header['grid']
    scale = header['motion_scale']
    if (len(grid) != 2 or any(type(v) is not int for v in grid)
            or not 4 <= grid[0] <= 64 or not 4 <= grid[1] <= 36 or scale not in (2, 4, 8)):
        raise ValueError('Invalid motion grid')
    shapes = dict(new_motion=(150, grid[1], grid[0], 2), old_motion=(30, grid[1], grid[0], 2),
        fish=(150, 288, 64))
    dtypes = dict(new_motion='<i2', old_motion='<i2', contours='<u2', fish='|u1')
    if list(header['sections']) != list(dtypes):
        raise ValueError('Unexpected auxiliary sections')
    offset = PREFIX.size+size
    arrays = {}
    for name, spec in header['sections'].items():
        shape = spec['shape']
        if (spec['dtype'] != dtypes[name] or len(shape) > 4
                or any(type(v) is not int or v < 1 or v > 1000000 for v in shape)
                or (name in shapes and tuple(shape) != shapes[name])
                or (name == 'contours' and len(shape) != 1)):
            raise ValueError('Invalid section geometry')
        dtype = np.dtype(spec['dtype'])
        raw_size = int(np.prod(shape))*dtype.itemsize
        if (not 0 < raw_size <= 4*1024*1024 or spec['raw_bytes'] != raw_size
                or not 0 < spec['bytes'] <= LIMIT or offset+spec['bytes'] > len(body)
                or tuple(spec['axes']) not in (PREDICTORS if name.endswith('motion') else ((),))):
            raise ValueError('Invalid compressed section bounds')
        decoder = lzma.LZMADecompressor(memlimit=64*1024*1024)
        payload = decoder.decompress(body[offset:offset+spec['bytes']], max_length=raw_size+1)
        if len(payload) != raw_size or not decoder.eof or decoder.unused_data:
            raise ValueError('Invalid compressed section length')
        offset += spec['bytes']
        value = np.frombuffer(payload, dtype=dtype).reshape(shape)
        for axis in reversed(spec['axes']):
            value = np.cumsum(value, axis=axis, dtype=dtype)
        if name.endswith('motion') and np.abs(value.astype(np.int32)).max() > 4096:
            raise ValueError('Decoded motion outside profile')
        arrays[name] = value
    if offset != len(body):
        raise ValueError('Trailing auxiliary payload')
    return header, arrays


def decode_masks(values):
    x0, y0, x1, y1 = CROP
    masks = np.zeros((180, y1-y0, x1-x0), np.uint8)
    pos = 0
    for mask in masks:
        if pos >= len(values):
            raise ValueError('Missing contour frame')
        count = int(values[pos]); pos += 1
        contours = []
        for _ in range(count):
            if pos >= len(values):
                raise ValueError('Missing contour size')
            points = int(values[pos]); pos += 1
            if not 1 <= points <= 4096 or pos+2*points > len(values):
                raise ValueError('Contour is out of bounds')
            contour = values[pos:pos+2*points].astype(np.int32).reshape(-1, 2)
            if (contour[:, 0] >= x1-x0).any() or (contour[:, 1] >= y1-y0).any():
                raise ValueError('Contour coordinate outside crop')
            contours.append(contour); pos += 2*points
        if contours:
            cv2.fillPoly(mask, contours, 1)
    if pos != len(values):
        raise ValueError('Trailing contour values')
    return masks


def render(raw, header, arrays):
    if raw.shape != (150, 288, 512, 3) or raw.dtype != np.uint8:
        raise ValueError('Unexpected base RGB')
    x0, y0, x1, y1 = CROP
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    fish = np.unpackbits(arrays['fish'], axis=2)
    masks = decode_masks(arrays['contours'])
    output = raw.copy()
    alphas = np.zeros(raw.shape[:3], np.float32)
    event_indices = {frame: k for k, frame in enumerate(EVENTS)}
    previous_key = 0

    def projection(flow, reference, mask, current):
        flow = cv2.resize(flow.astype(np.float32)/header['motion_scale'],
            (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR)
        mx, my = xx+flow[..., 0], yy+flow[..., 1]
        inside = (mx >= 2) & (mx < 510) & (my >= 2) & (my < 286)
        ref_fish = cv2.remap(fish[reference], mx, my, cv2.INTER_NEAREST)
        safe = mask.astype(bool) & inside & ~fish[current, y0:y1, x0:x1].astype(bool) & ~ref_fish.astype(bool)
        a = np.minimum(cv2.distanceTransform(safe.astype(np.uint8), cv2.DIST_L2, 5)/2., 1.)
        rgb = cv2.remap(raw[reference].astype(np.float32), mx, my, cv2.INTER_CUBIC)
        return rgb, a

    for i in range(len(raw)):
        if i in KEYS:
            if i:
                previous_key = anchor
            anchor = i
        fresh, a = projection(arrays['new_motion'][i], anchor, masks[i], i)
        current = raw[i, y0:y1, x0:x1].astype(np.float32)
        result = np.rint(np.clip(current*(1-a[..., None])+fresh*a[..., None], 0, 255))
        if i in event_indices:
            event = event_indices[i]
            old, shared = projection(arrays['old_motion'][event], previous_key, masks[150+event], i)
            weight = np.minimum(shared, a)*RAMP[event % len(RAMP)]
            result = np.rint(np.clip(result+weight[..., None]*(old-fresh), 0, 255))
        output[i, y0:y1, x0:x1] = result.astype(np.uint8)
        alphas[i, y0:y1, x0:x1] = a
    if not np.array_equal(output[alphas == 0], raw[alphas == 0]):
        raise ValueError('Zero-weight output changed')
    if not np.array_equal(output[fish.astype(bool)], raw[fish.astype(bool)]):
        raise ValueError('Transmitted foreground protection failed')
    return output, alphas
