"""GLCCSP01: per-frame motion with conservative grouped spatial support.

Whole-clip experiment format, not a real-time packet protocol. Predictors and
byte layouts are lossless; support reduction and motion resizing are not.
"""

import json
import lzma
from struct import Struct
import zlib

import cv2
import numpy as np

from fullframe_structure_stream import render as render_full


KEYS = (0, 15, 45, 75, 105, 135)
RAMP = (1., 5/6, 4/6, 3/6, 2/6, 1/6)
MAGIC = b'GLCCSP01'
PREFIX = Struct('<8sI')
CRC = Struct('<I')
LIMIT = 50000
AXES = ((0,), (0, 2), (1, 2), (0, 1, 2))
ORDERS = ((0, 1, 2), (1, 2, 0), (2, 1, 0))
GRIDS = ((24, 14), (20, 12), (16, 9), (12, 7))


def groups(span):
    intervals = list(zip(KEYS, (*KEYS[1:], 150)))+[(150+i, 156+i) for i in range(0, 30, 6)]
    return [(start, min(start+span, stop)) for begin, stop in intervals
            for start in range(begin, stop, span)]


def reduce_support(pixel_safe, step, span):
    if pixel_safe.shape != (180, 288, 512) or pixel_safe.dtype != bool:
        raise ValueError('Safety geometry')
    cells = pixel_safe.reshape(180, 288//step, step, 512//step, step).all(axis=(2, 4))
    return np.stack([cells[a:b].all(axis=0) for a, b in groups(span)])


def expand_support(compact, span):
    result = np.empty((180, *compact.shape[1:]), np.uint8)
    for mask, (a, b) in zip(compact, groups(span), strict=True):
        result[a:b] = mask
    return result


def motion_bytes(value, axes, layout):
    residual = value
    for axis in axes:
        residual = np.diff(residual, axis=axis,
                           prepend=np.zeros_like(np.take(residual, [0], axis=axis)))
    if layout == 'i16':
        return residual.astype('<i2').tobytes()
    r = residual.astype(np.int32)
    zigzag = ((r << 1) ^ (r >> 31)).astype('<u2')
    return zigzag.view('u1').reshape(-1, 2).T.tobytes()


def dependencies(mask, grid, tone=False):
    if tone:
        # Float32 GaussianBlur(sigma=12) uses 97 taps; include its full support.
        mask = cv2.dilate(mask.astype('u1'), np.ones((97, 97), 'u1')).astype(bool)
    y, x = np.nonzero(mask)
    y0 = np.floor((y+.5)*grid[1]/288-.5).astype(int)
    x0 = np.floor((x+.5)*grid[0]/512-.5).astype(int)
    needed = np.zeros((grid[1], grid[0]), bool)
    for dy in (0, 1):
        for dx in (0, 1):
            needed[(y0+dy).clip(0, grid[1]-1), (x0+dx).clip(0, grid[0]-1)] = True
    return needed


def required_nodes(compact, span, step, grid, fusion):
    cells = expand_support(compact, span)
    return np.stack([dependencies(np.repeat(np.repeat(c, step, 0), step, 1).astype(bool),
                                  grid, fusion == 'F') for c in cells])


def spatial_predict(values, needed, y, x, predictor):
    left = x > 0 and needed[y, x-1]
    up = y > 0 and needed[y-1, x]
    if left and up:
        a, b = values[y, x-1].astype(np.int32), values[y-1, x].astype(np.int32)
        if needed[y-1, x-1]:
            prediction = a+b-values[y-1, x-1]
            return np.clip(prediction, np.minimum(a, b), np.maximum(a, b)) if predictor == 'median' else prediction
        return (a+b)//2
    if left:
        return values[y, x-1].astype(np.int32)
    if up:
        return values[y-1, x].astype(np.int32)
    return np.zeros(2, np.int32)


def sparse_encode(motion, needed, predictor):
    residual = []
    for values, mask in zip(motion, needed, strict=True):
        for y, x in zip(*np.nonzero(mask)):
            residual.append(values[y, x].astype(np.int32)-spatial_predict(values, mask, y, x, predictor))
    return np.asarray(residual, dtype='<i2').reshape(-1, 2)


def sparse_decode(residual, needed, predictor):
    motion = np.zeros((*needed.shape, 2), dtype='<i2')
    pos = 0
    for values, mask in zip(motion, needed, strict=True):
        for y, x in zip(*np.nonzero(mask)):
            value = residual[pos].astype(np.int32)+spatial_predict(values, mask, y, x, predictor)
            if np.any(np.abs(value) > 4096):
                raise ValueError('Sparse motion range')
            values[y, x] = value
            pos += 1
    if pos != len(residual):
        raise ValueError('Sparse motion count')
    return motion


def pack(motion, compact, base_sha, *, step=4, span=3, fusion='E', sparse=False):
    motion_choices = []
    if sparse:
        needed = required_nodes(compact, span, step, (motion.shape[2], motion.shape[1]), fusion)
        for predictor in ('median', 'gradient'):
            residual = sparse_encode(motion, needed, predictor)
            payload = lzma.compress(motion_bytes(residual, (), 'zigzag'))
            motion_choices.append((payload, dict(axes=[], layout='zigzag', sparse=predictor, raw_bytes=residual.nbytes)))
    else:
        for axes in AXES:
            for layout in ('i16', 'zigzag'):
                payload = lzma.compress(motion_bytes(motion, axes, layout))
                motion_choices.append((payload, dict(axes=list(axes), layout=layout, raw_bytes=motion.nbytes)))
    mp, ms = min(motion_choices, key=lambda pair: len(pair[0]))
    support_choices = []
    for order in ORDERS:
        for xor in (False, True):
            v = compact.transpose(order).astype(np.uint8)
            if xor:
                original = v.copy()
                v[..., 1:] ^= original[..., :-1]
            raw = np.packbits(v, axis=-1).tobytes()
            support_choices.append((lzma.compress(raw), dict(order=list(order), xor_last=xor, raw_bytes=len(raw))))
    sp, ss = min(support_choices, key=lambda pair: len(pair[0]))
    ms.update(bytes=len(mp))
    ss.update(bytes=len(sp))
    header = dict(base_sha256=base_sha, width=512, height=288, frames=150, start=0, fps=15,
                  keys=KEYS, old_weights=RAMP, grid=[motion.shape[2], motion.shape[1]],
                  scale=2, mask_step=step, group_span=span, fusion=fusion,
                  sections=dict(motion=ms, support=ss))
    encoded = json.dumps(header, separators=(',', ':'), allow_nan=False).encode('ascii')
    body = PREFIX.pack(MAGIC, len(encoded))+encoded+mp+sp
    return body+CRC.pack(zlib.crc32(body)), header


def _decompress(payload, expected):
    decoder = lzma.LZMADecompressor(memlimit=64*1024*1024)
    try:
        raw = decoder.decompress(payload, max_length=expected+1)
    except lzma.LZMAError as error:
        raise ValueError('Compressed section') from error
    if len(raw) != expected or not decoder.eof or decoder.unused_data:
        raise ValueError('Section length')
    return raw


def unpack(data, base_sha):
    if not PREFIX.size+CRC.size < len(data) <= LIMIT:
        raise ValueError('File length or absolute lane cap')
    body = data[:-CRC.size]
    if CRC.unpack(data[-CRC.size:])[0] != zlib.crc32(body):
        raise ValueError('File CRC')
    magic, size = PREFIX.unpack_from(body)
    if magic != MAGIC or not 0 < size <= 4096 or PREFIX.size+size >= len(body):
        raise ValueError('File header')
    h = json.loads(body[PREFIX.size:PREFIX.size+size])
    expected = dict(base_sha256=base_sha, width=512, height=288, frames=150, start=0,
                    fps=15, keys=list(KEYS), old_weights=list(RAMP), scale=2)
    if any(h.get(k) != v for k, v in expected.items()):
        raise ValueError('Base binding or fixed profile')
    if (tuple(h['grid']) not in GRIDS or any(type(v) != int for v in h['grid'])
            or h['mask_step'] not in (2, 4, 8) or type(h['mask_step']) != int
            or h['group_span'] not in (1, 2, 3, 5) or type(h['group_span']) != int
            or h['fusion'] not in ('E', 'F')):
        raise ValueError('Unsupported profile')
    if list(h['sections']) != ['motion', 'support']:
        raise ValueError('Sections')
    gw, gh = h['grid']
    ms, ss = h['sections']['motion'], h['sections']['support']
    shape = (180, gh, gw, 2)
    length = int(np.prod(shape))*2
    sparse = ms.get('sparse')
    if (sparse not in (None, 'median', 'gradient')
            or (sparse is None and (tuple(ms['axes']) not in AXES or ms['raw_bytes'] != length))
            or (sparse is not None and (ms['axes'] != [] or ms['layout'] != 'zigzag'
                                      or type(ms['raw_bytes']) != int or not 0 <= ms['raw_bytes'] <= length))
            or ms['layout'] not in ('i16', 'zigzag') or tuple(ss['order']) not in ORDERS
            or type(ss['xor_last']) != bool):
        raise ValueError('Coding profile')
    compact_shape = (len(groups(h['group_span'])), 288//h['mask_step'], 512//h['mask_step'])
    transposed = tuple(compact_shape[i] for i in ss['order'])
    packed_shape = (*transposed[:-1], (transposed[-1]+7)//8)
    if ss['raw_bytes'] != int(np.prod(packed_shape)):
        raise ValueError('Support length')
    offset = PREFIX.size+size
    decoded = {}
    for name, spec in h['sections'].items():
        count = spec['bytes']
        if type(count) != int or not 0 < count <= LIMIT or offset+count > len(body):
            raise ValueError('Section bounds')
        decoded[name] = _decompress(body[offset:offset+count], spec['raw_bytes'])
        offset += count
    if offset != len(body):
        raise ValueError('Trailing bytes')
    support = np.unpackbits(np.frombuffer(decoded['support'], 'u1').reshape(packed_shape),
                            axis=-1)[..., :transposed[-1]]
    if ss['xor_last']:
        support = np.bitwise_xor.accumulate(support, axis=-1)
    support = support.transpose(np.argsort(ss['order']))
    needed = None
    if sparse:
        needed = required_nodes(support, h['group_span'], h['mask_step'], (gw, gh), h['fusion'])
        if ms['raw_bytes'] != int(needed.sum())*4:
            raise ValueError('Sparse dependency length')
        shape = (int(needed.sum()), 2)
    if ms['layout'] == 'i16':
        value = np.frombuffer(decoded['motion'], dtype='<i2').reshape(shape)
    else:
        b = np.frombuffer(decoded['motion'], dtype='u1').reshape(2, -1).T.copy()
        z = b.reshape(-1).view('<u2').reshape(shape).astype(np.int32)
        value = ((z >> 1) ^ -(z & 1)).astype('<i2')
    for axis in reversed(ms['axes']):
        value = np.cumsum(value, axis=axis, dtype='<i2')
    if sparse:
        value = sparse_decode(value, needed, sparse)
    if np.abs(value.astype(np.int32)).max() > 4096:
        raise ValueError('Motion range')
    return h, dict(motion=value, support=support)


def renderer_inputs(header, arrays):
    cells = expand_support(arrays['support'], header['group_span'])
    h = dict(header, feather_pixels=2)
    if header['fusion'] == 'F':
        h.update(feather_pixels=8, tone_sigma=12, flat_texture_threshold=3)
    a = dict(new_motion=arrays['motion'][:150], old_motion=arrays['motion'][150:],
             new_safe=np.packbits(cells[:150], axis=2), old_safe=np.packbits(cells[150:], axis=2))
    return h, a


def render(raw, header, arrays):
    h, a = renderer_inputs(header, arrays)
    return render_full(raw, h, a)
