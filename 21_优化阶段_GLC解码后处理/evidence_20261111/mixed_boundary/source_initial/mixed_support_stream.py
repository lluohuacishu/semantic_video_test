"""GLCCSP03 experimental sparse 2x2 boundary additions over a 4x4 support.

Motion, grouping and RGB rendering are inherited unchanged from GLCCSP02.
The receiver expands support before decoding necessary motion nodes. Validation
of the common profile and motion payload is delegated to the existing decoder.
This whole-clip format makes no real-time or packet-loss claim.
"""
import json
import lzma
import zlib

import numpy as np

import compact_support_stream as base
import predictive_support_stream as previous

MAGIC = b'GLCCSP03'


def codes_from_support(support):
    return (support[:, 0::2, 0::2].astype('u1')
            + 2 * support[:, 0::2, 1::2]
            + 4 * support[:, 1::2, 0::2]
            + 8 * support[:, 1::2, 1::2]).astype('u1')


def support_from_codes(codes):
    result = np.zeros((codes.shape[0], codes.shape[1] * 2, codes.shape[2] * 2), bool)
    for dy, dx, bit in ((0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)):
        result[:, dy::2, dx::2] = ((codes >> bit) & 1).astype(bool)
    return result


def varints(values):
    result = bytearray()
    for value in values:
        value = int(value)
        while value >= 128:
            result.append((value & 127) | 128)
            value >>= 7
        result.append(value)
    return bytes(result)


def read_varints(raw, count):
    result = []
    pos = 0
    for _ in range(count):
        value = shift = 0
        while True:
            if pos >= len(raw) or shift > 28:
                raise ValueError('Patch variable integer')
            byte = raw[pos]
            pos += 1
            value |= (byte & 127) << shift
            if byte < 128:
                break
            shift += 7
        result.append(value)
    return np.asarray(result, np.int64), pos


def mixed_candidates(support):
    codes = codes_from_support(support)
    for order in base.ORDERS:
        ordered = codes.transpose(order)
        flat = ordered.reshape(-1)
        indices = np.flatnonzero((flat > 0) & (flat < 15))
        gaps = np.diff(indices, prepend=-1).astype('<u4')
        detail = flat[indices].tobytes()
        for xor in (False, True):
            coarse = (ordered == 15).astype('u1')
            if xor:
                original = coarse.copy()
                coarse[..., 1:] ^= original[..., :-1]
            prefix = np.packbits(coarse, axis=-1).tobytes()
            for layout in ('varint', 'u32_planes'):
                delta = (varints(gaps) if layout == 'varint' else
                         gaps.view('u1').reshape(-1, 4).T.tobytes())
                raw = prefix + delta + detail
                spec = dict(codec='mixed', order=list(order), xor_last=xor,
                            patch_layout=layout, patch_count=len(indices), raw_bytes=len(raw))
                yield lzma.compress(raw), spec


def decode_mixed(raw, spec, shape):
    if (set(spec) != {'codec', 'order', 'xor_last', 'patch_layout', 'patch_count', 'raw_bytes', 'bytes'}
            or tuple(spec['order']) not in base.ORDERS or type(spec['xor_last']) != bool
            or spec['patch_layout'] not in ('varint', 'u32_planes')):
        raise ValueError('Mixed support profile')
    small = (shape[0], shape[1] // 2, shape[2] // 2)
    ordered = tuple(small[i] for i in spec['order'])
    n = int(np.prod(small))
    count = spec['patch_count']
    if type(count) != int or not 0 <= count <= n:
        raise ValueError('Patch count')
    packed_shape = (*ordered[:-1], (ordered[-1] + 7) // 8)
    prefix_size = int(np.prod(packed_shape))
    if len(raw) < prefix_size:
        raise ValueError('Coarse support length')
    coarse = np.unpackbits(np.frombuffer(raw[:prefix_size], 'u1').reshape(packed_shape), axis=-1)[..., :ordered[-1]]
    if spec['xor_last']:
        coarse = np.bitwise_xor.accumulate(coarse, axis=-1)
    tail = raw[prefix_size:]
    if spec['patch_layout'] == 'varint':
        gaps, consumed = read_varints(tail, count)
    else:
        consumed = count * 4
        if len(tail) < consumed:
            raise ValueError('Patch delta length')
        gaps = np.frombuffer(tail[:consumed], 'u1').reshape(4, count).T.copy().reshape(-1).view('<u4').astype(np.int64)
    detail = np.frombuffer(tail[consumed:], 'u1')
    if len(detail) != count or np.any(detail < 1) or np.any(detail > 14) or np.any(gaps < 1):
        raise ValueError('Patch detail or ordering')
    indices = np.cumsum(gaps) - 1
    if count and (indices[-1] >= n or np.any(coarse.reshape(-1)[indices])):
        raise ValueError('Patch bounds or coarse overlap')
    codes = coarse.reshape(-1).astype('u1') * 15
    codes[indices] = detail
    codes = codes.reshape(ordered).transpose(np.argsort(spec['order']))
    return support_from_codes(codes)


def assemble(header, motion_payload, support_payload, magic=MAGIC):
    encoded = json.dumps(header, separators=(',', ':'), allow_nan=False).encode('ascii')
    body = base.PREFIX.pack(magic, len(encoded)) + encoded + motion_payload + support_payload
    return body + base.CRC.pack(zlib.crc32(body))


def pack(motion, support, base_sha):
    original, header, probe = previous.pack(motion, support, base_sha, step=2)
    _, size = base.PREFIX.unpack_from(original)
    offset = base.PREFIX.size + size
    msize = header['sections']['motion']['bytes']
    motion_payload = original[offset:offset + msize]
    candidates = [(original[offset + msize:-base.CRC.size], header['sections']['support'])]
    for payload, spec in mixed_candidates(support):
        candidates.append((payload, dict(spec, bytes=len(payload))))
    measured = []
    for payload, spec in candidates:
        h = dict(header, sections=dict(motion=header['sections']['motion'], support=spec))
        data = assemble(h, motion_payload, payload)
        measured.append((len(data), data, h))
    _, data, header = min(measured, key=lambda item: item[0])
    probe['mixed_support'] = [dict(bytes=n, support=h['sections']['support']) for n, _, h in measured]
    return data, header, probe


def unpack(data, base_sha):
    if not base.PREFIX.size + base.CRC.size < len(data) <= base.LIMIT:
        raise ValueError('File length or 40 kbps cap')
    body = data[:-base.CRC.size]
    if base.CRC.unpack(data[-base.CRC.size:])[0] != zlib.crc32(body):
        raise ValueError('CRC')
    magic, size = base.PREFIX.unpack_from(body)
    if magic != MAGIC or not 0 < size <= 4096 or base.PREFIX.size + size >= len(body):
        raise ValueError('Header')
    h = json.loads(body[base.PREFIX.size:base.PREFIX.size + size])
    if list(h.get('sections', {})) != ['motion', 'support'] or h.get('mask_step') != 2:
        raise ValueError('Section order or support resolution')
    ms, ss = h['sections']['motion'], h['sections']['support']
    offset = base.PREFIX.size + size
    if any(type(s.get('bytes')) != int or not 0 < s['bytes'] <= base.LIMIT for s in (ms, ss)):
        raise ValueError('Section bounds')
    if offset + ms['bytes'] + ss['bytes'] != len(body):
        raise ValueError('Section length')
    motion_payload = body[offset:offset + ms['bytes']]
    support_payload = body[offset + ms['bytes']:]
    if ss.get('codec') != 'mixed':
        _, arrays = previous.unpack(assemble(h, motion_payload, support_payload, previous.MAGIC), base_sha)
        return h, arrays
    shape = (len(base.groups(5)), 144, 256)
    if type(ss.get('raw_bytes')) != int or not 0 < ss['raw_bytes'] <= int(np.prod(shape)) * 2:
        raise ValueError('Mixed raw length')
    support = decode_mixed(base._decompress(support_payload, ss['raw_bytes']), ss, shape)
    # Normalize only the decoded support representation; common profile and
    # motion validation remain the GLCCSP02 decoder's source of truth.
    normalized = np.packbits(support, axis=-1).tobytes()
    payload = lzma.compress(normalized)
    spec = dict(codec='bits', order=[0, 1, 2], xor_last=False, raw_bytes=len(normalized), bytes=len(payload))
    converted = dict(h, sections=dict(motion=ms, support=spec))
    _, arrays = previous.unpack(assemble(converted, motion_payload, payload, previous.MAGIC), base_sha)
    return h, arrays


render = base.render
