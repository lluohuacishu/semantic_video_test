"""Bounded wire format for quantized background motion and confidence grids."""

from struct import Struct
import zlib

import cv2
import numpy as np

from cache_motion_inputs import FPS, HEIGHT, LANE_LIMIT, WIDTH
from run_cache_motion import CACHE_FRAMES


MOTION = 'motion.glcflow'
GATE = 'reliability.glcgate'
FORMATS = {
    MOTION: (b'GLCFLW01', np.dtype('<i2'), (2,)),
    GATE: (b'GLCGAT01', np.dtype('u1'), ()),
}
HEADER = Struct('<8s32s10H')
CRC = Struct('<I')
PREDICTORS = ((), (0,), (2,), (0, 2))


def byte_limit(clip):
    return LANE_LIMIT*clip.count//(8*FPS)


def quantize(frames, editable, grid, scale, name):
    _, dtype, _ = FORMATS[name]
    nodes = cv2.resize(editable.astype(np.float32), grid, interpolation=cv2.INTER_AREA) > 0
    nodes = cv2.dilate(nodes.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    values = np.stack([cv2.resize(frame, grid, interpolation=cv2.INTER_AREA) for frame in frames])
    values[:, ~nodes] = 0
    values = np.rint(values*scale)
    limits = np.iinfo(dtype)
    if (not np.isfinite(values).all() or values.min() < limits.min or values.max() > limits.max):
        raise ValueError('Grid exceeds its fixed-point representation')
    return values.astype(dtype)


def pack_grid(values, name, clip, base_sha, scale):
    magic, dtype, channels = FORMATS[name]
    _, height, width = values.shape[:3]
    if (values.shape != (clip.count, height, width, *channels) or values.dtype != dtype
            or values[::CACHE_FRAMES].any()):
        raise ValueError('Unexpected quantized grid shape or anchor values')
    payloads = []
    for axes in PREDICTORS:
        residual = values
        for axis in axes:
            first = np.zeros_like(np.take(residual, [0], axis=axis))
            residual = np.diff(residual, axis=axis, prepend=first)
        payloads.append(zlib.compress(residual.tobytes(), level=9))
    predictor = min(range(len(payloads)), key=lambda i: len(payloads[i]))
    header = HEADER.pack(magic, bytes.fromhex(base_sha), WIDTH, HEIGHT, FPS, clip.start,
        clip.count, CACHE_FRAMES, width, height, scale, predictor)
    body = header + payloads[predictor]
    return body + CRC.pack(zlib.crc32(body))


def unpack_grid(data, name, clip, base_sha):
    magic, dtype, channels = FORMATS[name]
    if len(data) <= HEADER.size + CRC.size or len(data) > byte_limit(clip):
        raise ValueError('Invalid auxiliary file length')
    body, checksum = data[:-CRC.size], data[-CRC.size:]
    if CRC.unpack(checksum)[0] != zlib.crc32(body):
        raise ValueError('Auxiliary file checksum mismatch')
    kind, digest, w, h, fps, start, count, period, gw, gh, scale, pred = HEADER.unpack_from(body)
    if ((kind, digest.hex(), w, h, fps, start, count, period)
            != (magic, base_sha, WIDTH, HEIGHT, FPS, clip.start, clip.count, CACHE_FRAMES)
            or not 1 <= gw <= 96 or not 1 <= gh <= 54 or not 1 <= scale <= 8
            or pred >= len(PREDICTORS)):
        raise ValueError('Auxiliary file belongs to a different base or has an invalid profile')
    shape = (count, gh, gw, *channels)
    size = int(np.prod(shape))*dtype.itemsize
    decoder = zlib.decompressobj()
    payload = decoder.decompress(body[HEADER.size:], size+1)
    if len(payload) != size or not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
        raise ValueError('Invalid compressed auxiliary grid')
    values = np.frombuffer(payload, dtype=dtype).reshape(shape)
    for axis in reversed(PREDICTORS[pred]):
        values = np.cumsum(values, axis=axis, dtype=dtype)
    if values[::CACHE_FRAMES].any() or (name == GATE and values.max() > scale):
        raise ValueError('Invalid confidence or anchor grid values')
    return values.astype(np.float32)/scale


def expand(grid):
    return np.stack([cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR) for frame in grid])
