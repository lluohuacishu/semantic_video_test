"""Experimental frame transport; production GLCV files remain unchanged."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys
import zlib

SCRIPTS = Path(__file__).resolve().parents[1] / '04_外部依赖快照/semantic-video-training/scripts'
sys.path.insert(0, str(SCRIPTS))
from glc_container import Cursor, MAX_HEADER, frame_layout, pack_frame, pack_compact_frame, validate_header

HEADER, FRAME, END, READY = 1, 2, 3, 4
PREFIX = struct.Struct('!4sBII')
COMPACT_PREFIX = struct.Struct('!4sBHI')
MAX_RECORD = 16 * 1024 * 1024
OVERHEAD = PREFIX.size + 4 + 32


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def encode_record(kind, index, body, version=2):
    if version not in (1, 2):
        raise ValueError('Unsupported transport version')
    if kind not in (HEADER, FRAME, END, READY) or not 0 <= index <= 10000:
        raise ValueError('Invalid record kind or index')
    limit = MAX_RECORD if kind == FRAME else MAX_HEADER
    if not 0 < len(body) <= limit:
        raise ValueError('Invalid record size')
    layout = PREFIX if version == 1 else COMPACT_PREFIX
    prefix = layout.pack(b'GSP1' if version == 1 else b'GSP2', kind, index, len(body))
    check = (struct.pack('!I', zlib.crc32(prefix + body)) if version == 2 and kind == FRAME
             else hashlib.sha256(prefix + body).digest())
    return prefix + struct.pack('!I', zlib.crc32(prefix)) + body + check


def receive_record(connection, version=2):
    def exact(count):
        data = bytearray()
        while len(data) < count:
            chunk = connection.recv(count - len(data))
            if not chunk:
                raise EOFError('Stream closed before a complete record')
            data.extend(chunk)
        return bytes(data)

    magic = exact(4)
    if version not in (1, 2) or magic != (b'GSP1' if version == 1 else b'GSP2'):
        raise ValueError('Transport version mismatch')
    layout = PREFIX if version == 1 else COMPACT_PREFIX
    prefix = magic + exact(layout.size - 4)
    if struct.unpack('!I', exact(4))[0] != zlib.crc32(prefix):
        raise ValueError('Record header CRC failed')
    magic, kind, index, size = layout.unpack(prefix)
    limit = MAX_RECORD if kind == FRAME else MAX_HEADER
    if kind not in (HEADER, FRAME, END, READY) or index > 10000 or not 0 < size <= limit:
        raise ValueError('Invalid record header')
    body = exact(size)
    expected = (struct.pack('!I', zlib.crc32(prefix + body)) if version == 2 and kind == FRAME
                else hashlib.sha256(prefix + body).digest())
    if exact(len(expected)) != expected:
        raise ValueError('Record integrity check failed')
    return kind, index, body, layout.size + 4 + size + len(expected)


def receive_expected(connection, kind, index, version=2):
    actual_kind, actual_index, body, count = receive_record(connection, version)
    if (actual_kind, actual_index) != (kind, index):
        raise ValueError('Missing, reordered, duplicated or unexpected record')
    return body, count


def validate_trailer(trailer, frames, transcript):
    if not isinstance(trailer, dict) or set(trailer) != {'frames', 'frame_records_sha256', 'archive_sha256'}:
        raise ValueError('Invalid final record')
    if type(trailer['frames']) is not int or trailer['frames'] != frames or trailer['frame_records_sha256'] != transcript:
        raise ValueError('Final count or transcript hash mismatch')
    digest = trailer['archive_sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Invalid final archive digest')


def decode_header(body):
    header = json.loads(body)
    validate_header(header)
    if header.get('container_version') != 2:
        raise ValueError('Prototype requires compact archival container v2')
    return header


def encode_frame(header, index, packet, version=2):
    return pack_frame(header, index, packet) if version == 1 else pack_compact_frame(header, index, packet)


def decode_frame(header, index, body, version=2):
    expected_kind, expected_z, expected_parts, symbols = frame_layout(header, index)
    if version == 1:
        cursor = Cursor(body)
        inner = Cursor(cursor.take(cursor.integer()))
        cursor.done()
        actual, kind, parts, z_size = struct.unpack('!IBBI', inner.take(10))
        if (actual, kind, parts, z_size) != (index, expected_kind, expected_parts, expected_z):
            raise ValueError('Frame order or layout mismatch')
    else:
        inner = Cursor(body)
        kind, parts, z_size = expected_kind, expected_parts, expected_z
    packet = {'kind': kind, 'z': inner.take(z_size), 'y': []}
    for _ in range(parts):
        data = inner.take(inner.integer())
        if len(data) < 16 or struct.unpack_from('!I', data)[0] != symbols:
            raise ValueError('Invalid entropy stream')
        packet['y'].append(data)
    packet['latent_sha256'] = inner.take(32).hex() if version == 1 else None
    packet['rgb_sha256'] = inner.take(32).hex() if version == 1 else None
    inner.done()
    return packet
