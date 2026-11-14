"""Explicit experimental sender format, separate from strict GLCVLAB1/2.

GLCSND01 carries the unchanged compact z/y records, ordered reference-state
hashes and a whole-file checksum. It never claims a sender RGB checksum.
The existing production reader deliberately rejects this magic.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct

import sender_common  # Sets the existing adapter import path.
from glc_container import (Cursor, MAX_BYTES, MAX_HEADER, frame_layout,
                           pack_compact_frame, validate_header as validate_base_header)

MAGIC = b'GLCSND01'
PROFILE = 'reference-only-no-sender-rgb-v1'
VERSION = 3


def validate_header(header):
    if not isinstance(header, dict):
        raise ValueError('Sender header must be an object')
    if type(header.get('container_version')) is not int or header['container_version'] != VERSION:
        raise ValueError('GLCSND01 requires explicit container_version=3')
    if header.get('integrity_profile') != PROFILE or header.get('sender_rgb_reconstructed') is not False:
        raise ValueError('GLCSND01 must explicitly declare no sender RGB integrity')
    base = dict(header)
    del base['container_version']
    validate_base_header(base)


class ReferenceChecksums:
    def __init__(self):
        self.count = 0
        self.digest = hashlib.sha256(b'GLCSND01/ref_latent+ref_y/ordered-frame-sha256/v1\0')

    def add(self, index, frame):
        if index != self.count:
            raise ValueError('Reference checksum frame order mismatch')
        try:
            value = bytes.fromhex(frame['latent_sha256'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Missing or invalid reference checksum') from exc
        if len(value) != 32:
            raise ValueError('Invalid reference checksum length')
        self.digest.update(struct.pack('>I', index))
        self.digest.update(value)
        self.count += 1

    def values(self):
        return {'latent_sha256': self.digest.hexdigest()}

    def verify(self, expected):
        if self.values() != expected:
            raise ValueError('Ordered reference states differ from the sender')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate sender header key')
        result[key] = value
    return result


class SenderWriter:
    def __init__(self, path, header):
        validate_header(header)
        raw = json.dumps(header, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        if len(raw) > MAX_HEADER:
            raise ValueError('Sender header too large')
        self.path = Path(path)
        self.part = self.path.with_name(self.path.name + '.part')
        if self.path.exists():
            raise FileExistsError(self.path)
        self.header, self.count, self.size = header, 0, 0
        self.digest, self.references = hashlib.sha256(), ReferenceChecksums()
        self.handle = self.part.open('xb')
        self._write(MAGIC + struct.pack('>I', len(raw)) + raw)

    def _write(self, data):
        if self.size + len(data) + 32 > MAX_BYTES:
            raise ValueError('Sender file exceeds size limit')
        self.handle.write(data)
        self.digest.update(data)
        self.size += len(data)

    def add(self, frame):
        if self.count >= self.header['frames']:
            raise ValueError('Too many sender frames')
        if 'rgb_sha256' in frame:
            raise ValueError('GLCSND01 packets must not claim sender RGB reconstruction')
        data = pack_compact_frame(self.header, self.count, frame)
        self.references.add(self.count, frame)
        self._write(data)
        self.count += 1
        return len(data)

    def finish(self):
        if self.count != self.header['frames']:
            raise ValueError('Incomplete sender sequence')
        self._write(bytes.fromhex(self.references.values()['latent_sha256']))
        self.handle.write(self.digest.digest())
        self.handle.close()
        os.link(self.part, self.path)
        self.part.unlink()

    def close(self):
        self.handle.close()


def read_sender_stream(path):
    path = Path(path)
    if not 76 <= path.stat().st_size <= MAX_BYTES:
        raise ValueError('Invalid sender file size')
    data = path.read_bytes()
    if hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError('Sender whole-file checksum failed')
    cursor = Cursor(data[:-32])
    if cursor.take(8) != MAGIC:
        raise ValueError('Expected explicit GLCSND01 sender format')
    size = cursor.integer()
    if not 1 <= size <= MAX_HEADER:
        raise ValueError('Invalid sender header size')
    header = json.loads(cursor.take(size), object_pairs_hook=unique_object)
    validate_header(header)
    frames = []
    for index in range(header['frames']):
        kind, z_size, parts, symbols = frame_layout(header, index)
        frame = dict(kind=kind, z=cursor.take(z_size), y=[], latent_sha256=None, rgb_sha256=None)
        for _ in range(parts):
            stream = cursor.take(cursor.integer())
            if len(stream) < 16 or struct.unpack_from('>I', stream)[0] != symbols:
                raise ValueError('Invalid sender entropy symbol count')
            frame['y'].append(stream)
        frames.append(frame)
    expected = {'latent_sha256': cursor.take(32).hex()}
    cursor.done()
    return header, frames, dict(
        container_content_digest=data[-32:].hex(),
        expected_reference_checksums=expected,
        integrity_profile=PROFILE,
        sender_rgb_checksum_available=False,
    )
