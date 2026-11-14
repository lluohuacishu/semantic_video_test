"""GLCGDE02: variable-length guide windows bound to one continuous base stream."""
import json
from struct import Struct
import subprocess
import zlib

import cv2
import numpy as np

MAGIC = b'GLCGDE02'
PREFIX = Struct('<8sI')
CRC = Struct('<I')


def pack(payload, base_sha, start, frames, total):
    header = dict(base_sha256=base_sha, start_frame=start, frames=frames, total_frames=total,
                  fps=15, guide_size=[256,144], target_size=[512,288], codec='hevc-yuv420p',
                  color='RGB decode', upsample='cubic', payload_bytes=len(payload))
    data = json.dumps(header, separators=(',',':'), allow_nan=False).encode('ascii')
    body = PREFIX.pack(MAGIC, len(data)) + data + payload
    packet = body + CRC.pack(zlib.crc32(body))
    unpack(packet, base_sha, start, total)
    return packet, header


def unpack(packet, base_sha, start, total):
    if len(packet) <= PREFIX.size+CRC.size:
        raise ValueError('Truncated guide')
    body = packet[:-4]
    if CRC.unpack(packet[-4:])[0] != zlib.crc32(body):
        raise ValueError('Guide CRC')
    magic, length = PREFIX.unpack_from(body)
    if magic != MAGIC or not 0 < length <= 1024 or PREFIX.size+length >= len(body):
        raise ValueError('Guide header')
    header = json.loads(body[PREFIX.size:PREFIX.size+length])
    required = dict(base_sha256=base_sha, start_frame=start, total_frames=total, fps=15,
                    guide_size=[256,144], target_size=[512,288], codec='hevc-yuv420p',
                    color='RGB decode', upsample='cubic')
    if any(header.get(k) != v for k,v in required.items()) or len(base_sha) != 64:
        raise ValueError('Wrong base, timeline or profile')
    n = header.get('frames')
    if type(n) is not int or n != min(150,total-start) or not 0 < n <= 150 or start < 0 or start%150:
        raise ValueError('Invalid guide window')
    if len(packet)*8/(n/15) > 40000:
        raise ValueError('Auxiliary window exceeds 40kbps')
    payload = body[PREFIX.size+length:]
    if type(header.get('payload_bytes')) is not int or len(payload) != header['payload_bytes']:
        raise ValueError('Guide payload length')
    return header,payload


def decode(packet, base_sha, start, total, ffmpeg):
    header,payload = unpack(packet,base_sha,start,total)
    result = subprocess.run([str(ffmpeg),'-v','error','-threads','1','-f','hevc','-i','pipe:0',
                             '-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1'],
                            input=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    if len(result.stdout) != header['frames']*144*256*3:
        raise ValueError('Guide decoded frame count')
    low = np.frombuffer(result.stdout,np.uint8).reshape(header['frames'],144,256,3)
    full = np.stack([cv2.resize(x,(512,288),interpolation=cv2.INTER_CUBIC) for x in low])
    return header,full
