"""GLCGDE01: a base-bound, low-resolution structural RGB guide, fully billed."""

import json
from struct import Struct
import subprocess
import zlib

import cv2
import numpy as np

MAGIC = b'GLCGDE01'
PREFIX = Struct('<8sI')
CRC = Struct('<I')
LIMIT = 50000
SIZES = ((128,72),(256,144))


def pack(payload, base_sha, size):
    if len(base_sha)!=64 or tuple(size) not in SIZES:
        raise ValueError('Guide profile')
    h=dict(base_sha256=base_sha,frames=150,fps=15,guide_size=list(size),
           target_size=[512,288],codec='hevc-yuv420p',color='RGB decode',
           upsample='cubic',payload_bytes=len(payload))
    encoded=json.dumps(h,separators=(',',':'),allow_nan=False).encode('ascii')
    body=PREFIX.pack(MAGIC,len(encoded))+encoded+payload
    return body+CRC.pack(zlib.crc32(body)),h


def unpack(data, base_sha):
    if not PREFIX.size+CRC.size<len(data)<=LIMIT:
        raise ValueError('Guide length / 40 kbps auxiliary cap')
    body=data[:-CRC.size]
    if CRC.unpack(data[-CRC.size:])[0]!=zlib.crc32(body):
        raise ValueError('Guide CRC')
    magic,size=PREFIX.unpack_from(body)
    if magic!=MAGIC or not 0<size<=1024 or PREFIX.size+size>=len(body):
        raise ValueError('Guide header')
    h=json.loads(body[PREFIX.size:PREFIX.size+size])
    fixed=dict(base_sha256=base_sha,frames=150,fps=15,target_size=[512,288],
               codec='hevc-yuv420p',color='RGB decode',upsample='cubic')
    if any(h.get(k)!=v for k,v in fixed.items()) or tuple(h.get('guide_size',())) not in SIZES:
        raise ValueError('Wrong guide profile / base binding')
    payload=body[PREFIX.size+size:]
    if type(h.get('payload_bytes'))!=int or len(payload)!=h['payload_bytes']:
        raise ValueError('Guide payload length')
    return h,payload


def decode(data,base_sha,ffmpeg):
    h,payload=unpack(data,base_sha)
    w,height=h['guide_size']
    result=subprocess.run([str(ffmpeg),'-v','error','-threads','1','-f','hevc','-i','pipe:0',
        '-map','0:v:0','-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1'],
        input=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    if len(result.stdout)!=150*height*w*3:
        raise ValueError('Decoded guide geometry / frame count')
    low=np.frombuffer(result.stdout,np.uint8).reshape(150,height,w,3)
    full=np.stack([cv2.resize(v,(512,288),interpolation=cv2.INTER_CUBIC) for v in low])
    return h,low,full
