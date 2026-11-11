"""Full-frame auxiliary packet and source-independent display reconstruction."""

import json
import lzma
from struct import Struct
import zlib

import cv2
import numpy as np


KEYS=(0,30,60,90,120)
RAMP=(1.,5/6,4/6,3/6,2/6,1/6)
EVENTS=tuple(k+j for k in KEYS[1:] for j in range(6))
HEADER=Struct('<8sI')
CRC=Struct('<I')
MAGIC=b'GLCFUL01'
LIMIT=50000


def compress(value,axes):
    residual=value
    for axis in axes:
        residual=np.diff(residual,axis=axis,prepend=np.zeros_like(np.take(residual,[0],axis=axis)))
    payload=lzma.compress(residual.tobytes())
    return payload,dict(shape=list(value.shape),dtype=value.dtype.str,axes=list(axes),bytes=len(payload),raw_bytes=value.nbytes)


def pack(new_motion,old_motion,new_safe,old_safe,base_sha,*,tone_sigma=0,keys=KEYS,scale=4,feather=2,flat_only=False):
    fields=dict(new_motion=new_motion,old_motion=old_motion,
        new_safe=np.packbits(new_safe,axis=2),old_safe=np.packbits(old_safe,axis=2))
    chunks=[];sections={}
    for name,value in fields.items():
        choices=[compress(value,axes) for axes in (((0,),(0,2),(1,2),(0,1,2)) if name.endswith('motion') else ((),))]
        payload,spec=min(choices,key=lambda item:len(item[0]))
        chunks.append(payload);sections[name]=spec
    header=dict(base_sha256=base_sha,width=512,height=288,frames=150,start=0,fps=15,
        keys=keys,old_weights=RAMP,grid=[new_motion.shape[2],new_motion.shape[1]],
        scale=scale,mask_grid=[64,36],feather_pixels=feather,full_frame=True,sections=sections)
    if tone_sigma:
        if tone_sigma!=12:raise ValueError('Unsupported tone profile')
        header['tone_sigma']=tone_sigma
    if flat_only:header['flat_texture_threshold']=3
    encoded=json.dumps(header,separators=(',',':'),allow_nan=False).encode('ascii')
    body=HEADER.pack(MAGIC,len(encoded))+encoded+b''.join(chunks)
    return body+CRC.pack(zlib.crc32(body)),header


def unpack(data,base_sha):
    if not HEADER.size+CRC.size<len(data)<=LIMIT:raise ValueError('Packet length/budget')
    body=data[:-4]
    if CRC.unpack(data[-4:])[0]!=zlib.crc32(body):raise ValueError('Packet CRC')
    magic,size=HEADER.unpack_from(body)
    if magic!=MAGIC or not 0<size<=8192 or HEADER.size+size>=len(body):raise ValueError('Packet header')
    h=json.loads(body[HEADER.size:HEADER.size+size])
    expected=dict(base_sha256=base_sha,width=512,height=288,frames=150,start=0,fps=15,
        old_weights=list(RAMP),mask_grid=[64,36],full_frame=True)
    if any(h.get(k)!=v for k,v in expected.items()):raise ValueError('Wrong base/profile')
    if h['keys'] not in ([0,30,60,90,120],[0,15,45,75,105,135]) or h['scale'] not in (2,4) or h['feather_pixels'] not in (2,8):
        raise ValueError('Schedule/scale/feather')
    if h.get('tone_sigma',0) not in (0,12):raise ValueError('Unsupported tone profile')
    if h.get('flat_texture_threshold',3)!=3:raise ValueError('Flat tone profile')
    gw,gh=h['grid']
    if type(gw)!=int or type(gh)!=int or not 4<=gw<=64 or not 4<=gh<=36:raise ValueError('Invalid grid')
    event_count=(len(h['keys'])-1)*6
    shapes=dict(new_motion=(150,gh,gw,2),old_motion=(event_count,gh,gw,2),new_safe=(150,36,8),old_safe=(event_count,36,8))
    if list(h['sections'])!=list(shapes):raise ValueError('Wrong sections')
    offset=HEADER.size+size;arrays={}
    for name,spec in h['sections'].items():
        dtype=np.dtype('<i2' if name.endswith('motion') else 'u1')
        axes=spec['axes']
        choices=((0,),(0,2),(1,2),(0,1,2)) if name.endswith('motion') else ((),)
        length=int(np.prod(shapes[name]))*dtype.itemsize
        if (spec['shape']!=list(shapes[name]) or spec['dtype']!=dtype.str or tuple(axes) not in choices
                or spec['raw_bytes']!=length or not 0<spec['bytes']<=LIMIT or offset+spec['bytes']>len(body)):
            raise ValueError('Invalid section bounds')
        decoder=lzma.LZMADecompressor(memlimit=64*1024*1024)
        payload=decoder.decompress(body[offset:offset+spec['bytes']],max_length=length+1)
        if len(payload)!=length or not decoder.eof or decoder.unused_data:raise ValueError('Section length')
        offset+=spec['bytes']
        value=np.frombuffer(payload,dtype).reshape(shapes[name])
        for axis in reversed(axes):value=np.cumsum(value,axis=axis,dtype=dtype)
        if name.endswith('motion') and np.abs(value.astype(np.int32)).max()>4096:raise ValueError('Invalid motion')
        arrays[name]=value
    if offset!=len(body):raise ValueError('Trailing payload')
    return h,arrays


def expanded_motion(grid,scale=4):
    return cv2.resize(grid.astype(np.float32)/scale,(512,288),interpolation=cv2.INTER_LINEAR)


def render(raw,header,arrays):
    if raw.shape!=(150,288,512,3) or raw.dtype!=np.uint8:raise ValueError('Base geometry')
    y,x=np.mgrid[:288,:512].astype(np.float32)
    masks={k:np.unpackbits(v,axis=2) for k,v in arrays.items() if k.endswith('safe')}
    result=raw.copy();alpha=np.zeros(raw.shape[:3],np.float32)
    keys=header['keys'];event_list=[k+j for k in keys[1:] for j in range(6)]
    events={i:j for j,i in enumerate(event_list)}
    def project(flow,bits,key,current_low,flat):
        motion=expanded_motion(flow,header['scale']);mx,my=x+motion[...,0],y+motion[...,1]
        mask=cv2.resize(bits,(512,288),interpolation=cv2.INTER_NEAREST).astype(bool)
        mask&=(mx>=2)&(mx<510)&(my>=2)&(my<286)
        a=np.minimum(cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)/header['feather_pixels'],1.)
        rgb=cv2.remap(raw[key].astype(np.float32),mx,my,cv2.INTER_CUBIC)
        if current_low is not None:
            rgb=rgb+flat*(current_low-cv2.GaussianBlur(rgb,(0,0),header['tone_sigma']))
        return rgb,a
    for i in range(150):
        key=keys[np.searchsorted(keys,i,side='right')-1]
        sigma=header.get('tone_sigma',0)
        current_low=cv2.GaussianBlur(raw[i].astype(np.float32),(0,0),sigma) if sigma else None
        flat=1.
        if 'flat_texture_threshold' in header:
            luminance=cv2.cvtColor(raw[i].astype(np.float32),cv2.COLOR_RGB2GRAY)
            energy=cv2.GaussianBlur(np.abs(luminance-cv2.GaussianBlur(luminance,(0,0),1.2)),(0,0),3)
            flat=np.clip((3-energy)/3,0,1)[...,None]
        new,a=project(arrays['new_motion'][i],masks['new_safe'][i],key,current_low,flat)
        frame=np.rint(np.clip(raw[i]*(1-a[...,None])+new*a[...,None],0,255))
        if i in events:
            j=events[i]
            old,oa=project(arrays['old_motion'][j],masks['old_safe'][j],keys[j//6],current_low,flat)
            weight=np.minimum(a,oa)*RAMP[j%6]
            frame=np.rint(np.clip(frame+weight[...,None]*(old-new),0,255))
        result[i]=frame.astype(np.uint8);alpha[i]=a
    if not np.array_equal(result[alpha==0],raw[alpha==0]):raise ValueError('Zero-weight mismatch')
    return result,alpha
