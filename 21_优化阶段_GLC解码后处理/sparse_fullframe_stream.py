"""Temporally sampled motion and full-frame conservative masks on the wire."""

import json
import lzma
from struct import Struct
import zlib

import cv2
import numpy as np

from fullframe_structure_stream import compress


HEADER=Struct('<8sI');CRC=Struct('<I');MAGIC=b'GLCFSP01';LIMIT=50000
RAMP=(1.,5/6,4/6,3/6,2/6,1/6)


def sampled_indices(keys):
    result=[]
    for first,stop in zip(keys,(*keys[1:],150),strict=True):
        result.extend(sorted(set((*range(first,stop,3),stop-1))))
    return result


def interpolate(samples,indices,count,keys):
    output=np.empty((count,*samples.shape[1:]),np.float32)
    for first,stop in zip(keys,(*keys[1:],count),strict=True):
        local=[j for j,i in enumerate(indices) if first<=i<stop]
        if not local or indices[local[0]]!=first or indices[local[-1]]!=stop-1:
            raise ValueError('Temporal samples do not cover segment endpoints')
        for i in range(first,stop):
            right=next((j for j in local if indices[j]>=i),local[-1])
            left=max(local[0],right-1)
            if indices[right]==i:
                output[i]=samples[right]
            else:
                fraction=(i-indices[left])/(indices[right]-indices[left])
                output[i]=samples[left].astype(np.float32)*(1-fraction)+samples[right]*fraction
    return output


def pack(new_motion,old_motion,new_masks,old_masks,base_sha,keys,*,soft=False):
    fields=dict(new_motion=new_motion,old_motion=old_motion,
        new_masks=np.packbits(new_masks,axis=2),old_masks=np.packbits(old_masks,axis=2))
    chunks=[];sections={}
    for name,value in fields.items():
        choices=[compress(value,axes) for axes in (((0,),(0,2),(1,2),(0,1,2)) if name.endswith('motion') else ((),))]
        payload,description=min(choices,key=lambda item:len(item[0]))
        chunks.append(payload);sections[name]=description
    old_count=(len(keys)-1)*6
    old_indices=[j for j in range(old_count) if j%6 in (0,3,5)]
    h=dict(base_sha256=base_sha,width=512,height=288,frames=150,start=0,fps=15,keys=keys,
        old_weights=RAMP,new_indices=sampled_indices(keys),old_indices=old_indices,
        grid=[new_motion.shape[2],new_motion.shape[1]],mask_grid=[new_masks.shape[2],new_masks.shape[1]],
        motion_scale=4,feather_pixels=8 if soft else 2,flat_tone_sigma=12 if soft else 0,
        flat_texture_threshold=3,sections=sections)
    encoded=json.dumps(h,separators=(',',':'),allow_nan=False).encode('ascii')
    body=HEADER.pack(MAGIC,len(encoded))+encoded+b''.join(chunks)
    return body+CRC.pack(zlib.crc32(body)),h


def unpack(data,base_sha):
    if not HEADER.size+4<len(data)<=LIMIT:raise ValueError('Packet length/budget')
    body=data[:-4]
    if CRC.unpack(data[-4:])[0]!=zlib.crc32(body):raise ValueError('CRC')
    magic,size=HEADER.unpack_from(body)
    if magic!=MAGIC or not 0<size<8192 or HEADER.size+size>=len(body):raise ValueError('Header')
    h=json.loads(body[HEADER.size:HEADER.size+size])
    expected=dict(base_sha256=base_sha,width=512,height=288,frames=150,start=0,fps=15,
        old_weights=list(RAMP),motion_scale=4,flat_texture_threshold=3)
    if any(h.get(k)!=v for k,v in expected.items()):raise ValueError('Base/profile')
    keys=h['keys']
    if keys not in ([0,30,60,90,120],[0,15,45,75,105,135]):raise ValueError('Keys')
    old_count=(len(keys)-1)*6
    expected_old=[j for j in range(old_count) if j%6 in (0,3,5)]
    if h['new_indices']!=sampled_indices(keys) or h['old_indices']!=expected_old:raise ValueError('Sample schedule')
    gw,gh=h['grid'];mw,mh=h['mask_grid']
    if (gw,gh) not in ((32,18),(24,14),(16,9)) or (mw,mh) not in ((128,72),(64,36)):raise ValueError('Grid')
    if (h['feather_pixels'],h['flat_tone_sigma']) not in ((2,0),(8,12)):raise ValueError('Blend profile')
    shapes=dict(new_motion=(len(h['new_indices']),gh,gw,2),old_motion=(len(expected_old),gh,gw,2),
        new_masks=(150,mh,mw//8),old_masks=(old_count,mh,mw//8))
    if list(h['sections'])!=list(shapes):raise ValueError('Sections')
    arrays={};offset=HEADER.size+size
    for name,spec in h['sections'].items():
        dtype=np.dtype('<i2' if name.endswith('motion') else 'u1');axes=spec['axes']
        valid_axes=((0,),(0,2),(1,2),(0,1,2)) if name.endswith('motion') else ((),)
        length=int(np.prod(shapes[name]))*dtype.itemsize
        if (spec['shape']!=list(shapes[name]) or spec['dtype']!=dtype.str or tuple(axes) not in valid_axes
                or spec['raw_bytes']!=length or not 0<spec['bytes']<=LIMIT or offset+spec['bytes']>len(body)):
            raise ValueError('Section bounds')
        decoder=lzma.LZMADecompressor(memlimit=64*1024*1024)
        payload=decoder.decompress(body[offset:offset+spec['bytes']],max_length=length+1)
        if len(payload)!=length or not decoder.eof or decoder.unused_data:raise ValueError('Section decompression')
        offset+=spec['bytes'];value=np.frombuffer(payload,dtype).reshape(shapes[name])
        for axis in reversed(axes):value=np.cumsum(value,axis=axis,dtype=dtype)
        if name.endswith('motion') and np.abs(value.astype(np.int32)).max()>4096:raise ValueError('Flow magnitude')
        arrays[name]=value
    if offset!=len(body):raise ValueError('Trailing payload')
    return h,arrays


def render(raw,h,arrays):
    if raw.shape!=(150,288,512,3) or raw.dtype!=np.uint8:raise ValueError('RGB geometry')
    keys=h['keys'];events=[k+j for k in keys[1:] for j in range(6)]
    normal=interpolate(arrays['new_motion'],h['new_indices'],150,keys)
    old=interpolate(arrays['old_motion'],h['old_indices'],len(events),list(range(0,len(events),6)))
    new_masks=np.unpackbits(arrays['new_masks'],axis=2);old_masks=np.unpackbits(arrays['old_masks'],axis=2)
    event_map={i:j for j,i in enumerate(events)}
    y,x=np.mgrid[:288,:512].astype(np.float32)
    result=raw.copy();alphas=np.zeros(raw.shape[:3],np.float32)
    for i in range(150):
        key=keys[np.searchsorted(keys,i,side='right')-1]
        base=raw[i].astype(np.float32)
        if h['flat_tone_sigma']:
            low=cv2.GaussianBlur(base,(0,0),h['flat_tone_sigma'])
            luminance=cv2.cvtColor(base,cv2.COLOR_RGB2GRAY)
            energy=cv2.GaussianBlur(np.abs(luminance-cv2.GaussianBlur(luminance,(0,0),1.2)),(0,0),3)
            flat=np.clip((h['flat_texture_threshold']-energy)/h['flat_texture_threshold'],0,1)[...,None]
        def project(qflow,mask,reference):
            flow=cv2.resize(qflow/4.,(512,288),interpolation=cv2.INTER_LINEAR)
            mx,my=x+flow[...,0],y+flow[...,1]
            safe=cv2.resize(mask,(512,288),interpolation=cv2.INTER_NEAREST).astype(bool)
            safe&=(mx>=2)&(mx<510)&(my>=2)&(my<286)
            a=np.minimum(cv2.distanceTransform(safe.astype(np.uint8),cv2.DIST_L2,5)/h['feather_pixels'],1.)
            rgb=cv2.remap(raw[reference].astype(np.float32),mx,my,cv2.INTER_CUBIC)
            if h['flat_tone_sigma']:
                rgb+=flat*(low-cv2.GaussianBlur(rgb,(0,0),h['flat_tone_sigma']))
            return rgb,a
        new,a=project(normal[i],new_masks[i],key)
        frame=np.rint(np.clip(base*(1-a[...,None])+new*a[...,None],0,255))
        if i in event_map:
            j=event_map[i];old_rgb,oa=project(old[j],old_masks[j],keys[j//6])
            weight=np.minimum(a,oa)*RAMP[j%6]
            frame=np.rint(np.clip(frame+weight[...,None]*(old_rgb-new),0,255))
        result[i]=frame.astype(np.uint8);alphas[i]=a
    if not np.array_equal(result[alphas==0],raw[alphas==0]):raise ValueError('Zero weight preservation')
    return result,alphas
