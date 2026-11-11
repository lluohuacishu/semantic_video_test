"""GLCCSP02 lossless motion predictors and hierarchical support coding.

Support decodes before sparse motion, so there is no motion/mask dependency cycle.
Predictors reset at reference boundaries and use only decoded necessary nodes.
The RGB renderer and motion quantization are unchanged from GLCCSP01.
"""

import json
import lzma
import zlib

import numpy as np

import compact_support_stream as old


MAGIC = b'GLCCSP02'
STARTS = set(old.KEYS) | set(range(150, 180, 6))
PREDICTORS = ('spatial_median', 'spatial_gradient', 'time_median', 'time_gradient',
              'velocity_median', 'time_only')


def temporal_base(index, decoded, needed, mode):
    result = np.zeros(decoded.shape[1:], np.int32)
    if index in STARTS or mode.startswith('spatial'):
        return result
    available = needed[index-1]
    result[available] = decoded[index-1][available]
    if mode.startswith('velocity') and index-1 not in STARTS:
        both = available & needed[index-2]
        result[both] += decoded[index-1][both].astype(np.int32)-decoded[index-2][both]
    return result


def predict_residual(field, mask, y, x, mode):
    if mode == 'time_only':
        return np.zeros(2, np.int32)
    return old.spatial_predict(field, mask, y, x, 'gradient' if mode.endswith('gradient') else 'median')


def encode_sparse(motion, needed, mode):
    values = []
    for i, mask in enumerate(needed):
        base = temporal_base(i, motion, needed, mode)
        residual = motion[i].astype(np.int32)-base
        for y, x in zip(*np.nonzero(mask)):
            value = residual[y,x]-predict_residual(residual,mask,y,x,mode)
            if np.any(value < -32768) or np.any(value > 32767):
                raise ValueError('Predictor residual overflow')
            values.append(value)
    return np.asarray(values,dtype='<i2').reshape(-1,2)


def decode_sparse(residual, needed, mode):
    decoded = np.zeros((*needed.shape,2),dtype='<i2')
    pos = 0
    for i,mask in enumerate(needed):
        base=temporal_base(i,decoded,needed,mode)
        field=np.zeros(decoded.shape[1:],np.int32)
        for y,x in zip(*np.nonzero(mask)):
            field[y,x]=residual[pos].astype(np.int32)+predict_residual(field,mask,y,x,mode)
            value=field[y,x]+base[y,x]
            if np.any(np.abs(value)>4096):raise ValueError('Motion range')
            decoded[i,y,x]=value
            pos+=1
    if pos!=len(residual):raise ValueError('Motion count')
    return decoded


def support_candidates(compact):
    choices=[]
    for order in old.ORDERS:
        for xor in (False,True):
            value=compact.transpose(order).astype('u1')
            if xor:
                original=value.copy()
                value[...,1:]^=original[...,:-1]
            raw=np.packbits(value,axis=-1).tobytes()
            choices.append((lzma.compress(raw),dict(codec='bits',order=list(order),xor_last=xor,raw_bytes=len(raw))))
        code=(compact[:,0::2,0::2].astype('u1')+2*compact[:,0::2,1::2]
              +4*compact[:,1::2,0::2]+8*compact[:,1::2,1::2]).astype('u1')
        flat=code.transpose(order).reshape(-1)
        active=flat!=0
        full=flat[active]==15
        raw=np.packbits(active).tobytes()+np.packbits(full).tobytes()+flat[active][~full].tobytes()
        choices.append((lzma.compress(raw),dict(codec='quad',order=list(order),raw_bytes=len(raw))))
    return choices


def decode_support(raw,spec,shape):
    order=spec['order']
    if spec['codec']=='bits':
        transposed=tuple(shape[i] for i in order)
        packed_shape=(*transposed[:-1],(transposed[-1]+7)//8)
        if len(raw)!=int(np.prod(packed_shape)):raise ValueError('Bit support length')
        value=np.unpackbits(np.frombuffer(raw,'u1').reshape(packed_shape),axis=-1)[...,:transposed[-1]]
        if spec['xor_last']:value=np.bitwise_xor.accumulate(value,axis=-1)
        return value.transpose(np.argsort(order)).astype(bool)
    small=(shape[0],shape[1]//2,shape[2]//2)
    n=int(np.prod(small));size=(n+7)//8
    if len(raw)<size:raise ValueError('Quad header')
    active=np.unpackbits(np.frombuffer(raw[:size],'u1'))[:n].astype(bool)
    count=int(active.sum());full_size=(count+7)//8
    if len(raw)<size+full_size:raise ValueError('Quad full field')
    full=np.unpackbits(np.frombuffer(raw[size:size+full_size],'u1'))[:count].astype(bool)
    detail=np.frombuffer(raw[size+full_size:],'u1')
    if len(detail)!=count-int(full.sum()) or np.any(detail<1) or np.any(detail>14):
        raise ValueError('Quad detail')
    codes=np.zeros(n,'u1');values=np.full(count,15,'u1');values[~full]=detail;codes[active]=values
    codes=codes.reshape(tuple(small[i] for i in order)).transpose(np.argsort(order))
    value=np.zeros(shape,bool)
    for dy,dx,bit in ((0,0,0),(0,1,1),(1,0,2),(1,1,3)):
        value[:,dy::2,dx::2]=((codes>>bit)&1).astype(bool)
    return value


def pack(motion,compact,base_sha,*,step,span=5):
    grid=(motion.shape[2],motion.shape[1])
    needed=old.required_nodes(compact,span,step,grid,'E')
    choices=[]
    motion_probe=[]
    for mode in PREDICTORS:
        value=encode_sparse(motion,needed,mode)
        payload=lzma.compress(old.motion_bytes(value,(),'zigzag'))
        spec=dict(codec=mode,raw_bytes=value.nbytes)
        choices.append((payload,spec))
        motion_probe.append(dict(codec=mode,bytes=len(payload)))
    for axes in old.AXES:
        for layout in ('i16','zigzag'):
            payload=lzma.compress(old.motion_bytes(motion,axes,layout))
            spec=dict(codec='full',axes=list(axes),layout=layout,raw_bytes=motion.nbytes)
            choices.append((payload,spec))
            motion_probe.append(dict(**spec,bytes=len(payload)))
    mp,ms=min(choices,key=lambda v:len(v[0])+len(json.dumps(v[1],separators=(',',':'))))
    support_options=support_candidates(compact)
    sp,ss=min(support_options,key=lambda v:len(v[0])+len(json.dumps(v[1],separators=(',',':'))))
    ms=dict(ms,bytes=len(mp));ss=dict(ss,bytes=len(sp))
    h=dict(base_sha256=base_sha,width=512,height=288,frames=150,start=0,fps=15,
           keys=list(old.KEYS),old_weights=list(old.RAMP),grid=list(grid),scale=2,
           mask_step=step,group_span=span,fusion='E',sections=dict(motion=ms,support=ss))
    encoded=json.dumps(h,separators=(',',':'),allow_nan=False).encode('ascii')
    body=old.PREFIX.pack(MAGIC,len(encoded))+encoded+mp+sp
    data=body+old.CRC.pack(zlib.crc32(body))
    probes=dict(motion=motion_probe,support=[dict(**s,bytes=len(p)) for p,s in support_options],
                necessary_node_fraction=float(needed.mean()))
    return data,h,probes


def unpack(data,base_sha):
    if not old.PREFIX.size+old.CRC.size<len(data)<=old.LIMIT:raise ValueError('File length or 40 kbps cap')
    body=data[:-old.CRC.size]
    if old.CRC.unpack(data[-old.CRC.size:])[0]!=zlib.crc32(body):raise ValueError('CRC')
    magic,size=old.PREFIX.unpack_from(body)
    if magic!=MAGIC or not 0<size<=4096 or old.PREFIX.size+size>=len(body):raise ValueError('Header')
    h=json.loads(body[old.PREFIX.size:old.PREFIX.size+size])
    fixed=dict(base_sha256=base_sha,width=512,height=288,frames=150,start=0,fps=15,
               keys=list(old.KEYS),old_weights=list(old.RAMP),scale=2,fusion='E')
    if any(h.get(k)!=v for k,v in fixed.items()):raise ValueError('Base binding or fixed profile')
    if (tuple(h['grid']) not in old.GRIDS or any(type(v)!=int for v in h['grid'])
            or type(h['mask_step'])!=int or h['mask_step'] not in (2,4)
            or h['group_span']!=5 or type(h['group_span'])!=int
            or list(h['sections'])!=['motion','support']):raise ValueError('Profile')
    ms,ss=h['sections']['motion'],h['sections']['support']
    shape=(len(old.groups(5)),288//h['mask_step'],512//h['mask_step'])
    if tuple(ss['order']) not in old.ORDERS or ss['codec'] not in ('bits','quad'):
        raise ValueError('Support codec')
    if ss['codec']=='bits' and type(ss['xor_last'])!=bool:raise ValueError('Xor mode')
    if type(ss['raw_bytes'])!=int or not 0<ss['raw_bytes']<=int(np.prod(shape)):
        raise ValueError('Support size')
    gw,gh=h['grid'];full_shape=(180,gh,gw,2);maximum=int(np.prod(full_shape))*2
    if (ms['codec'] not in (*PREDICTORS,'full') or type(ms['raw_bytes'])!=int
            or not 0<=ms['raw_bytes']<=maximum or ms['raw_bytes']%4):raise ValueError('Motion size or codec')
    if ms['codec']=='full' and (tuple(ms['axes']) not in old.AXES or ms['layout'] not in ('i16','zigzag')
                              or ms['raw_bytes']!=maximum):raise ValueError('Full motion profile')
    offset=old.PREFIX.size+size;decoded={}
    for name,spec in h['sections'].items():
        count=spec['bytes']
        if type(count)!=int or not 0<count<=old.LIMIT or offset+count>len(body):raise ValueError('Section bounds')
        decoded[name]=old._decompress(body[offset:offset+count],spec['raw_bytes'])
        offset+=count
    if offset!=len(body):raise ValueError('Trailing bytes')
    support=decode_support(decoded['support'],ss,shape)
    needed=old.required_nodes(support,5,h['mask_step'],(gw,gh),'E')
    full=ms['codec']=='full'
    motion_shape=full_shape if full else (int(needed.sum()),2)
    if not full and ms['raw_bytes']!=int(needed.sum())*4:raise ValueError('Dependency length')
    if full and ms['layout']=='i16':
        value=np.frombuffer(decoded['motion'],'<i2').reshape(motion_shape)
    else:
        b=np.frombuffer(decoded['motion'],'u1').reshape(2,-1).T.copy()
        z=b.reshape(-1).view('<u2').reshape(motion_shape).astype(np.int32)
        value=((z>>1)^-(z&1)).astype('<i2')
    if full:
        for axis in reversed(ms['axes']):value=np.cumsum(value,axis=axis,dtype='<i2')
    else:value=decode_sparse(value,needed,ms['codec'])
    if np.max(np.abs(value.astype(np.int32)))>4096:raise ValueError('Motion range')
    return h,dict(motion=value,support=support)


render=old.render
