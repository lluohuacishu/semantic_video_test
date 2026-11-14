"""One fixed receiver-only texture-confidence revision of the guide method.

Uses locally computed, hash-bound guide flow caches; they are not transmitted.
The fixed std >= 3 rule is part of this receiver implementation for every clip.
"""

import argparse
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from receiver_structure import file_sha,load_json,save_json,render,reliable
import structure_guide_stream as guide


def texture_gate(images,base_gate=reliable):
    fields=[]
    for rgb in images:
        gray=cv2.cvtColor(rgb.astype(np.float32),cv2.COLOR_RGB2GRAY)
        mean=cv2.GaussianBlur(gray,(0,0),3)
        variance=cv2.GaussianBlur(gray*gray,(0,0),3)-mean*mean
        fields.append(np.sqrt(np.maximum(variance,0)))
    texture=np.stack(fields)
    def gate(values,current,reference,back,forward):
        alpha,mx,my=base_gate(values,current,reference,back,forward)
        reference_texture=cv2.remap(texture[reference],mx,my,cv2.INTER_LINEAR)
        safe=(alpha>0)&(texture[current]>=3)&(reference_texture>=3)
        feather=np.minimum(cv2.distanceTransform(safe.astype(np.uint8),cv2.DIST_L2,5)/2.,1.)
        return np.minimum(alpha,feather),mx,my
    return gate


def receive(base,packet,cache,ffmpeg,out):
    out.mkdir(parents=True,exist_ok=False)
    allowed={p.resolve() for p in (base/'base.glcsnd',base/'base_receive/decoded_rgb.npy',
        base/'base_receive/result.json',packet,cache/'receipt.json',cache/'guide_rgb.npy',
        cache/'to_anchor.npy',cache/'from_anchor.npy',cache/'transition_flow.npy')}
    observed=set();data_root=base.parents[2]
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p in allowed:observed.add(str(p));return
            if p.is_relative_to(out):return
            if p.is_relative_to(data_root):raise RuntimeError('Untransmitted texture receiver input: '+str(p))
    sys.addaudithook(audit)
    blocked=[]
    for p in (base/'source.npy',base/'base_cut.json',base/'flow/to_anchor.npy',base/'flow/from_anchor.npy',
              packet.parent/'send.json',packet.parent/'sender_expected_guide.npy',
              base.parents[1]/'support_refinement_20261111_01'/base.name/'K/support.glccsp2'):
        try:p.read_bytes()
        except RuntimeError:blocked.append(str(p))
        else:raise ValueError('Texture receiver input guard failed')
    original=load_json(cache/'receipt.json');receipt=load_json(base/'base_receive/result.json')
    digest=file_sha(base/'base.glcsnd');size=(base/'base.glcsnd').stat().st_size
    if (not original['source_reads_blocked'] or len(original['reads'])!=5 or len(original['blocked_inputs'])!=7
        or original['auxiliary_sha256']!=file_sha(packet) or original['base_sha256']!=digest
        or original['flow_pairs']!=174 or digest!=receipt['stream_sha256']
        or size!=receipt['stream_bytes'] or not receipt['reference_verified']
        or not receipt['source_media_reads_forbidden']
        or file_sha(base/'base_receive/decoded_rgb.npy')!=receipt['decoded_rgb_sha256']):
        raise ValueError('Base / local flow provenance mismatch')
    for name in ('guide_rgb.npy','to_anchor.npy','from_anchor.npy','transition_flow.npy'):
        if file_sha(cache/name)!=original['outputs_sha256'][name]:raise ValueError('Local guide flow changed')
    raw=np.load(base/'base_receive/decoded_rgb.npy',mmap_mode='r')
    header,_,images=guide.decode(packet.read_bytes(),digest,ffmpeg)
    if not np.array_equal(images,np.load(cache/'guide_rgb.npy',mmap_mode='r')):
        raise ValueError('Actual packet does not match local flow input')
    np.save(out/'guide_rgb.npy',images,allow_pickle=False)
    motion=[np.load(cache/(name+'.npy'),mmap_mode='r') for name in ('to_anchor','from_anchor','transition_flow')]
    started=time.perf_counter()
    values=render(raw,images,*motion,gate_fn=texture_gate(images))
    elapsed=time.perf_counter()-started
    for name,value in values.items():np.save(out/(name+'.npy'),value,allow_pickle=False)
    save_json(out/'receipt.json',dict(complete=True,source_reads_blocked=True,reads=sorted(observed),blocked_inputs=blocked,
        base_sha256=digest,base_bytes=size,auxiliary_sha256=file_sha(packet),auxiliary_bytes=packet.stat().st_size,
        auxiliary_kbps=packet.stat().st_size*8/10000,total_kbps=(size+packet.stat().st_size)*8/10000,
        guide_header=header,receiver_revision='Fixed guide luminance std >=3 at sigma3, at both current and projected reference',
        no_per_clip_threshold_or_mask_transmitted=True,receiver_local_flow_cache=str(cache),
        receiver_local_flow_receipt_sha256=file_sha(cache/'receipt.json'),flow_cache_source_free_provenance_verified=True,
        zero_weight_unchanged=True,coverage=float((values['alpha']>0).mean()),
        guide_decode_seconds=0,flow_seconds=original['flow_seconds'],render_seconds=elapsed,
        runtime_note='Flow time inherited from locally computed identical guide flow; not rerun for this CPU gate revision. Decode timing not measured here.',
        peak_allocated_mib=original['peak_allocated_mib'],outputs_sha256={p.name:file_sha(p) for p in out.glob('*.npy')},
        code_sha256=file_sha(__file__),guide_codec_sha256=file_sha(Path(guide.__file__))))
    print(base.name,packet.parent.name,'texture-only receiver revision complete; coverage',float((values['alpha']>0).mean()),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','packet','cache','ffmpeg','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    receive(a.base.resolve(),a.packet.resolve(),a.cache.resolve(),a.ffmpeg.resolve(),a.output.resolve())
