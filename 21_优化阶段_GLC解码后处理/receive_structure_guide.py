"""Reconstruct stable GLC display using only the base and a transmitted guide."""

import argparse
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from receiver_structure import file_sha,load_json,save_json,render,KEYS,RAMP,EVENTS,WEIGHT_SHA,ITERATIONS
import structure_guide_stream as guide


def receive(base,packet,upstream,ffmpeg,out):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0,str(upstream))
    from src.models.stage_1.core.raft import RAFT
    out.mkdir(parents=True,exist_ok=False)
    weight=upstream/'pretrained_weights/raft-things.pth'
    allowed={p.resolve() for p in (base/'base.glcsnd',base/'base_receive/decoded_rgb.npy',
        base/'base_receive/result.json',packet,weight)}
    data_root=base.parents[2];observed=set()
    protected={'.npy','.npz','.glcsnd','.glccsp2','.glccsp3','.glcguide','.mp4','.hevc'}
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):
            return
        p=Path(os.fsdecode(args[0])).resolve()
        if p in allowed:
            observed.add(str(p));return
        if p.is_relative_to(out):return
        runtime=p.is_relative_to(upstream) and p.suffix.lower() in {'.py','.pyc','.pyd','.dll'}
        if not runtime and (p.is_relative_to(data_root) or p.suffix.lower() in protected):
            raise RuntimeError('Untransmitted guide receiver input: '+str(p))
    sys.addaudithook(audit)
    blocked=[]
    for p in (base/'source.npy',base/'base_cut.json',base/'flow/to_anchor.npy',base/'flow/from_anchor.npy',
              packet.parent/'send.json',packet.parent/'sender_expected_guide.npy',
              base.parents[1]/'support_refinement_20261111_01'/base.name/'K/support.glccsp2'):
        try:p.read_bytes()
        except RuntimeError:blocked.append(str(p))
        else:raise ValueError('Receiver guard failed')
    base_receipt=load_json(base/'base_receive/result.json')
    digest=file_sha(base/'base.glcsnd');size=(base/'base.glcsnd').stat().st_size
    rgb_path=base/'base_receive/decoded_rgb.npy'
    if (digest!=base_receipt['stream_sha256'] or size!=base_receipt['stream_bytes']
        or file_sha(rgb_path)!=base_receipt['decoded_rgb_sha256'] or not base_receipt['reference_verified']
        or not base_receipt['source_media_reads_forbidden'] or base_receipt['header']['fps']!=15):
        raise ValueError('Base independent-receive binding')
    raw=np.load(rgb_path,mmap_mode='r',allow_pickle=False)
    if raw.shape!=(150,288,512,3) or raw.dtype!=np.uint8 or size*8/10>40000:
        raise ValueError('Base geometry / budget')
    started=time.perf_counter()
    header,_,images=guide.decode(packet.read_bytes(),digest,ffmpeg)
    decode_seconds=time.perf_counter()-started
    np.save(out/'guide_rgb.npy',images,allow_pickle=False)
    if file_sha(weight)!=WEIGHT_SHA:raise ValueError('RAFT weight mismatch')
    save_json(out/'inputs.json',dict(base_sha256=digest,base_bytes=size,
        auxiliary_sha256=file_sha(packet),auxiliary_bytes=packet.stat().st_size,
        guide_header=header,decoded_guide_sha256=file_sha(out/'guide_rgb.npy'),
        weights_sha256=WEIGHT_SHA,reference_RGB='Historical raw GLC, never the guide',
        keys=KEYS,old_weights=RAMP,source_RGB_available=False))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=RAFT(argparse.Namespace(small=False,mixed_precision=False,alternate_corr=False))
    state=torch.load(weight,map_location='cpu',weights_only=True)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    model.cuda().eval()
    frames=torch.from_numpy(images).permute(0,3,1,2).to(device='cuda',dtype=torch.float32)
    back,forward=[np.lib.format.open_memmap(out/name,mode='w+',dtype=np.float32,shape=(150,288,512,2))
        for name in ('to_anchor.npy','from_anchor.npy')]
    transition=np.lib.format.open_memmap(out/'transition_flow.npy',mode='w+',dtype=np.float32,
                                         shape=(len(EVENTS),2,288,512,2))
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter();inferred=0
    def pair(i,key):
        q=frames[[i,key]]
        _,prediction=model(q,q.flip(0),iters=ITERATIONS,test_mode=True)
        value=prediction.permute(0,2,3,1).contiguous().cpu().numpy()
        if value.shape!=(2,288,512,2) or not np.isfinite(value).all():raise ValueError('Guide flow')
        return value
    with torch.inference_mode():
        for key,stop in zip(KEYS,(*KEYS[1:],150),strict=True):
            back[key]=forward[key]=0
            for i in range(key+1,stop):
                back[i],forward[i]=pair(i,key);inferred+=1
            print(base.name,packet.parent.name,'guide RAFT anchor',key,'complete',flush=True)
        for j,i in enumerate(EVENTS):
            transition[j]=pair(i,KEYS[j//6]);inferred+=1
    torch.cuda.synchronize();flow_seconds=time.perf_counter()-started
    for value in (back,forward,transition):value.flush()
    peak=torch.cuda.max_memory_allocated()/1024**2;gpu=torch.cuda.get_device_name()
    del model,frames,state;torch.cuda.empty_cache()
    started=time.perf_counter()
    values=render(raw,images,back,forward,transition)
    render_seconds=time.perf_counter()-started
    for name,value in values.items():np.save(out/(name+'.npy'),value,allow_pickle=False)
    hashes={p.name:file_sha(p) for p in out.glob('*.npy')}
    save_json(out/'receipt.json',dict(complete=True,base_sha256=digest,base_bytes=size,
        auxiliary_bytes=packet.stat().st_size,auxiliary_sha256=file_sha(packet),
        auxiliary_kbps=packet.stat().st_size*8/10000,total_kbps=(size+packet.stat().st_size)*8/10000,
        source_reads_blocked=True,audit_scope='Python open audit; FFmpeg receives only actual HEVC payload through stdin',
        reads=sorted(observed),blocked_inputs=blocked,no_sender_receipt=True,
        motion_and_photometry_input='Locally decoded transmitted low-resolution guide, cubic upsampled',
        output_reference_RGB='Historical raw GLC; no guide or source pixel replacement',
        zero_weight_unchanged=True,flow_pairs=inferred,iterations=ITERATIONS,
        coverage=float((values['alpha']>0).mean()),guide_decode_seconds=decode_seconds,
        flow_seconds=flow_seconds,render_seconds=render_seconds,peak_allocated_mib=peak,
        gpu=gpu,torch=str(torch.__version__),torch_CPU_threads=1,
        outputs_sha256=hashes,code_sha256=file_sha(__file__),
        guide_codec_sha256=file_sha(Path(guide.__file__)),
        renderer_sha256=file_sha(Path(__file__).with_name('receiver_structure.py'))))
    print(base.name,packet.parent.name,'GUIDE RECEIVE COMPLETE; aux kbps',packet.stat().st_size*8/10000,
          'coverage',float((values['alpha']>0).mean()),'flow seconds',flow_seconds,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','packet','upstream','ffmpeg','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    receive(a.base.resolve(),a.packet.resolve(),a.upstream.resolve(),a.ffmpeg.resolve(),a.output.resolve())
