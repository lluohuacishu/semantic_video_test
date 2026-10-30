"""Fresh diagnostic sender/independent receiver; optionally replace only input encoders."""
from __future__ import annotations
import argparse
import hashlib
import time
from experiment_common import *
from encoder_variants import install
from role_codec import RoleCodec,ref
from sender_container import PROFILE,SenderWriter,ReferenceChecksums,read_sender_stream
from trial_worker import payload_fingerprint
import torch


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--clip',required=True)
    p.add_argument('--role',choices=['sender','receiver'],required=True)
    p.add_argument('--variant',default='eager');p.add_argument('--fill',choices=['on','off'],required=True)
    p.add_argument('--stream',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--save-rgb',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    plan=load_json(args.plan)
    if args.role=='sender':frozen(plan)
    else:
        # Fail closed: a receiver never opens cached/source media or metric targets.
        forbidden={str(Path(c['source']).resolve()).casefold() for c in plan['clips']}
        def audit(event,values):
            if event=='open' and values and isinstance(values[0],(str,bytes)):
                name=str(values[0]);n=name.replace('\\','/').lower()
                if str(Path(name).resolve()).casefold() in forbidden or n.endswith(('.npy','.png','.jpg','.jpeg','.mkv','.mp4','.avi')):
                    # Creating this receiver's RGB result is allowed; reading media is not.
                    if Path(name).resolve()==(args.output/'decoded_rgb.npy').resolve():
                        return
                    raise RuntimeError('Independent receiver attempted source/media read')
        sys.addaudithook(audit)
    clip=next(c for c in plan['clips'] if c['id']==args.clip)
    begin=time.perf_counter();codec=RoleCodec(plan['assets'],args.role)
    torch.utils.deterministic.fill_uninitialized_memory=args.fill=='on'
    if not torch.are_deterministic_algorithms_enabled():raise ValueError('Determinism disabled')
    if args.role=='sender':
        source=np.load(clip['source'],mmap_mode='r')
        header={k:clip[k] for k in ['width','height','frames','fps','q']}
        header.update(intra_period=64,reset_interval=32,revision=ref.REVISION,entropy=ref.ENTROPY_ID,
            table_sha256=codec.entropy.table_sha256,arithmetic=codec.profile,
            weights={k:v['sha256'] for k,v in ref.ASSETS.items()},container_version=3,
            integrity_profile=PROFILE,sender_rgb_reconstructed=False)
        example=input_tensor(source[0]);geometry=f'{example.shape[-1]}x{example.shape[-2]}'
        setup=install(codec,example,args.variant,args.plan.parent/'engines'/geometry)
    else:
        if args.variant!='eager':raise ValueError('Receiver implementation must remain unchanged')
        header,packets,integrity=read_sender_stream(args.stream)
        if header['arithmetic']!=codec.profile or header['table_sha256']!=codec.entropy.table_sha256:
            raise ValueError('Receiver arithmetic/entropy mismatch')
        setup={}
    torch.cuda.synchronize();init=time.perf_counter()-begin
    def frame(i):
        if args.role=='sender':return codec.frame(header,i,x=input_tensor(source[i]),render_rgb=False)
        return codec.frame(header,i,packet=packets[i])
    for i in range(min(plan['warmup_frames'],header['frames'])):frame(i)
    torch.cuda.synchronize();codec.dpb=dict(ref_latent=None,ref_y=None)
    torch.cuda.reset_peak_memory_stats();rows=[];checks=ReferenceChecksums()
    path=args.output/'stream.glcsnd' if args.role=='sender' else args.stream
    writer=SenderWriter(path,header) if args.role=='sender' else None
    rgb_file=(args.output/'decoded_rgb.npy').open('wb') if args.role=='receiver' and args.save_rgb else None
    if rgb_file:
        np.lib.format.write_array_header_1_0(rgb_file,dict(descr='|u1',fortran_order=False,shape=(header['frames'],header['height'],header['width'],3)))
    entropy_ops=[]
    op=codec.entropy.encode if args.role=='sender' else codec.entropy.decode
    def entropy(values,indexes):
        v=op(values,indexes);symbols=values if args.role=='sender' else v
        entropy_ops.append(dict(symbols_sha256=array_sha(np.asarray(symbols,dtype='<i4')),
                                cdf_sha256=array_sha(np.asarray(indexes,dtype='<i4'))))
        return v
    if args.role=='sender':codec.entropy.encode=entropy
    else:codec.entropy.decode=entropy
    task=time.perf_counter()
    try:
        for i in range(header['frames']):
            x=input_tensor(source[i]) if args.role=='sender' else None
            torch.cuda.synchronize();entropy_ops.clear();tick=time.perf_counter_ns()
            rgb,packet=codec.frame(header,i,x=x,render_rgb=False) if args.role=='sender' else codec.frame(header,i,packet=packets[i])
            torch.cuda.synchronize();ns=time.perf_counter_ns()-tick
            if writer:writer.add(packet)
            checks.add(i,packet)
            if rgb_file:rgb_file.write(rgb.tobytes())
            rows.append(dict(index=i,kind=packet['kind'],seconds=ns/1e9,latent_sha256=packet['latent_sha256'],
                payload_sha256=payload_fingerprint(packet),payload_bytes=len(packet['z'])+sum(map(len,packet['y'])),
                rgb_sha256=packet.get('rgb_sha256'),entropy=list(entropy_ops)))
            if (i+1)%64==0:print(f'{args.clip} {args.role} {args.variant} {i+1}/{header["frames"]}',flush=True)
        if writer:writer.finish()
        else:checks.verify(integrity['expected_reference_checksums'])
    finally:
        if writer:writer.close()
        if rgb_file:rgb_file.close()
    result=dict(clip=clip['id'],role=args.role,variant=args.variant,fill=args.fill,header=header,
        setup=setup,init_seconds=init,core_seconds=sum(r['seconds'] for r in rows),task_seconds=time.perf_counter()-task,
        records=rows,stream_sha256=file_sha(path),stream_bytes=path.stat().st_size,
        payload_bytes=sum(r['payload_bytes'] for r in rows),reference_checksums=checks.values(),
        reference_verified=args.role=='receiver',source_media_reads_forbidden=args.role=='receiver',
        deterministic_algorithms=True,tf32_outside_encoder=False,
        encoder_tf32=args.variant=='tf32',encoder_autocast=args.variant=='amp_fp16',
        arithmetic_header_scope='Unchanged reference/entropy/decoder FP32 profile; experimental input encoder implementation is recorded in this result, not asserted by the legacy stream arithmetic field',
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        plan_sha256=file_sha(args.plan),weights_changed=False,training_steps=0)
    if rgb_file:result['decoded_rgb_sha256']=file_sha(args.output/'decoded_rgb.npy')
    save_json(args.output/'result.json',result)
    print(f'{args.role} {args.variant} fill={args.fill}: {result["core_seconds"]:.3f}s',flush=True)


if __name__=='__main__':main()
