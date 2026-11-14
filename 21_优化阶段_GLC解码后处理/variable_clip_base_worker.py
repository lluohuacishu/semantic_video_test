"""Length-only adaptation of the frozen FP32 GLC file-chain worker, 1..150 frames."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from runtime_chain_worker import Worker, emit
from runtime_speed_common import file_sha,load_json

class VariableClipWorker(Worker):
    def send(self,r,out):
        source_path=Path(r['source'])
        source=np.load(source_path,mmap_mode='r',allow_pickle=False)
        n=len(source)
        assert 0<n<=150 and source.shape==(n,288,512,3) and source.dtype==np.uint8
        assert file_sha(source_path)==r['source_sha256']
        codec=self.model;codec.dpb=dict(ref_latent=None,ref_y=None)
        header=dict(width=512,height=288,frames=n,fps=15,q=0,intra_period=64,reset_interval=32,
            revision=self.ref.REVISION,entropy=self.ref.ENTROPY_ID,table_sha256=codec.entropy.table_sha256,
            arithmetic=codec.profile,weights={k:v['sha256'] for k,v in self.ref.ASSETS.items()},
            container_version=3,integrity_profile=self.profile,sender_rgb_reconstructed=False)
        path=out/'base.glcsnd';writer=self.Writer(path,header);times=[]
        try:
            for i,rgb in enumerate(source):
                torch.cuda.synchronize();tick=time.perf_counter()
                _,packet=codec.frame(header,i,x=self.input_tensor(rgb),render_rgb=False)
                writer.add(packet);torch.cuda.synchronize();times.append(time.perf_counter()-tick)
            writer.finish()
        finally:
            writer.close()
        codec.dpb=dict(ref_latent=None,ref_y=None)
        return dict(complete=True,stream_sha256=file_sha(path),stream_bytes=path.stat().st_size,
            core_with_input_and_write_seconds=sum(times),frame_seconds=times,header=header,
            input_sha256=r['source_sha256'],warmup_frames=0,frames=n,no_duplicate_padding=True)

    def receive(self,r,out):
        tick=time.perf_counter();header,packets,integrity=self.read_stream(r['base'])
        codec=self.model;n=header['frames']
        assert 0<n<=150 and header['width']==512 and header['height']==288
        assert header['arithmetic']==codec.profile and header['table_sha256']==codec.entropy.table_sha256
        parse_seconds=time.perf_counter()-tick
        codec.dpb=dict(ref_latent=None,ref_y=None);checks=self.Checks();times=[]
        raw=np.empty((n,288,512,3),np.uint8)
        count=0
        for i,packet in enumerate(packets):
            assert i<n
            torch.cuda.synchronize();tick=time.perf_counter()
            rgb,value=codec.frame(header,i,packet=packet)
            checks.add(i,value);raw[i]=rgb
            torch.cuda.synchronize();times.append(time.perf_counter()-tick);count+=1
        assert count==n
        checks.verify(integrity['expected_reference_checksums'])
        tick=time.perf_counter();path=out/'decoded_rgb.npy';np.save(path,raw,allow_pickle=False)
        elapsed=time.perf_counter()-tick;codec.dpb=dict(ref_latent=None,ref_y=None)
        return dict(complete=True,stream_sha256=file_sha(r['base']),stream_bytes=Path(r['base']).stat().st_size,
            decoded_rgb_sha256=file_sha(path),reference_verified=True,source_media_reads_forbidden=True,
            parse_seconds=parse_seconds,core_seconds=sum(times),frame_seconds=times,
            output_save_seconds=elapsed,backend='eager',warmup_frames=0,frames=n,no_duplicate_padding=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',choices=('sender','receiver'),required=True)
    p.add_argument('--request',type=Path,required=True)
    args=p.parse_args()
    # Only metadata is read here. All source pixels remain forbidden to the receiver.
    request=load_json(args.request)
    worker=VariableClipWorker(args.role)
    if args.role=='receiver':
        worker.guard.authorize(Path(request['output']),[request['base']])
        try:Path(request['forbidden_source']).read_bytes()
        except RuntimeError:worker.guard.blocked.append(request['forbidden_source'])
        else:raise AssertionError('Actual source read was not blocked')
    worker.job(request)
    emit('complete',role=args.role)

