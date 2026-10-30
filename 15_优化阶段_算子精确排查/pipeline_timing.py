"""Cold real-file path, timed separately from operator sampling."""
from __future__ import annotations

import argparse
import functools
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(REPO/'14_优化阶段_部署精简'))
import deploy_codec
import role_codec
from sender_common import load_json,save_json
import c1_io
import torch


class Timings:
    def __init__(self):self.rows=[];self.stack=[]
    def wrap(self,owner,attribute,name):
        original=getattr(owner,attribute)
        @functools.wraps(original)
        def call(*args,**kwargs):
            parent='/'.join(self.stack);self.stack.append(name)
            begin,thread,process=time.perf_counter_ns(),time.thread_time_ns(),time.process_time_ns()
            try:return original(*args,**kwargs)
            finally:
                self.rows.append(dict(name=name,parent=parent,wall_ns=time.perf_counter_ns()-begin,
                                      thread_cpu_ns=time.thread_time_ns()-thread,process_cpu_ns=time.process_time_ns()-process))
                self.stack.pop()
        setattr(owner,attribute,call)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--role',choices=['sender','receiver'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plan=load_json(args.plan);clip=plan['clip']
    timing=Timings()
    for owner,attribute,name in [
        (role_codec,'models_on_meta','init.meta_construction'),(role_codec,'prune','init.role_prune'),
        (role_codec,'validate_state','init.validate_tensors'),(role_codec,'file_sha','init.asset_hash'),
        (role_codec.ref,'check_source','init.source_check'),(role_codec.ref,'configure','init.cuda_configuration'),
        (torch,'load','init.torch_load'),(torch.nn.Module,'load_state_dict','init.assign_weights'),
        (torch.nn.Module,'to','init.to_device'),
        (c1_io.MemoryFrames,'__next__','input.read_resize_rgb'),
        (deploy_codec.SenderWriter,'add','container.add'),(deploy_codec.SenderWriter,'finish','container.finish'),
        (deploy_codec,'read_sender_stream','container.read_verify'),
        (c1_io.RawPreview,'add','video.queue_feed'),(c1_io.RawPreview,'finish','video.finish_publish'),
        (role_codec.ref,'reference_hash','reference_hash'),(role_codec.ref,'rgb_array','rgb_array'),
        (deploy_codec,'file_sha','output.file_hash')]:timing.wrap(owner,attribute,name)
    original=deploy_codec.RoleCodec
    def factory(*arguments,**kwargs):
        codec=original(*arguments,**kwargs)
        timing.wrap(codec,'frame','codec.frame')
        for method in ['indexes','encode','decode']:timing.wrap(codec.entropy,method,'entropy.'+method)
        for method in ['glc_rans_encode','glc_rans_decode']:timing.wrap(codec.entropy.lib,method,'native_c.'+method)
        return codec
    deploy_codec.RoleCodec=factory
    action='encode' if args.role=='sender' else 'decode'
    source=clip['original_source'] if args.role=='sender' else clip['stream']
    sys.argv=[str(Path(deploy_codec.__file__)),action,'--assets',plan['assets'],'--input',source,'--output',str(args.output)]
    if args.role=='sender':sys.argv+=['--frames','66','--width','640','--stride','2','--q','0']
    deploy_codec.main()
    save_json(args.output/'timing_scopes.json',dict(rows=timing.rows,
        note='Nested wall scopes overlap; thread CPU excludes helper threads, process CPU includes them. GPU waits are wall time.'))


if __name__=='__main__':main()
