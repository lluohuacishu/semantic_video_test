"""Additional cold-file scopes for media/FFmpeg startup and input tensor setup."""
from contextlib import contextmanager
import functools
import sys
import time

import pipeline_timing
from pipeline_timing import c1_io,deploy_codec,torch,save_json


class DetailedTimings(pipeline_timing.Timings):
    def __init__(self):
        super().__init__()
        self.wrap(c1_io.MemoryFrames,'__init__','input.open_media')
        self.wrap(c1_io.RawPreview,'__init__','video.open_ffmpeg')
        self.wrap(deploy_codec,'encode','pipeline.encode')
        self.wrap(deploy_codec,'decode','pipeline.decode')
        for owner,name in [(torch,'from_numpy'),(torch.Tensor,'to'),(torch.Tensor,'permute'),
                           (torch.Tensor,'__getitem__'),(torch.Tensor,'float'),(torch.Tensor,'div'),
                           (torch.Tensor,'mul'),(torch.Tensor,'sub'),(deploy_codec.ref.F,'pad')]:
            original=getattr(owner,name)
            def call(*args,_original=original,_name=name,**kwargs):
                if 'pipeline.encode' not in self.stack or 'codec.frame' in self.stack:
                    return _original(*args,**kwargs)
                begin,thread,process=time.perf_counter_ns(),time.thread_time_ns(),time.process_time_ns()
                try:return _original(*args,**kwargs)
                finally:self.rows.append(dict(name='input.tensor.'+_name,parent='/'.join(self.stack),
                    wall_ns=time.perf_counter_ns()-begin,thread_cpu_ns=time.thread_time_ns()-thread,
                    process_cpu_ns=time.process_time_ns()-process))
            setattr(owner,name,call)


if __name__=='__main__':
    pipeline_timing.Timings=DetailedTimings
    pipeline_timing.main()
