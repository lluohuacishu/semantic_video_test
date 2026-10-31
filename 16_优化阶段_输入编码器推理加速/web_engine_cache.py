"""Stage first-time engines privately; cancellation cannot publish an incomplete cache."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import time
from experiment_common import load_json,save_json


def build_cached(model,example,variant,cache,staging,check,factory=None):
    if factory is None:
        from extra_variants import build
        factory=build
    cache,staging=Path(cache),Path(staging)
    name=variant.removesuffix('_stream')+'.engine'
    if not variant.startswith('trt_') or (cache/name).exists():return factory(model,example,variant,cache)
    began=time.perf_counter();wrapped=factory(model,example,variant,staging);check()
    cache.mkdir(parents=True,exist_ok=True);target=(cache/name).resolve()
    if not target.is_relative_to(cache.resolve()):raise ValueError('Invalid engine cache target')
    if not target.exists():
        partial=cache/(name+'.part');shutil.copyfile(staging/name,partial);check();os.replace(partial,target)
        info=staging/(variant.removesuffix('_stream')+'_build.json')
        if info.exists():
            data=load_json(info);data['onnx_source_path']=str((staging/'encoder.onnx').resolve())
            save_json(cache/info.name,data)
    wrapped.setup_seconds=time.perf_counter()-began
    return wrapped
