"""Connect CPU phases, module paths and ATen operators to CUPTI correlations."""
from __future__ import annotations

from contextlib import contextmanager
import functools
import itertools
import importlib
import json
import math
import threading
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from cupti_trace import union_ns


def shapes(value):
    if isinstance(value,float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device), stride=list(value.stride()))
    if isinstance(value, (tuple, list)):
        return [shapes(item) for item in value[:12]]
    if isinstance(value, dict):
        return {str(key):shapes(item) for key,item in value.items()}
    return value if isinstance(value, (str, int, float, bool, type(None))) else str(value)


class DispatchObserver(TorchDispatchMode):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        observer = self.owner
        if not observer.active:
            return func(*args, **(kwargs or {}))
        row = dict(id=observer.next_id(), name=str(func), module=observer.modules[-1] if observer.modules else '',
                   phase='/'.join(observer.scopes), inputs=shapes(args), kwargs=shapes(kwargs or {}),
                   thread_id=threading.get_native_id(),operator_type='aten')
        observer.cupti.push(row['id'])
        start, thread = time.perf_counter_ns(), time.thread_time_ns()
        try:
            return func(*args, **(kwargs or {}))
        finally:
            row.update(start_perf_ns=start, end_perf_ns=time.perf_counter_ns(), thread_cpu_ns=time.thread_time_ns()-thread)
            observer.cupti.pop(row['id'])
            observer.operators.append(row)


class Observer:
    def __init__(self, cupti, native_nvtx=False):
        self.cupti = cupti
        self.native_nvtx=native_nvtx
        self.counter = itertools.count(1)
        self.active = False
        self.modules, self.scopes = [], []
        self.operators, self.phases, self.module_events = [], [], []
        self.handles = []

    def next_id(self):
        return next(self.counter)

    @contextmanager
    def scope(self, name):
        if not self.active:
            yield
            return
        row = dict(id=self.next_id(), name=name, parent='/'.join(self.scopes),
                   thread_id=threading.get_native_id())
        self.scopes.append(name)
        self.cupti.push(row['id'])
        if self.native_nvtx:torch.cuda.nvtx.range_push('glc.phase:'+name)
        start, thread = time.perf_counter_ns(), time.thread_time_ns()
        try:
            yield
        finally:
            row.update(start_perf_ns=start, end_perf_ns=time.perf_counter_ns(), thread_cpu_ns=time.thread_time_ns()-thread)
            if self.native_nvtx:torch.cuda.nvtx.range_pop()
            self.cupti.pop(row['id'])
            if self.scopes.pop() != name:
                raise RuntimeError('CPU scope stack mismatch')
            self.phases.append(row)

    def wrap(self, owner, attribute, name):
        original = getattr(owner, attribute)
        @functools.wraps(original)
        def call(*args, **kwargs):
            with self.scope(name):
                return original(*args, **kwargs)
        setattr(owner, attribute, call)

    def install(self, codec, ref):
        # TorchScript fusion can launch kernels without an ATen dispatch.
        # Tag the actual scripted function explicitly rather than dropping it.
        script_module=importlib.import_module('src.models.vqgan_arch')
        original_swish=script_module.swish
        def scripted_swish(*args,**kwargs):
            if not self.active:return original_swish(*args,**kwargs)
            if self.native_nvtx:
                torch.cuda.nvtx.range_push('glc.script:swish|sizes='+json.dumps([list(x.shape) for x in args if isinstance(x,torch.Tensor)]))
                try:return original_swish(*args,**kwargs)
                finally:torch.cuda.nvtx.range_pop()
            row=dict(id=self.next_id(),name='torchscript.swish',operator_type='torchscript',
                module=self.modules[-1] if self.modules else '',phase='/'.join(self.scopes),
                inputs=shapes(args),kwargs=shapes(kwargs),thread_id=threading.get_native_id())
            self.cupti.push(row['id']);start,thread=time.perf_counter_ns(),time.thread_time_ns()
            try:return original_swish(*args,**kwargs)
            finally:
                row.update(start_perf_ns=start,end_perf_ns=time.perf_counter_ns(),thread_cpu_ns=time.thread_time_ns()-thread)
                self.cupti.pop(row['id']);self.operators.append(row)
        script_module.swish=scripted_swish
        for prefix, model in [('I', codec.image), ('P', codec.video)]:
            for path, module in model.named_modules():
                name = prefix + ('.' + path if path else '')
                pending = []
                def enter(_module, _args, _name=name, _pending=pending):
                    if not self.active:
                        return
                    row = dict(name=_name, start_perf_ns=time.perf_counter_ns(), thread_start_ns=time.thread_time_ns(),
                               thread_id=threading.get_native_id(), parent=self.modules[-1] if self.modules else '')
                    _pending.append(row)
                    self.modules.append(_name)
                    if self.native_nvtx:torch.cuda.nvtx.range_push('glc.module:'+_name)
                def leave(_module, _args, _output, _name=name, _pending=pending):
                    if not self.active:
                        return
                    row = _pending.pop()
                    if self.native_nvtx:torch.cuda.nvtx.range_pop()
                    row.update(end_perf_ns=time.perf_counter_ns(), thread_cpu_ns=time.thread_time_ns()-row.pop('thread_start_ns'))
                    if self.modules.pop() != _name:
                        raise RuntimeError('Module stack mismatch')
                    self.module_events.append(row)
                self.handles.extend([module.register_forward_pre_hook(enter), module.register_forward_hook(leave)])
        for name in ['codebook', 'prior']:
            self.wrap(codec, name, name)
        for name in ['indexes', 'encode', 'decode']:
            self.wrap(codec.entropy, name, 'entropy.'+name)
        for name in ['glc_rans_encode', 'glc_rans_decode']:
            self.wrap(codec.entropy.lib, name, 'native_c.'+name)
        for name in ['rgb_array', 'reference_hash', 'pack_indices', 'unpack_indices']:
            self.wrap(ref, name, name)
        for model in [codec.image, codec.video]:
            for name in ['context_generation', 'res_prior_param_decoder']:
                if hasattr(model, name):
                    self.wrap(model, name, 'P.'+name)

    def reset(self):
        if self.modules or self.scopes:
            raise RuntimeError('Profiling scope leaked across frames')
        self.operators, self.phases, self.module_events = [], [], []

    def summarize(self, records, begin, end, correlation_override=None):
        external = {row['correlationId']:row['externalId'] for row in records if row['kind']==39}
        if correlation_override:external.update(correlation_override)
        known = {row['id']:row for row in self.operators}
        phase_ids = {row['id']:row for row in self.phases}
        groups, kernels, copies, runtime, unattributed = {}, [], [], [], []
        for record in records:
            kind = record['kind']
            if kind not in (1,2,4,5,10,38):
                continue
            row = dict(record)
            row['external_id'] = external.get(row['correlationId'])
            operator = known.get(row['external_id'])
            if operator:
                row.update(operator=operator['name'], module=operator['module'], phase=operator['phase'])
            else:
                phase = phase_ids.get(row['external_id'])
                row.update(operator='', module='', phase=(phase['parent']+'/'+phase['name']).strip('/') if phase else '')
            if 'start' in row and 'end' in row:
                row['duration_ns'] = row['end']-row['start']
                if row['start'] and row['end'] and row['duration_ns'] < 0:
                    raise RuntimeError('Invalid CUPTI timestamp interval')
                if kind in (1,2,10) and (not row['start'] or not row['end']):
                    raise RuntimeError('Missing GPU activity timestamp')
            if kind == 10:
                kernels.append(row)
                if not operator:
                    unattributed.append(row)
            elif kind in (1,2):
                copies.append(row)
            elif kind == 5:
                runtime.append(row)
        for operator in self.operators:
            key = (operator['name'], operator['module'], operator['phase'], json.dumps(operator['inputs'], sort_keys=True))
            item = groups.setdefault(key, dict(name=key[0],module=key[1],phase=key[2],inputs=operator['inputs'],calls=0,
                cpu_wall_ns=0,thread_cpu_ns=0,kernel_count=0,kernel_sum_ns=0,memcpy_bytes=0,memcpy_sum_ns=0,runtime_sync_wall_ns=0))
            item['calls'] += 1
            item['cpu_wall_ns'] += operator['end_perf_ns']-operator['start_perf_ns']
            item['thread_cpu_ns'] += operator['thread_cpu_ns']
            operator['_group'] = key
        for row in kernels+copies+runtime:
            operator = known.get(row['external_id'])
            if not operator:
                continue
            item = groups[operator['_group']]
            if row['kind']==10:
                item['kernel_count']+=1
                item['kernel_sum_ns']+=row['duration_ns']
            elif row['kind']==1:
                item['memcpy_bytes']+=row['bytes']
                item['memcpy_sum_ns']+=row['duration_ns']
            elif row['kind']==5 and 'Synchronize' in row['name']:
                item['runtime_sync_wall_ns']+=row['duration_ns']
        for operator in self.operators:
            operator.pop('_group')
        active = [row for row in kernels+copies if row.get('start') and row.get('end') and row['start']>=begin and row['end']<=end]
        return dict(
            window_start_ns=begin,window_end_ns=end,window_ns=end-begin,
            kernel_count=len(kernels),kernel_sum_ns=sum(row['duration_ns'] for row in kernels),
            gpu_activity_union_ns=union_ns((row['start'],row['end']) for row in active),
            unattributed_kernel_count=len(unattributed),
            runtime_sync_wall_ns=sum(row['duration_ns'] for row in runtime if 'Synchronize' in row['name']),
            runtime_api_wall_ns=sum(row['duration_ns'] for row in runtime),
            memcpy_bytes=sum(row['bytes'] for row in copies if row['kind']==1),
            memcpy_sum_ns=sum(row['duration_ns'] for row in copies if row['kind']==1),
            operators=sorted(groups.values(),key=lambda row:row['kernel_sum_ns'],reverse=True),
            cpu_operators=self.operators,phases=self.phases,module_events=self.module_events,
            kernels=kernels,copies=copies,runtime=runtime,driver=[row for row in records if row['kind']==4],
            capture_dropped_records=self.cupti.dropped_records,capture_errors=list(self.cupti.errors))
