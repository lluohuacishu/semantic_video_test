"""Follow-up to measured default-stream overhead and FP16 layout interaction."""
from __future__ import annotations
import time
import torch
from torch import nn
from encoder_variants import EncoderVariant,install as original_install


class StreamTensorRT(EncoderVariant):
    def __init__(self,model,example,variant,cache):
        super().__init__(model,example,variant.removesuffix('_stream'),cache)
        self.execution_stream=torch.cuda.Stream()

    def forward(self,x):
        current=torch.cuda.current_stream();self.execution_stream.wait_stream(current)
        with torch.cuda.stream(self.execution_stream):
            x=x.contiguous()
            self.trt_context.set_tensor_address('x',x.data_ptr())
            self.trt_context.set_tensor_address('latent',self.trt_output.data_ptr())
            if not self.trt_context.execute_async_v3(self.execution_stream.cuda_stream):
                raise RuntimeError('TensorRT non-default-stream execution failed')
            output=self.trt_output.clone()
            x.record_stream(self.execution_stream)
        current.wait_stream(self.execution_stream)
        return output


class AMPChannelsLast(nn.Module):
    def __init__(self,model,example,variant,cache):
        super().__init__();start=time.perf_counter();self.model=model.to(memory_format=torch.channels_last)
        torch.cuda.synchronize();self.setup_seconds=time.perf_counter()-start

    def forward(self,x):
        with torch.autocast('cuda',dtype=torch.float16):
            y=self.model(x.contiguous(memory_format=torch.channels_last))
        return y.float().contiguous()


class ContiguousInput(nn.Module):
    def __init__(self,model,example,variant,cache):
        super().__init__();self.model=model;self.setup_seconds=0.

    def forward(self,x):return self.model(x.contiguous()).contiguous()


def build(model,example,variant,cache):
    if variant in ['trt_fp32_stream','trt_fp16_stream']:return StreamTensorRT(model,example,variant,cache)
    if variant=='amp_fp16_cl':return AMPChannelsLast(model,example,variant,cache)
    if variant=='contiguous_nchw':return ContiguousInput(model,example,variant,cache)
    return EncoderVariant(model,example,variant,cache)


def install(codec,example,variant,cache):
    if variant not in ['trt_fp32_stream','trt_fp16_stream','amp_fp16_cl','contiguous_nchw']:
        return original_install(codec,example,variant,cache)
    results={}
    for name in ['image','video']:
        target=getattr(codec,name).vqgan
        wrapper=build(target.encoder,example,variant,cache/name);target.encoder=wrapper
        results[name]=dict(setup_seconds=wrapper.setup_seconds)
    return results
