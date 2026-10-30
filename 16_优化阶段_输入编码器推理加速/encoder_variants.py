"""Diagnostic encoder-only alternatives. Restore the unchanged codec arithmetic."""
from __future__ import annotations
from contextlib import nullcontext
import os
import sys
import time
from experiment_common import PACKAGES, file_sha, save_json
import torch
from torch import nn


class EncoderVariant(nn.Module):
    def __init__(self, model, example, variant, cache=None):
        super().__init__();self.model=model;self.variant=variant
        self.setup_seconds=0.;self.graph=None
        begin=time.perf_counter()
        if variant=='channels_last':self.model.to(memory_format=torch.channels_last)
        elif variant=='trace':
            self.model=torch.jit.freeze(torch.jit.trace(model,example,check_trace=False).eval())
        elif variant=='graph':
            self.static_input=example.clone();side=torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(8):self.static_output=model(self.static_input)
            torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize()
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.static_output=model(self.static_input)
        elif variant.startswith('trt_'):
            if cache is None:raise ValueError('TensorRT requires a local engine cache')
            self._trt(model,example,cache)
        elif variant not in ['eager','fill_on','amp_fp16','tf32']:
            raise ValueError('Unknown encoder variant')
        torch.cuda.synchronize();self.setup_seconds=time.perf_counter()-begin

    def _trt(self,model,example,cache):
        sys.path.insert(0,str(PACKAGES))
        import tensorrt as trt
        cache.mkdir(parents=True,exist_ok=True)
        onnx=cache/'encoder.onnx';engine=cache/(self.variant+'.engine')
        logger=trt.Logger(trt.Logger.WARNING)
        if not engine.exists():
            torch.onnx.export(model,example,str(onnx),opset_version=17,input_names=['x'],output_names=['latent'],
                              do_constant_folding=True,dynamo=False)
            builder=trt.Builder(logger);network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser=trt.OnnxParser(network,logger)
            if not parser.parse(onnx.read_bytes()):
                raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            config=builder.create_builder_config();config.clear_flag(trt.BuilderFlag.TF32)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,512*1024**2)
            config.builder_optimization_level=3
            if self.variant=='trt_fp16':config.set_flag(trt.BuilderFlag.FP16)
            blob=builder.build_serialized_network(network,config)
            if blob is None:raise RuntimeError('TensorRT failed to build encoder')
            engine.write_bytes(bytes(blob))
            save_json(cache/(self.variant+'_build.json'),dict(tensorrt=trt.__version__,engine_sha256=file_sha(engine),
                onnx_sha256=file_sha(onnx),input_shape=list(example.shape),tf32=False,fp16_enabled=self.variant=='trt_fp16',
                workspace_limit_bytes=512*1024**2,device=torch.cuda.get_device_name(0),
                note='fixed shape, local RTX4060 Windows engine; not an Orin artifact'))
        self.trt_runtime=trt.Runtime(logger);self.trt_engine=self.trt_runtime.deserialize_cuda_engine(engine.read_bytes())
        if self.trt_engine is None:raise RuntimeError('TensorRT deserialize failed')
        self.trt_context=self.trt_engine.create_execution_context()
        self.trt_output=torch.empty(tuple(self.trt_engine.get_tensor_shape('latent')),device=example.device,dtype=torch.float32)

    def forward(self,x):
        if self.variant=='channels_last':return self.model(x.contiguous(memory_format=torch.channels_last)).contiguous()
        if self.variant=='graph':
            self.static_input.copy_(x);self.graph.replay();return self.static_output.clone()
        if self.variant=='amp_fp16':
            with torch.autocast('cuda',dtype=torch.float16):y=self.model(x)
            return y.float().contiguous()
        if self.variant=='tf32':
            conv,mm=torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=True
                return self.model(x).contiguous()
            finally:
                torch.backends.cudnn.allow_tf32=conv;torch.backends.cuda.matmul.allow_tf32=mm
        if self.variant.startswith('trt_'):
            x=x.contiguous()
            self.trt_context.set_tensor_address('x',x.data_ptr())
            self.trt_context.set_tensor_address('latent',self.trt_output.data_ptr())
            if not self.trt_context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
                raise RuntimeError('TensorRT execution failed')
            return self.trt_output.clone()
        return self.model(x)


def install(codec,example,variant,cache):
    values={}
    if variant in ['eager','fill_on']:return values
    for name in ['image','video']:
        target=getattr(codec,name).vqgan
        wrapper=EncoderVariant(target.encoder,example,variant,cache/name)
        target.encoder=wrapper;values[name]=dict(setup_seconds=wrapper.setup_seconds)
    return values
