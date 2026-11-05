"""Fixed-buffer CUDA graph around only an already-validated RGB engine."""
from __future__ import annotations
import torch
from torch import nn


class GraphRGB(nn.Module):
    def __init__(self, runner, example):
        super().__init__()
        self.runner = runner
        self.shape = tuple(example.shape)
        self.input = example.clone()
        self.output = runner.output
        self.context = runner.context
        self.stream = torch.cuda.Stream()
        expected = runner(example).clone()
        torch.cuda.synchronize()
        self.context.set_tensor_address('latent', self.input.data_ptr())
        self.context.set_tensor_address('rgb', self.output.data_ptr())
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                if not self.context.execute_async_v3(self.stream.cuda_stream):
                    raise RuntimeError('RGB graph warmup failed')
        self.stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError('RGB graph capture failed')
        self.stream.synchronize()
        if not torch.equal(expected, self(example)):
            raise ValueError('RGB graph changes engine output')

    def forward(self, value):
        if value.dtype != torch.float32 or not value.is_cuda or tuple(value.shape) != self.shape:
            raise ValueError('RGB graph input differs')
        current = torch.cuda.current_stream()
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            self.input.copy_(value)
            value.record_stream(self.stream)
            self.graph.replay()
            result = self.output.clone()
        current.wait_stream(self.stream)
        result.record_stream(current)
        return result
