"""Original-weight, fixed-shape TensorRT RGB rendering; reference tensors stay FP32."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

from decoder_common import PACKAGES, file_sha, load_json, save_json
import torch
from torch import nn


def state_sha(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)], separators=(',', ':')).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def identity(model, example, backend, role, trt_version):
    if backend not in ('trt_fp32', 'trt_fp16_safe') or role not in ('image', 'video'):
        raise ValueError('Only I/P RGB generators support these backends')
    if example.dtype != torch.float32 or not example.is_cuda or example.ndim != 4:
        raise ValueError('Generator input must be a four-dimensional CUDA FP32 tensor')
    return dict(schema='glc-rgb-generator-trt-v1', role=role, backend=backend,
                model_state_sha256=state_sha(model), implementation_sha256=file_sha(__file__),
                input_shape=list(example.shape), input_dtype='float32', output_dtype='float32',
                tensorrt=trt_version, torch=str(torch.__version__), torch_cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(example.device),
                compute_capability=list(torch.cuda.get_device_capability(example.device)),
                tf32=False, workspace_bytes=512 * 1024**2, optimization_level=3, onnx_opset=17,
                precision_policy='strongly_typed_fp16_bounded_convolutions_fp32_remainder_v4' if backend == 'trt_fp16_safe' else 'fp32')


def mixed_onnx(source, output):
    """Explicit casts bound each eligible convolution; the remaining graph stays FP32.

    Weak precision constraints allowed a float->half reformat before GroupNorm,
    overflowing the >125,000 residual. Strong typing preserves these boundaries.
    """
    sys.path.insert(0, str(PACKAGES))
    import onnx
    model = onnx.load(str(source))
    graph_outputs = {value.name for value in model.graph.output}
    nodes, policy = [], []
    for index, node in enumerate(model.graph.node):
        if node.op_type != 'Conv':
            nodes.append(node)
            continue
        protected = '/conv/Conv' in node.name or any(name in graph_outputs for name in node.output)
        policy.append(dict(name=node.name, compute='float32' if protected else 'float16',
                           protected_upsample_or_output=protected))
        if protected:
            nodes.append(node)
            continue
        for i, name in enumerate(list(node.input)):
            if not name:
                continue
            converted = f'glc_half_conv_{index}_input_{i}'
            nodes.append(onnx.helper.make_node('Cast', [name], [converted], to=onnx.TensorProto.FLOAT16,
                                             name=converted))
            node.input[i] = converted
        casts = []
        for i, name in enumerate(list(node.output)):
            converted = f'glc_half_conv_{index}_output_{i}'
            node.output[i] = converted
            casts.append(onnx.helper.make_node('Cast', [converted], [name], to=onnx.TensorProto.FLOAT,
                                               name=f'glc_float_conv_{index}_output_{i}'))
        nodes.append(node)
        nodes.extend(casts)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    if not any(row['compute'] == 'float16' for row in policy):
        raise ValueError('Mixed generator contains no accelerated convolutions')
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output))
    return policy


class GeneratorTensorRT(nn.Module):
    """Separate execution stream and private output buffer; never writes the input latent."""

    def __init__(self, model, example, backend, role, cache, check=lambda: None):
        super().__init__()
        started = time.perf_counter()
        sys.path.insert(0, str(PACKAGES))
        import tensorrt as trt

        self.identity = identity(model, example, backend, role, trt.__version__)
        key = hashlib.sha256(json.dumps(self.identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        folder = Path(cache) / key
        folder.mkdir(parents=True, exist_ok=True)
        engine_file, receipt = folder / 'generator.engine', folder / 'build.json'
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.built = not (engine_file.exists() and receipt.exists())
        if not self.built:
            meta = load_json(receipt)
            if meta['identity'] != self.identity or meta['engine_sha256'] != file_sha(engine_file):
                raise ValueError('Generator engine cache identity/checksum mismatch')
        else:
            check()
            onnx_file = folder / 'generator.onnx'
            print(f'Export original {role} generator: {backend}, {list(example.shape)}', flush=True)
            torch.onnx.export(model, example, str(onnx_file), opset_version=17,
                              input_names=['latent'], output_names=['rgb'],
                              do_constant_folding=True, dynamo=False)
            check()
            policy=[]
            parsed_file=onnx_file
            flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            if backend=='trt_fp16_safe':
                parsed_file=folder/'generator.mixed.onnx'
                policy=mixed_onnx(onnx_file,parsed_file)
                flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
            builder = trt.Builder(self.logger)
            network = builder.create_network(flags)
            parser = trt.OnnxParser(network, self.logger)
            if not parser.parse(parsed_file.read_bytes()):
                raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            config = builder.create_builder_config()
            config.clear_flag(trt.BuilderFlag.TF32)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.identity['workspace_bytes'])
            config.builder_optimization_level = self.identity['optimization_level']
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            print(f'Build original {role} generator: {backend}', flush=True)
            blob = builder.build_serialized_network(network, config)
            if blob is None:
                raise RuntimeError('TensorRT could not build the RGB generator')
            check()
            partial = folder / 'generator.engine.part'
            partial.write_bytes(bytes(blob))
            engine_sha = file_sha(partial)
            check()
            os.replace(partial, engine_file)
            save_json(receipt, dict(identity=self.identity, engine_sha256=engine_sha,
                                   onnx_sha256=file_sha(onnx_file), parsed_onnx_sha256=file_sha(parsed_file),
                                   strongly_typed=backend=='trt_fp16_safe', build_seconds=time.perf_counter() - started,
                                   convolution_precision=policy,
                                   note='Local RTX 4060 Windows engine; not an Orin deployment artifact'))
        check()
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_file.read_bytes())
        if self.engine is None:
            raise RuntimeError('TensorRT generator deserialization failed')
        if (tuple(self.engine.get_tensor_shape('latent')) != tuple(example.shape) or
                self.engine.get_tensor_dtype('latent') != trt.float32 or
                self.engine.get_tensor_dtype('rgb') != trt.float32):
            raise ValueError('Generator engine input/output contract mismatch')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('TensorRT generator context creation failed')
        self.output = torch.empty(tuple(self.engine.get_tensor_shape('rgb')),
                                  device=example.device, dtype=torch.float32)
        self.execution_stream = torch.cuda.Stream(device=example.device)
        self.input_shape = tuple(example.shape)
        self.device = example.device
        self.engine_file = str(engine_file)
        self.engine_sha256 = file_sha(engine_file)
        if self.built:
            inspector = self.engine.create_engine_inspector()
            (folder / 'layers.json').write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON), encoding='utf-8')
        torch.cuda.synchronize()
        self.setup_seconds = time.perf_counter() - started

    def forward(self, latent):
        if tuple(latent.shape) != self.input_shape or latent.dtype != torch.float32 or latent.device != self.device:
            raise ValueError('RGB generator received an incompatible reference latent')
        current = torch.cuda.current_stream(self.device)
        self.execution_stream.wait_stream(current)
        with torch.cuda.stream(self.execution_stream):
            value = latent.contiguous()
            self.context.set_tensor_address('latent', value.data_ptr())
            self.context.set_tensor_address('rgb', self.output.data_ptr())
            if not self.context.execute_async_v3(self.execution_stream.cuda_stream):
                raise RuntimeError('TensorRT generator execution failed')
            # Private storage prevents the next invocation from overwriting a returned image.
            result = self.output.clone().float()
            value.record_stream(self.execution_stream)
        current.wait_stream(self.execution_stream)
        result.record_stream(current)
        return result

    def receipt(self):
        return dict(identity=self.identity, engine_file=self.engine_file, engine_sha256=self.engine_sha256,
                    setup_seconds=self.setup_seconds, built=self.built, fp32_output_before_pixel_quantization=True)
