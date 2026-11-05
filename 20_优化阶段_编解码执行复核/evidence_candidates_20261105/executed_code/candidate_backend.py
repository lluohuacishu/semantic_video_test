"""Experimental RGB-only execution graphs; original reference arithmetic is frozen."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WORKSPACE = REPO.parents[2]
DATA = WORKSPACE / '02_实验数据'
sys.path.insert(0, str(REPO / '17_优化阶段_解码器推理加速'))
from decoder_common import PACKAGES, ASSETS, file_sha, load_json, save_json
sys.path.insert(0, str(PACKAGES))
import numpy as np
import torch
from torch import nn
import tensorrt as trt
from generator_backend import mixed_onnx, state_sha

MODES = ('safe_tf32', 'safe_layernorm', 'safe_layernorm_tf32', 'safe_bf16', 'safe_bf16_layernorm')
BASE_KEY = '340bde8808e40b72192150f2efd28d78bfaeda4159c754a818b845b757b7cbdd'
BASE_ENGINE = DATA / f'优化_解码器推理/decoder_20261031_05/engines/{BASE_KEY}/generator.engine'


def bf16_upsample_convolutions(source, target):
    """Only four protected upsample Conv operators; output Conv stays FP32."""
    import onnx
    # Conv-22 adds BF16 to the ONNX type contract. Convert other operators too,
    # rather than only changing the opset number or bypassing the checker.
    model = onnx.version_converter.convert_version(onnx.load(str(source)), 22)
    nodes, names = [], []
    for index, node in enumerate(model.graph.node):
        if node.op_type != 'Conv' or '/conv/Conv' not in node.name:
            nodes.append(node)
            continue
        names.append(node.name)
        for i, name in enumerate(list(node.input)):
            if name:
                value = f'glc_bf16_conv_{index}_input_{i}'
                nodes.append(onnx.helper.make_node('Cast',[name],[value],to=onnx.TensorProto.BFLOAT16,name=value))
                node.input[i] = value
        after = []
        for i,name in enumerate(list(node.output)):
            value = f'glc_bf16_conv_{index}_output_{i}'
            node.output[i] = value
            after.append(onnx.helper.make_node('Cast',[value],[name],to=onnx.TensorProto.FLOAT,name=value+'_float'))
        nodes.append(node)
        nodes.extend(after)
    if len(names) != 4:
        raise ValueError('Expected four protected upsample convolutions')
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model,full_check=True)
    onnx.save(model,str(target))
    return names


def replace_groupnorm_standardization(source, target):
    """GroupNorm's [N,32,L] InstanceNorm -> LayerNorm on L, affine unchanged."""
    import onnx
    from onnx import helper, numpy_helper
    model = onnx.shape_inference.infer_shapes(onnx.load(str(source)))
    g = model.graph
    shapes = {v.name: len(v.type.tensor_type.shape.dim) for v in list(g.value_info) + list(g.input)}
    constants = {v.name: numpy_helper.to_array(v) for v in g.initializer}
    for node in g.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.name == 'value':
                    constants[node.output[0]] = numpy_helper.to_array(attr.t)
        elif node.op_type == 'Identity' and node.input[0] in constants:
            constants[node.output[0]] = constants[node.input[0]]
    one, zero = 'glc_gn_standardize_one', 'glc_gn_standardize_zero'
    g.initializer.extend([numpy_helper.from_array(np.ones(1, dtype=np.float32), one),
                          numpy_helper.from_array(np.zeros(1, dtype=np.float32), zero)])
    changed = []
    for node in g.node:
        if node.op_type != 'InstanceNormalization':
            continue
        scale, bias = constants.get(node.input[1]), constants.get(node.input[2])
        if (shapes.get(node.input[0]) != 3 or scale is None or bias is None or
                scale.shape != (32,) or bias.shape != (32,) or
                not np.all(scale == 1) or not np.all(bias == 0)):
            raise ValueError('Only the verified GroupNorm standardization is eligible: ' + node.name)
        eps = next((float(a.f) for a in node.attribute if a.name == 'epsilon'), 1e-5)
        if abs(eps - 1e-6) > 1e-12:
            raise ValueError('Unexpected GroupNorm epsilon')
        node.op_type = 'LayerNormalization'
        del node.input[1:]
        node.input.extend([one, zero])
        del node.attribute[:]
        node.attribute.extend([helper.make_attribute('axis', -1), helper.make_attribute('epsilon', eps),
                               helper.make_attribute('stash_type', 1)])
        changed.append(node.name)
    if not changed:
        raise ValueError('No eligible GroupNorm operators found')
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(target))
    return dict(replaced_count=len(changed), names=changed, groups=32, epsilon=1e-6,
                affine='original per-channel Mul/Add after reshape retained', statistics='FP32, last axis of rank3 tensor')


class EngineRunner(nn.Module):
    def __init__(self, engine_file, expected_sha=None):
        super().__init__()
        self.engine_file = Path(engine_file)
        self.engine_sha256 = file_sha(self.engine_file)
        if expected_sha is not None and self.engine_sha256 != expected_sha:
            raise ValueError('Engine checksum mismatch')
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_file.read_bytes())
        if self.engine is None:
            raise RuntimeError('Engine deserialization failed')
        self.context = self.engine.create_execution_context()
        self.shape = tuple(self.engine.get_tensor_shape('latent'))
        if self.engine.get_tensor_dtype('rgb') != trt.float32:
            raise ValueError('RGB output must be FP32 before pixel conversion')
        self.output = torch.empty(tuple(self.engine.get_tensor_shape('rgb')), device='cuda', dtype=torch.float32)
        self.stream = torch.cuda.Stream()

    def forward(self, x):
        if x.dtype != torch.float32 or tuple(x.shape) != self.shape or not x.is_cuda:
            raise ValueError('Invalid generator latent')
        current = torch.cuda.current_stream()
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            value = x.contiguous()
            self.context.set_tensor_address('latent', value.data_ptr())
            self.context.set_tensor_address('rgb', self.output.data_ptr())
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError('Generator execution failed')
            result = self.output.clone()
            value.record_stream(self.stream)
        current.wait_stream(self.stream)
        result.record_stream(current)
        return result


def build_candidate(model, example, mode, role, cache):
    if mode not in MODES or role not in ('image', 'video'):
        raise ValueError('Unknown experimental RGB backend')
    if trt.__version__ != '10.5.0':
        raise ValueError('This experiment is pinned to existing TensorRT10.5')
    started = time.perf_counter()
    model_sha = state_sha(model)
    root = Path(cache)
    source_dir = root / 'onnx' / f'{role}_{example.shape[-1]}x{example.shape[-2]}_{model_sha[:12]}'
    source_dir.mkdir(parents=True, exist_ok=True)
    original, source_meta = source_dir / 'original.onnx', source_dir / 'source.json'
    if not original.is_file():
        print(f'Export {role} FP32 generator {list(example.shape)}', flush=True)
        torch.onnx.export(model, example, str(original), opset_version=17, input_names=['latent'],
                          output_names=['rgb'], do_constant_folding=True, dynamo=False)
        save_json(source_meta, dict(model_state_sha256=model_sha, onnx_sha256=file_sha(original)))
    else:
        meta = load_json(source_meta)
        if meta != dict(model_state_sha256=model_sha, onnx_sha256=file_sha(original)):
            raise ValueError('ONNX source provenance differs')
    mixed = source_dir / 'safe_mixed.onnx'
    if not mixed.is_file():
        policy = mixed_onnx(original, mixed)
        save_json(source_dir / 'mixed_policy.json', policy)
    norm_meta = None
    parsed = mixed
    if 'bf16' in mode:
        parsed = source_dir / 'safe_bf16.onnx'
        if not parsed.is_file():
            names = bf16_upsample_convolutions(mixed, parsed)
            save_json(source_dir/'bf16_policy.json', dict(convolutions=names,final_rgb_conv='float32'))
    if 'layernorm' in mode:
        norm_source = parsed
        parsed = source_dir / ('safe_bf16_layernorm.onnx' if 'bf16' in mode else 'safe_layernorm.onnx')
        if not parsed.is_file():
            norm_meta = replace_groupnorm_standardization(norm_source, parsed)
            save_json(parsed.with_suffix('.policy.json'), norm_meta)
        else:
            policy_path = parsed.with_suffix('.policy.json')
            norm_meta = load_json(policy_path if policy_path.is_file() else source_dir/'norm_policy.json')
    identity = dict(schema='glc-rgb-candidate-v1', mode=mode, role=role, shape=list(example.shape),
        model_state_sha256=model_sha, parsed_onnx_sha256=file_sha(parsed),
        implementation_sha256=file_sha(__file__), tf32=mode.endswith('tf32'),
        norm_policy=norm_meta, tensorrt=trt.__version__, torch=str(torch.__version__),
        gpu=torch.cuda.get_device_name(0), workspace_bytes=512*1024**2, optimization_level=3,
        reference_path='unchanged eager FP32; TF32 remains disabled in PyTorch')
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    folder = root / 'engines' / key
    folder.mkdir(parents=True, exist_ok=True)
    engine_file = folder / 'generator.engine'
    receipt_file = folder / 'build.json'
    if engine_file.is_file() and receipt_file.is_file():
        receipt = load_json(receipt_file)
        if receipt['identity'] != identity or file_sha(engine_file) != receipt['engine_sha256']:
            raise ValueError('Candidate engine identity differs')
    else:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) | (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(parsed.read_bytes()):
            raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        if identity['tf32']:
            config.set_flag(trt.BuilderFlag.TF32)
        else:
            config.clear_flag(trt.BuilderFlag.TF32)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512*1024**2)
        config.builder_optimization_level = 3
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        print(f'Build candidate {role} {mode}', flush=True)
        blob = builder.build_serialized_network(network, config)
        if blob is None:
            raise RuntimeError('Candidate engine construction failed')
        part = folder / 'generator.engine.part'
        part.write_bytes(bytes(blob))
        os.replace(part, engine_file)
        receipt = dict(identity=identity, engine_sha256=file_sha(engine_file),
                       build_seconds=time.perf_counter()-started)
        save_json(receipt_file, receipt)
    runner = EngineRunner(engine_file, receipt['engine_sha256'])
    inspector = runner.engine.create_engine_inspector()
    (folder / 'layers.json').write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON), encoding='utf-8')
    runner.setup_receipt = dict(receipt, engine_file=str(engine_file), setup_seconds=time.perf_counter()-started)
    return runner
