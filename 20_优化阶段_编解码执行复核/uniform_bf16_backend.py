"""BF16 generator body, FP32 final RGB convolution and FP32 external I/O."""
from __future__ import annotations
import copy
import hashlib
import json
import time
from candidate_backend import *


class FP32Output(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv.float()

    def forward(self, x):
        return self.conv(x.float())


class FP32Norm(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.norm = copy.deepcopy(original).float()

    def forward(self, x):
        return self.norm(x.float()).bfloat16()


class FP32Resize(nn.Module):
    def __init__(self, upsample):
        super().__init__()
        self.conv = upsample.conv

    def forward(self, x):
        x = torch.nn.functional.interpolate(x.float(),scale_factor=2.0,mode='nearest').bfloat16()
        return self.conv(x)


class BodyBF16(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.renderer = copy.deepcopy(original).bfloat16()
        # The pinned TRT parser rejects BF16 InstanceNormalization. Keep both
        # normalization arithmetic and its original affine parameters FP32.
        for name,module in list(self.renderer.named_modules()):
            if isinstance(module,nn.GroupNorm):
                prefix,_,leaf=name.rpartition('.')
                parent=self.renderer.get_submodule(prefix) if prefix else self.renderer
                setattr(parent,leaf,FP32Norm(original.get_submodule(name)))
            elif module.__class__.__name__=='Upsample':
                prefix,_,leaf=name.rpartition('.')
                parent=self.renderer.get_submodule(prefix) if prefix else self.renderer
                setattr(parent,leaf,FP32Resize(module))
        if not isinstance(original.blocks[-1], nn.Conv2d) or original.blocks[-1].out_channels != 3:
            raise ValueError('Expected the original final RGB convolution')
        self.renderer.blocks[-1] = FP32Output(copy.deepcopy(original.blocks[-1]))

    def forward(self, x):
        return self.renderer(x.bfloat16()).float()


@torch.inference_mode()
def build_uniform(model, example, mode, role, cache):
    if mode != 'uniform_bf16':
        raise ValueError('Unknown uniform-body precision')
    import onnx
    began = time.perf_counter()
    identity = dict(schema='glc-rgb-uniform-bf16-v1', mode=mode, role=role, input_shape=list(example.shape),
        original_state_sha256=state_sha(model), implementation_sha256=file_sha(__file__),
        gpu=torch.cuda.get_device_name(0), tensorrt=trt.__version__, torch=str(torch.__version__),
        tf32=False, input_output='float32', body='bfloat16', normalization='original FP32 affine and statistics; BF16 output',
        resize='FP32 nearest with BF16 output',final_rgb_conv='original float32', onnx_opset=22)
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    folder = Path(cache)/'uniform_bf16'/key
    folder.mkdir(parents=True,exist_ok=True)
    engine_file,receipt_file = folder/'generator.engine',folder/'build.json'
    if engine_file.is_file() and receipt_file.is_file():
        receipt=load_json(receipt_file)
        if receipt['identity']!=identity or file_sha(engine_file)!=receipt['engine_sha256']:
            raise ValueError('BF16 body cache differs')
    else:
        shadow=BodyBF16(model).eval()
        test=shadow(example)
        if not torch.isfinite(test).all():
            raise ValueError('PyTorch BF16-body shadow output is non-finite')
        original=folder/'export_opset17.onnx'
        print(f'Export BF16 body {role}',flush=True)
        torch.onnx.export(shadow,example,str(original),opset_version=17,input_names=['latent'],output_names=['rgb'],
                          do_constant_folding=True,dynamo=False)
        graph=onnx.version_converter.convert_version(onnx.load(str(original)),22)
        onnx.checker.check_model(graph,full_check=True)
        parsed=folder/'generator_opset22.onnx'
        onnx.save(graph,str(parsed))
        del shadow,test
        torch.cuda.empty_cache()
        logger=trt.Logger(trt.Logger.WARNING)
        builder=trt.Builder(logger)
        flags=(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))|(1<<int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        network=builder.create_network(flags)
        parser=trt.OnnxParser(network,logger)
        if not parser.parse(parsed.read_bytes()):
            raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config=builder.create_builder_config()
        config.clear_flag(trt.BuilderFlag.TF32)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,512*1024**2)
        config.builder_optimization_level=3
        config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
        print(f'Build BF16 body {role}',flush=True)
        blob=builder.build_serialized_network(network,config)
        if blob is None:
            raise RuntimeError('BF16 body engine build failed')
        part=folder/'generator.engine.part'
        part.write_bytes(bytes(blob));part.replace(engine_file)
        receipt=dict(identity=identity,engine_sha256=file_sha(engine_file),
                     parsed_onnx_sha256=file_sha(parsed),build_seconds=time.perf_counter()-began)
        save_json(receipt_file,receipt)
    runner=EngineRunner(engine_file,receipt['engine_sha256'])
    inspector=runner.engine.create_engine_inspector()
    (folder/'layers.json').write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON),encoding='utf-8')
    runner.setup_receipt=dict(receipt,engine_file=str(engine_file),setup_seconds=time.perf_counter()-began)
    return runner
