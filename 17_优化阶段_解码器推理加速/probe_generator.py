"""Numerical diagnostic only: real latent/module ranges and parsed TRT layer types."""
from __future__ import annotations
import argparse
import sys
from decoder_common import *
import torch
from role_codec import RoleCodec
from sender_container import read_sender_stream


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=TRIAL/'numerical_probe')
    out=p.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    codec = RoleCodec(ASSETS, 'receiver')
    torch.utils.deterministic.fill_uninitialized_memory = False
    header, packets, _ = read_sender_stream(ARCHIVE / 'compact/long108_q0_640/eager/stream.glcsnd')
    examples, rows, handles = {}, [], []
    def hook(name):
        def record(module, args, result):
            if name.endswith('generator') and name.split('.')[0] not in examples:
                examples[name.split('.')[0]] = args[0].clone()
                value=args[0]
                rows.append(dict(name=name+'.INPUT',minimum=float(value.min()),maximum=float(value.max()),
                                 max_abs=float(value.abs().max()),nonfinite=int((~torch.isfinite(value)).sum())))
            if isinstance(result,torch.Tensor):
                rows.append(dict(name=name,minimum=float(result.min()),maximum=float(result.max()),
                                 max_abs=float(result.abs().max()),nonfinite=int((~torch.isfinite(result)).sum())))
        return record
    for name in ('image','video'):
        for path,module in getattr(codec,name).vqgan.generator.named_modules():
            handles.append(module.register_forward_hook(hook(name+'.generator'+('.'+path if path else ''))))
    for i in range(2):codec.frame(header,i,packet=packets[i])
    for h in handles:h.remove()
    save_json(out/'eager_ranges.json',rows)
    amp=[]
    for name,value in examples.items():
        with torch.autocast('cuda',dtype=torch.float16):
            actual=getattr(codec,name).vqgan.generator(value)
        finite=bool(torch.isfinite(actual).all())
        amp.append(dict(model=name,nonfinite=int((~torch.isfinite(actual)).sum()),
                        maximum=float(actual.max()) if finite else None,minimum=float(actual.min()) if finite else None))
    save_json(out/'amp_probe.json',amp)
    sys.path.insert(0,str(PACKAGES))
    import tensorrt as trt
    all_layers=[]
    for receipt in (TRIAL/'engines').glob('*/build.json'):
        meta=load_json(receipt)
        if meta['identity']['backend']!='trt_fp16_safe':continue
        logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger)
        network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser=trt.OnnxParser(network,logger)
        if not parser.parse((receipt.parent/'generator.onnx').read_bytes()):raise ValueError('Parse failed')
        layers=[dict(name=l.name,type=str(l.type),outputs=[str(l.get_output(i).dtype) for i in range(l.num_outputs)]) for l in network]
        all_layers.append(dict(model=meta['identity']['role'],layers=layers))
    save_json(out/'parsed_layers.json',all_layers)
    print('Largest original activations:', sorted(rows,key=lambda x:x['max_abs'],reverse=True)[:8],flush=True)
    print('PyTorch AMP:',amp,flush=True)


if __name__=='__main__':main()
