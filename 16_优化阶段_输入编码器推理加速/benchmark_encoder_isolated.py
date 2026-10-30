"""Preserve an independent original-layout baseline for channels-last comparison."""
from __future__ import annotations
import copy
import sys
from pathlib import Path
import benchmark_encoder
from experiment_common import load_json,save_json
from extra_variants import build

original=benchmark_encoder.EncoderVariant
probe={}


def factory(model,example,variant,cache):
    probe.update(input_stride=list(example.stride()),input_contiguous=example.is_contiguous(),
                 input_channels_last=example.is_contiguous(memory_format=benchmark_encoder.torch.channels_last),
                 original_first_weight_stride=list(next(model.parameters()).stride()))
    for _ in range(8):model(example)
    # Layout conversion is in-place on a Module. Retain an original NCHW control.
    if variant in ['channels_last','amp_fp16_cl']:model=copy.deepcopy(model)
    return build(model,example,variant,cache)


if __name__=='__main__':
    benchmark_encoder.EncoderVariant=factory
    benchmark_encoder.main()
    out=Path(sys.argv[sys.argv.index('--output')+1]);r=load_json(out/'result.json')
    r['baseline_note']='Independent original-input/original-weight-layout eager control; channels-last candidate has a separate equal-weight copy'
    r['input_storage']=probe
    save_json(out/'result.json',r)
