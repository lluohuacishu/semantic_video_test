"""Attach experimental input-encoder identity without changing reference arithmetic."""
from __future__ import annotations
import sys
import codec_worker
from experiment_common import file_sha
from extra_variants import install


class ExperimentWriter(codec_worker.SenderWriter):
    def __init__(self,path,header):
        variant=sys.argv[sys.argv.index('--variant')+1]
        if variant!='eager':
            header['input_encoder_experiment']=dict(variant=variant,training_steps=0,
                scope='Input image feature extraction only; arithmetic field identifies unchanged reference/entropy/decoder',
                fp16='fp16' in variant,tf32=variant=='tf32',
                encoder_variant_code_sha256=file_sha(__file__.replace('candidate_worker.py','encoder_variants.py')),
                extra_variant_code_sha256=file_sha(__file__.replace('candidate_worker.py','extra_variants.py')))
        super().__init__(path,header)


if __name__=='__main__':
    codec_worker.SenderWriter=ExperimentWriter
    codec_worker.install=install
    codec_worker.main()
