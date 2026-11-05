"""Independent receiver entry for the BF16-body candidate."""
from candidate_receiver import main
from uniform_bf16_backend import build_uniform, HERE

if __name__=='__main__':
    main(backend_builder=build_uniform,modes=('uniform_bf16',),backend_source=HERE/'uniform_bf16_backend.py')
