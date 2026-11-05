"""Use the same real-input screen protocol for the BF16-body candidate."""
from screen_candidates import main
from uniform_bf16_backend import build_uniform, Path, HERE

if __name__=='__main__':
    main(backend_builder=build_uniform,modes=('uniform_bf16',),backend_source=HERE/'uniform_bf16_backend.py')
