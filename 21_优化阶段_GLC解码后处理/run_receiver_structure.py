"""Freeze and reproduce the receiver-only experiment in a new output directory."""

import argparse
from pathlib import Path
import subprocess
import sys

from receiver_structure import file_sha, save_json, KEYS, RAMP


def run(workspace, output):
    stage = Path(__file__).resolve().parent
    data = workspace/'02_实验数据/优化_GLC后处理'
    batch = data/'source_dense_flow_three_clips_20261109_01'
    upstream = data/'deflicker_reef80_20261108_01/upstream'
    output.mkdir(parents=True, exist_ok=False)
    cases = ('video1','video2','video3')
    inputs = []
    for case in cases:
        base = batch/case
        old = data/'fullframe_reference_refined_20261110_01'/case
        k = data/'support_refinement_20261111_01'/case
        inputs += [base/p for p in ('base.glcsnd','base_receive/decoded_rgb.npy','base_receive/result.json','source.npy')]
        inputs += [old/p for p in ('flow/to_anchor.npy','flow/from_anchor.npy','flow/flow.json',
            'transition_flow.npy','transition_flow.json','offline/rgb.npy','offline/alpha.npy','offline/shared.npy')]
        inputs += [k/p for p in ('K/support.glccsp2','K/send.json','K/receive/rgb.npy','K/receive/alpha.npy',
            'K/receive/receipt.json','evaluation/summary.json')]
    inputs += [upstream/'pretrained_weights/raft-things.pth',
               data/'video1_persistent_aux_20261110_01/g17q4/receive/rgb.npy']
    save_json(output/'execution_plan.json',dict(status='frozen_before_receiver',date='2026-11-11',
        cases=cases,frames=150,fps=15,keys=KEYS,old_weights=RAMP,
        main_candidate='RR: decoded RGB motion and decoded RGB photometric reliability; zero auxiliary bytes',
        controls={'SS':'source motion + source photometric reliability; offline only',
                  'SR':'source motion + decoded RGB photometric reliability; offline only',
                  'RS':'decoded RGB motion + source photometric reliability; offline only'},
        geometry_tests='Follow the motion source; FB <1.5px, gradient <.75, inside margin 2px',
        photometric_tests='sigma3 max RGB diff <16; sigma1 max RGB diff <24; 3x3 erosion; 2px inward feather',
        reference_RGB='Historical raw GLC decoded RGB; never source RGB or recursively filtered RGB',
        gate_representation='Dense pixel-level every frame; no auxiliary compression constraints',
        K_comparison='Deployment comparison, not a pure motion-source comparison; use dense 2x2 controls for attribution',
        freeze='No parameter or case-specific tuning in this run',
        evaluation='Lossless RGB, fixed source-derived support, full/quadrants plus existing video1 coral box',
        receiver_guard='Fresh process; allow base bitstream, base receiver receipt and RGB, shared weights only',
        billing='Full base file; auxiliary 0 bytes; preinstalled code/weights and local diagnostics excluded',
        limits='Three development clips; not unseen data, full-length, or realtime certification',
        input_sha256={str(p):file_sha(p) for p in inputs},
        receiver_code_sha256=file_sha(stage/'receiver_structure.py')))
    for case in cases:
        subprocess.run([sys.executable,'-B','-u','-X','utf8',str(stage/'receiver_structure.py'),
            '--base-dir',str(batch/case),'--upstream',str(upstream),'--output',str(output/case/'RR')],check=True)
        subprocess.run([sys.executable,'-B','-u','-X','utf8',str(stage/'evaluate_receiver_structure.py'),
            '--run',str(output),'--ffmpeg',str(workspace/'01_项目/运行环境/FFmpeg/ffmpeg.exe'),
            '--case',case],check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    run(a.workspace.resolve(),a.output.resolve())
