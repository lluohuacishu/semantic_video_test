"""Optimistic byte probe for receiver-flow prediction with K preserved exactly.

This is not a deployable new packet: the GLCCSP02 surrogate omits new predictor
signaling/checksum overhead. If it is already larger than K, that overhead
cannot rescue this particular coding design. No source-free quality claim.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

import compact_support_stream as geometry
import predictive_support_stream as codec
from receiver_structure import file_sha, load_json, save_json


def probe(run, case):
    folder = run/case/'prediction_byte_probe'; folder.mkdir(exist_ok=False)
    original = run.parent/'support_refinement_20261111_01'/case/'K'
    rr = run/case/'RR'
    base = run.parent/'source_dense_flow_three_clips_20261109_01'/case
    packet = (original/'support.glccsp2').read_bytes()
    digest = file_sha(base/'base.glcsnd')
    h,a = codec.unpack(packet,digest)
    save_json(folder/'plan.json',dict(frozen_before_probe=True,case=case,
        only_candidate='Half-pixel K grid prediction from resized decoded-RGB RAFT; same K support',
        compression='Same existing lossless sparse spatial/time/velocity and full-grid codecs',
        preserved='All needed K motion, support, and output RGB/alpha exactly',
        billing='Optimistic surrogate complete bytes; new prediction metadata would add bytes',
        K_sha256=file_sha(original/'support.glccsp2'),
        local_flow_sha256={n:file_sha(rr/n) for n in ('to_anchor.npy','transition_flow.npy')}))
    flows = np.concatenate([np.load(rr/'to_anchor.npy',mmap_mode='r'),
                            np.load(rr/'transition_flow.npy',mmap_mode='r')[:,0]])
    pred = np.stack([np.rint(cv2.resize(v,tuple(h['grid']),interpolation=cv2.INTER_LINEAR)*2).astype(np.int32)
                     for v in flows])
    needed = geometry.required_nodes(a['support'],5,4,tuple(h['grid']),'E')
    delta = np.zeros(a['motion'].shape,dtype='<i2')
    residual = a['motion'][needed].astype(np.int32)-pred[needed]
    if np.abs(residual).max() > 4096:
        raise ValueError('Existing surrogate codec range exceeded')
    delta[needed] = residual
    baseline,_,_ = codec.pack(a['motion'],a['support'],digest,step=4,span=5)
    surrogate,header,choices = codec.pack(delta,a['support'],digest,step=4,span=5)
    dh,da = codec.unpack(surrogate,digest)
    recovered = np.zeros(a['motion'].shape,dtype='<i2')
    recovered[needed] = (da['motion'][needed].astype(np.int32)+pred[needed]).astype('<i2')
    assert np.array_equal(recovered[needed],a['motion'][needed])
    assert np.array_equal(da['support'],a['support'])
    raw = np.load(base/'base_receive/decoded_rgb.npy',mmap_mode='r')
    rgb,alpha = codec.render(raw,h,dict(motion=recovered,support=da['support']))
    assert np.array_equal(rgb,np.load(original/'receive/rgb.npy',mmap_mode='r'))
    assert np.array_equal(alpha,np.load(original/'receive/alpha.npy',mmap_mode='r'))
    result = dict(complete=True,case=case,K_bytes=len(packet),canonical_K_repack_bytes=len(baseline),
        optimistic_surrogate_bytes=len(surrogate),delta_bytes=len(surrogate)-len(packet),
        optimistic_aux_kbps=len(surrogate)*8/10000,
        K_motion_payload_bytes=h['sections']['motion']['bytes'],
        residual_motion_payload_bytes=header['sections']['motion']['bytes'],
        support_payload_bytes=header['sections']['support']['bytes'],
        decoder_prediction_metadata_included=False,deployed_candidate=False,
        K_RGB_alpha_bit_exact=True,needed_motion_exact=True,support_exact=True,
        header=header,encoding_choices=choices,code_sha256=file_sha(__file__))
    save_json(folder/'summary.json',result)
    print(case,'receiver-predicted correction probe:',len(packet),'->',len(surrogate),
          'B before new predictor metadata; K RGB/alpha exact',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--case',choices=('video1','video2','video3'),required=True)
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    probe(a.run.resolve(),a.case)
