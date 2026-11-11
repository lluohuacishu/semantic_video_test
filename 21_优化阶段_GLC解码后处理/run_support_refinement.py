"""Fixed-motion support refinement and independent GLCCSP02 receiver."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import cv2
from coral_reference_sidecar import file_sha,load_json,save_json
from compact_support_stream import reduce_support,unpack as unpack_old,required_nodes
from fullframe_structure_stream import unpack as unpack_full
from run_compact_support import paths,safety
from predictive_support_stream import pack,unpack,render
from experiment_common import array_sha
from cache_motion_inputs import base_frames
from run_dense_batch import CASES



def prepare(workspace,run):
    root=workspace/'02_实验数据/优化_GLC后处理'
    run.mkdir(exist_ok=False)
    selection=load_json(root/'compact_support_20261110_01/selected.json')['profiles']
    save_json(run/'plan.json',dict(date='2026-11-11',status='frozen_before_prepare',
        basis='Spatial cell loss exceeds temporal intersection loss in support_factorial_20261111_01.',
        cases=['video1','video2','video3'],variants={'K':'Bit-exact repacking of L','N':'2x2 safety cells, same five-frame intersections'},
        fixed='Same selected L motion grid, half-pixel values, keys, blending and existing safety thresholds for each clip; no refitting or score-based grid choice.',
        predictor_selection='Choose complete lossless section plus metadata bytes; necessary motion and renderer output must match unpruned motion.',
        budget='40 kbps absolute cap. Compare against current L bytes; above L must be labeled higher-rate, below base is preferred.',
        visual_review='pending',no_new_source_rgb_at_receiver=True))
    batch,prior=paths(workspace)
    records={}
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    for case in ('video1','video2','video3'):
        folder=run/case;folder.mkdir()
        base=batch/case
        digest=file_sha(base/'base.glcsnd')
        old_dir=root/'compact_support_20261110_01'/case/selection[case]['E']['name']
        lh,la=unpack_old((old_dir/'support.glccsp').read_bytes(),digest)
        _,fa=unpack_full((prior/case/'g24q2_E/fullframe.glcfull').read_bytes(),digest)
        frozen=np.concatenate([fa['new_motion'],fa['old_motion']])
        motion=np.stack([np.rint(cv2.resize(q.astype(np.float32),tuple(lh['grid']),interpolation=cv2.INTER_LINEAR)).astype('<i2') for q in frozen])
        source=np.load(base/'source.npy',mmap_mode='r')
        safe=safety(source,prior/case,motion)
        raw=np.load(base/'base_receive/decoded_rgb.npy',mmap_mode='r')
        assert np.array_equal(reduce_support(safe,4,5),la['support'])
        records[case]={}
        for name,step in [('K',4),('N',2)]:
            target=folder/name;target.mkdir()
            support=reduce_support(safe,step,5)
            payload,h,probe=pack(motion,support,digest,step=step)
            (target/'support.glccsp2').write_bytes(payload)
            record=dict(bytes=len(payload),kbps=len(payload)*8/10000,sha256=file_sha(target/'support.glccsp2'),
                        base_bytes=(base/'base.glcsnd').stat().st_size,old_L_bytes=selection[case]['E']['bytes'],
                        under_40kbps=len(payload)<=50000,below_base=len(payload)<(base/'base.glcsnd').stat().st_size,
                        at_or_below_L_bytes=len(payload)<=selection[case]['E']['bytes'],header=h,probe=probe,
                        input_sha256={str(p):file_sha(p) for p in [base/'base.glcsnd',base/'source.npy',base/'base_receive/decoded_rgb.npy',old_dir/'support.glccsp',prior/case/'g24q2_E/fullframe.glcfull']})
            if len(payload)<=50000:
                dh,da=unpack(payload,digest)
                needed=required_nodes(support,5,step,tuple(lh['grid']),'E')
                assert np.array_equal(da['support'],support)
                assert np.array_equal(da['motion'][needed],motion[needed])
                rgb,alpha=render(raw,dh,da)
                check,check_alpha=render(raw,dh,dict(motion=motion,support=support))
                assert np.array_equal(rgb,check) and np.array_equal(alpha,check_alpha)
                if name=='K':
                    assert np.array_equal(rgb,np.load(old_dir/'receive/rgb.npy',mmap_mode='r'))
                    assert np.array_equal(alpha,np.load(old_dir/'receive/alpha.npy',mmap_mode='r'))
                record.update(required_motion_exact=True,unpruned_RGB_alpha_exact=True,
                    old_L_RGB_alpha_exact=name=='K',expected_rgb_sha256=array_sha(rgb),expected_alpha_sha256=array_sha(alpha))
            save_json(target/'send.json',record)
            records[case][name]=record
            print(case,name,len(payload),'bytes',record['kbps'],'kbps',h['sections'],flush=True)
        save_json(run/'prepare_progress.json',records)
    save_json(run/'prepare_summary.json',dict(complete=True,cases=records))


def receive(workspace,run,case,name):
    batch,_=paths(workspace)
    base=batch/case
    folder=run/case/name
    packet=folder/'support.glccsp2'
    out=folder/'receive'
    out.mkdir(exist_ok=False)
    allowed={p.resolve() for p in (base/'base.glcsnd',base/'base_cut.json',
             base/'base_receive/decoded_rgb.npy',base/'base_receive/result.json',packet)}
    observed=set()
    protected={'.json','.npy','.npz','.png','.jpg','.mp4','.mkv','.glcsnd',
               '.glcfull','.glccsp','.glccsp2','.glcflow','.glcgate','.glcstruct'}
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.suffix.lower() in protected:
                if p not in allowed and not p.is_relative_to(out):
                    raise RuntimeError('Untransmitted receiver input: '+str(p))
                if p in allowed:observed.add(str(p))
    sys.addaudithook(audit)
    blocked=[]
    for p in (base/'source.npy',folder/'send.json',run/'untransmitted_flow.npy',run/'untransmitted_mask.npy'):
        try:p.read_bytes()
        except RuntimeError:blocked.append(str(p))
        else:raise ValueError('Read guard failed')
    raw,digest,size=base_frames(base,CASES[case][0])
    h,a=unpack(packet.read_bytes(),digest)
    rgb,alpha=render(raw,h,a)
    np.save(out/'rgb.npy',rgb,allow_pickle=False)
    np.save(out/'alpha.npy',alpha,allow_pickle=False)
    save_json(out/'receipt.json',dict(complete=True,reads=sorted(observed),blocked_inputs=blocked,
        source_files_unavailable=True,independently_verified_base=True,
        base_sha256=digest,base_bytes=size,aux_bytes=packet.stat().st_size,aux_sha256=file_sha(packet),
        rgb_sha256=array_sha(rgb),alpha_sha256=array_sha(alpha),
        zero_weight_unchanged=bool(np.array_equal(rgb[alpha==0],raw[alpha==0])),
        outputs_sha256={p.name:file_sha(p) for p in out.glob('*.npy')},
        code_sha256=file_sha(__file__),codec_sha256=file_sha(Path(__file__).with_name('predictive_support_stream.py'))))
    print(case,name,'independent receive PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','receive'))
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--case',choices=('video1','video2','video3'))
    p.add_argument('--name',choices=('K','N'))
    args=p.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    if args.action=='prepare':prepare(args.workspace.resolve(),args.run.resolve())
    else:
        if not args.case or not args.name:p.error('receive needs --case and --name')
        receive(args.workspace.resolve(),args.run.resolve(),args.case,args.name)
