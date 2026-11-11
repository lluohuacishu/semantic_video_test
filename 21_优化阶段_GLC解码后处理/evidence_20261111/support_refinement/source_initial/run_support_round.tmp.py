from pathlib import Path
import sys

stage=Path(__file__).parent/'21_优化阶段_GLC解码后处理'
sys.path.insert(0,str(stage))
import numpy as np
import cv2
from coral_reference_sidecar import file_sha,load_json,save_json
from compact_support_stream import reduce_support,unpack as unpack_old,required_nodes
from fullframe_structure_stream import unpack as unpack_full
from run_compact_support import paths,safety
from predictive_support_stream import pack,unpack,render
from experiment_common import array_sha

workspace=Path(__file__).resolve().parents[3]
root=workspace/'02_实验数据/优化_GLC后处理'
run=root/'support_refinement_20261111_01'
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
