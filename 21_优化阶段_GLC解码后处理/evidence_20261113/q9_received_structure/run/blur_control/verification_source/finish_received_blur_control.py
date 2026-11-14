"""Independent final check of the diagnostic control; preserve the frozen experiment."""
import argparse
from pathlib import Path
import shutil
import cv2
import numpy as np
from runtime_speed_common import STAGE, file_sha, load_json, save_json, check_inputs
from audit_received_structure_blur import CASES, NAME, CONFIG, amplitude, match, media
from finish_q9_ghost_exit import scores_from_csv


def verify(run):
    root = run/'blur_control'
    plan = load_json(root/'execution_plan.json')
    reg = load_json(run/'case_registration.json')
    first = load_json(run/'summary.json')
    assert load_json(root/'summary.json')['complete']
    check_inputs(plan['input_sha256'])
    for n,h in plan['code_sha256'].items():
        assert file_sha(STAGE/n)==h==file_sha(root/'source_snapshot'/n)
    for n in ('plan','summary','verification'):
        p=run/('execution_plan.json' if n=='plan' else n+'.json')
        assert file_sha(p)==plan['first_phase_'+n+'_sha256']
    assert file_sha(STAGE/'当前采用配置.json')==CONFIG
    checks={}
    for case in CASES:
        out=root/case
        r=load_json(out/'receive/receipt.json')
        assert r['complete'] and r['source_reads_blocked'] and len(r['blocked'])==5
        assert set(r['reads'])==set(r['allowed']) and not set(r['reads'])&set(r['blocked'])
        q=np.load(run/case/'receive/Q9_R25/rgb.npy',mmap_mode='r')
        g=np.load(run/case/'receive/GS25/rgb.npy',mmap_mode='r')
        a=np.load(out/'receive/rgb.npy',mmap_mode='r')
        assert a.shape==(150,288,512,3) and a.dtype==np.uint8
        assert file_sha(out/'receive/rgb.npy')==r['output_sha256']
        target=amplitude(g)
        (err,sigma,expected),search=match(q,target)
        assert sigma==r['sigma'] and search==r['search'] and np.array_equal(expected,a)
        assert target==r['target_amplitude'] and amplitude(a)==r['matched_amplitude']
        assert err/target==r['relative_matching_error']<plan['max_relative_matching_error']
        for i in range(150):
            expected=np.rint(np.clip(cv2.GaussianBlur(q[i].astype(np.float32),(0,0),sigma),0,255)).astype(np.uint8)
            assert np.array_equal(expected,a[i])
        summary=load_json(out/'evaluation/summary.json')
        scores_from_csv(out/'evaluation',summary)
        for scope,methods in summary['scores'].items():
            for n in ('source','raw','Q9_R25','GS25'):
                assert methods[n]==first['cases'][case]['scores'][scope][n]
        for n in ('source','raw','Q9_R25','GS25'):
            assert summary['dynamic'][n]==first['cases'][case]['dynamic'][n]
        source=np.load(reg['cases'][case]['source'],mmap_mode='r')
        for label,(x0,y0,x1,y1) in load_json(run/'execution_plan.json')['review_regions'][case].items():
            for n in ('source','raw','Q9_R25','GS25'):
                assert summary['current_regions'][label][n]==first['cases'][case]['current_regions'][label][n]
            # Evaluation converts each per-frame float32 scalar to Python float before averaging.
            values=[float(np.abs(a[i,y0:y1,x0:x1].astype(np.float32)-source[i,y0:y1,x0:x1]).mean()) for i in range(150)]
            assert float(np.mean(values))==summary['current_regions'][label][NAME]['mean_rgb_mae']
        checks[case]=dict(complete=True,matching_replayed_exact=True,source_guard_verified=True,
            independent_pixel_formula_exact=True,baselines_exact=True,csv_reaggregated=True,
            region_directly_recomputed=True,matched_amplitude_relative_error=r['relative_matching_error'])
        print('CONTROL VERIFIED',case,flush=True)
    snap=root/'verification_source'
    snap.mkdir(exist_ok=False)
    shutil.copy2(__file__,snap/Path(__file__).name)
    save_json(root/'verification.json',dict(complete=True,cases=checks,
        first_phase_unchanged=True,current_config_unchanged=True,new_outputs=3,
        verifier_sha256=file_sha(__file__),repair_note='Original verifier used float32 mean for region summary while evaluator uses float64 aggregation of per-frame scalars; independent verifier mirrors recorded aggregation without changing outputs or experiment code.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    verify(args.run)
    media(args.run)

