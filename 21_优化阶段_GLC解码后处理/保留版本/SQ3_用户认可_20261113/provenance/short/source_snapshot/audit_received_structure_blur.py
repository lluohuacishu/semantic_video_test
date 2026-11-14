"""Post-registered matched-fine-amplitude blur control; not a deployment candidate."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from runtime_speed_common import STAGE, REPO, DATA, FFMPEG, file_sha, load_json, save_json, check_inputs
from probe_reef import bands
from run_p0 import preview

CASES = ('video1', 'video2', 'video3')
NAME = 'BLUR_MATCH'
CODE = ('audit_received_structure_blur.py', 'probe_reef.py',
        'resume_q9_archive_transfer.py', 'evaluate_structure_guide.py',
        'finish_q9_ghost_exit.py', 'run_p0.py', 'runtime_speed_common.py')
CONFIG = 'b5eeb844d0adcdfe97648351a0654a2575a98942e1627c98896ba302555f7b04'


def amplitude(video):
    return sum(float(np.abs(bands(f)[..., 2]).sum()) for f in video) / np.prod(video.shape[:3])


def blur(video, sigma):
    if sigma == 0:
        return np.asarray(video).copy()
    return np.stack([np.rint(np.clip(cv2.GaussianBlur(f.astype(np.float32), (0, 0), sigma),
                                   0, 255)).astype(np.uint8) for f in video])


def match(q, target):
    lo, hi = 0., 1.5
    samples = []
    best = None
    for sigma in (lo, hi):
        arr = blur(q, sigma)
        value = amplitude(arr)
        samples.append(dict(sigma=sigma, amplitude=value))
        item = (abs(value - target), sigma, arr)
        if best is None or item[:2] < best[:2]:
            best = item
    assert samples[0]['amplitude'] >= target >= samples[1]['amplitude']
    for _ in range(12):
        sigma = (lo + hi) / 2
        arr = blur(q, sigma)
        value = amplitude(arr)
        samples.append(dict(sigma=sigma, amplitude=value))
        item = (abs(value - target), sigma, arr)
        if item[:2] < best[:2]:
            best = item
        if value > target:
            lo = sigma
        else:
            hi = sigma
    assert best[0] / target < .005
    return best, samples


def freeze(run):
    root = run / 'blur_control'
    if root.exists():
        assert not any(root.iterdir()), 'Refuse to overwrite a registered control'
    else:
        root.mkdir()
    plan = load_json(run / 'execution_plan.json')
    assert load_json(run / 'verification.json')['complete']
    inputs = {}
    dependency = REPO / '11_优化阶段_纹理稳定性/temporal_filter.py'
    inputs[str(dependency)] = file_sha(dependency)
    for case in CASES:
        for name in ('Q9_R25', 'GS25'):
            p = run / case / 'receive' / name / 'rgb.npy'
            inputs[str(p)] = file_sha(p)
    p = dict(date='2026-11-13', diagnostic_only=True, adaptive_postregistration=True,
             reason='First-phase error reductions accompany reduced fine texture and motion amplitude. Compare GS25 with spatial smoothing at matched full-frame fine amplitude.',
             first_phase_plan_sha256=file_sha(run / 'execution_plan.json'),
             first_phase_summary_sha256=file_sha(run / 'summary.json'),
             first_phase_verification_sha256=file_sha(run / 'verification.json'),
             cases=list(CASES), new_outputs=3, new_deployment_candidates=0,
             matching_inputs_only=['Q9_R25 receiver RGB', 'GS25 receiver RGB'],
             target='150-frame full-image mean abs(gray - Gaussian_sigma1.2(gray)), identical to scoring fine_amplitude',
             matching_rule='Gaussian on float32 Q9 then clip/rint/uint8; endpoints [0,1.5], 12 bisections; minimum absolute amplitude mismatch, lower sigma breaks ties.',
             max_relative_matching_error=.005, per_clip_sigma=True,
             per_clip_sigma_note='Diagnostic nuisance matching only, not per-clip source tuning or a deployment profile.',
             source_only_used_after_output=True, extra_transmission_bytes=0,
             main_outputs_and_evidence_unchanged=True, automatic_selection=None,
             current_config_sha256=plan['current_config_sha256'],
             input_sha256=inputs, code_sha256={n:file_sha(STAGE/n) for n in CODE})
    save_json(root / 'execution_plan.json', p)
    (root / 'source_snapshot').mkdir()
    for n in CODE:
        shutil.copy2(STAGE/n, root/'source_snapshot'/n)
    print('BLUR CONTROL FROZEN: 3 diagnostic outputs', flush=True)


def worker(run, case):
    root = run / 'blur_control'
    plan = load_json(root/'execution_plan.json')
    spec = load_json(run/'case_registration.json')['cases'][case]
    assert all(file_sha(STAGE/n) == h for n,h in plan['code_sha256'].items())
    out = root / case / 'receive'
    out.mkdir(parents=True, exist_ok=False)
    qp = run/case/'receive/Q9_R25/rgb.npy'
    gp = run/case/'receive/GS25/rgb.npy'
    allowed = {qp.resolve(), gp.resolve()}
    reads = set()
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if p in allowed:
            reads.add(str(p))
            return
        if p.is_relative_to(out):
            return
        if p.suffix.lower() not in {'.py','.pyc','.pyd','.dll'} and (
                p.is_relative_to(DATA) or p.suffix.lower() in {'.npy','.npz','.mp4','.glcguide','.glcsnd'}):
            raise RuntimeError('Forbidden matching input: ' + str(p))
    sys.addaudithook(audit)
    blocked = []
    for p in (Path(spec['source']), Path(spec['send_receipt']), run/'summary.json',
              run/'diagnosis/current_structure_errors.json', run/case/'receive/received_guide.npy'):
        try:
            p.read_bytes()
        except RuntimeError:
            blocked.append(str(p))
        else:
            raise AssertionError('Guard failed')
    for p in (qp, gp):
        assert file_sha(p) == plan['input_sha256'][str(p)]
    q = np.load(qp, mmap_mode='r')
    g = np.load(gp, mmap_mode='r')
    target = amplitude(g)
    (error, sigma, arr), search = match(q, target)
    np.save(out/'rgb.npy', arr)
    save_json(out/'receipt.json', dict(complete=True, case=case, sigma=sigma,
        target_amplitude=target, matched_amplitude=amplitude(arr),
        relative_matching_error=error/target, search=search, output_sha256=file_sha(out/'rgb.npy'),
        allowed=sorted(map(str,allowed)), reads=sorted(reads), blocked=blocked,
        source_reads_blocked=True, diagnostic_only=True, per_clip_deployment_tuning=False))
    print('MATCHED', case, 'sigma', sigma, 'relative mismatch', error/target, flush=True)


def evaluate(run, case):
    from resume_q9_archive_transfer import score_valid_support, SCOPES
    from evaluate_structure_guide import motion_change_diagnostic
    from compare_deflicker import save_csv
    spec = load_json(run/'case_registration.json')['cases'][case]
    plan = load_json(run/'execution_plan.json')
    out = run/'blur_control'/case/'evaluation'
    out.mkdir(exist_ok=False)
    seq = {n:np.load(spec[n],mmap_mode='r') for n in ('source','raw')}
    for n in ('Q9_R25','GS25'):
        seq[n] = np.load(run/case/'receive'/n/'rgb.npy',mmap_mode='r')
    seq[NAME] = np.load(run/'blur_control'/case/'receive/rgb.npy',mmap_mode='r')
    SCOPES.pop('reef',None)
    if spec.get('reef'):
        SCOPES['reef'] = tuple(spec['reef'])
    scores,tr,sr,missing = score_valid_support(seq)
    dynamic,dr = motion_change_diagnostic(seq)
    save_csv(out/'temporal.csv',tr)
    save_csv(out/'spatial.csv',sr)
    save_csv(out/'dynamic.csv',dr)
    regions,rows = {},[]
    for label,(x0,y0,x1,y1) in plan['review_regions'][case].items():
        regions[label] = {}
        for n,a in seq.items():
            vals,low = [],[]
            for i in range(150):
                x=a[i,y0:y1,x0:x1].astype(np.float32)
                y=seq['source'][i,y0:y1,x0:x1].astype(np.float32)
                v=float(np.abs(x-y).mean())
                l=float(np.abs(cv2.GaussianBlur(x,(0,0),1)-cv2.GaussianBlur(y,(0,0),1)).mean())
                vals.append(v);low.append(l)
                rows.append(dict(region=label,method=n,frame=i,rgb_mae=v,sigma1_mae=l))
            regions[label][n]=dict(mean_rgb_mae=float(np.mean(vals)),mean_sigma1_mae=float(np.mean(low)),
                review_frames={str(i):dict(rgb_mae=vals[i],sigma1_mae=low[i]) for i in plan['review_frames']})
    save_csv(out/'current_fish_regions.csv',rows)
    result=dict(complete=True,case=case,scores=scores,dynamic=dynamic,low_support=missing,
                current_regions=regions,diagnostic_only=True)
    save_json(out/'summary.json',result)
    print('CONTROL EVALUATED',case,flush=True)
    return result


def run_all(run):
    root=run/'blur_control'
    plan=load_json(root/'execution_plan.json')
    check_inputs(plan['input_sha256'])
    cases={}
    for case in CASES:
        subprocess.run([sys.executable,'-B','-u','-X','utf8',str(Path(__file__).resolve()),
                        'worker','--run',str(run),'--case',case],check=True,cwd=REPO)
        cases[case]=evaluate(run,case)
    save_json(root/'summary.json',dict(complete=True,cases=cases,new_outputs=3,
                                      diagnostic_only=True,automatic_selection=None))


def verify(run):
    from finish_q9_ghost_exit import scores_from_csv
    root=run/'blur_control'
    plan=load_json(root/'execution_plan.json')
    first=load_json(run/'summary.json')
    reg=load_json(run/'case_registration.json')
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
            value=float(np.mean([np.abs(a[i,y0:y1,x0:x1].astype(np.float32)-source[i,y0:y1,x0:x1]).mean() for i in range(150)]))
            assert value==summary['current_regions'][label][NAME]['mean_rgb_mae']
        checks[case]=dict(complete=True,matching_replayed_exact=True,source_guard_verified=True,
                         baselines_exact=True,csv_reaggregated=True,region_directly_recomputed=True)
        print('CONTROL VERIFIED',case,flush=True)
    save_json(root/'verification.json',dict(complete=True,cases=checks,
        first_phase_unchanged=True,current_config_unchanged=True,new_outputs=3,
        verifier_sha256=file_sha(__file__)))


def media(run):
    from finish_q9_ghost_exit import inspect
    root=run/'blur_control';out=root/'previews'
    out.mkdir(exist_ok=False)
    assert load_json(root/'verification.json')['complete']
    reg=load_json(run/'case_registration.json')
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    checks={}
    for case in CASES:
        panels=[np.load(reg['cases'][case]['source'],mmap_mode='r'),
                np.load(run/case/'receive/Q9_R25/rgb.npy',mmap_mode='r'),
                np.load(root/case/'receive/rgb.npy',mmap_mode='r'),
                np.load(run/case/'receive/GS25/rgb.npy',mmap_mode='r')]
        names=['Source','Original Q9_R25','Matched blur / diagnostic only','GS25 / NOT adopted']
        p=out/(case+'_matched_blur_control.mp4')
        preview(FFMPEG,p,panels,15,0,names)
        checks[p.name]=inspect(p,panels)
        i=61
        canvas=Image.new('RGB',(1024,628),(24,35,48));d=ImageDraw.Draw(canvas)
        for j,arr in enumerate(panels):
            x,y=(j%2)*512,(j//2)*314
            d.text((x+5,y+4),names[j]+' | f61 4.07s',font=font,fill='white')
            canvas.paste(Image.fromarray(arr[i]),(x,y+26))
        canvas.save(out/(case+'_matched_blur_frame61.png'))
        print('CONTROL MEDIA',case,flush=True)
    save_json(out/'verification.json',dict(complete=True,videos=checks,video_count=3,
         panel_order='source,Q9_R25,BLUR_MATCH,GS25',user_original_speed_review='pending'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('freeze','worker','run','verify','media'))
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--case')
    args=p.parse_args()
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    if args.action=='worker':
        worker(args.run,args.case)
    else:
        {'freeze':freeze,'run':run_all,'verify':verify,'media':media}[args.action](args.run)

