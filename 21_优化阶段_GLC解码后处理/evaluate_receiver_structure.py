"""Separate source-assisted evaluation of the zero-sidecar receiver experiment."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import receiver_structure as receiver
from receiver_structure import file_sha, load_json, save_json, KEYS
from compare_deflicker import save_csv
import evaluate_fullframe_reference as scoring
from run_fullframe_factorial import BOX
from run_fullframe_reference import reliable as legacy_reliable
from run_p0 import preview
from run_support_factorial import timing
from verify_video1_persistent_structure import inspect_preview


def load_arrays(folder, names):
    return [np.load(folder/(n+'.npy'), mmap_mode='r', allow_pickle=False) for n in names]


def mask_summary(alpha, scopes):
    non_key = [i for i in range(150) if i not in KEYS]
    return {s:dict(active=float((alpha[:,y0:y1,x0:x1]>0).mean()),
                   mean_alpha=float(alpha[:,y0:y1,x0:x1].mean()),
                   non_key_active=float((alpha[non_key,y0:y1,x0:x1]>0).mean()))
            for s,(x0,y0,x1,y1) in scopes.items()}


def evaluate(run, case, ffmpeg):
    data = run.parent
    batch = data/'source_dense_flow_three_clips_20261109_01'/case
    prior = data/'fullframe_reference_refined_20261110_01'/case
    kfolder = data/'support_refinement_20261111_01'/case
    rr = run/case/'RR'
    out = run/case/'evaluation'; out.mkdir(exist_ok=False)
    plan = load_json(run/'execution_plan.json')
    receipt = load_json(rr/'receipt.json')
    assert file_sha(Path(receiver.__file__)) == receipt['code_sha256'] == plan['receiver_code_sha256']
    assert receipt['complete'] and receipt['auxiliary_bytes'] == 0 and receipt['source_reads_blocked']
    assert len(receipt['reads']) == 4 and len(receipt['blocked_inputs']) == 7
    assert receipt['flow_pairs'] == 174
    for name,digest in receipt['outputs_sha256'].items():
        assert file_sha(rr/name) == digest
    source = np.load(batch/'source.npy', mmap_mode='r')
    raw = np.load(batch/'base_receive/decoded_rgb.npy', mmap_mode='r')
    rback,rforward,rtransition = load_arrays(rr, ('to_anchor','from_anchor','transition_flow'))
    sback,sforward = load_arrays(prior/'flow', ('to_anchor','from_anchor'))
    stransition = np.load(prior/'transition_flow.npy', mmap_mode='r')
    sequences = dict(source=source, raw=raw,
        K=np.load(kfolder/'K/receive/rgb.npy', mmap_mode='r'),
        RR=np.load(rr/'rgb.npy', mmap_mode='r'))
    alphas = dict(K=np.load(kfolder/'K/receive/alpha.npy', mmap_mode='r'),
                  RR=np.load(rr/'alpha.npy', mmap_mode='r'))
    checks = dict(receiver_source_read_guard=True, receiver_outputs_hash_verified=True)
    # Recompute using the original reliability implementation, including its warp.
    replay = receiver.render(raw, raw, rback, rforward, rtransition, gate_fn=legacy_reliable)
    for name,value in replay.items():
        assert np.array_equal(value, np.load(rr/(name+'.npy'), mmap_mode='r')), ('RR replay',name)
    checks['RR_CPU_replay_legacy_gate_bit_exact'] = True
    del replay
    for name,images,motion in (('SS',source,(sback,sforward,stransition)),
                               ('SR',raw,(sback,sforward,stransition)),
                               ('RS',source,(rback,rforward,rtransition))):
        folder = run/case/name; folder.mkdir(exist_ok=False)
        values = receiver.render(raw, images, *motion)
        if name == 'SS':
            for field,value in values.items():
                assert np.array_equal(value, np.load(prior/'offline'/(field+'.npy'), mmap_mode='r')), ('SS',field)
            checks['SS_archived_RGB_alpha_shared_bit_exact'] = True
        for field,value in values.items():
            np.save(folder/(field+'.npy'), value, allow_pickle=False)
        sequences[name] = np.load(folder/'rgb.npy', mmap_mode='r')
        alphas[name] = np.load(folder/'alpha.npy', mmap_mode='r')
        save_json(folder/'receipt.json',dict(complete=True,offline_source_assisted=True,
            deployed_candidate=False, gate_photometric_input='source' if name in ('SS','RS') else 'GLC decoded RGB',
            motion_input='source' if name in ('SS','SR') else 'GLC decoded RGB',
            reference_RGB='Historical raw GLC decoded RGB',
            outputs_sha256={p.name:file_sha(p) for p in folder.glob('*.npy')}))
        del values
        print(case, name, 'offline control complete', flush=True)

    scoring.SCOPES.pop('coral', None)
    if case == 'video1':
        scoring.SCOPES['coral'] = BOX
        sequences['U'] = np.load(data/'video1_persistent_aux_20261110_01/g17q4/receive/rgb.npy', mmap_mode='r')
    scores, temporal, spatial = scoring.score(sequences)
    archived = load_json(kfolder/'evaluation/summary.json')['scores']
    for scope in scoring.SCOPES:
        for name in ('raw','K'):
            for field in ('coarse_mae','fine_ncc','psnr_db','valid_pixels'):
                assert abs(scores[scope][name][field]-archived[scope][name][field]) < 1e-9
    checks['raw_K_archived_metrics_exact'] = True
    save_csv(out/'temporal.csv', temporal); save_csv(out/'spatial.csv', spatial)
    coverage = {name:mask_summary(a,scoring.SCOPES) for name,a in alphas.items()}
    source_gate_disagreement = {}
    motion_difference = {}
    for scope,(x0,y0,x1,y1) in scoring.SCOPES.items():
        ar = alphas['RR'][:,y0:y1,x0:x1] > 0
        ass = alphas['RS'][:,y0:y1,x0:x1] > 0
        source_gate_disagreement[scope] = dict(receiver_active=int(ar.sum()),
            rejected_by_source_photometry=int((ar & ~ass).sum()),
            rejected_fraction_among_receiver_active=float((ar & ~ass).sum()/max(1,ar.sum())),
            explanation='Same receiver motion and geometry tests; replace only photometric RGB input. Not semantic ground truth.')
        samples = []
        for i in range(150):
            if i in KEYS:
                continue
            active = alphas['SS'][i,y0:y1,x0:x1] >= .99
            epe = np.linalg.norm(rback[i,y0:y1,x0:x1]-sback[i,y0:y1,x0:x1], axis=2)
            samples.append(epe[active])
        values = np.concatenate(samples)
        motion_difference[scope] = dict(pixels=int(values.size),mean=float(values.mean()),
            median=float(np.median(values)),p95=float(np.quantile(values,.95)),
            explanation='Receiver-to-source RAFT discrepancy on source-safe non-key pixels; source RAFT is not ground-truth flow.')

    groups = {}
    main_scope = 'coral' if case == 'video1' else 'full'
    for category in ('refresh_with_exit','group_update','group_interior'):
        groups[category] = {}
        for name in sequences:
            rows = [r for r in temporal if r['scope'] == main_scope and r['method'] == name and timing(r['frame']) == category]
            n = sum(r['pixels'] for r in rows)
            groups[category][name] = dict(pairs=len(rows),pixels=n,
                coarse_mae=sum(r['coarse_sum'] for r in rows)/n,
                fine_ncc=sum(r['fine_ncc']*r['pixels'] for r in rows)/n)
    factors = {}
    for scope,methods in scores.items():
        v = {n:methods[n]['coarse_mae'] for n in ('SS','SR','RS','RR')}
        factors[scope] = dict(receiver_motion_at_source_photo=v['RS']-v['SS'],
            receiver_motion_at_receiver_photo=v['RR']-v['SR'],
            receiver_photo_at_source_motion=v['SR']-v['SS'],
            receiver_photo_at_receiver_motion=v['RR']-v['RS'],
            interaction=v['RR']-v['RS']-v['SR']+v['SS'])
    labels = dict(source='Source RGB [evaluation only]',raw='Original GLC',
        K=f'K / aux {load_json(kfolder/"K/send.json")["kbps"]:.2f} kbps',
        RR='R0 / receiver only / aux 0 kbps',SS='SS / source flow + source photo [offline]',
        SR='SR / source flow + receiver photo [offline]',RS='RS / receiver flow + source photo [offline]')
    specs = [('K_vs_R0_full.mp4',('K','RR'),None),
             ('source_raw_K_R0.mp4',('source','raw','K','RR'),None)]
    if case == 'video1':
        specs += [('K_vs_R0_coral.mp4',('K','RR'),(250,0,512,132)),
                  ('raw_vs_R0_coral.mp4',('raw','RR'),(250,0,512,132)),
                  ('offline_factorial_coral.mp4',('SS','SR','RS','RR'),(250,0,512,132))]
    previews = {}
    for filename,panel_names,crop in specs:
        panels = [sequences[n] for n in panel_names]
        region = None if crop is None else np.s_[crop[1]:crop[3],crop[0]:crop[2]]
        preview(ffmpeg,out/filename,panels,15,0,[labels[n] for n in panel_names],crop=region)
        previews[filename] = inspect_preview(ffmpeg,out/filename,panels,crop=crop)
    canvas = Image.new('RGB',(512*4,312*3),(24,35,48)); draw = ImageDraw.Draw(canvas)
    for row,i in enumerate((20,80,149)):
        for col,name in enumerate(('source','raw','K','RR')):
            canvas.paste(Image.fromarray(np.asarray(sequences[name][i])),(col*512,row*312+24))
            draw.text((col*512+6,row*312+5),f'{case} {name} f{i}',fill='white')
    canvas.save(out/'source_raw_K_R0.png')
    kbytes = (kfolder/'K/support.glccsp2').stat().st_size
    summary = dict(complete=True,case=case,scores=scores,checks=checks,coverage=coverage,
        dense_factor_effects=factors,source_gate_disagreement=source_gate_disagreement,
        motion_discrepancy_px=motion_difference,time_groups=groups,
        rates=dict(base_bytes=receipt['base_bytes'],K_aux_bytes=kbytes,R0_aux_bytes=0,
            K_total_kbps=(receipt['base_bytes']+kbytes)*8/10000,R0_total_kbps=receipt['total_kbps'],
            total_bytes_saved_fraction=kbytes/(receipt['base_bytes']+kbytes)),
        source_only_for_evaluation_and_labeled_controls=True,
        previews=previews,visual_review='pending',
        timing=dict(flow_seconds=receipt['flow_seconds'],render_seconds=receipt['render_seconds'],
                    peak_allocated_mib=receipt['peak_allocated_mib']),
        code_sha256=file_sha(__file__))
    save_json(out/'summary.json',summary)
    print(case, main_scope, {n:(round(v['coarse_mae'],6),round(v['fine_ncc'],6))
                            for n,v in scores[main_scope].items()},flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--ffmpeg',type=Path,required=True)
    p.add_argument('--case',choices=('video1','video2','video3'))
    a = p.parse_args(); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    run = a.run.resolve()
    results = {}
    for case in ((a.case,) if a.case else ('video1','video2','video3')):
        results[case] = evaluate(run,case,a.ffmpeg.resolve())
    results = {case:load_json(run/case/'evaluation/summary.json')
               for case in ('video1','video2','video3') if (run/case/'evaluation/summary.json').exists()}
    save_json(run/'evaluation_progress.json',results)
    if len(results) != 3:
        raise SystemExit(0)
    manifest = load_json(run/'execution_plan.json')['input_sha256']
    for path,digest in manifest.items():
        assert file_sha(Path(path)) == digest, path
    save_json(run/'final_summary.json',dict(complete=True,cases=results,
        frozen_inputs_unchanged=True,input_hash_count=len(manifest),
        default='K pending assessment of R0 tradeoff',visual_review='pending',
        no_new_H265_comparison=True,unseen_long_realtime_tested=False))
