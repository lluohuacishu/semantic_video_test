"""Read-only retrospective of L: grouped support losses and residual locations.

No new candidate RGB is rendered. Conditional error partitions are descriptive,
not causal percentages; the spatial and temporal support reductions are coupled.
"""

import argparse
import csv
from pathlib import Path
import shutil

import cv2
import numpy as np

from audit_deflicker_texture import source_match
from compact_support_stream import KEYS, RAMP, expand_support, unpack
from compare_deflicker import save_csv
from coral_reference_sidecar import file_sha, load_json, save_json
from diagnose_fullframe_factorial import coefficients
from probe_reef import bands
from run_fullframe_factorial import BOX, feather
from temporal_filter import warp


REFRESH = {key+j for key in KEYS[1:] for j in range(7)}


def timing(i):
    if i in REFRESH:
        return 'anchor_refresh_with_exit'
    return 'nonrefresh_group_boundary' if i % 5 == 0 else 'nonrefresh_group_interior'


def aggregate_saved(run):
    path = run/'video1/evaluation_sparse/temporal.csv'
    with path.open(encoding='utf8', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['scope'] == 'coral']
    result = {}
    for group in ('all', 'anchor_refresh_with_exit', 'nonrefresh_group_boundary', 'nonrefresh_group_interior'):
        result[group] = {}
        for method in ('raw', 'E', 'F', 'U', 'GP', 'S24_B4_T3_E', 'S24_B4_T5_E', 'S24_B4_T5_F'):
            selected = [r for r in rows if r['method'] == method and
                        (group == 'all' or timing(int(r['frame'])) == group)]
            n = sum(int(r['pixels']) for r in selected)
            result[group][method] = dict(pairs=len(selected), pixels=n,
                coarse_sum=sum(float(r['coarse_sum']) for r in selected),
                coarse_mae=sum(float(r['coarse_sum']) for r in selected)/n,
                fine_ncc=sum(float(r['fine_ncc'])*int(r['pixels']) for r in selected)/n)
    return result


def run(workspace, output):
    output.mkdir(parents=True, exist_ok=False)
    root = workspace/'02_实验数据/优化_GLC后处理'
    latest = root/'compact_support_20261110_01'
    factorial = root/'fullframe_factorial_20261110_01'
    batch = root/'source_dense_flow_three_clips_20261109_01/video1'
    local = root/'video1_persistent_aux_20261110_01/g17q4/receive'
    selected = load_json(latest/'selected.json')['profiles']
    rates = {}
    for case, choice in selected.items():
        send = load_json(latest/case/choice['E']['name']/'send.json')
        h = send['header']
        motion = h['sections']['motion']['bytes']
        support = h['sections']['support']['bytes']
        rates[case] = dict(aux_bytes=send['bytes'], base_bytes=send['base_bytes'],
            auxiliary_to_base_ratio=send['bytes']/send['base_bytes'],
            auxiliary_share_of_total=send['bytes']/(send['base_bytes']+send['bytes']),
            motion_bytes=motion, support_bytes=support,
            framing_bytes=send['bytes']-motion-support,
            needed_node_fraction=send['needed_node_fraction'],
            grid=h['grid'], aux_kbps=send['kbps'])
    folder = latest/'video1'/selected['video1']['E']['name']
    packet = folder/'support.glccsp'
    h, arrays = unpack(packet.read_bytes(), file_sha(batch/'base.glcsnd'))
    masks = expand_support(arrays['support'], h['group_span'])
    masks = np.repeat(np.repeat(masks, h['mask_step'], 1), h['mask_step'], 2).astype(bool)
    pixels = np.concatenate([np.load(factorial/'gates'/f'{k}_PG.npy') for k in ('new','old')])
    spatial = pixels.reshape(180,72,4,128,4).all(axis=(2,4))
    spatial = np.repeat(np.repeat(spatial,4,1),4,2)
    assert np.all(~masks | spatial) and np.all(~spatial | pixels)
    scope_masks = {'full': np.ones((288,512),bool), 'coral': np.zeros((288,512),bool)}
    x0,y0,x1,y1 = BOX
    scope_masks['coral'][y0:y1,x0:x1] = True
    coverage = {}
    for kind, frames in (('new',np.s_[:150]),('old',np.s_[150:])):
        coverage[kind] = {}
        for scope, area in scope_masks.items():
            p,b,l = pixels[frames][:,area], spatial[frames][:,area], masks[frames][:,area]
            coverage[kind][scope] = dict(total_pixel_frames=int(p.size),
                pixel_safe=int(p.sum()), spatial_4_safe=int(b.sum()), grouped_5_safe=int(l.sum()),
                lost_by_spatial_cell=int((p & ~b).sum()), lost_by_temporal_group=int((b & ~l).sum()))
    files = dict(source=batch/'source.npy',raw=batch/'base_receive/decoded_rgb.npy',
                 U=local/'rgb.npy', GP=factorial/'native_GP/rgb.npy',
                 E=factorial/'native_GB/rgb.npy', L=folder/'receive/rgb.npy')
    seq = {name:np.load(path,mmap_mode='r') for name,path in files.items()}
    alphas = {name:np.load(path,mmap_mode='r') for name,path in
              dict(L=folder/'receive/alpha.npy',GP=factorial/'native_GP/alpha.npy').items()}
    for i in range(150):
        assert np.array_equal(feather(masks[i]),alphas['L'][i])
        assert np.all(alphas['L'][i] <= alphas['GP'][i]+1e-6)
    old_weights = {}
    for j, i in enumerate(key+d for key in KEYS[1:] for d in range(6)):
        old_weights[i] = np.minimum(alphas['L'][i],feather(masks[150+j]))*RAMP[j%6]
    zero = np.zeros((288,512),np.float32)
    prior_coeff = coefficients(alphas['L'][0],old_weights.get(0,zero),0,KEYS)
    previous = {name:bands(video[0]) for name,video in seq.items()}
    group_rows, cause_rows, frame_rows, core_rows = [], [], [], []
    for i in range(1,150):
        current = {name:bands(video[i]) for name,video in seq.items()}
        mx,my,common = source_match(seq['source'][i-1],seq['source'][i],previous['source'],
                                    current['source'],np.s_[:,:])
        valid = common & scope_masks['coral']
        aligned = {name:warp(b,mx,my) for name,b in previous.items()}
        source_delta = current['source'][...,0]-aligned['source'][...,0]
        errors = {name:np.abs(current[name][...,0]-aligned[name][...,0]-source_delta)
                  for name in ('raw','E','GP','L','U')}
        last_alpha = {name:warp(a[i-1],mx,my) for name,a in alphas.items()}
        held = {name:(a[i] >= .99) & (last_alpha[name] >= .99) for name,a in alphas.items()}
        groups = dict(both_held=held['GP'] & held['L'],
                      GP_held_L_not=held['GP'] & ~held['L'],
                      L_held_GP_not=held['L'] & ~held['GP'],
                      neither_held=~held['GP'] & ~held['L'])
        causes = dict(pixel_gate_rejected=~pixels[i], spatial_cell_collateral=pixels[i] & ~spatial[i],
                      temporal_group_collateral=spatial[i] & ~masks[i], L_active=masks[i])
        for partition, rows in ((groups,group_rows),(causes,cause_rows)):
            assert np.array_equal(sum(v.astype('u1') for v in partition.values()),np.ones((288,512),'u1'))
            for name, selector in partition.items():
                area = valid & selector
                row = dict(frame=i,timing=timing(i),group=name,pixels=int(area.sum()))
                row.update({method+'_sum':float(error[area].sum()) for method,error in errors.items()})
                rows.append(row)
        alpha = alphas['L'][i]
        coeff = coefficients(alpha,old_weights.get(i,zero),i,KEYS)
        tv = np.abs(coeff-warp(prior_coeff,mx,my)).sum(axis=2)/2
        alpha_delta = np.abs(alpha-last_alpha['L'])
        image_delta = np.abs(alpha-alphas['L'][i-1])
        prior_on = cv2.remap((alphas['L'][i-1]>0).astype('u1'),mx,my,cv2.INTER_NEAREST).astype(bool)
        on = alpha>0
        row = dict(frame=i,timing=timing(i),pixels=int(valid.sum()),
                   switch_pixels=int((valid & (on!=prior_on)).sum()),
                   image_coordinate_alpha_delta_sum=float(image_delta[valid].sum()),
                   aligned_alpha_delta_sum=float(alpha_delta[valid].sum()),
                   reference_identity_tv_sum=float(tv[valid].sum()))
        row.update({method+'_sum':float(error[valid].sum()) for method,error in errors.items()})
        frame_rows.append(row)
        if i not in REFRESH:
            interior = valid.copy()
            for name,a in alphas.items():
                here = cv2.distanceTransform((a[i]>=.99).astype('u1'),cv2.DIST_L2,5)>14
                before = cv2.distanceTransform((a[i-1]>=.99).astype('u1'),cv2.DIST_L2,5)>14
                interior &= here & (warp(before.astype(np.float32),mx,my)>=.999)
            row = dict(frame=i,pixels=int(interior.sum()))
            row.update({method+'_sum':float(error[interior].sum()) for method,error in errors.items()})
            core_rows.append(row)
        previous,prior_coeff = current,coeff
        if i%50 == 0:
            print('Retrospective correspondence',i,'/149',flush=True)
    save_csv(output/'support_groups.csv',group_rows)
    save_csv(output/'current_frame_causes.csv',cause_rows)
    save_csv(output/'temporal_frames.csv',frame_rows)
    save_csv(output/'common_interior.csv',core_rows)
    timing_summary = aggregate_saved(latest)
    expected = timing_summary['all']['S24_B4_T5_E']
    n = sum(r['pixels'] for r in frame_rows)
    assert n == expected['pixels'] == 1499191
    assert abs(sum(r['L_sum'] for r in frame_rows)/n-expected['coarse_mae']) < 1e-5
    def summarize(rows):
        total = sum(r['pixels'] for r in rows)
        result = dict(pairs=len(rows),pixels=total,pixel_fraction=total/n)
        for method in ('raw','E','GP','L','U'):
            result[method+'_mae'] = sum(r[method+'_sum'] for r in rows)/total if total else None
        result['L_minus_GP_sum'] = sum(r['L_sum']-r['GP_sum'] for r in rows)
        return result
    grouped = {name:summarize([r for r in group_rows if r['group']==name]) for name in groups}
    causal = {name:summarize([r for r in cause_rows if r['group']==name]) for name in causes}
    total_gap = sum(r['L_sum']-r['GP_sum'] for r in frame_rows)
    for rows in (grouped,causal):
        for values in rows.values():
            values['fraction_of_net_L_minus_GP_gap'] = values['L_minus_GP_sum']/total_gap
        assert abs(sum(v['pixel_fraction'] for v in rows.values())-1)<1e-9
        assert abs(sum(v['fraction_of_net_L_minus_GP_gap'] for v in rows.values())-1)<1e-6
    by_time = {}
    for name in ('anchor_refresh_with_exit','nonrefresh_group_boundary','nonrefresh_group_interior'):
        rr = [r for r in frame_rows if r['timing']==name]
        result = summarize(rr)
        for metric in ('image_coordinate_alpha_delta_sum','aligned_alpha_delta_sum','reference_identity_tv_sum','switch_pixels'):
            result[metric+'_per_pixel'] = sum(r[metric] for r in rr)/result['pixels']
        result['fraction_of_L_error'] = sum(r['L_sum'] for r in rr)/sum(r['L_sum'] for r in frame_rows)
        result['fraction_of_net_L_minus_GP_gap'] = result['L_minus_GP_sum']/total_gap
        by_time[name] = result
    save_json(output/'summary.json',dict(complete=True,analysis_only=True,new_candidate_rgb=False,
        rates=rates,coverage_chain=coverage,timing_from_archived_csv=timing_summary,
        observed_timing=by_time,held_groups=grouped,current_frame_rejections=causal,
        common_interior=summarize(core_rows),L_score_reproduced=True,
        warning='Conditional groups locate error; they are not causal fractions. Non-refresh still contains I/reset. Image-coordinate support holding is not material-coordinate tracking.',
        inputs_sha256={str(p):file_sha(p) for p in [*files.values(),packet,
            factorial/'gates/new_PG.npy',factorial/'gates/old_PG.npy',
            latest/'video1/evaluation_sparse/temporal.csv',folder/'receive/alpha.npy',
            factorial/'native_GP/alpha.npy']},code_sha256=file_sha(__file__)))
    shutil.copyfile(__file__,output/Path(__file__).name)
    print('TIMING',by_time,flush=True)
    print('GROUPS',grouped,flush=True)
    print('CAUSES',causal,flush=True)
    print('COVERAGE',coverage['new']['coral'],flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    run(args.workspace.resolve(),args.output.resolve())
