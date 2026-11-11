"""Observe support, reference identities, rejection causes and diagnostic mask size."""

import argparse
import lzma
from pathlib import Path
import shutil

import cv2
import numpy as np

from audit_deflicker_texture import normalized, source_match
from coral_reference_sidecar import file_sha, load_json, save_json
from compare_deflicker import save_csv
from evaluate_fullframe_factorial import sequences
from fullframe_structure_stream import expanded_motion, unpack as unpack_full
from persistent_structure_stream import CROP, decode_masks, unpack as unpack_local
from probe_reef import bands
from run_fullframe_factorial import BOX, H, W, T, X, Y, REASONS
from temporal_filter import warp


def baseline_states(workspace, run, records, prior):
    root = workspace/'02_实验数据/优化_GLC后处理'
    local = root/'video1_persistent_aux_20261110_01/g17q4'
    states = {}
    for name in ('DP', 'DB', 'GP', 'GB', 'native_DP'):
        folder = run/records['cases'][name]['canonical']
        states[name] = tuple(np.load(folder/(n+'.npy'), mmap_mode='r')
                             for n in ('alpha', 'old_weight', 'new_reference_gap'))
    keys = records['keys']
    events = [(k+j, n*6+j, keys[n]) for n, k in enumerate(keys[1:]) for j in range(6)]
    base_sha = load_json(run/'frozen_plan.json')['frozen_inputs']['base_stream']['sha256']
    state_dir = run/'diagnostics/baseline_states'
    state_dir.mkdir()
    for name, path in (('U', local/'receive/alpha.npy'), ('dense', prior/'offline/alpha.npy'),
                       ('F', prior/'g24q2_F/receive/alpha.npy')):
        alpha = np.load(path, mmap_mode='r')
        old_weight = np.lib.format.open_memmap(state_dir/(name+'_old_weight.npy'),
                                              mode='w+', dtype=np.float32, shape=(T, H, W))
        old_weight[:] = 0
        if name == 'dense':
            shared = np.load(prior/'offline/shared.npy', mmap_mode='r')
            for i, j, _ in events:
                old_weight[i] = shared[j]*records['old_reference_ramp'][j % 6]
        elif name == 'F':
            header, arrays = unpack_full((prior/'g24q2_F/fullframe.glcfull').read_bytes(), base_sha)
            masks = np.unpackbits(arrays['old_safe'], axis=2)
            for i, j, _ in events:
                f = expanded_motion(arrays['old_motion'][j], header['scale'])
                mx, my = X+f[..., 0], Y+f[..., 1]
                safe = cv2.resize(masks[j], (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                safe &= (mx >= 2) & (mx < 510) & (my >= 2) & (my < 286)
                a = np.minimum(cv2.distanceTransform(safe.astype(np.uint8), cv2.DIST_L2, 5)/8., 1.)
                old_weight[i] = np.minimum(a, alpha[i])*records['old_reference_ramp'][j % 6]
        else:
            header, arrays = unpack_local((local/'structure.glcstruct').read_bytes(), base_sha)
            masks = decode_masks(arrays['contours'])
            fish = np.unpackbits(arrays['fish'], axis=2)
            x0, y0, x1, y1 = CROP
            for i, j, key in events:
                f = cv2.resize(arrays['old_motion'][j].astype(np.float32)/header['motion_scale'],
                               (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR)
                mx, my = X[y0:y1, x0:x1]+f[..., 0], Y[y0:y1, x0:x1]+f[..., 1]
                safe = masks[150+j].astype(bool) & (mx >= 2) & (mx < 510) & (my >= 2) & (my < 286)
                safe &= ~fish[i, y0:y1, x0:x1].astype(bool)
                safe &= ~cv2.remap(fish[key], mx, my, cv2.INTER_NEAREST).astype(bool)
                a = np.minimum(cv2.distanceTransform(safe.astype(np.uint8), cv2.DIST_L2, 5)/2., 1.)
                old_weight[i, y0:y1, x0:x1] = np.minimum(a, alpha[i, y0:y1, x0:x1])*records['old_reference_ramp'][j % 6]
        old_weight.flush()
        assert np.all(old_weight >= 0) and np.all(old_weight <= alpha)
        states[name] = (alpha, old_weight, None)
    return states


def coefficients(alpha, old_weight, frame, keys):
    result = np.zeros((H, W, len(keys)+1), np.float32)
    result[..., 0] = 1-alpha
    index = np.searchsorted(keys, frame, side='right')-1
    result[..., index+1] = alpha-old_weight
    if index:
        result[..., index] = old_weight
    assert np.allclose(result.sum(axis=2), 1, atol=1e-6)
    return result


def diagnose(workspace, run):
    out = run/'diagnostics'
    out.mkdir(exist_ok=False)
    seq, records, prior = sequences(workspace, run)
    states = baseline_states(workspace, run, records, prior)
    keys = records['keys']
    refresh = {k+j for k in keys[1:] for j in range(7)}
    x0, y0, x1, y1 = BOX
    region = np.zeros((H, W), bool)
    region[y0:y1, x0:x1] = True
    previous = {k: bands(v[0]) for k, v in seq.items()}
    previous_coeff = {k: coefficients(a[0], w[0], 0, keys) for k, (a, w, _) in states.items()}
    rows, interior_rows, rejection_rows, alpha_samples = [], [], [], {}
    for name in states:
        alpha_samples[name] = {'all': [], 'non_refresh': [], 'refresh_with_exit': []}
    reasons = np.load(run/'gates/new_G_reasons.npy', mmap_mode='r')
    for i in range(1, T):
        current = {k: bands(v[i]) for k, v in seq.items()}
        mx, my, common = source_match(seq['source'][i-1], seq['source'][i],
                                      previous['source'], current['source'], np.s_[:, :])
        valid = common & region
        n = int(valid.sum())
        timing = 'refresh_with_exit' if i in refresh else 'non_refresh'
        aligned = {k: warp(v, mx, my) for k, v in previous.items()}
        source_delta = current['source']-aligned['source']
        held = {}
        interiors = []
        for name, (a, old, gap) in states.items():
            last = warp(a[i-1], mx, my)
            last_on = cv2.remap((a[i-1] > 0).astype(np.uint8), mx, my, cv2.INTER_NEAREST).astype(bool)
            on = a[i] > 0
            always = valid & on & last_on
            delta = np.abs(a[i]-last)
            coeff = coefficients(a[i], old[i], i, keys)
            coeff_delta = np.abs(coeff-warp(previous_coeff[name], mx, my)).sum(axis=2)/2
            held[name] = (a[i] >= .99) & (last >= .99)
            sample = delta[always].copy()
            alpha_samples[name]['all'].append(sample)
            alpha_samples[name][timing].append(sample)
            rows.append(dict(frame=i, timing=timing, method=name, pixels=n,
                enter_pixels=int((valid & on & ~last_on).sum()), exit_pixels=int((valid & ~on & last_on).sum()),
                held_pixels=int((valid & held[name]).sum()), always_on_pixels=int(always.sum()),
                alpha_delta_sum=float(delta[valid].sum()), always_on_alpha_delta_sum=float(sample.sum()),
                source_identity_tv_sum=float(coeff_delta[valid].sum()),
                alpha_gap_risk_sum=None if gap is None else float((delta*gap[i].astype(np.float32))[valid].sum())))
            previous_coeff[name] = coeff
            if name in ('U', 'DP', 'DB', 'GP', 'GB'):
                # 12px coarse filter neighborhood plus a 2px interpolation margin.
                here = cv2.distanceTransform((a[i] >= .99).astype(np.uint8), cv2.DIST_L2, 5) > 14
                prior_core = cv2.distanceTransform((a[i-1] >= .99).astype(np.uint8), cv2.DIST_L2, 5) > 14
                interiors.append(here & (warp(prior_core.astype(np.float32), mx, my) >= .999))
        if i not in refresh:
            core = valid & np.logical_and.reduce(interiors)
            count = int(core.sum())
            if count:
                for name in seq:
                    a, b = aligned[name][core], current[name][core]
                    e = np.abs(b-a-source_delta[core])
                    interior_rows.append(dict(frame=i, method=name, pixels=count,
                        coarse_sum=float(e[:, 0].sum()),
                        fine_ncc=float((normalized(a[:, 2:3])*normalized(b[:, 2:3])).mean())))
        groups = {'all': valid, 'U_held_GB_not': valid & held['U'] & ~held['GB']}
        for group, mask in groups.items():
            alive = mask.copy()
            rr = dict(frame=i, group=group, pixels=int(mask.sum()))
            for bit, reason in enumerate(REASONS):
                fail = (reasons[i] & (1 << bit)) != 0
                rr[reason+'_overlapping'] = int((fail & mask).sum())
                rr[reason+'_exclusive'] = int((fail & alive).sum())
                alive &= ~fail
            rr['block_safe'] = int(alive.sum())
            rejection_rows.append(rr)
        previous = current
        if i % 50 == 0:
            print(f'State diagnostics {i}/149', flush=True)
    support = {}
    for timing in ('all', 'non_refresh', 'refresh_with_exit'):
        support[timing] = {}
        for name in states:
            chosen = [r for r in rows if r['method'] == name and (timing == 'all' or r['timing'] == timing)]
            n = sum(r['pixels'] for r in chosen)
            on = sum(r['always_on_pixels'] for r in chosen)
            samples = np.concatenate(alpha_samples[name][timing])
            support[timing][name] = dict(pixels=n, always_on_pixels=on,
                enter_fraction=sum(r['enter_pixels'] for r in chosen)/n,
                exit_fraction=sum(r['exit_pixels'] for r in chosen)/n,
                switch_fraction=sum(r['enter_pixels']+r['exit_pixels'] for r in chosen)/n,
                held_fraction=sum(r['held_pixels'] for r in chosen)/n,
                mean_alpha_delta=sum(r['alpha_delta_sum'] for r in chosen)/n,
                always_on_mean_alpha_delta=sum(r['always_on_alpha_delta_sum'] for r in chosen)/on,
                always_on_p95_alpha_delta=float(np.percentile(samples, 95)),
                source_identity_mean_tv=sum(r['source_identity_tv_sum'] for r in chosen)/n,
                mean_alpha_gap_risk=None if chosen[0]['alpha_gap_risk_sum'] is None else sum(r['alpha_gap_risk_sum'] for r in chosen)/n)
    interior = {}
    for name in seq:
        chosen = [r for r in interior_rows if r['method'] == name]
        n = sum(r['pixels'] for r in chosen)
        interior[name] = dict(pairs=len(chosen), valid_pixels=n,
            coarse_mae=None if not n else sum(r['coarse_sum'] for r in chosen)/n,
            fine_ncc=None if not n else sum(r['fine_ncc']*r['pixels'] for r in chosen)/n)
    causes = {}
    for group in ('all', 'U_held_GB_not'):
        chosen = [r for r in rejection_rows if r['group'] == group]
        causes[group] = {k: sum(r[k] for r in chosen) for k in chosen[0] if k not in ('frame', 'group')}
    # Byte probe only: no new receiver format, no end-to-end packet claim.
    plan = load_json(run/'frozen_plan.json')
    h, _ = unpack_full(Path(plan['frozen_inputs']['E_packet']['path']).read_bytes(),
                       plan['frozen_inputs']['base_stream']['sha256'])
    motion_bytes = sum(h['sections'][k]['bytes'] for k in ('new_motion', 'old_motion'))
    mask_bytes = {}
    for kind in ('new', 'old'):
        mask = np.load(run/'gates'/(kind+'_PG.npy'), mmap_mode='r')
        packed = np.packbits(mask, axis=2)
        payload = lzma.compress(packed.tobytes())
        (out/(kind+'_pixel_mask.bits.xz')).write_bytes(payload)
        decoded = np.frombuffer(lzma.decompress(payload), np.uint8).reshape(packed.shape)
        assert np.array_equal(decoded, packed)
        mask_bytes[kind] = len(payload)
    byte_probe = dict(existing_motion_sections_bytes=motion_bytes, pixel_mask_sections_bytes=mask_bytes,
        payload_bytes_excluding_header=motion_bytes+sum(mask_bytes.values()), lane_limit_bytes=50000,
        over_budget_before_header=motion_bytes+sum(mask_bytes.values()) > 50000,
        representation='Full-resolution bitpacked pixel safety, independent LZMA sections; no new codec/packet implemented',
        roundtrip_exact=True)
    old_diagnosis = load_json(workspace/'01_项目/代码/semantic_video_test/21_优化阶段_GLC解码后处理/evidence_20261110/strategy_diagnosis/summary.json')
    for name, old_name in (('U', 'U'), ('GB', 'E'), ('F', 'F'), ('dense', 'dense_offline')):
        expected = old_diagnosis['motion_compensated_support']['all'][old_name]
        assert abs(support['all'][name]['switch_fraction']-expected['switch_fraction']) < 1e-9
        assert abs(support['all'][name]['mean_alpha_delta']-expected['mean_alpha_change']) < 1e-6
    save_csv(out/'support.csv', rows)
    save_csv(out/'common_interior.csv', interior_rows)
    save_csv(out/'rejection_on_evaluation_support.csv', rejection_rows)
    result = dict(complete=True, support=support, common_interior=interior,
        interior_definition='Non-refresh pairs; U/DP/DB/GP/GB alpha>=0.99 in both frames, >14px inside each full-weight support, shared source textured mask.',
        rejection_causes=causes, pixel_mask_byte_probe=byte_probe,
        caveats=['Conditional always-on denominators vary by method.', 'Weight identity is base class plus actual anchor frame IDs; color correction is not represented.',
                 'Alpha*reference discrepancy is a risk proxy, not additive causal attribution.', 'Byte probe is not an interoperable independent receiver file.'],
        existing_support_statistics_reproduced=True, code_sha256=file_sha(__file__))
    save_json(out/'summary.json', result)
    shutil.copyfile(__file__, run/Path(__file__).name)
    print('SUPPORT '+str({k: (round(v['switch_fraction'], 5), round(v['mean_alpha_delta'], 5),
                              round(v['always_on_mean_alpha_delta'], 5)) for k, v in support['all'].items()}), flush=True)
    print('INTERIOR '+str(interior), flush=True)
    print('BYTES '+str(byte_probe), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    diagnose(args.workspace.resolve(), args.run.resolve())


if __name__ == '__main__':
    main()
