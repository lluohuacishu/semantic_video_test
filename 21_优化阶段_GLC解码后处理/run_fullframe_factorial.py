"""Frozen-motion, pixel/block gate interventions; source-assisted diagnostics only."""

import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np

from coral_reference_sidecar import file_sha, load_json, save_json
from compare_deflicker import save_csv
from fullframe_structure_stream import RAMP, expanded_motion, unpack
from temporal_filter import correspondence


H, W, T = 288, 512, 150
BOX = (342, 10, 502, 105)
Y, X = np.mgrid[:H, :W].astype(np.float32)
REASONS = ('outside_dense', 'cycle', 'source_low', 'source_mid',
           'motion_gradient', 'erosion', 'base_support', 'grid_error',
           'represented_photo', 'represented_outside', 'block_collateral')


def save_array(path, value):
    np.save(path, value, allow_pickle=False)


def blockify(mask):
    cells = mask.reshape(H//8, 8, W//8, 8).all(axis=(1, 3))
    return np.repeat(np.repeat(cells, 8, axis=0), 8, axis=1)


def feather(mask):
    return np.minimum(cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)/2., 1.)


def dense_reasons(source, i, key, back, forward):
    mx, my, inside, cycle = correspondence(back, forward)
    low = cv2.GaussianBlur(source[i].astype(np.float32), (0, 0), 3)
    old_low = cv2.GaussianBlur(source[key].astype(np.float32), (0, 0), 3)
    mid = cv2.GaussianBlur(source[i].astype(np.float32), (0, 0), 1)
    old_mid = cv2.GaussianBlur(source[key].astype(np.float32), (0, 0), 1)
    dlow = np.max(np.abs(low-cv2.remap(old_low, mx, my, cv2.INTER_LINEAR)), axis=2)
    dmid = np.max(np.abs(mid-cv2.remap(old_mid, mx, my, cv2.INTER_LINEAR)), axis=2)
    grad = np.maximum(np.linalg.norm(np.gradient(back, axis=0), axis=2),
                      np.linalg.norm(np.gradient(back, axis=1), axis=2))
    flags = [~inside, cycle >= 1.5, dlow >= 16, dmid >= 24, grad >= .75]
    initial = ~np.logical_or.reduce(flags)
    eroded = cv2.erode(initial.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    flags.append(initial & ~eroded)
    bits = np.zeros((H, W), np.uint16)
    for j, failed in enumerate(flags):
        bits |= failed.astype(np.uint16) << j
    return feather(eroded), bits


def pixel_gate(source, i, key, dense, represented, base, dense_bits):
    mx, my = X+represented[..., 0], Y+represented[..., 1]
    difference = np.linalg.norm(represented-dense, axis=2)
    current = cv2.GaussianBlur(source[i].astype(np.float32), (0, 0), 1)
    prior = cv2.GaussianBlur(source[key].astype(np.float32), (0, 0), 1)
    photo = np.max(np.abs(current-cv2.remap(prior, mx, my, cv2.INTER_LINEAR)), axis=2)
    checks = (base > 0, difference < 1.25, photo < 24,
              (mx >= 2) & (mx < 510) & (my >= 2) & (my < 286))
    safe = np.logical_and.reduce(checks)
    bits = dense_bits.copy()
    for j, passed in enumerate(checks, 6):
        bits |= (~passed).astype(np.uint16) << j
    block = blockify(safe)
    bits |= (safe & ~block).astype(np.uint16) << 10
    return safe, bits


def render(raw, new_motion, old_motion, new_safe, old_safe, keys, folder):
    folder.mkdir(exist_ok=False)
    output = np.lib.format.open_memmap(folder/'rgb.npy', mode='w+', dtype=np.uint8, shape=raw.shape)
    alpha = np.lib.format.open_memmap(folder/'alpha.npy', mode='w+', dtype=np.float32, shape=raw.shape[:3])
    old_weight = np.lib.format.open_memmap(folder/'old_weight.npy', mode='w+', dtype=np.float32, shape=raw.shape[:3])
    gap = np.lib.format.open_memmap(folder/'new_reference_gap.npy', mode='w+', dtype=np.float16, shape=raw.shape[:3])
    events = {k+j: n*6+j for n, k in enumerate(keys[1:]) for j in range(6)}

    def project(motion, safe, key):
        mx, my = X+motion[..., 0], Y+motion[..., 1]
        mask = safe & (mx >= 2) & (mx < 510) & (my >= 2) & (my < 286)
        a = feather(mask)
        rgb = cv2.remap(raw[key].astype(np.float32), mx, my, cv2.INTER_CUBIC)
        return rgb, a

    for i in range(T):
        key = keys[np.searchsorted(keys, i, side='right')-1]
        fresh, a = project(new_motion[i], new_safe[i], key)
        frame = np.rint(np.clip(raw[i]*(1-a[..., None])+fresh*a[..., None], 0, 255))
        weight = np.zeros_like(a)
        if i in events:
            j = events[i]
            old, oa = project(old_motion[j], old_safe[j], keys[j//6])
            weight = np.minimum(a, oa)*RAMP[j % 6]
            frame = np.rint(np.clip(frame+weight[..., None]*(old-fresh), 0, 255))
        output[i] = frame.astype(np.uint8)
        alpha[i] = a
        old_weight[i] = weight
        gap[i] = np.abs(cv2.cvtColor(fresh, cv2.COLOR_RGB2GRAY)
                       -cv2.cvtColor(raw[i].astype(np.float32), cv2.COLOR_RGB2GRAY))
        assert np.array_equal(output[i][a == 0], raw[i][a == 0])
        assert np.all(weight >= 0) and np.all(weight <= a)
    for value in (output, alpha, old_weight, gap):
        value.flush()
    return dict(outputs_sha256={p.name: file_sha(p) for p in folder.glob('*.npy')},
                source_assisted_offline=True, zero_weight_unchanged=True,
                weight_identity='base class plus actual anchor frame IDs; gap is uncorrected new-reference luminance difference')


def run(workspace, plan_path, output):
    plan = load_json(plan_path)
    assert plan['status'] == 'planned_not_run'
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(plan_path, output/'frozen_plan.json')
    for name, spec in plan['frozen_inputs'].items():
        if file_sha(Path(spec['path'])) != spec['sha256']:
            raise ValueError('Frozen input changed: '+name)
    inp = {k: Path(v['path']) for k, v in plan['frozen_inputs'].items()}
    source = np.load(inp['source_rgb'], mmap_mode='r')
    raw = np.load(inp['base_rgb'], mmap_mode='r')
    dense = np.load(inp['dense_new_motion'], mmap_mode='r')
    forward_path = inp['dense_new_motion'].with_name('from_anchor.npy')
    flow_record = load_json(forward_path.parent/'flow.json')
    assert file_sha(forward_path) == flow_record['flow_file_sha256']['from_anchor.npy']
    forward = np.load(forward_path, mmap_mode='r')
    transition = np.load(inp['dense_transition_motion'], mmap_mode='r')
    initial = np.load(inp['dense_alpha'], mmap_mode='r')
    shared = np.load(inp['dense_shared'], mmap_mode='r')
    header, arrays = unpack(inp['E_packet'].read_bytes(), plan['frozen_inputs']['base_stream']['sha256'])
    keys = header['keys']
    assert keys == plan['fixed']['keys'] and header['grid'] == [24, 14] and header['scale'] == 2
    events = [k+j for k in keys[1:] for j in range(6)]
    represented = {kind: np.stack([expanded_motion(q, header['scale']) for q in arrays[kind+'_motion']])
                   for kind in ('new', 'old')}
    gate_root = output/'gates'
    gate_root.mkdir()
    gates, rows = {}, []
    scopes = {'full': np.ones((H, W), bool), 'coral': np.zeros((H, W), bool)}
    x0, y0, x1, y1 = BOX
    scopes['coral'][y0:y1, x0:x1] = True
    source_checks = 0
    for kind, frame_ids in (('new', list(range(T))), ('old', events)):
        gates[kind] = {name: np.zeros((len(frame_ids), H, W), bool) for name in ('PD', 'PG', 'C', 'BD', 'BG', 'BC')}
        reason_arrays = {m: np.lib.format.open_memmap(gate_root/(kind+'_'+m+'_reasons.npy'),
                         mode='w+', dtype=np.uint16, shape=(len(frame_ids), H, W)) for m in ('D', 'G')}
        for j, i in enumerate(frame_ids):
            key = keys[np.searchsorted(keys, i, side='right')-1] if kind == 'new' else keys[j//6]
            d = dense[j] if kind == 'new' else transition[j, 0]
            f = forward[j] if kind == 'new' else transition[j, 1]
            a, bits = dense_reasons(source, i, key, d, f)
            support = initial[j] if kind == 'new' else shared[j]
            expected = a if kind == 'new' else np.minimum(initial[i], a)
            assert np.array_equal(expected, support), (kind, i, 'dense support reproduction')
            source_checks += 1
            for name, motion in (('D', d), ('G', represented[kind][j])):
                safe, reasons = pixel_gate(source, i, key, d, motion, support, bits)
                gates[kind]['P'+name][j] = safe
                gates[kind]['B'+name][j] = blockify(safe)
                reason_arrays[name][j] = reasons
                for scope, area in scopes.items():
                    alive = area.copy()
                    row = dict(frame=i, reference=key, stream=kind, motion=name, scope=scope,
                               pixels=int(area.sum()), pixel_safe=int((safe & area).sum()),
                               block_safe=int((blockify(safe) & area).sum()))
                    for bit, reason in enumerate(REASONS):
                        fail = (reasons & (1 << bit)) != 0
                        row[reason+'_overlapping'] = int((fail & area).sum())
                        row[reason+'_exclusive'] = int((fail & alive).sum())
                        alive &= ~fail
                    assert int(alive.sum()) == row['block_safe']
                    rows.append(row)
            gates[kind]['C'][j] = gates[kind]['PD'][j] & gates[kind]['PG'][j]
            gates[kind]['BC'][j] = blockify(gates[kind]['C'][j])
        transmitted = np.unpackbits(arrays[kind+'_safe'], axis=2).astype(bool)
        actual = gates[kind]['BG'].reshape(len(frame_ids), H//8, 8, W//8, 8).all(axis=(2, 4))
        assert np.array_equal(actual, transmitted), kind+' E safety blocks'
        for name, values in gates[kind].items():
            save_array(gate_root/(kind+'_'+name+'.npy'), values)
        for value in reason_arrays.values():
            value.flush()
        print(kind+' safety and source masks exactly reproduced', flush=True)
    save_csv(output/'gate_reasons.csv', rows)
    records = {}
    definitions = [('native_GB', 'G', 'BG'), ('native_DP', 'D', 'PD'), ('native_GP', 'G', 'PG'),
                   ('DP', 'D', 'C'), ('DB', 'D', 'BC'), ('GP', 'G', 'C'), ('GB', 'G', 'BC')]
    expected_rgb = np.load(inp['E_packet'].parent/'receive/rgb.npy', mmap_mode='r')
    expected_alpha = np.load(inp['E_packet'].parent/'receive/alpha.npy', mmap_mode='r')
    for name, motion, gate in definitions:
        duplicate = None
        for old_name, old_record in records.items():
            if (old_record['motion'] == motion and
                    all(np.array_equal(gates[k][gate], gates[k][old_record['gate']]) for k in ('new', 'old'))):
                duplicate = old_record['canonical']
                break
        folder = output/(duplicate or name)
        if duplicate is None:
            new = dense if motion == 'D' else represented['new']
            old = transition[:, 0] if motion == 'D' else represented['old']
            receipt = render(raw, new, old, gates['new'][gate], gates['old'][gate], keys, folder)
            save_json(folder/'receipt.json', receipt)
        records[name] = dict(motion=motion, gate=gate, canonical=duplicate or name,
                             alias_verified_by_exact_masks=duplicate is not None)
        if name == 'native_GB':
            assert np.array_equal(np.load(folder/'rgb.npy', mmap_mode='r'), expected_rgb)
            assert np.array_equal(np.load(folder/'alpha.npy', mmap_mode='r'), expected_alpha)
            print('native_GB RGB and alpha are bit-exact with E', flush=True)
        print(name+' -> '+records[name]['canonical'], flush=True)
    save_json(output/'render_summary.json', dict(complete=True, offline_source_assisted=True,
        actual_transmitted_candidate=False, E_rgb_alpha_bit_exact=True,
        E_blocks_bit_exact=True, dense_support_frames_bit_exact=source_checks,
        keys=keys, old_reference_ramp=RAMP, cases=records,
        gate_file_hashes={p.name: file_sha(p) for p in gate_root.glob('*.npy')},
        reason_bits=REASONS, input_hashes_checked=True,
        script_sha256=file_sha(__file__), plan_sha256=file_sha(plan_path)))
    shutil.copyfile(__file__, output/Path(__file__).name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    run(args.workspace.resolve(), args.plan.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
