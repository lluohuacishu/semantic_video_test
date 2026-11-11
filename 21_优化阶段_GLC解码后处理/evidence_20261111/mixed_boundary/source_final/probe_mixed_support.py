"""Frozen finite byte probe: sparse safe boundary refinements near K."""
import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np

import compact_support_stream as base
import predictive_support_stream as previous
import mixed_support_stream as codec
from coral_reference_sidecar import file_sha, load_json, save_json
from fullframe_structure_stream import expanded_motion, unpack as unpack_full
from run_compact_support import paths

FRACTIONS = (0.01, 0.03, 0.06, 0.12)


def priority(raw, motion, coarse, fine):
    """Estimate usable structural correction without source RGB or ROI labels."""
    codes = codec.codes_from_support(fine)
    assert np.array_equal(codes == 15, coarse)
    adjacent = np.stack([cv2.dilate(m.astype('u1'), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], 'u1')).astype(bool) for m in coarse])
    eligible = (codes > 0) & (codes < 15) & adjacent
    gray = np.stack([cv2.GaussianBlur(cv2.cvtColor(f.astype(np.float32), cv2.COLOR_RGB2GRAY), (0, 0), 3) for f in raw])
    yy, xx = np.mgrid[:288, :512].astype(np.float32)
    frame_ids = list(range(150)) + [k + j for k in base.KEYS[1:] for j in range(6)]
    scores = np.zeros(coarse.shape, np.float32)
    for g, (start, end) in enumerate(base.groups(5)):
        kmask = np.repeat(np.repeat(coarse[g], 4, 0), 4, 1)
        nmask = np.repeat(np.repeat(fine[g], 2, 0), 2, 1)
        ka = np.minimum(cv2.distanceTransform(kmask.astype('u1'), cv2.DIST_L2, 5) / 2, 1)
        na = np.minimum(cv2.distanceTransform(nmask.astype('u1'), cv2.DIST_L2, 5) / 2, 1)
        for j in range(start, end):
            i = frame_ids[j]
            key = (base.KEYS[np.searchsorted(base.KEYS, i, side='right') - 1]
                   if j < 150 else base.KEYS[(j - 150) // 6])
            flow = expanded_motion(motion[j], 2)
            reference = cv2.remap(gray[key], xx + flow[..., 0], yy + flow[..., 1], cv2.INTER_CUBIC)
            value = (na - ka) * np.abs(reference - gray[i])
            scores[g] += value.reshape(72, 4, 128, 4).mean(axis=(1, 3)) / (end - start)
    scores[~eligible] = 0
    return codes, eligible, scores


def run(workspace, output):
    output.mkdir(exist_ok=False)
    plan = dict(date='2026-11-11', status='frozen_before_probe', cases=['video1', 'video2', 'video3'],
        fractions=list(FRACTIONS), max_render_candidates=2,
        selection='For each group select the highest scoring 1/3/6/12 percent of mixed 4x4 parents adjoining K. Score is the mean potential alpha gain times absolute sigma3 luminance correction to the same decoded reference. No source RGB, manual ROI or evaluation metrics enter ranking.',
        fixed='K motion grid/values, keys, old-reference transition, 2px feather, five-frame intersection; additions are a subset of already verified N safe support.',
        coding='Existing lossless motion candidates plus finite coarse-bitmask and sparse child-nibble layouts, all header and checksum bytes counted.',
        budget='K actual bytes are the comparison point, not a new hard cap. Report every byte increase separately; 40 kbps absolute cap; below base preferred.',
        decision='Inspect byte frontier first. Render at most two globally consistent fractions. Stop if visible benefit does not justify added bytes. No promotion without human original-speed review.',
        source_inputs_unchanged=True, realtime_validated=False)
    save_json(output / 'plan.json', plan)
    snapshot = output / 'source_initial'; snapshot.mkdir()
    for name in ('probe_mixed_support.py', 'mixed_support_stream.py', 'predictive_support_stream.py', 'compact_support_stream.py', 'fullframe_structure_stream.py'):
        shutil.copyfile(Path(__file__).with_name(name), snapshot / name)
    save_json(snapshot / 'manifest.json', {p.name: file_sha(p) for p in snapshot.iterdir()})
    batch, prior = paths(workspace)
    previous_run = output.parent / 'support_refinement_20261111_01'
    results = {}
    for case in plan['cases']:
        folder = output / case; folder.mkdir()
        digest = file_sha(batch / case / 'base.glcsnd')
        old = previous_run / case
        kh, ka = previous.unpack((old / 'K/support.glccsp2').read_bytes(), digest)
        _, na = previous.unpack((old / 'N/support.glccsp2').read_bytes(), digest)
        _, full = unpack_full((prior / case / 'g24q2_E/fullframe.glcfull').read_bytes(), digest)
        frozen = np.concatenate([full['new_motion'], full['old_motion']])
        motion = np.stack([np.rint(cv2.resize(q.astype(np.float32), tuple(kh['grid']), interpolation=cv2.INTER_LINEAR)).astype('<i2') for q in frozen])
        raw = np.load(batch / case / 'base_receive/decoded_rgb.npy', mmap_mode='r')
        codes, eligible, scores = priority(raw, motion, ka['support'], na['support'])
        np.save(folder / 'priority.npy', scores, allow_pickle=False)
        coarse_codes = ka['support'].astype('u1') * 15
        records = {}
        for fraction in FRACTIONS:
            name = 'M' + str(round(fraction * 100)).zfill(2)
            selected = np.zeros(eligible.shape, bool)
            for g, mask in enumerate(eligible):
                ids = np.flatnonzero(mask)
                count = int(np.ceil(len(ids) * fraction))
                order = np.argsort(-scores[g].reshape(-1)[ids], kind='stable')
                selected[g].reshape(-1)[ids[order[:count]]] = True
            mixed = coarse_codes.copy(); mixed[selected] = codes[selected]
            support = codec.support_from_codes(mixed)
            kfine = np.repeat(np.repeat(ka['support'], 2, 1), 2, 2)
            assert np.all(support[kfine]) and not np.any(support & ~na['support'])
            target = folder / name; target.mkdir()
            payload, h, probe = codec.pack(motion, support, digest)
            (target / 'support.glccsp3').write_bytes(payload)
            dh, da = codec.unpack(payload, digest)
            needed = base.required_nodes(support, 5, 2, tuple(kh['grid']), 'E')
            assert np.array_equal(da['support'], support)
            assert np.array_equal(da['motion'][needed], motion[needed])
            for packed, spec in codec.mixed_candidates(support):
                import lzma
                spec = dict(spec, bytes=len(packed))
                assert np.array_equal(codec.decode_mixed(lzma.decompress(packed), spec, support.shape), support)
            baseline_bytes = (old / 'K/support.glccsp2').stat().st_size
            record = dict(bytes=len(payload), kbps=len(payload)*8/10000, delta_K_bytes=len(payload)-baseline_bytes,
                delta_K_percent=(len(payload)/baseline_bytes-1)*100,
                base_bytes=(batch / case / 'base.glcsnd').stat().st_size, K_bytes=baseline_bytes,
                selected_parent_groups=int(selected.sum()), eligible_parent_groups=int(eligible.sum()),
                added_2x2_cell_groups=int(support.sum()-kfine.sum()), fraction=fraction,
                safe_subset_of_N=True, preserves_all_K_support=True, required_motion_exact=True,
                every_mixed_layout_roundtrip=True, header=h, probe=probe, sha256=file_sha(target / 'support.glccsp3'))
            save_json(target / 'probe.json', record)
            records[name] = record
            print(case, name, len(payload), 'B', record['kbps'], 'kbps', 'delta_K', record['delta_K_bytes'], 'codec', h['sections']['support']['codec'], flush=True)
        results[case] = records
        save_json(output / 'progress.json', results)
    save_json(output / 'summary.json', dict(complete=True, byte_probe_only=True, cases=results,
        no_rendering_or_quality_claim=True, code_sha256={p.name:file_sha(p) for p in snapshot.iterdir() if p.suffix=='.py'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    run(args.workspace.resolve(), args.output.resolve())
