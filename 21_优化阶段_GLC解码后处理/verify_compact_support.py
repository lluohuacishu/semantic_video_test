"""Independent readback of byte limits, masks, motion dependencies and receipts."""

import argparse
import json
from pathlib import Path
import shutil
import zlib

import cv2
import numpy as np

from compact_support_stream import (CRC, PREFIX, expand_support, pack, render, required_nodes, unpack)
from coral_reference_sidecar import file_sha, load_json, save_json
from experiment_common import array_sha
from fullframe_structure_stream import unpack as old_unpack
from run_compact_support import paths, safety


def altered_header(data, edit):
    magic, size = PREFIX.unpack_from(data)
    h = json.loads(data[PREFIX.size:PREFIX.size+size])
    edit(h)
    encoded = json.dumps(h, separators=(',', ':')).encode('ascii')
    body = PREFIX.pack(magic, len(encoded))+encoded+data[PREFIX.size+size:-CRC.size]
    return body+CRC.pack(zlib.crc32(body))


def verify(workspace, run):
    batch, prior = paths(workspace)
    selected = load_json(run/'selected.json')['profiles']
    report = {}
    hashed = {}
    for case, selection in selected.items():
        root = run/case
        digest = file_sha(batch/case/'base.glcsnd')
        base_bytes = (batch/case/'base.glcsnd').stat().st_size
        assert base_bytes <= 50000
        case_checks = {}
        for folder in sorted(root.iterdir()):
            packet = folder/'support.glccsp'
            if not packet.exists():
                continue
            send = load_json(folder/'send.json')
            assert packet.stat().st_size == send['bytes'] and file_sha(packet) == send['sha256']
            for filename, expected in send['inputs_sha256'].items():
                if filename not in hashed:
                    hashed[filename] = file_sha(Path(filename))
                assert hashed[filename] == expected
            assert packet.stat().st_size <= 50000
            h, a = unpack(packet.read_bytes(), digest)
            assert h == send['header']
            receive = folder/'receive/receipt.json'
            received = receive.exists()
            if received:
                r = load_json(receive)
                assert r['source_files_unavailable'] and r['independently_verified_base']
                assert r['aux_bytes'] == send['bytes'] and r['aux_sha256'] == send['sha256']
                assert r['rgb_array_sha256'] == send['sender_expected_rgb_sha256']
                assert r['alpha_array_sha256'] == send['sender_expected_alpha_sha256']
                for name, expected in r['outputs_sha256'].items():
                    assert file_sha(folder/'receive'/name) == expected
                assert len(r['reads']) == 5
            case_checks[folder.name] = dict(bytes=send['bytes'], readback_pass=True,
                independent_receive=received, safety_violations=send['unsafe_pixels'])
        source = np.load(batch/case/'source.npy', mmap_mode='r')
        raw = np.load(batch/case/'base_receive/decoded_rgb.npy', mmap_mode='r')
        _, old = old_unpack((prior/case/'g24q2_E/fullframe.glcfull').read_bytes(), digest)
        original = np.concatenate([old['new_motion'], old['old_motion']])
        selection_checks = {}
        for fusion, choice in selection.items():
            folder = root/choice['name']
            packet = (folder/'support.glccsp').read_bytes()
            h, a = unpack(packet, digest)
            assert len(packet) == choice['bytes'] < base_bytes
            full_motion = np.stack([np.rint(cv2.resize(m.astype(np.float32), tuple(h['grid']),
                                    interpolation=cv2.INTER_LINEAR)).astype('<i2') for m in original])
            safe = safety(source, prior/case, full_motion)
            cells = expand_support(a['support'], h['group_span'])
            mask = np.repeat(np.repeat(cells, h['mask_step'], 1), h['mask_step'], 2).astype(bool)
            assert not np.any(mask & ~safe)
            needed = required_nodes(a['support'], h['group_span'], h['mask_step'], tuple(h['grid']), fusion)
            assert np.array_equal(a['motion'][needed], full_motion[needed])
            expected, expected_alpha = render(raw, h, dict(motion=full_motion, support=a['support']))
            received = np.load(folder/'receive/rgb.npy', mmap_mode='r')
            alpha = np.load(folder/'receive/alpha.npy', mmap_mode='r')
            assert np.array_equal(expected, received) and np.array_equal(expected_alpha, alpha)
            assert np.array_equal(received[alpha == 0], raw[alpha == 0])
            tests = {}
            edits = dict(grid=lambda header: header.update(grid=[25, 14]),
                         schedule=lambda header: header.update(keys=[0, 30]),
                         source_binding=lambda header: header.update(base_sha256='0'*64),
                         support_length=lambda header: header['sections']['support'].update(raw_bytes=1),
                         sparse_length=lambda header: header['sections']['motion'].update(raw_bytes=0))
            for label, edit in edits.items():
                try:
                    unpack(altered_header(packet, edit), digest)
                except ValueError:
                    tests[label] = True
                else:
                    raise ValueError('Malformed CRC-valid header accepted: '+label)
            hevc = load_json(root/'hevc/result.json')['controls']['h265_'+fusion]
            assert 0 < hevc['bytes'] <= base_bytes+len(packet) == hevc['target_bytes']
            # CRF is discrete: require near-full use while never exceeding the budget.
            assert hevc['bytes']/hevc['target_bytes'] > .98
            assert file_sha(root/'hevc'/('h265_'+fusion+'.mp4')) == hevc['sha256']
            assert array_sha(np.load(root/'hevc'/('h265_'+fusion+'.npy'), mmap_mode='r')) == hevc['rgb_sha256']
            selection_checks[fusion] = dict(bytes=len(packet), below_base=True, unsafe_pixels=0,
                required_motion_exact=True, unpruned_rgb_alpha_exact=True, zero_weight_unchanged=True,
                malformed_header_checks=tests, hevc_bytes=hevc['bytes'], hevc_target=hevc['target_bytes'])
        report[case] = dict(files=case_checks, selected=selection_checks)
        print(case, 'complete readback PASS', flush=True)
    final = load_json(run/'final_summary.json')
    assert final['complete']
    for case, previews in final['preview_checks'].items():
        for name, check in previews.items():
            assert file_sha(run/case/'final_evaluation'/name) == check['sha256']
    save_json(run/'verification.json', dict(complete=True, cases=report, input_files_verified=len(hashed),
              source_assistance_only_at_sender_and_evaluation=True, all_selected_aux_below_base=True,
              every_file_under_40kbps=True, malformed_crc_valid_headers_rejected=True,
              previews_verified=True, visual_review='pending', code_sha256=file_sha(__file__)))
    snap = run/'source_final'
    snap.mkdir(exist_ok=False)
    names = ['compact_support_stream.py', 'run_compact_support.py', 'evaluate_compact_support.py',
             'finish_compact_support.py', 'verify_compact_support.py', 'probe_compact_support.py',
             'probe_support_motion_fill.py', 'fullframe_structure_stream.py', 'run_fullframe_factorial.py',
             'cache_hevc_baseline.py', 'evaluate_fullframe_reference.py', 'audit_deflicker_texture.py',
             'cache_motion_inputs.py', 'run_p0.py', 'verify_video1_persistent_structure.py']
    for name in names:
        shutil.copyfile(Path(__file__).with_name(name), snap/name)
    save_json(snap/'manifest.json', {name:file_sha(snap/name) for name in names})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    verify(args.workspace.resolve(), args.run.resolve())
