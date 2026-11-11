"""Validate actual refined-support files, score all three clips and make previews."""

import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np

import compact_support_stream as old
import predictive_support_stream as codec
from cache_hevc_baseline import encode_controls
from coral_reference_sidecar import load_json,save_json,file_sha
from compare_deflicker import save_csv
from experiment_common import array_sha
from fullframe_structure_stream import unpack as unpack_full
import evaluate_fullframe_reference as scoring
from run_compact_support import paths,safety
from run_dense_batch import CASES
from run_fullframe_factorial import BOX
from run_p0 import preview
from verify_compact_support import altered_header
from verify_video1_persistent_structure import inspect_preview


def verify(workspace,run):
    batch,prior=paths(workspace)
    previous=run.parent/'compact_support_20261110_01'
    selection=load_json(previous/'selected.json')['profiles']
    records={}
    rng=np.random.default_rng(20261111)
    needed=rng.random((180,3,4))>.3
    needed[12:16]=False
    values=rng.integers(-200,201,(180,3,4,2),dtype=np.int16)
    predictor_checks={}
    for mode in codec.PREDICTORS:
        residual=codec.encode_sparse(values,needed,mode)
        decoded=codec.decode_sparse(residual,needed,mode)
        assert np.array_equal(decoded[needed],values[needed])
        assert not decoded[~needed].any()
        predictor_checks[mode]=True
    for case in CASES:
        base=batch/case
        digest=file_sha(base/'base.glcsnd')
        old_dir=previous/case/selection[case]['E']['name']
        raw=np.load(base/'base_receive/decoded_rgb.npy',mmap_mode='r')
        source=np.load(base/'source.npy',mmap_mode='r')
        _,a=unpack_full((prior/case/'g24q2_E/fullframe.glcfull').read_bytes(),digest)
        frozen=np.concatenate([a['new_motion'],a['old_motion']])
        grid=load_json(run/case/'K/send.json')['header']['grid']
        motion=np.stack([np.rint(cv2.resize(q.astype(np.float32),tuple(grid),interpolation=cv2.INTER_LINEAR)).astype('<i2') for q in frozen])
        safe=safety(source,prior/case,motion)
        records[case]={}
        for name in ('K','N'):
            folder=run/case/name
            send=load_json(folder/'send.json')
            recv=load_json(folder/'receive/receipt.json')
            packet=folder/'support.glccsp2'
            data=packet.read_bytes()
            assert len(data)==send['bytes']==recv['aux_bytes']<=50000
            assert file_sha(packet)==send['sha256']==recv['aux_sha256']
            assert send['expected_rgb_sha256']==recv['rgb_sha256']
            assert send['expected_alpha_sha256']==recv['alpha_sha256']
            assert recv['source_files_unavailable'] and recv['independently_verified_base']
            assert len(recv['reads'])==5 and len(recv['blocked_inputs'])==4
            for p,d in recv['outputs_sha256'].items():assert file_sha(folder/'receive'/p)==d
            for p,d in send['input_sha256'].items():assert file_sha(Path(p))==d
            h,a=codec.unpack(data,digest)
            assert h==send['header']
            support=old.reduce_support(safe,h['mask_step'],5)
            assert np.array_equal(support,a['support'])
            needed=old.required_nodes(support,5,h['mask_step'],tuple(grid),'E')
            assert np.array_equal(a['motion'][needed],motion[needed])
            masks=old.expand_support(support,5)
            masks=np.repeat(np.repeat(masks,h['mask_step'],1),h['mask_step'],2).astype(bool)
            assert not np.any(masks & ~safe)
            expected,alpha=old.render(raw,h,dict(motion=motion,support=support))
            actual=np.load(folder/'receive/rgb.npy',mmap_mode='r')
            actual_alpha=np.load(folder/'receive/alpha.npy',mmap_mode='r')
            assert np.array_equal(actual,expected) and np.array_equal(actual_alpha,alpha)
            assert np.array_equal(actual[alpha==0],raw[alpha==0])
            if name=='K':
                assert np.array_equal(actual,np.load(old_dir/'receive/rgb.npy',mmap_mode='r'))
                assert np.array_equal(alpha,np.load(old_dir/'receive/alpha.npy',mmap_mode='r'))
            # Exercise every support layout, including unused hierarchical alternatives.
            import lzma
            for payload,spec in codec.support_candidates(support):
                assert np.array_equal(codec.decode_support(lzma.decompress(payload),spec,support.shape),support)
            damaged=bytearray(data);damaged[-8]^=1
            bad=[('CRC',damaged,digest),('truncated',data[:-1],digest),
                 ('wrong_base',data,'0'*64),('over_cap',data+b'0'*50000,digest)]
            edits=dict(grid=lambda h:h.update(grid=[25,14]),schedule=lambda h:h.update(keys=[0,30]),
                       support_length=lambda h:h['sections']['support'].update(raw_bytes=1),
                       motion_length=lambda h:h['sections']['motion'].update(raw_bytes=0),
                       predictor=lambda h:h['sections']['motion'].update(codec='unknown'),
                       group_span=lambda h:h.update(group_span=3))
            bad.extend((label,altered_header(data,edit),digest) for label,edit in edits.items())
            tests={}
            for label,packet_bytes,binding in bad:
                try:codec.unpack(packet_bytes,binding)
                except ValueError:tests[label]=True
                else:raise ValueError('Malformed packet accepted: '+label)
            records[case][name]=dict(bytes=len(data),kbps=len(data)*8/10000,
                safety_violations=0,needed_motion_exact=True,unpruned_RGB_alpha_exact=True,
                old_L_RGB_alpha_exact=name=='K',independent_receiver=True,
                negative_packet_checks=tests,all_support_layouts_exact=True)
        print(case,'roundtrip/safety/standalone verification PASS',flush=True)
    save_json(run/'verification.json',dict(complete=True,cases=records,
        missing_node_and_anchor_reset_predictor_checks=predictor_checks,
        no_source_information_at_receiver=True,all_auxiliary_under_40kbps=True))


def evaluate(workspace,run,ffmpeg):
    batch,_=paths(workspace)
    previous=run.parent/'compact_support_20261110_01'
    selection=load_json(previous/'selected.json')['profiles']
    all_scores={};all_previews={}
    for case in CASES:
        out=run/case/'evaluation'
        out.mkdir(exist_ok=False)
        base=batch/case
        seq=dict(source=np.load(base/'source.npy',mmap_mode='r'),
                 raw=np.load(base/'base_receive/decoded_rgb.npy',mmap_mode='r'))
        for name in ('K','N'):
            seq[name]=np.load(run/case/name/'receive/rgb.npy',mmap_mode='r')
        seq['L']=np.load(previous/case/selection[case]['E']['name']/'receive/rgb.npy',mmap_mode='r')
        assert np.array_equal(seq['L'],seq['K'])
        rates={name:load_json(run/case/name/'send.json') for name in ('K','N')}
        budgets={'h265_'+k:v['base_bytes']+v['bytes'] for k,v in rates.items()}
        hevc_dir=run/case/'hevc'
        encode_controls(ffmpeg,hevc_dir,CASES[case][0],seq['source'],budgets)
        hr=load_json(hevc_dir/'result.json')['controls']
        for name in ('K','N'):
            seq['H'+name]=np.load(hevc_dir/('h265_'+name+'.npy'),mmap_mode='r')
            assert hr['h265_'+name]['bytes']<=budgets['h265_'+name]
            assert hr['h265_'+name]['bytes']/budgets['h265_'+name]>.98
        scoring.SCOPES.pop('coral',None)
        if case=='video1':
            scoring.SCOPES['coral']=BOX
            seq['U']=np.load(run.parent/'video1_persistent_aux_20261110_01/g17q4/receive/rgb.npy',mmap_mode='r')
            seq['GP']=np.load(run.parent/'fullframe_factorial_20261110_01/native_GP/rgb.npy',mmap_mode='r')
            seq['P5']=np.load(run.parent/'support_factorial_20261111_01/P5/rgb.npy',mmap_mode='r')
        scores,temporal,spatial=scoring.score(seq)
        save_csv(out/'temporal.csv',temporal)
        save_csv(out/'spatial.csv',spatial)
        for scope in scores:
            assert scores[scope]['K']==scores[scope]['L']
        baseline=load_json(previous/case/'final_evaluation/summary.json') if (previous/case/'final_evaluation/summary.json').exists() else None
        save_json(out/'summary.json',dict(complete=True,scores=scores,rates={k:{s:v[s] for s in ('bytes','kbps','base_bytes','old_L_bytes')} for k,v in rates.items()},
            hevc=hr,K_and_L_RGB_bit_exact=True,visual_review='pending'))
        labels=dict(L=f'L / aux {selection[case]["E"]["kbps"]:.2f} kbps',
                    N=f'N / aux {rates["N"]["kbps"]:.2f} kbps',
                    K=f'K / aux {rates["K"]["kbps"]:.2f} kbps',
                    HN=f'H.265 / total {hr["h265_N"]["kbps"]:.2f} kbps',
                    HK=f'H.265 / total {hr["h265_K"]["kbps"]:.2f} kbps')
        specs=[('L_vs_N_full.mp4',('L','N'),None)]
        if case=='video1':
            specs.extend([('L_vs_N_coral.mp4',('L','N'),(250,0,512,132)),
                          ('N_vs_HEVC_full.mp4',('N','HN'),None),('K_vs_HEVC_full.mp4',('K','HK'),None)])
        checks={}
        for filename,methods,crop in specs:
            panels=[seq[k] for k in methods]
            sl=None if crop is None else np.s_[crop[1]:crop[3],crop[0]:crop[2]]
            panel_labels=[labels[k] for k in methods]
            if methods[1] in ('HN','HK'):
                rate=rates[methods[0]]
                panel_labels[0]=f'{methods[0]} / total {(rate["base_bytes"]+rate["bytes"])*8/10000:.2f} kbps'
            preview(ffmpeg,out/filename,panels,15,0,panel_labels,crop=sl)
            checks[filename]=inspect_preview(ffmpeg,out/filename,panels,crop=crop)
        save_json(out/'preview_verification.json',dict(complete=True,previews=checks))
        all_scores[case]=scores;all_previews[case]=checks
        scope='coral' if case=='video1' else 'full'
        print(case,scope,{k:(round(v['coarse_mae'],5),round(v['fine_ncc'],5)) for k,v in scores[scope].items()},flush=True)
    save_json(run/'final_summary.json',dict(complete=True,scores=all_scores,previews=all_previews,
        K='Bit-exact L with reduced actual auxiliary bytes',N='Higher-rate spatial refinement; no claim of equal-budget gain',
        visual_review='pending'))
    snap=run/'source_final';snap.mkdir()
    for name in ('predictive_support_stream.py','run_support_refinement.py','finish_support_refinement.py',
                 'compact_support_stream.py','fullframe_structure_stream.py','run_compact_support.py',
                 'evaluate_fullframe_reference.py','cache_hevc_baseline.py'):
        shutil.copyfile(Path(__file__).with_name(name),snap/name)
    save_json(snap/'manifest.json',{p.name:file_sha(p) for p in snap.iterdir()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('verify','evaluate'))
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--ffmpeg',type=Path)
    args=p.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    if args.action=='verify':verify(args.workspace.resolve(),args.run.resolve())
    else:
        if not args.ffmpeg:p.error('--ffmpeg required')
        evaluate(args.workspace.resolve(),args.run.resolve(),args.ffmpeg.resolve())
