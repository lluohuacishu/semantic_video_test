"""Read-back receipts, source bindings, packet budgets and full-frame support."""

import argparse
import ast
from pathlib import Path

import numpy as np

from compare_source_dense_flow import load_flow
from coral_reference_sidecar import file_sha,load_json,save_json
from experiment_common import array_sha
from fullframe_structure_stream import unpack as unpack_initial
from run_dense_batch import CASES
from run_fullframe_reference import get_inputs
from sparse_fullframe_stream import unpack as unpack_refined


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True);parser.add_argument('--initial',type=Path,required=True)
    parser.add_argument('--refined',type=Path,required=True)
    parser.add_argument('--include-perframe',action='store_true')
    args=parser.parse_args();records={}
    selection=load_json(args.refined/'refinement_selection.json')['selected']
    for case in CASES:
        source,raw,base_sha,base_bytes=get_inputs(args.batch,case)
        if base_bytes>50000:raise ValueError('Base lane exceeds budget')
        controls=[];variants={}
        for root in (args.initial,args.refined):
            keys=tuple(load_json(root/case/'plan.json')['keys'])
            load_flow(root/case/'flow',source,CASES[case][0],anchors=keys)
            flow=load_json(root/case/'transition_flow.json')
            if flow['source_rgb_sha256']!=array_sha(source) or file_sha(root/case/'transition_flow.npy')!=flow['flow_sha256']:
                raise ValueError('Transition source/geometry changed')
            offline=load_json(root/case/'offline/summary.json')
            for name,d in offline['outputs_sha256'].items():
                if file_sha(root/case/'offline'/name)!=d:raise ValueError('Offline output hash')
        cases=[('A',args.initial/case/'grid16x9','fullframe.glcfull',unpack_initial),
            ('B',args.initial/case/'grid16x9_tone12','fullframe.glcfull',unpack_initial),
            ('C',args.refined/case/(selection+'_C'),'fullframe.glcfsp',unpack_refined),
            ('D',args.refined/case/(selection+'_D'),'fullframe.glcfsp',unpack_refined)]
        if args.include_perframe:
            perframe=load_json(args.refined/'perframe_selection.json')['selected']
            cases.extend((v,args.refined/case/(perframe+'_'+v),'fullframe.glcfull',unpack_initial) for v in ('E','F'))
        unpacked={}
        for variant,folder,name,unpack in cases:
            send=load_json(folder/'send.json');receipt=load_json(folder/'receive/receipt.json')
            path=folder/name
            if path.stat().st_size!=send['bytes'] or file_sha(path)!=send['sha256']:
                raise ValueError('Actual packet changed')
            h,a=unpack(path.read_bytes(),base_sha);unpacked[variant]=a
            rgb=np.load(folder/'receive/rgb.npy',mmap_mode='r',allow_pickle=False)
            alpha=np.load(folder/'receive/alpha.npy',mmap_mode='r',allow_pickle=False)
            if rgb.shape!=raw.shape or rgb.dtype!=np.uint8 or alpha.shape!=raw.shape[:3] or alpha.dtype!=np.float32:
                raise ValueError('Unexpected output geometry')
            if not np.isfinite(alpha).all() or alpha.min()<0 or alpha.max()>1:raise ValueError('Invalid alpha')
            if not np.array_equal(rgb[alpha==0],raw[alpha==0]):raise ValueError('Zero weight changed')
            expected=send.get('expected_rgb_sha256',send.get('sender_expected_rgb_sha256'))
            if array_sha(rgb)!=expected:raise ValueError('Sender/receiver differ')
            for filename,digest in receipt['outputs_sha256'].items():
                if file_sha(folder/'receive'/filename)!=digest:raise ValueError('Output hash changed')
            support={f'{r}_{c}':float((alpha[:,r*144:(r+1)*144,c*256:(c+1)*256]>0).mean())
                for r in range(2) for c in range(2)}
            if min(support.values())<=0:raise ValueError('A quadrant has no processing')
            variants[variant]=dict(base_bytes=base_bytes,auxiliary_bytes=path.stat().st_size,
                base_kbps=base_bytes*8/10000,auxiliary_kbps=path.stat().st_size*8/10000,
                output_hash=receipt['outputs_sha256']['rgb.npy'],packet_hash=file_sha(path),
                sender_receiver_bit_exact=True,zero_weight_unchanged=True,quadrant_support=support)
        for first,second in (('A','B'),('C','D'),*((('E','F'),) if args.include_perframe else ())):
            if any(not np.array_equal(unpacked[first][k],v) for k,v in unpacked[second].items()):
                raise ValueError('Blend control changed motion or masks')
        directories=[args.initial/case/'evaluation',args.initial/case/'evaluation_tone',args.refined/case/'evaluation']
        if args.include_perframe:directories.append(args.refined/case/'evaluation_perframe')
        for directory in directories:
            summary=load_json(directory/'summary.json')
            controls.append(summary['scores']['full']['raw'])
            for scope,methods in summary['scores'].items():
                if any(row['pairs']!=149 for row in methods.values()):raise ValueError('Incomplete pair evaluation')
            previews=summary.get('preview_verification',summary.get('previews'))
            for name,record in previews.items():
                if file_sha(directory/name)!=record['sha256']:raise ValueError('Verified preview changed')
                if int(record['probe']['nb_read_frames'])!=150 or record['probe']['avg_frame_rate']!='15/1':
                    raise ValueError('Preview timing')
        if any(row!=controls[0] for row in controls):raise ValueError('Evaluation baseline/masks drifted')
        records[case]=dict(variants=variants,shared_control_metrics_exact=True,
            source_npy_hash=file_sha(args.batch/case/'source.npy'),base_sha256=base_sha,
            flow_hashes_verified=True,source_and_sender_receipts_unavailable_at_receivers=True)
        print(case,'A/B/C/D read-back and full-frame checks PASS',flush=True)
    stage=Path(__file__).parent
    names=('run_fullframe_reference.py','fullframe_structure_stream.py','run_fullframe_auxiliary.py',
        'evaluate_fullframe_reference.py','sparse_fullframe_stream.py','run_fullframe_refinement.py',
        'evaluate_fullframe_refinement.py','run_fullframe_perframe.py','verify_fullframe_transfer.py')
    for name in names:ast.parse((stage/name).read_text(encoding='utf-8'),filename=name)
    save_json(args.refined/('verification_all_variants.json' if args.include_perframe else 'verification.json'),dict(complete=True,clips=records,
        frames_each=150,fps=15,billing_seconds=10,all_lanes_within_40000_bps=True,
        code_sha256={name:file_sha(stage/name) for name in names},python_syntax_checked=True))


if __name__=='__main__':main()
