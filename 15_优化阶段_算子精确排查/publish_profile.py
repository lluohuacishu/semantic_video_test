"""Validate producer hashes/results and publish compact tables, plots, provenance."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parent
sys.path.insert(0,str(REPO/'13_优化阶段_发送端免RGB重建'))
from sender_common import load_json,save_json,file_sha
from run_profile import compare


def compact_analysis(analysis):
    result=json.loads(json.dumps(analysis))
    for role,item in result['roles'].items():
        for frame in item['frames']:
            frame.pop('gpu_layer_operator_shapes',None)
            frame.pop('gpu_layer_operators',None)
            frame['complete_layer_operator_shape_table']=f'gpu_layers_{role}_{frame["index"]:03d}.csv'
    result['table_storage']='Full per-layer/operator/shape data in companion CSVs; duplicate JSON arrays omitted.'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve();dest=args.destination.resolve()
    plan=load_json(output/'plan.json');summary=load_json(output/'summary.json');analysis=load_json(output/'analysis.json')
    if not summary['complete'] or not analysis['graphs_exact']:raise ValueError('Incomplete analysis')
    for name,digest in plan['code_sha256'].items():
        if file_sha(REPO/name)!=digest:raise ValueError('Capture source changed: '+name)
    if file_sha(output/'plan.json')!=summary['plan_sha256']:raise ValueError('Plan changed')
    teacher=load_json(plan['clip']['teacher_result']);expected_video='6f0e6c889eac3511ff0f4bffbb35423517f61c0d556e05a8af467dfa430a0d6e'
    baselines,fill_rows,capture_validation=[],[],[]
    result_paths=sorted(output.rglob('result.json'))
    for path in result_paths:
        value=load_json(path)
        if 'records' in value:
            compare(teacher,value,value['role'])
            if value['role']=='sender':
                if file_sha(path.parent/'stream.glcsnd')!=plan['clip']['stream_sha256']:raise ValueError('Sender stream changed')
            if value.get('capture_dropped_records') not in (0,None):raise ValueError('CUPTI dropped records')
            if 'fill_uninitialized_memory' in value:
                fill_rows.append(dict(path=path.relative_to(output).as_posix(),**value))
            elif value['mode']=='baseline':baselines.append(dict(path=path.relative_to(output).as_posix(),**value))
            capture_validation.append(dict(path=path.relative_to(output).as_posix(),role=value['role'],mode=value['mode'],
                exact=True,dropped=value.get('capture_dropped_records'),profiles=value['profiles']))
        elif value.get('action')=='encode':
            if value['stream_sha256']!=plan['clip']['stream_sha256']:raise ValueError('Real-file stream differs')
        elif value.get('action')=='decode':
            if value['video_sha256']!=expected_video:raise ValueError('Real-file viewing output differs')
    verification=load_json(output/'verification.json')
    if any(row['returncode'] for row in verification['checks']):raise ValueError('Regression failed')
    dest.mkdir(parents=True,exist_ok=False)
    for name,value in [('analysis',compact_analysis(analysis)),('plan',plan),('provenance',load_json(output/'provenance.json')),
                       ('baseline_frames',baselines),('fill_ablation_frames',fill_rows),('capture_validation',capture_validation),
                       ('verification',verification),('fill_kernel_validation',load_json(output/'fill_kernel_validation.json'))]:
        save_json(dest/(name+'.json'),value)
    fill_summary=load_json(output/'fill_ablation/summary.json')
    save_json(dest/'fill_summary.json',dict(complete=fill_summary['complete'],default_modified=fill_summary['default_modified'],
        deterministic_algorithms_preserved=fill_summary['deterministic_algorithms_preserved'],
        results={role:{key:value for key,value in item.items() if key!='pairs'} for role,item in fill_summary['results'].items()},
        code_sha256=fill_summary['code_sha256']))
    for role,item in analysis['roles'].items():
        for frame in item['frames']:
            rows=[]
            for row in frame['gpu_layer_operator_shapes']:
                rows.append(dict(role=role,index=frame['index'],module=row['module'],operator=row['operator'],phase=row['phase'],
                    input_shapes=json.dumps(row['inputs']),operator_calls=row['operator_calls'],kernel_count=row['kernel_count'],
                    gpu_kernel_ms=row['kernel_ns']/1e6,share_percent=100*row['fraction_of_kernel_sum'],
                    attribution='CUPTI-only timing; validated native-NVTX per-stream signature mapping'))
            path=dest/f'gpu_layers_{role}_{frame["index"]:03d}.csv'
            with path.open('w',encoding='utf-8-sig',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
        for folder in ['pipeline','pipeline_detailed']:
            save_json(dest/f'{folder}_{role}.json',dict(result=load_json(output/folder/role/'result.json'),
                scopes=load_json(output/folder/role/'timing_scopes.json')))
    for filename in ['operator_overview.png','operator_overview.svg','P002_timeline.png','P002_timeline.svg']:
        (dest/filename).write_bytes((output/filename).read_bytes())
    for row in verification['checks']:
        (dest/row['log']).write_text((output/row['log']).read_text('utf-8'),encoding='utf-8',newline='\n')
    raw={}
    for path in output.rglob('*'):
        if path.is_file() and path.suffix in ('.json','.csv','.log','.glcsnd','.mp4','.png','.svg'):
            raw[path.relative_to(output).as_posix()]=dict(sha256=file_sha(path),bytes=path.stat().st_size)
    save_json(dest/'manifest.json',dict(raw_output=str(output),raw_execution_artifacts=raw,
        capture_code_sha256=plan['code_sha256'],
        analysis_code_sha256={path.relative_to(REPO).as_posix():file_sha(path) for path in HERE.glob('*.py')},
        public_artifacts={path.name:dict(sha256=file_sha(path),bytes=path.stat().st_size) for path in dest.iterdir() if path.is_file()},
        note='No raw big traces or NVIDIA headers redistributed. Public text is UTF-8 LF; timing scopes may overlap.'))
    print(f'Published {len(result_paths)} validated full runs, 64 sampled windows, 16 shape/operation tables')


if __name__=='__main__':main()
