"""Validate activity-only/native graphs and summarize timing without double counts."""
from __future__ import annotations

import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path
import statistics
import sys

REPO=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(REPO/'13_优化阶段_发送端免RGB重建'))
from sender_common import load_json,save_json


def signature(row):
    return tuple(row[key] for key in ['name','gridX','gridY','gridZ','blockX','blockY','blockZ',
                                      'registersPerThread','staticSharedMemory','dynamicSharedMemory'])


def project(activity,native):
    left,right=defaultdict(list),defaultdict(list)
    for row in activity['kernels']:left[row['streamId']].append(row)
    for row in native['kernels']:right[row['streamId']].append(row)
    for values in [left,right]:
        for stream in values:values[stream].sort(key=lambda row:(row['start'],row['end']))
    projected=[];pairs=[]
    native_ops={row['id']:row for row in native['cpu_operators']}
    for stream,rows in left.items():
        expected=[signature(row) for row in rows]
        candidates=[key for key,values in right.items() if [signature(row) for row in values]==expected]
        if len(candidates)!=1:raise ValueError(f'Kernel graph/stream ambiguity at stream {stream}: {candidates}')
        source=right[candidates[0]];pairs.append((stream,candidates[0]))
        for actual,label in zip(rows,source):
            projected.append(dict(actual,operator=label['operator'],module=label['module'],phase=label['phase'],
                inputs=native_ops[label['external_id']]['inputs'],native_operator_id=label['external_id'],
                attribution='validated identical per-stream kernel-signature sequence; labels from native NVTX'))
    if len(projected)!=len(native['kernels']):raise ValueError('Unmatched native stream')
    return projected,pairs


def validate_fill_deletion(original,changed):
    original=sorted(original,key=lambda row:(row['streamId'],row['start']))
    changed=sorted(changed,key=lambda row:(row['streamId'],row['start']))
    cursor,removed=0,[]
    for row in original:
        if cursor<len(changed) and signature(row)==signature(changed[cursor]):cursor+=1
        elif row['operator']=='aten::fill_':removed.append(row)
        else:raise ValueError('A non-fill GPU kernel changed in the ablation')
    if cursor!=len(changed):raise ValueError('Unexpected new GPU kernels in the ablation')
    return removed


def block(row):
    module=row['module'];phase=row['phase']
    for target in ['vqgan.encoder','vqgan.generator']:
        if target in module:return module.split('.')[0]+'.'+target
    if module:return '.'.join(module.split('.')[:2])
    if 'codebook' in phase:return 'codebook'
    if 'reference_hash' in phase:return 'reference_hash'
    if 'rgb_array' in phase:return 'rgb_array'
    if 'input_tensor' in phase:return 'input_tensor'
    return phase or 'other'


def aggregate(rows,key):
    result={}
    for row in rows:
        name=key(row)
        item=result.setdefault(name,dict(name=name,kernel_count=0,kernel_ns=0))
        item['kernel_count']+=1;item['kernel_ns']+=row['duration_ns']
    return sorted(result.values(),key=lambda row:row['kernel_ns'],reverse=True)


def percentile(values,percent):
    values=sorted(values);position=(len(values)-1)*percent/100
    low=int(position);high=min(len(values)-1,low+1)
    return values[low]+(values[high]-values[low])*(position-low)


def chrome_trace(profile,path):
    origin=profile['window_start_ns'];events=[]
    for row in profile['kernels']+profile['copies']:
        if not row.get('start') or not row.get('end'):continue
        label=row.get('name') or ('memcpy' if row['kind']==1 else 'memset')
        events.append(dict(name=label,cat='gpu',ph='X',pid=1,tid=row['streamId'],ts=(row['start']-origin)/1000,
                           dur=(row['end']-row['start'])/1000,args={key:row.get(key) for key in ['operator','module','phase','bytes','copyKind','correlationId']}))
    for row in profile.get('nvtx_ranges',[]):
        events.append(dict(name=row['name'],cat='native_nvtx',ph='X',pid=0,tid=row['thread_id'],
            ts=(row['start']-origin)/1000,dur=(row['end']-row['start'])/1000))
    for row in profile.get('runtime',[]):
        events.append(dict(name=row['name'],cat='cuda_runtime_cpu',ph='X',pid=0,tid=row['threadId'],
            ts=(row['start']-origin)/1000,dur=(row['end']-row['start'])/1000,args={'correlation_id':row['correlationId']}))
    save_json(path,dict(traceEvents=events,displayTimeUnit='ms'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve();plan=load_json(output/'plan.json');summary=load_json(output/'summary.json')
    if not summary['complete']:raise ValueError('Campaign is incomplete')
    analysis=dict(roles={},graphs_exact=True,attribution='CUPTI-only GPU intervals; per-stream graph-validated native NVTX labels',
                  profiling_times_are_not_normal_speed=True)
    for role in ['sender','receiver']:
        result=summary['results'][role]
        baselines=result['baseline'];all_rows=[row for item in baselines for row in item['records']]
        statistics_by_kind={}
        for label,kind in [('I',0),('P',1)]:
            values=[row['seconds']*1000 for row in all_rows if row['kind']==kind]
            statistics_by_kind[label]=dict(count=len(values),mean_ms=statistics.mean(values),median_ms=statistics.median(values),
                p95_ms=percentile(values,95),min_ms=min(values),max_ms=max(values))
        frames=[]
        for index in plan['samples']:
            activity=load_json(output/role/'activity_1/frames'/f'{index:03d}'/'profile.json')
            natives=[load_json(output/role/f'nvtx_{n}/frames'/f'{index:03d}'/'profile.json') for n in [1,2]]
            projections=[project(activity,native) for native in natives]
            first,second=[rows for rows,_ in projections]
            if [(row['operator'],row['module']) for row in first]!=[(row['operator'],row['module']) for row in second]:
                raise ValueError('Operator/module labels disagree between two native captures')
            kernel_total=sum(row['duration_ns'] for row in first)
            api_groups=defaultdict(lambda:dict(calls=0,wall_ns=0))
            api_by_id={row['correlationId']:row for row in activity['runtime']+activity['driver']}
            for row in activity['runtime']:
                api_groups[row['name']]['calls']+=1;api_groups[row['name']]['wall_ns']+=row['duration_ns']
            transfer_pairs=[]
            for row in activity['copies']:
                if row['kind']!=1:continue
                api=api_by_id.get(row['correlationId'])
                following=sorted([value for value in activity['runtime'] if api and value['threadId']==api['threadId'] and
                    value['start']>=api['end'] and 'StreamSynchronize' in value['name']],key=lambda value:value['start'])
                sync=following[0] if following else None
                transfer_pairs.append(dict(bytes=row['bytes'],copy_kind=row['copyKind'],gpu_ns=row['duration_ns'],
                    api_name=api['name'] if api else None,api_wall_ns=api['duration_ns'] if api else None,
                    correlation_id=row['correlationId'],phase=row['phase'],
                    following_stream_sync_ns=sync['duration_ns'] if sync else None,
                    following_sync_encloses_gpu_copy=bool(sync and sync['start']<=row['start'] and sync['end']>=row['end'])))
            baseline_ms=statistics.median(value['records'][index]['seconds']*1000 for value in baselines)
            data=dict(index=index,kind='I' if index%64==0 else 'P',baseline_core_median_ms=baseline_ms,
                activity_core_ms=activity['core_ns']/1e6,native_core_ms=[native['core_ns']/1e6 for native in natives],
                activity_to_baseline_ratio=activity['core_ns']/1e6/baseline_ms,
                native_to_baseline_ratios=[native['core_ns']/1e6/baseline_ms for native in natives],
                kernel_count=len(first),kernel_ms=kernel_total/1e6,gpu_union_ms=activity['gpu_activity_union_ns']/1e6,
                frame_window_ms=activity['window_ns']/1e6,memcpy_ms=activity['memcpy_sum_ns']/1e6,memcpy_bytes=activity['memcpy_bytes'],
                runtime_sync_ms=activity['runtime_sync_wall_ns']/1e6,
                runtime_apis=sorted([dict(name=name,**value) for name,value in api_groups.items()],key=lambda r:r['wall_ns'],reverse=True),
                copies=transfer_pairs,
                gpu_blocks=aggregate(first,block),gpu_operators=aggregate(first,lambda row:row['operator']),
                gpu_layer_operators=aggregate(first,lambda row:row['module']+' / '+row['operator']),
                graphs_exact=True,label_agreement_between_two_native_runs=True,
                dropped_records=[activity['capture_dropped_records'],*[native['capture_dropped_records'] for native in natives]])
            for group in data['gpu_blocks']+data['gpu_operators']+data['gpu_layer_operators']:
                group['fraction_of_kernel_sum']=group['kernel_ns']/kernel_total
            shape_groups={}
            for row in first:
                key=(row['operator'],row['module'],row['phase'],json.dumps(row['inputs'],sort_keys=True))
                group=shape_groups.setdefault(key,dict(operator=key[0],module=key[1],phase=key[2],inputs=row['inputs'],kernel_ns=0,kernel_count=0,native_ids=set()))
                group['kernel_ns']+=row['duration_ns'];group['kernel_count']+=1;group['native_ids'].add(row['native_operator_id'])
            data['gpu_layer_operator_shapes']=[]
            for group in shape_groups.values():
                group['operator_calls']=len(group.pop('native_ids'));group['fraction_of_kernel_sum']=group['kernel_ns']/kernel_total
                data['gpu_layer_operator_shapes'].append(group)
            data['gpu_layer_operator_shapes'].sort(key=lambda row:row['kernel_ns'],reverse=True)
            fill_path=output/'fill_activity'/role/'frames'/f'{index:03d}'/'profile.json'
            if fill_path.exists():
                changed=load_json(fill_path);removed=validate_fill_deletion(first,changed['kernels'])
                data['fill_graph_validation']=dict(only_fill_kernels_removed=True,baseline_kernels=len(first),
                    fill_off_kernels=changed['kernel_count'],removed_fill_kernels=len(removed),
                    removed_fill_kernel_ns=sum(row['duration_ns'] for row in removed),
                    remaining_compute_kernels_identical=True)
            frames.append(data)
            if index==2:
                chrome_trace(natives[0],output/role/'P002_native_trace.json')
                chrome_trace(dict(activity,kernels=first),output/role/'P002_activity_trace.json')
        pipeline_scopes=load_json(output/'pipeline'/role/'timing_scopes.json')['rows'];groups={}
        for row in pipeline_scopes:
            item=groups.setdefault(row['name'],dict(name=row['name'],calls=0,wall_ns=0,thread_cpu_ns=0,process_cpu_ns=0))
            item['calls']+=1
            for metric in ['wall_ns','thread_cpu_ns','process_cpu_ns']:item[metric]+=row[metric]
        analysis['roles'][role]=dict(
            baseline_seconds=[item['core_seconds'] for item in baselines],baseline_median_seconds=statistics.median(item['core_seconds'] for item in baselines),
            input_seconds=[item['input_seconds'] for item in baselines],baseline_frame_statistics=statistics_by_kind,frames=frames,
            pipeline_scopes=sorted(groups.values(),key=lambda row:row['wall_ns'],reverse=True),pipeline=result['pipeline'])
        detailed=output/'pipeline_detailed'/role
        if (detailed/'timing_scopes.json').exists():
            groups={}
            for row in load_json(detailed/'timing_scopes.json')['rows']:
                item=groups.setdefault(row['name'],dict(name=row['name'],calls=0,wall_ns=0,thread_cpu_ns=0,process_cpu_ns=0))
                item['calls']+=1
                for metric in ['wall_ns','thread_cpu_ns','process_cpu_ns']:item[metric]+=row[metric]
            analysis['roles'][role]['pipeline_detailed_scopes']=sorted(groups.values(),key=lambda row:row['wall_ns'],reverse=True)
            analysis['roles'][role]['pipeline_detailed_result']=load_json(detailed/'result.json')
    if (output/'fill_ablation/summary.json').exists():
        raw=load_json(output/'fill_ablation/summary.json')
        analysis['fill_ablation']={role:{key:value for key,value in data.items() if key!='pairs'} for role,data in raw['results'].items()}
    save_json(output/'analysis.json',analysis)
    print('All 16 activity frames match both native graphs and labels; low-overhead GPU timing assigned')
    for role,item in analysis['roles'].items():
        f=next(row for row in item['frames'] if row['index']==2)
        print(role,'baseline',item['baseline_median_seconds'],'P2 GPU',f['kernel_ms'],
              'copy',f['memcpy_ms'],'top blocks',[(row['name'],round(row['kernel_ns']/1e6,3)) for row in f['gpu_blocks'][:4]])


if __name__=='__main__':main()
