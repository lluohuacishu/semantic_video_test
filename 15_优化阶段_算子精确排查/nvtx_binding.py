"""Associate native C++ ATen NVTX ranges with CUDA APIs and GPU correlations."""
from __future__ import annotations

import ast
from collections import defaultdict
from cupti_trace import union_ns


def pair_ranges(records):
    starts,ranges={},[]
    for row in records:
        if row['kind']!=12:continue
        if row['flags'] & 2:
            if row['id'] in starts:raise ValueError('Duplicate NVTX range start')
            starts[row['id']]=row
        elif row['flags'] & 4:
            start=starts.pop(row['id'],None)
            if start is None:continue
            if row['timestamp']<start['timestamp']:raise ValueError('Negative NVTX range')
            ranges.append(dict(id=start['id'],name=start['name'],start=start['timestamp'],end=row['timestamp'],
                               process_id=start['objectId'][0],thread_id=start['objectId'][1]))
    return ranges


def inputs_from_name(name):
    for delimiter in ['sizes = ','|sizes=']:
        if delimiter in name:
            value=name.split(delimiter,1)[1].split(', input_op_ids',1)[0]
            try:return ast.literal_eval(value)
            except (ValueError,SyntaxError):return value
    return []


def native_binding(observer,records,begin,end):
    ranges=pair_ranges(records)
    by_thread=defaultdict(list)
    for row in ranges:by_thread[row['thread_id']].append(row)
    operators,by_id=[],{}
    for row in ranges:
        name=row['name']
        if not (name.startswith(('aten::','prim::','glc.script:')) and row['start']>=begin and row['end']<=end):continue
        containers=[r for r in by_thread[row['thread_id']] if r['start']<=row['start'] and r['end']>=row['end']]
        modules=[r for r in containers if r['name'].startswith('glc.module:')]
        phases=sorted([r for r in containers if r['name'].startswith('glc.phase:')],key=lambda r:r['start'])
        module=max(modules,key=lambda r:r['start'])['name'][11:] if modules else ''
        label=name.split(', ',1)[0] if not name.startswith('glc.script:') else 'torchscript.'+name.split(':',1)[1].split('|',1)[0]
        item=dict(id=1000000000+row['id'],name=label,operator_type='native_nvtx',
                  module=module,phase='/'.join(r['name'][10:] for r in phases),inputs=inputs_from_name(name),
                  kwargs={},thread_id=row['thread_id'],start_perf_ns=row['start'],end_perf_ns=row['end'],
                  thread_cpu_ns=0,thread_cpu_measured=False,nvtx_range_id=row['id'],native_label=name,
                  clock_domain='cupti_normalized_ns')
        # Inclusive CPU wall is retained, with child intervals separately excluded.
        children=[(r['start'],r['end']) for r in containers if r['id']!=row['id'] and
                  r['start']>=row['start'] and r['end']<=row['end']]
        item['self_cpu_wall_ns']=row['end']-row['start']-union_ns(children)
        operators.append(item);by_id[row['id']]=item
    # Recompute self wall against all contained ranges, not only ancestors.
    for item in operators:
        children=[(r['start'],r['end']) for r in by_thread[item['thread_id']] if
                  r['id']!=item['nvtx_range_id'] and r['start']>=item['start_perf_ns'] and r['end']<=item['end_perf_ns']]
        item['self_cpu_wall_ns']=item['end_perf_ns']-item['start_perf_ns']-union_ns(children)
    mapping,launches={},{}
    for row in records:
        if row['kind'] not in (4,5):continue
        candidates=[r for r in by_thread[row['threadId']] if r['id'] in by_id and
                    r['start']<=row['start'] and r['end']>=row['end']]
        if candidates:
            target=max(candidates,key=lambda r:(r['start'],-r['end']))
            mapping[row['correlationId']]=by_id[target['id']]['id']
            launches[row['correlationId']]=dict(api=row['name'],kind=row['kind'],thread_id=row['threadId'],
                start=row['start'],end=row['end'],nvtx_range_id=target['id'])
    observer.operators=operators
    result=observer.summarize(records,begin,end,correlation_override=mapping)
    result['nvtx_ranges']=ranges
    result['operator_binding']='GPU correlation -> CUDA runtime/driver API -> innermost native ATen/JIT NVTX range on same CPU thread'
    result['native_operator_thread_cpu_measured']=False
    for row in result['kernels']+result['copies']:
        row['launch_binding']=launches.get(row['correlationId'])
    for group in result['operators']:
        ids={row['id'] for row in operators if row['name']==group['name'] and row['module']==group['module'] and
             row['phase']==group['phase'] and row['inputs']==group['inputs']}
        group['self_cpu_wall_ns']=sum(row['self_cpu_wall_ns'] for row in operators if row['id'] in ids)
        group['thread_cpu_ns']=None
    return result
