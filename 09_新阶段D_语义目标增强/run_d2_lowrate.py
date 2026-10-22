"""Bounded original-rate pilot; actual complete bytes gate every selected output."""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
from prepare_d2 import read_json, save_json

HERE=Path(__file__).resolve().parent


def dispatch(root,runtime,role,tasks,phase):
    if (root/'STOP').exists(): raise RuntimeError('Stopped before worker dispatch')
    config=root/(phase+'_config.json')
    if config.exists(): raise FileExistsError('Phase already dispatched: '+phase)
    save_json(config,{'phase':phase,'role':role,'run_root':str(root),'runtime':runtime,
                      'tasks':tasks,'record_legacy_q32_anchor':False})
    print('START '+phase,flush=True)
    with (root/(phase+'.log')).open('x',encoding='utf-8') as log:
        child=subprocess.Popen([runtime['python'],'-u',str(HERE/'d2_codec_worker.py'),'--config',str(config)],
                               stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUTF8='1'))
        save_json(root/'progress.json',{'state':'running','phase':phase,'pid':child.pid,'points_this_phase':len(tasks)})
        code=child.wait()
    if code: raise RuntimeError(phase+' failed; retain its output and inspect its log')
    print('DONE '+phase,flush=True)


def task_for(root,clip,size,method,q):
    width,height=size; label=f'v{clip["video_id"]:03d}_{width}x{height}_{method}_q{q:02d}'
    if method=='A': paths=[f['resized'][f'{width}x{height}']['rgb'] for f in clip['frames']]
    else: paths=[str(root/'controls'/f'v{clip["video_id"]:03d}'/method/f'{f["index"]:04d}.png') for f in clip['frames']]
    return {'id':label,'video_id':clip['video_id'],'method':method,'q':q,'width':width,'height':height,
            'fps':clip['fps'],'frames':paths,'output':str(root/'points'/label/'encoder')}


def point(root,task):
    d=read_json(Path(task['output'])/'encode.json')
    return {'id':task['id'],'video_id':task['video_id'],'method':task['method'],'q':task['q'],
            'file_bytes':d['file_bytes'],'output':task['output'],'width':task['width'],'height':task['height']}


def next_q(records,ceiling):
    allowed=[r for r in records if r['file_bytes']<=ceiling]
    if not allowed: return None
    best=max(allowed,key=lambda r:(r['file_bytes'],-r['q']))
    if (ceiling-best['file_bytes'])/ceiling<=.02: return None
    low=max(r['q'] for r in allowed)
    overs=[r for r in records if r['file_bytes']>ceiling and r['q']>low]
    high=min((r['q'] for r in overs),default=64)
    remaining=[q for q in range(low+1,high) if q not in {r['q'] for r in records}]
    if not remaining: return None
    ordered=sorted(records,key=lambda r:r['q']); slope=.016
    left,right=ordered[0],ordered[-1]
    if right['q']>left['q'] and right['file_bytes']>left['file_bytes']:
        slope=math.log(right['file_bytes']/left['file_bytes'])/(right['q']-left['q'])
    estimate=best['q']+math.log(ceiling/best['file_bytes'])/slope
    return min(remaining,key=lambda q:(abs(q-estimate),q))


def probe(root,plan,clips,runtime):
    if (root/'probe_trials.json').exists(): raise FileExistsError('Probe already started; inspect checkpoint')
    trials=[]
    for size in plan['delivery']['resolution_ladder']:
        tasks=[task_for(root,c,size,'A',0) for c in clips]
        dispatch(root,runtime,'encode',tasks,f'probe_{size[0]}x{size[1]}')
        results=[point(root,t) for t in tasks]
        for p,c in zip(results,clips):
            p['rate_cap_bytes']=c['rate_cap_bytes'];p['within_rate_cap']=p['file_bytes']<=c['rate_cap_bytes']
            trials.append(p);print(f'{p["id"]}: {p["file_bytes"]} B / cap {p["rate_cap_bytes"]} B',flush=True)
        save_json(root/'probe_trials.json',trials)
        if all(p['within_rate_cap'] for p in results):
            anchors={str(c['video_id']):{'S0':p['file_bytes'],'task':p['id'],
                     'rate_cap_bytes':c['rate_cap_bytes'],'ceiling':min(c['rate_cap_bytes'],11*p['file_bytes']//10)} for c,p in zip(clips,results)}
            save_json(root/'rate_gate.json',{'state':'passed','size':size,'anchors':anchors,
                      'selection_before_semantics':True,'actual_complete_bytes':True})
            save_json(root/'progress.json',{'state':'baseline_rate_gate_passed','size':size})
            return
    save_json(root/'rate_gate.json',{'state':'failed','max_probes':len(trials),'semantic_work_started':False})
    save_json(root/'progress.json',{'state':'baseline_rate_gate_failed'})


def compare(root,plan,clips,runtime):
    gate=read_json(root/'rate_gate.json')
    if gate['state']!='passed': raise ValueError('Ordinary baseline must meet the hard rate cap first')
    if read_json(root/'controls.json')['state']!='completed': raise ValueError('Frozen controls not ready')
    if (root/'comparison_started.json').exists(): raise FileExistsError('Comparison already dispatched')
    save_json(root/'comparison_started.json',{'date':'2026-10-22','size':gate['size'],'max_points_per_method_clip':4})
    size=gate['size']; registry={};selected={};selection_details={}
    for c in clips:
        key=str(c['video_id']);registry[key]={};selected[key]={};selection_details[key]={}
        for method in 'ABC':
            ceiling=gate['anchors'][key]['ceiling'] if method!='C' else min(registry[key][m][selected[key][m]]['file_bytes'] for m in 'AB')
            t=task_for(root,c,size,method,0)
            if method!='A': dispatch(root,runtime,'encode',[t],t['id']+'_encode')
            points={0:point(root,t)};tasks={0:t}
            while len(points)<plan['codec']['rate_match_max_points_per_method_clip']:
                q=next_q(list(points.values()),ceiling)
                if q is None: break
                t=task_for(root,c,size,method,q)
                dispatch(root,runtime,'encode',[t],t['id']+'_encode')
                points[q]=point(root,t);tasks[q]=t
            valid=[p for p in points.values() if p['file_bytes']<=ceiling]
            if not valid:
                save_json(root/'comparison_failure.json',{'clip':key,'method':method,'ceiling':ceiling,'points':list(points.values()),'reason':'No point within both rate/growth and comparator-byte caps'})
                raise RuntimeError('No compliant '+method+' point; do not expand search')
            best=max(valid,key=lambda p:(p['file_bytes'],-p['q']))
            registry[key][method]=points;selected[key][method]=best['q']
            selection_details[key][method]={'target_bytes':ceiling,'selected':best,'unused_bytes':ceiling-best['file_bytes'],'tested':list(points.values()),'quality_used':False}
            save_json(root/'rate_selection.json',selection_details)
            print(f'SELECT {best["id"]}: {best["file_bytes"]} / {ceiling} B',flush=True)
    tasks=[]
    for c in clips:
        for m,q in selected[str(c['video_id'])].items():
            t=task_for(root,c,size,m,q)
            tasks.append({'id':t['id'],'input':str(Path(t['output'])/'stream.glcv'),'output':str(Path(t['output']).parent/'decoder')})
    dispatch(root,runtime,'decode',tasks,'selected_independent_decode')
    save_json(root/'progress.json',{'state':'codec_complete','selected_points':len(tasks),'size':size,'rate_cap_bps':40000})


def run(args):
    root=args.run.resolve(); plan=read_json(root/'plan.json');clips=read_json(root/'inputs.json')['clips']
    sys.path.insert(0,str(HERE.parent/'08_新阶段C_编解码加速与网页展示'))
    from c4_config import load_settings
    runtime={k:str(v) for k,v in load_settings()['runtime'].items()};save_json(root/'runtime.json',runtime)
    if args.phase=='probe': probe(root,plan,clips,runtime)
    else: compare(root,plan,clips,runtime)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--phase',choices=['probe','compare'],required=True);args=parser.parse_args()
    try: run(args)
    except BaseException as exc:
        save_json(args.run/'supervisor_failure.json',{'phase':args.phase,'type':type(exc).__name__,'message':str(exc)})
        save_json(args.run/'progress.json',{'state':'failed','phase':args.phase,'message':str(exc)})
        raise
