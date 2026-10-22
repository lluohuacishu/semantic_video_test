"""Decode one existing compliant O point; preserve the strict matching failure."""
from __future__ import annotations
import argparse
from pathlib import Path
from prepare_d2 import read_json,save_json
from run_d2_lowrate import dispatch


def run(root):
    if read_json(root/'progress.json')['state']!='codec_complete':raise ValueError('Wait for original bounded decode batch')
    if (root/'evaluation.json').exists():raise ValueError('This rate-only amendment must precede quality evaluation')
    if (root/'selection_before_amendment.json').exists():raise FileExistsError('Amendment already started')
    plan=read_json(Path(__file__).with_name('d21_selection_amendment_v2.json'))
    s=read_json(root/'selection.json');p=next(p for p in s['registry']['21']['O'] if p['q']==25)
    if (p['file_bytes'],s['selected']['21']['A']['file_bytes'],s['selected']['21']['B']['file_bytes'],s['selected']['21']['O']['q'])!=(9670,9720,9660,0):raise ValueError('Unexpected rate checkpoint')
    gate=read_json(root/'rate_gate.json')
    if p['file_bytes']>gate['anchors']['21']['ceiling'] or s['new_decodes']+1>read_json(root/'codec_plan.json')['max_new_independent_decodes']:raise ValueError('Registered hard cap exceeded')
    save_json(root/'selection_before_amendment.json',s);save_json(root/'selection_amendment_v2.json',plan)
    decoder=root/'decodes'/p['id']
    dispatch(root,read_json(root/'runtime.json'),'decode',[{'id':p['id'],'input':str(Path(p['output'])/'stream.glcv'),'output':str(decoder)}],'amended_O21_independent_decode')
    s['selected']['21']['O']=p
    for e in s['entries']:
        if e['video_id']==21 and e['method']=='O' and e['category']=='selected':
            e['point']=p;e['decoder']=str(decoder);e['size_selection_amended']=True
    s['matching']['21_O']['strict_selected']=s['matching']['21_O']['selected']
    s['matching']['21_O']['selected']=p;s['matching']['21_O']['amendment']='selection_amendment_v2.json'
    s['new_decodes']+=1;s['selection_amendment']='selection_amendment_v2.json';save_json(root/'selection.json',s)
    save_json(root/'progress.json',{'state':'codec_complete','new_encodes':s['new_encodes'],'new_decodes':s['new_decodes'],
        'unique_evaluated_streams':len({e['point']['output'] for e in s['entries']}),'comparison_entries':len(s['entries']),
        'selection_amendment':'selection_amendment_v2.json'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);run(p.parse_args().run.resolve())
