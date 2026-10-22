"""Complete the measured, budget-compliant but explicitly unmatched selection."""
from __future__ import annotations
import argparse
from fractions import Fraction
from pathlib import Path
from prepare_d2 import read_json,save_json
from run_d2_lowrate import dispatch


def run(args):
    root=args.run.resolve();amendment=read_json(Path(__file__).with_name('d2_lowrate_amendment_v2.json'))
    gate=read_json(root/'rate_gate.json');selection=read_json(root/'rate_selection.json')
    if (root/'selection_amendment_v2.json').exists():raise FileExistsError('Amended decode already dispatched')
    failure=read_json(root/'comparison_failure.json');candidate=failure['points'][0]
    if failure['clip']!='21' or candidate['id']!='v021_512x288_C_q00' or candidate['file_bytes']!=9372:
        raise ValueError('Unexpected rate-only checkpoint; inspect before proceeding')
    expected={'11':{'A':('v011_512x288_A_q05',10126),'B':('v011_512x288_B_q16',10636),
                    'C':('v011_512x288_C_q02',9978)},
              '21':{'A':('v021_512x288_A_q06',9094),'B':('v021_512x288_B_q08',8785)}}
    if set(selection)!=set(expected) or gate['state']!='passed' or gate['size']!=[512,288]:
        raise ValueError('Amendment applies only to the recorded six-point experiment')
    for video,methods in expected.items():
        if set(selection[video])!=set(methods):
            raise ValueError('Unexpected pre-amendment method selection')
        anchor=gate['anchors'][video]
        if (anchor['S0'],anchor['ceiling'])!={'11':(9865,10677),'21':(8846,9730)}[video]:
            raise ValueError('Frozen budget differs from the recorded amendment')
        for method,(point_id,size) in methods.items():
            selected=selection[video][method]['selected']
            if (selected['id'],selected['file_bytes'])!=(point_id,size):
                raise ValueError('Selected checkpoint differs from the recorded amendment')
    if failure['method']!='C' or failure['ceiling']!=8785 or len(failure['points'])!=1:
        raise ValueError('Comparator failure differs from the recorded amendment')
    save_json(root/'rate_selection_before_amendment.json',selection)
    save_json(root/'selection_amendment_v2.json',amendment)
    selection['21']['C']={'target_bytes':gate['anchors']['21']['ceiling'],'selected':candidate,
                        'unused_bytes':gate['anchors']['21']['ceiling']-candidate['file_bytes'],
                        'tested':[candidate],'quality_used':False,'comparator_matching_failed':True,
                        'strict_comparator_ceiling':failure['ceiling'],'amendment':'selection_amendment_v2.json'}
    clips={str(c['video_id']):c for c in read_json(root/'inputs.json')['clips']};tasks=[]
    for video,methods in selection.items():
        for method,item in methods.items():
            p=item['selected'];size=p['file_bytes'];anchor=gate['anchors'][video]
            if size>int(40000*Fraction(clips[video]['duration_exact'])/8) or 10*size>11*anchor['S0']:
                raise ValueError('Hard user budget would be violated')
            folder=root/'points'/p['id']
            tasks.append({'id':p['id'],'input':str(folder/'encoder/stream.glcv'),'output':str(folder/'decoder')})
    if len(tasks)!=6:raise ValueError('Exactly six selected outputs expected')
    save_json(root/'rate_selection.json',selection)
    dispatch(root,read_json(root/'runtime.json'),'decode',tasks,'selected_independent_decode')
    save_json(root/'progress.json',{'state':'codec_complete','selected_points':6,'size':gate['size'],'rate_cap_bps':40000,
             'strict_rate_matching_insufficient_for_021':True,'amendment':'selection_amendment_v2.json'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);run(p.parse_args())
