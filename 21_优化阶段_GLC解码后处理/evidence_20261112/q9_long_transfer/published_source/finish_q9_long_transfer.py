"""Verify frozen Q9 transfer outputs and build bounded-memory review media."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from runtime_speed_common import STAGE,FFMPEG,WEIGHT,file_sha,load_json,save_json,check_inputs
from receiver_structure import EVENTS,render
from mild_reliability_gate import mild_gate
from finish_generalization_factorial import sample_sheet
import q9_chunk_protocol as protocol
import structure_guide_stream as guide


def verify_long(run):
    plan=load_json(run/'execution_plan.json');check_inputs(plan['input_sha256'])
    for name,h in plan['code_sha256'].items():
        assert file_sha(STAGE/name)==h==file_sha(run/'source_snapshot'/name)
    rec=load_json(run/'receive/receipt.json');send=load_json(run/'sender_receipt.json')
    manifest_path=run/'guide/manifest.json';manifest=load_json(manifest_path)
    assert rec['complete'] and rec['source_reads_blocked'] and len(rec['blocked'])==4
    assert set(rec['reads'])<=set(rec['allowed'])
    assert all(str(Path(p).resolve()) not in rec['reads'] for p in rec['blocked'])
    assert rec['code_sha256']==plan['code_sha256']['run_q9_long_transfer.py']
    assert rec['protocol_sha256']==plan['code_sha256']['q9_chunk_protocol.py']
    assert send['manifest_sha256']==rec['manifest_sha256']==file_sha(manifest_path)
    base_sha=file_sha(plan['base']);raw=np.load(plan['raw'],mmap_mode='r',allow_pickle=False)
    assert base_sha==manifest['base_sha256']==rec['base_sha256']
    assert file_sha(plan['raw'])==rec['raw_sha256'] and raw.shape==(1239,288,512,3)
    cursor=0;packet_bytes=0;packets=[];negative=[]
    for item in manifest['packets']:
        path=run/'guide'/item['file'];packet=path.read_bytes()
        assert file_sha(path)==item['sha256'] and len(packet)==item['bytes']
        assert item['start']==cursor
        header,images=protocol.decode(packet,base_sha,cursor,1239,FFMPEG)
        assert images.shape==(item['frames'],288,512,3) and images.dtype==np.uint8
        assert header['frames']==min(150,1239-cursor)
        packets.append(dict(start=cursor,frames=header['frames'],bytes=len(packet),decode_verified=True))
        if cursor==0:
            bad=bytearray(packet);bad[-5]^=1
            for label,args in [('crc',(bytes(bad),base_sha,0,1239)),('truncated',(packet[:-1],base_sha,0,1239)),
                               ('base_binding',(packet,'0'*64,0,1239)),('start',(packet,base_sha,150,1239)),
                               ('total',(packet,base_sha,0,1240))]:
                try:protocol.unpack(*args)
                except (ValueError,TypeError):negative.append(label)
                else:raise AssertionError('Invalid packet accepted: '+label)
        cursor+=header['frames'];packet_bytes+=len(packet)
    assert cursor==1239 and len(packets)==9 and packets[-1]['frames']==39
    index_bytes=manifest_path.stat().st_size;duration=1239/15
    assert send['packet_bytes']==packet_bytes and send['index_bytes']==index_bytes
    assert send['auxiliary_bytes']==packet_bytes+index_bytes
    assert send['base_bytes']==Path(plan['base']).stat().st_size
    assert abs(send['total_kbps']-(packet_bytes+index_bytes+send['base_bytes'])*8/duration/1000)<1e-10
    assert send['auxiliary_kbps']<40 and send['base_bytes']*8/duration/1000<40
    arrays={};mask_stats={}
    for mode in ('continuous','reset'):
        values={}
        for name,h in rec['outputs_sha256'][mode].items():
            path=run/'receive'/mode/(name+'.npy');assert file_sha(path)==h
            a=np.load(path,mmap_mode='r',allow_pickle=False)
            assert a.shape==((1239,288,512,3) if name=='rgb' else (1239,288,512))
            assert a.dtype==(np.uint8 if name=='rgb' else np.float32);values[name]=a
        active=0;weight=0.;shared_total=0.
        for start in range(0,1239,150):
            sl=slice(start,start+150);a=values['alpha'][sl];s=values['shared'][sl]
            assert np.isfinite(a).all() and np.isfinite(s).all()
            assert np.all((a>=0)&(a<=1)) and np.all((s>=0)&(s<=a))
            assert np.array_equal(values['rgb'][sl][a==0],raw[sl][a==0])
            active+=int((a>0).sum());weight+=float(a.sum(dtype=np.float64));shared_total+=float(s.sum(dtype=np.float64))
        arrays[mode]=values;pixels=1239*288*512
        mask_stats[mode]=dict(active_fraction=active/pixels,mean_alpha=weight/pixels,mean_shared=shared_total/pixels)
    changed=[]
    for i in range(1239):
        same=all(np.array_equal(arrays['continuous'][name][i],arrays['reset'][name][i]) for name in ('rgb','alpha','shared'))
        if not same:changed.append(i)
        if i<150 or i%150>=21:assert same,('Control differs outside declared reset region',i)
    records=rec['records'];assert rec['max_window_frames']<=180
    assert rec['first_window_legacy_exact'] and records[0]['first_window_legacy_exact']
    result=dict(complete=True,frozen_inputs_and_listed_inference_code_unchanged=True,
        first_window_original_implementation_exact=rec['first_window_legacy_exact'],
        independently_redecoded_packets=packets,invalid_packets_rejected=negative,
        input_allowlist_verified=True,zero_alpha_unchanged=True,shared_never_exceeds_alpha=True,
        control_exact_outside_each_boundary_first21_frames=True,different_frames=changed,coverage=mask_stats,
        actual_rate_accounting=dict(packet_bytes=packet_bytes,index_bytes=index_bytes,auxiliary_kbps=send['auxiliary_kbps'],total_kbps=send['total_kbps']),
        memory=dict(max_cached_frames=rec['max_window_frames'],peak_gpu_allocated_mib=max(r['peak_gpu_allocated_mib'] for r in records),
            working_set_after_release_mib=[r['process_memory_after_release']['working_set_mib'] for r in records],
            caveat='GPU/input cache is bounded. Full-length memory-mapped outputs retain resident pages; total process RAM is not demonstrated constant.'),
        timing_caveat='Dual-output diagnostic run includes first-window legacy replay, model load, hashes and IO; not a single-mode or end-to-end speed benchmark.',
        verifier_sha256=file_sha(__file__))
    save_json(run/'verification.json',result)
    print('LONG VERIFIED',len(changed),'changed control frames; GPU peak',result['memory']['peak_gpu_allocated_mib'],flush=True)


def inspect_samples(path,panels,indices,timeline):
    probe=json.loads(subprocess.check_output([str(FFMPEG.with_name('ffprobe.exe')),'-v','error','-count_frames',
        '-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,nb_read_frames,duration','-of','json',str(path)]))['streams'][0]
    assert int(probe['nb_read_frames'])==len(timeline) and Fraction(probe['avg_frame_rate'])==15
    assert (probe['width'],probe['height'])==(1024,314*((len(panels)+1)//2))
    assert abs(float(probe['duration'])-len(timeline)/15)<.001
    indices=sorted(set(indices));selection='+'.join(f'eq(n\\,{i})' for i in indices)
    data=subprocess.check_output([str(FFMPEG),'-v','error','-threads','1','-i',str(path),'-vf','select='+selection,
        '-fps_mode','vfr','-pix_fmt','rgb24','-threads','1','-f','rawvideo','pipe:1'])
    frames=np.frombuffer(data,np.uint8).reshape(len(indices),probe['height'],probe['width'],3)
    errors=[]
    for j,seq in enumerate(panels):
        x,y=(j%2)*512,(j//2)*314+26;values=[]
        for k,output_index in enumerate(indices):
            ref=seq[timeline[output_index]]
            values.append(float(np.abs(frames[k,y:y+288,x:x+512].astype(np.float32)-ref).mean()))
        assert max(values)<6,(str(path),j,max(values));errors.append(values)
    return dict(probe=probe,sampled_output_frames=indices,source_frames=[timeline[i] for i in indices],
                panel_rgb_mae=errors,sha256=file_sha(path),only_sampled_frames_materialized=True)


def write_preview(path,panels,labels,timeline,boundary=False):
    width=1024;height=314*((len(panels)+1)//2)
    cmd=[str(FFMPEG),'-v','error','-nostdin','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
        '-r','15','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','14','-threads','2',
        '-pix_fmt','yuv420p','-movflags','+faststart','-n',str(path)]
    with path.with_suffix('.ffmpeg.log').open('xb') as log:
        process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        try:
            canvas=np.full((height,width,3),(24,35,48),np.uint8)
            for out_i,i in enumerate(timeline):
                for j,seq in enumerate(panels):
                    x,y=(j%2)*512,(j//2)*314;canvas[y:y+26,x:x+512]=(24,35,48)
                    suffix=f' | f{i} / {i/15:.2f}s'
                    if boundary:suffix=f' | win {out_i//45+1}/8 | {i/15:.2f}s'
                    cv2.putText(canvas,labels[j]+suffix,(x+7,y+18),cv2.FONT_HERSHEY_SIMPLEX,.44,(235,244,252),1,cv2.LINE_AA)
                    canvas[y+26:y+314,x:x+512]=seq[i]
                process.stdin.write(canvas.tobytes())
            process.stdin.close();assert process.wait(timeout=90)==0
        finally:
            if process.poll() is None:process.kill();process.wait()
    if boundary:indices=[k*45+d for k in range(8) for d in (0,14,15,20,35,44)]
    else:indices=[0,619,1238]+[b+d for b in range(150,1239,150) for d in (-1,0,5,20)]
    return inspect_samples(path,panels,indices,timeline)


def media(run):
    assert load_json(run/'verification.json')['complete']
    plan=load_json(run/'execution_plan.json');out=run/'previews';out.mkdir(exist_ok=False)
    seq=dict(source=np.load(plan['source'],mmap_mode='r'),raw=np.load(plan['raw'],mmap_mode='r'),
             continuous=np.load(run/'receive/continuous/rgb.npy',mmap_mode='r'),reset=np.load(run/'receive/reset/rgb.npy',mmap_mode='r'))
    records={};timeline=list(range(1239))
    specifications=[('long_source_raw_continuous_reset.mp4',list(seq),False),('long_raw_vs_Q9.mp4',['raw','continuous'],False),
                    ('all8_boundaries_continuous_vs_reset.mp4',['continuous','reset'],True)]
    labels=dict(source='Source',raw='Original GLC',continuous='Q9_R25 continuous',reset='Q9_R25 reset')
    for filename,names,is_boundary in specifications:
        times=[i for b in range(150,1239,150) for i in range(b-15,b+30)] if is_boundary else timeline
        records[filename]=write_preview(out/filename,[seq[n] for n in names],[labels[n] for n in names],times,is_boundary)
        print('PREVIEW VERIFIED',filename,flush=True)
    sample_sheet(out/'long_first_middle_tail.png','82.6s',seq,dict(source='原片',raw='原GLC',continuous='Q9连续',reset='Q9块重置'),(0,619,1238))
    for group,ids in [('boundaries_1_4',(150,300,450,600)),('boundaries_5_8',(750,900,1050,1200))]:
        sample_sheet(out/(group+'.png'),group,seq,dict(source='原片',raw='原GLC',continuous='Q9连续',reset='Q9块重置'),ids)
    save_json(out/'verification.json',dict(complete=True,videos=records,original_frame_rate=15,
        boundary_edit='8 chronological 45-frame windows, [b-15,b+30); jumps between windows are edits, not model flicker.',
        subjective_review='pending',verifier_sha256=file_sha(__file__)))


def verify_archive(parent):
    run=parent/'archive_transfer';registration=load_json(run/'registration.json');records={}
    assert registration['parent_plan_sha256']==file_sha(parent/'execution_plan.json')
    for name,h in registration['code_sha256'].items():
        assert file_sha(STAGE/name)==h==file_sha(run/'source_snapshot'/name)
    for case,spec in registration['cases'].items():
        root=run/case;source_rec=load_json(root/'source_receipt.json');rec=load_json(root/'post/receipt.json')
        assert file_sha(spec['video'])==spec['video_sha256']==source_rec['video_sha256']
        assert file_sha(spec['source'])==source_rec['source_sha256'] and source_rec['frames']==150
        assert rec['complete'] and rec['source_reads_blocked'] and len(rec['blocked'])==4
        assert set(rec['reads'])<=set(rec['allowed']) and str(WEIGHT.resolve()) in rec['allowed']
        assert rec['code_sha256']==registration['code_sha256']['run_q9_archive_transfer.py']
        base=root/'send/base.glcsnd';raw_path=root/'base_receive/decoded_rgb.npy';base_rec=load_json(root/'base_receive/receipt.json')
        assert file_sha(base)==base_rec['stream_sha256']==rec['base_sha256']
        assert base_rec['reference_verified'] and base_rec['source_media_reads_forbidden']
        assert file_sha(raw_path)==base_rec['decoded_rgb_sha256']
        packet=root/'guide/guide.glcguide';send=load_json(root/'guide/send.json')
        assert file_sha(packet)==rec['guide_sha256']==send['sha256'] and packet.stat().st_size==send['bytes']
        header,low,images=guide.decode(packet.read_bytes(),file_sha(base),FFMPEG)
        assert header==rec['guide_header'] and np.array_equal(images,np.load(root/'post/guide_rgb.npy',mmap_mode='r'))
        cached={}
        for name,h in rec['local_cache_sha256'].items():
            path=root/'post'/(name+'.npy');assert file_sha(path)==h
            cached[name]=np.load(path,mmap_mode='r');assert np.isfinite(cached[name]).all()
        raw=np.load(raw_path,mmap_mode='r');values={}
        for name,h in rec['outputs_sha256'].items():
            path=root/'post'/(name+'.npy');assert file_sha(path)==h
            a=np.load(path,mmap_mode='r');values[name]=a
            assert a.shape==dict(rgb=(150,288,512,3),alpha=(150,288,512),shared=(30,288,512))[name]
            assert a.dtype==(np.uint8 if name=='rgb' else np.float32) and np.isfinite(a).all()
        a=values['alpha'];s=values['shared'];assert np.all((a>=0)&(a<=1)) and np.all((s>=0)&(s<=a[list(EVENTS)]))
        assert np.array_equal(values['rgb'][a==0],raw[a==0])
        replay=render(raw,images,*(cached[n] for n in ('to_anchor','from_anchor','transition_flow')),gate_fn=mild_gate(images,raw,'R25'))
        assert all(np.array_equal(values[n],replay[n]) for n in values)
        aggregate=load_json(run/'summary.json')
        summary=load_json(root/aggregate['cases'][case].get('evaluation_dir','evaluation')/'summary.json');assert summary['complete']
        assert abs(summary['total_kbps']-(base.stat().st_size+packet.stat().st_size)*8/10000)<1e-10
        assert all(summary['scores'][scope]['Q9_R25']['pairs']+
                   sum(r['scope']==scope for r in summary.get('low_support',[]))==149 for scope in summary['scores'])
        records[case]=dict(complete=True,source_and_frozen_code_unchanged=True,guide_redecode_exact=True,
            reference_verified=True,input_allowlist_verified=True,stored_flow_rerender_exact=True,zero_alpha_unchanged=True,
            shared_never_exceeds_alpha=True,rate_bytes_verified=True,per_channel_budget_pass=summary['per_channel_budget_pass'])
        print('ARCHIVE VERIFIED',case,flush=True)
    save_json(run/'verification.json',dict(complete=True,cases=records,verifier_sha256=file_sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('verify','media','archive'))
    p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    dict(verify=verify_long,media=media,archive=verify_archive)[args.action](args.run)
