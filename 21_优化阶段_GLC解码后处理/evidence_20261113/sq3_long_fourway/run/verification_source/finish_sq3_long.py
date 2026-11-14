"""Replay full SQ3 outputs, measure basic diagnostics and create four-way media."""
import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from runtime_speed_common import STAGE,FFMPEG,file_sha,load_json,save_json,check_inputs
from short_history_filter import blend_frame,quantize
from received_structure_fusion import fuse_received_structure
from run_sq3_long import CONFIG
import q9_chunk_protocol as protocol


def sequences(run):
    p=load_json(run/'execution_plan.json')
    return {k:np.load(v,mmap_mode='r') for k,v in dict(source=p['source'],H265=run/'h265/rgb.npy',
        raw=p['raw'],Q9=p['q9'],GS25=run/'receive/GS25.npy',SQ3=run/'receive/SQ3.npy').items()}


def verify(run):
    p=load_json(run/'execution_plan.json');check_inputs(p['input_sha256'])
    assert file_sha(STAGE/'当前采用配置.json')==CONFIG
    for name,h in p['code_sha256'].items():assert file_sha(STAGE/name)==h==file_sha(run/'source_snapshot'/name)
    guard_extension=load_json(run/'receiver_guard_extension.json')
    assert file_sha(run/'execution_plan.json')==guard_extension['original_plan_sha256']
    assert file_sha(STAGE/'run_sq3_long_receiver.py')==guard_extension['wrapper_sha256']==file_sha(run/'source_snapshot/run_sq3_long_receiver.py')
    regression=load_json(run/'regression/verification.json');assert regression['complete']
    rec=load_json(run/'receive/receipt.json');h265=load_json(run/'h265/receipt.json')
    assert rec['complete'] and h265['complete'] and rec['frames']==p['frames']
    assert rec['source_reads_blocked'] and len(rec['blocked'])==4 and set(rec['reads'])<=set(rec['allowed'])
    assert not set(rec['reads'])&set(rec['blocked']) and rec['extra_transmission_bytes']==0
    seq=sequences(run);guide=np.load(run/'receive/guide.npy',mmap_mode='r');n=p['frames']
    assert all(a.shape==(n,288,512,3) and a.dtype==np.uint8 for a in seq.values())
    for name,h in rec['output_sha256'].items():assert file_sha(run/'receive'/(name+'.npy'))==h
    assert file_sha(run/'h265/rgb.npy')==h265['rgb_sha256'] and file_sha(run/'h265/h265_matched.mp4')==h265['stream_sha256']
    assert (run/'h265/h265_matched.mp4').stat().st_size==h265['bytes']<=p['total_bytes']
    assert h265['relative_shortfall']<=.02 and h265['source_sha256']==file_sha(p['source'])
    manifest=load_json(p['manifest']);cursor=0;tested=[]
    for block,r in zip(manifest['packets'],rec['records'],strict=True):
        start=block['start'];count=block['frames'];end=start+count;begin=max(0,start-2)
        assert start==cursor==r['start'] and count==r['frames'] and r['history_begin']==begin;cursor=end
        packet=Path(p['guide_root'])/block['file'];header,decoded=protocol.decode(packet.read_bytes(),file_sha(p['base']),start,n,FFMPEG)
        assert np.array_equal(decoded,guide[start:end])
        assert file_sha(run/'receive'/r['flow_file'])==r['flow_sha256']
        flows=np.load(run/'receive'/r['flow_file'],mmap_mode='r')
        assert flows.shape==(end-begin,2,2,288,512,2)
        for t in range(start,end):
            i=t-begin;assert np.isfinite(flows[i]).all()
            expected=fuse_received_structure(seq['Q9'][t],guide[t],'GS25')[0]
            assert np.array_equal(expected,seq['GS25'][t]),('GS25',t)
            x,stats=blend_frame(seq['GS25'][begin:end],guide[begin:end],flows,i)
            assert np.array_equal(quantize(x),seq['SQ3'][t]),('SQ3',t)
            assert stats['history_weight_max']<=.5+1e-6
        # Local frame 2 at every non-initial window uses both preceding frames.
        if start:
            assert r['frame_stats'][0]['frame']==2 and len(r['frame_stats'][0]['lags'])==2
        tested.append(dict(start=start,frames=count,history_begin=begin,all_frames_replayed=True))
        flows._mmap.close();del flows
        print('VERIFIED WINDOW',start,end,flush=True)
    assert cursor==n and np.array_equal(seq['SQ3'][0],seq['GS25'][0])
    save_json(run/'verification.json',dict(complete=True,frames=n,windows=tested,prior_input_hashes_unchanged=True,
        all_frames_replayed_exact=True,cross_window_two_frame_history_verified=True,guide_redecode_exact=True,
        source_read_guard_verified=True,short_window_regression=regression,current_config_sha256=CONFIG,
        historical_metadata_guard_extension=guard_extension,
        h265_complete_file_match=h265,verifier_sha256=file_sha(__file__),subjective_review='pending'))
    snap=run/'verification_source';snap.mkdir();shutil.copy2(__file__,snap/Path(__file__).name)


def measurements(run):
    seq=sequences(run);n=len(seq['source']);rows=[];previous=None
    for i in range(n):
        current={name:a[i].astype(np.float32) for name,a in seq.items()}
        source=current['source'];source_delta=None if previous is None else source-previous['source']
        for name,x in current.items():
            error=x-source
            gray=cv2.cvtColor(seq[name][i],cv2.COLOR_RGB2GRAY).astype(np.float32)
            fine=gray-cv2.GaussianBlur(gray,(0,0),1.2)
            row=dict(frame=i,seconds=i/15,method=name,pixels=288*512,
                rgb_sse=float(np.square(error).sum()),rgb_abs_sum=float(np.abs(error).sum()),
                fine_amplitude=float(np.abs(fine).mean()))
            if previous is None:row.update(change_mean=None,source_change_mean=None,change_error_mean=None)
            else:row.update(change_mean=float(np.abs(x-previous[name]).mean()),
                source_change_mean=float(np.abs(source_delta).mean()),
                change_error_mean=float(np.abs(x-previous[name]-source_delta).mean()))
            rows.append(row)
        previous=current
    out=run/'evaluation';out.mkdir()
    with (out/'per_frame.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result={}
    for start,end,label in [(0,n,'full')]+[(i,min(i+150,n),f'{i:06d}') for i in range(0,n,150)]:
        result[label]={}
        for name in seq:
            rr=[r for r in rows if r['method']==name and start<=r['frame']<end]
            mse=sum(r['rgb_sse'] for r in rr)/(len(rr)*288*512*3)
            result[label][name]=dict(frames=len(rr),rgb_mae=sum(r['rgb_abs_sum'] for r in rr)/(len(rr)*288*512*3),
                psnr_db=float(10*np.log10(255**2/mse)) if mse else None,
                fine_amplitude=float(np.mean([r['fine_amplitude'] for r in rr])))
    source_rows=[r for r in rows if r['method']=='source' and r['frame']>0]
    peaks=[]
    for row in sorted(source_rows,key=lambda r:r['source_change_mean'],reverse=True):
        if all(abs(row['frame']-x['frame'])>=15 for x in peaks):peaks.append(dict(frame=row['frame'],seconds=row['seconds'],source_change_mean=row['source_change_mean']))
        if len(peaks)==5:break
    save_json(out/'summary.json',dict(complete=True,scores=result,source_change_review_points=peaks,
        source_change_points_are_not_verified_scene_cuts=True,
        measurement_note='Basic same-frame fidelity, fine amplitude and unaligned RGB changes; not comparable to prior source-flow temporal MAE or motion-ROI amplitude. No subjective adoption.'))
    print('BASIC MEASUREMENTS COMPLETE',flush=True)


def media(run):
    assert load_json(run/'verification.json')['complete']
    plan=load_json(run/'execution_plan.json');hr=load_json(run/'h265/receipt.json');seq=sequences(run);n=plan['frames']
    out=run/'previews';out.mkdir();path=out/'82秒_四宫格_原片_H265_原GLC_SQ3.mp4'
    width,height=1024,688;positions=[(0,56),(512,56),(0,400),(512,400)]
    names=('source','H265','raw','SQ3');panels=[seq[k] for k in names]
    base=plan['base_bytes']*8/plan['seconds']/1000;aux=plan['auxiliary_bytes']*8/plan['seconds']/1000
    headings=['原片（同分辨率采样）',f'H.265   {hr["kbps"]:.2f} kbps',f'原始 GLC   {base:.2f} kbps',f'SQ3   {plan["total_kbps"]:.2f} kbps']
    notes=['82.6 秒 | 512×288 | 15 FPS','完整文件匹配SQ3总码率 | medium','仅基础码流',f'本体 {base:.2f} + 辅助 {aux:.2f} kbps']
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    canvas=Image.new('RGB',(width,height),(20,29,40));draw=ImageDraw.Draw(canvas)
    for j,(title,note) in enumerate(zip(headings,notes,strict=True)):
        x,y=(j%2)*512,(j//2)*344
        assert draw.textbbox((0,0),title,font=font)[2]<492 and draw.textbbox((0,0),note,font=small)[2]<492
        draw.text((x+10,y+4),title,font=font,fill=(239,246,252));draw.text((x+10,y+31),note,font=small,fill=(169,198,218))
    ids=sorted({0,619,1238,*[t for b in range(150,n,150) for t in (b-1,b,b+1)]})
    reviewed=load_json(run/'evaluation/summary.json')['source_change_review_points']
    ids=sorted(set(ids)|{t for p in reviewed for t in (p['frame']-1,p['frame'],p['frame']+1) if 0<=t<n})
    command=[str(FFMPEG),'-hide_banner','-loglevel','error','-nostdin','-f','rawvideo','-pixel_format','rgb24',
        '-video_size',f'{width}x{height}','-framerate','15','-i','pipe:0','-an',
        '-vf','scale=out_color_matrix=bt709:out_range=tv,format=yuv420p','-color_range','tv','-colorspace','bt709',
        '-color_primaries','bt709','-color_trc','bt709','-c:v','libx264','-preset','fast','-crf','12','-threads','4','-movflags','+faststart','-n',str(path)]
    with (out/'encode.log').open('wb') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        try:
            for i in range(n):
                draw.rectangle((0,29,511,55),fill=(20,29,40))
                draw.text((10,31),f'15 FPS | {i/15:05.2f} / 82.60 秒 | 帧 {i}',font=small,fill=(169,198,218))
                for panel,pos in zip(panels,positions,strict=True):canvas.paste(Image.fromarray(panel[i]),pos)
                if i in ids:
                    assembled=np.asarray(canvas)
                    for panel,(x,y) in zip(panels,positions,strict=True):assert np.array_equal(assembled[y:y+288,x:x+512],panel[i])
                process.stdin.write(canvas.tobytes())
                if i and i%300==0:print('FOURWAY ENCODE',i,'/',n,flush=True)
            process.stdin.close();assert process.wait(timeout=60)==0
        finally:
            if process.poll() is None:process.kill();process.wait()
    stream=json.loads(subprocess.check_output([str(FFMPEG.with_name('ffprobe.exe')),'-v','error','-select_streams','v:0','-count_frames',
        '-show_entries','stream=width,height,avg_frame_rate,nb_read_frames,duration','-of','json',str(path)]))['streams'][0]
    assert int(stream['nb_read_frames'])==n and Fraction(stream['avg_frame_rate'])==15
    assert (stream['width'],stream['height'])==(width,height) and abs(float(stream['duration'])-82.6)<.001
    errors={};contact=Image.new('RGB',(1024,344*3),(20,29,40));event_contact=Image.new('RGB',(1024,344*5),(20,29,40))
    eventids=[p['frame'] for p in reviewed]
    with (out/'decode_verify.log').open('wb') as log:
        process=subprocess.Popen([str(FFMPEG),'-v','error','-i',str(path),'-vf','scale=in_color_matrix=bt709:in_range=tv,format=rgb24','-f','rawvideo','pipe:1'],stdout=subprocess.PIPE,stderr=log)
        try:
            for i in range(n):
                buf=process.stdout.read(height*width*3);assert len(buf)==height*width*3
                frame=np.frombuffer(buf,np.uint8).reshape(height,width,3)
                if i in ids:
                    e=[float(np.abs(frame[y:y+288,x:x+512].astype(np.float32)-panel[i]).mean()) for panel,(x,y) in zip(panels,positions,strict=True)]
                    assert max(e)<6,(i,e);errors[str(i)]=e
                if i in (0,619,1238):
                    im=Image.fromarray(frame);im.save(out/f'verified_frame_{i:06d}.png')
                    contact.paste(im.resize((512,344)),(0,(0,619,1238).index(i)*344))
                    # Keep full-width four-way layout in a separate native PNG.
                    ImageDraw.Draw(contact).text((520,(0,619,1238).index(i)*344+12),f'{i/15:.2f}s / frame {i}',font=font,fill='white')
                if i in eventids:
                    event_contact.paste(Image.fromarray(frame).resize((512,344)),(0,eventids.index(i)*344))
                    ImageDraw.Draw(event_contact).text((520,eventids.index(i)*344+12),f'Review {i/15:.2f}s / f{i}',font=font,fill='white')
            assert not process.stdout.read(1) and process.wait(timeout=30)==0
        finally:
            if process.poll() is None:process.kill();process.wait()
    contact.save(out/'首中尾四宫格.png');event_contact.save(out/'高变化时段核对.png')
    save_json(out/'verification.json',dict(complete=True,path=str(path),frames=n,fps=15,duration_seconds=82.6,stream=stream,
        layout=list(names),headings=headings,notes=notes,positions=positions,preencode_tiles_exact=True,
        all_frames_decoded=True,checked_frames=ids,panel_MAE=errors,sha256=file_sha(path),bytes=path.stat().st_size,
        source_change_review_points=reviewed,original_speed_subjective_review='pending',
        viewing_encoding='x264 fast CRF12 BT709 YUV420; viewing-file size is not communication bitrate'))
    print('FOURWAY VERIFIED',path,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);verify(a.run);measurements(a.run);media(a.run)
