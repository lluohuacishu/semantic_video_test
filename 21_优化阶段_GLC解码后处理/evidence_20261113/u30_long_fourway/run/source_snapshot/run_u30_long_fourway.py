"""User-requested full U30 viewing extension; does not change prior screens."""
import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from runtime_speed_common import STAGE,REPO,DATA,WORKSPACE,FFMPEG,file_sha,load_json,save_json,check_inputs
from preserve_sq3_accepted import DEST,LONG,CONFIG
from sq3_detail_filter import SETTINGS,enhance

PRIOR=DATA/'sq3_detail_20261113_02'
RUN=DATA/'sq3_u30_long_fourway_20261113_01'
VIEW=WORKSPACE/'05_成果与观看/GLC_U30长片四宫格_20261113'
MOVIE='82秒_四宫格_原片_U30_原GLC_SQ3.mp4'
REPORT=STAGE/'U30长片四宫格_原片_U30_GLC_SQ3_20261113.md'
QUOTE='你跑一下长视频我看看呢，还是四宫格对照，但是把右上角的h.265替换成这个U30'

def freeze():
    RUN.mkdir(exist_ok=False)
    old=load_json(LONG/'execution_plan.json');prior=load_json(PRIOR/'execution_plan.json')
    assert load_json(LONG/'verification.json')['complete']
    assert file_sha(STAGE/'sq3_detail_filter.py')==prior['code_sha256']['sq3_detail_filter.py']
    assert file_sha(STAGE/'当前采用配置.json')==CONFIG
    sq=DEST/'outputs/long_SQ3.npy'
    assert file_sha(sq)==load_json(DEST/'回退配置.json')['outputs']['long_SQ3.npy']['sha256']
    files=[Path(old['source']),Path(old['raw']),sq,Path(old['source_metadata']),
        LONG/'execution_plan.json',LONG/'verification.json',PRIOR/'execution_plan.json',PRIOR/'short_screen.json',DEST/'manifest.json']
    plan=dict(date='2026-11-13',frozen_before_outputs=True,request=QUOTE,
        reason='Explicit user request to view U30 on the full video despite prior numeric screening failure. Prior thresholds/results are preserved; this is not automatic adoption.',
        profile='SQ3_U30',gain=.30,settings=SETTINGS,
        source=old['source'],raw=old['raw'],sq3=str(sq),source_metadata=old['source_metadata'],
        frames=old['frames'],fps=old['fps'],seconds=old['seconds'],
        base_bytes=old['base_bytes'],total_bytes=old['total_bytes'],total_kbps=old['total_kbps'],
        layout=['source','SQ3_U30','raw_GLC','SQ3'],panel_pixels=[512,288],canvas_pixels=[1024,688],
        no_new_transmission_bytes=True,lookahead_frames_added=0,new_flow_inference=False,
        no_source_pixels_at_receiver=True,automatic_adoption=False,preferred_fallback='SQ3',
        current_config_sha256=CONFIG,input_sha256={str(p):file_sha(p) for p in files},
        code_sha256={n:file_sha(STAGE/n) for n in ('run_u30_long_fourway.py','sq3_detail_filter.py')})
    save_json(RUN/'execution_plan.json',plan)
    for name in plan['code_sha256']:
        d=RUN/'source_snapshot'/name;d.parent.mkdir(exist_ok=True);shutil.copy2(STAGE/name,d)
    sp=REPO/'05_资料与索引/会话收束_20261110/接续状态.json';shutil.copy2(sp,RUN/'state_before_run.json')
    state=load_json(sp);state.update(status='U30_long_fourway_running',active_experiment_plan=str(RUN/'execution_plan.json'),
        user_latest_decision=QUOTE,current_session_work='User explicitly requests full U30 in top-right; preserve source/rawGLC/SQ3 remaining panels.',
        experiment_execution_gate=dict(status='running',currently_running=True,automatic_next_run=False))
    save_json(sp,state)
    print('FROZEN full U30 four-way',plan['frames'],plan['seconds'],flush=True)

def receive():
    p=load_json(RUN/'execution_plan.json')
    for name,h in p['code_sha256'].items():assert file_sha(STAGE/name)==h
    out=RUN/'receive';out.mkdir();src=Path(p['sq3']).resolve();reads=set()
    forbidden={Path(p[k]).resolve() for k in ('source','raw','source_metadata')}
    def guard(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        f=Path(os.fsdecode(args[0])).resolve()
        if f==src:reads.add(str(f));return
        if f.is_relative_to(out):return
        if f in forbidden or f.is_relative_to(DATA) or (f.is_relative_to(DEST) and f!=src) or f.suffix.lower() in {'.npy','.npz','.mp4','.glcsnd','.glcg2','.glcguide'}:
            raise RuntimeError('U30 receiver forbidden input: '+str(f))
    sys.addaudithook(guard);blocked=[]
    for f in forbidden:
        try:f.read_bytes()
        except RuntimeError:blocked.append(str(f))
        else:raise AssertionError('Source guard did not block input')
    assert file_sha(src)==p['input_sha256'][str(src)]
    sq=np.load(src,mmap_mode='r',allow_pickle=False);assert sq.shape==(p['frames'],288,512,3) and sq.dtype==np.uint8
    a=np.lib.format.open_memmap(out/'U30.npy',mode='w+',dtype=np.uint8,shape=sq.shape)
    tick=time.perf_counter();changed=0;maxdiff=0
    for i,x in enumerate(sq):
        a[i]=enhance(x,p['gain']);d=a[i].astype(np.int16)-x
        assert np.all(d==d[...,:1]);maxdiff=max(maxdiff,int(np.abs(d).max()));assert maxdiff<=3
        changed+=int(np.any(d!=0,axis=2).sum())
        if i and i%300==0:print('U30',i,'/',len(sq),flush=True)
    a.flush();seconds=time.perf_counter()-tick
    save_json(out/'receipt.json',dict(complete=True,frames=len(sq),fps=p['fps'],seconds=seconds,
        input_sha256=file_sha(src),output_sha256=file_sha(out/'U30.npy'),source_reads_blocked=True,
        blocked=blocked,reads=sorted(reads),maximum_RGB_change=maxdiff,
        changed_pixel_fraction=changed/(len(sq)*288*512),all_pixels_shared_integer_RGB_correction=True,
        extra_transmission_bytes=0,added_temporal_state=False))
    print('U30 RECEIVED',len(sq),'frames',round(seconds,2),'seconds',flush=True)

def verify():
    p=load_json(RUN/'execution_plan.json');check_inputs(p['input_sha256'])
    rec=load_json(RUN/'receive/receipt.json');assert rec['complete'] and rec['frames']==p['frames']
    assert rec['source_reads_blocked'] and len(rec['blocked'])==3 and rec['reads']==[p['sq3']]
    sq=np.load(p['sq3'],mmap_mode='r');a=np.load(RUN/'receive/U30.npy',mmap_mode='r')
    assert a.shape==sq.shape and a.dtype==np.uint8 and file_sha(RUN/'receive/U30.npy')==rec['output_sha256']
    for i in range(len(a)):assert np.array_equal(a[i],enhance(sq[i],.30)),i
    regression={}
    for case in ('video1','video2','video3','video4'):
        prior=np.load(PRIOR/case/'receive/SQ3_U30/rgb.npy',mmap_mode='r')
        baseline=np.load(DEST/'outputs'/f'{case}_SQ3.npy',mmap_mode='r')
        for i in range(len(prior)):assert np.array_equal(prior[i],enhance(baseline[i],.30))
        regression[case]=dict(frames=len(prior),prior_U30_replayed_exact=True)
    assert file_sha(STAGE/'当前采用配置.json')==CONFIG
    save_json(RUN/'verification.json',dict(complete=True,all_1239_frames_replayed_exact=True,
        short_regression=regression,input_hashes_unchanged=True,source_guard_verified=True,
        formal_config_unchanged=True,prior_failed_numeric_screen_preserved=True,automatic_adoption=False))
    print('VERIFIED complete long U30 and all four prior clips',flush=True)

def media():
    assert load_json(RUN/'verification.json')['complete'];p=load_json(RUN/'execution_plan.json')
    seq=dict(source=np.load(p['source'],mmap_mode='r'),U30=np.load(RUN/'receive/U30.npy',mmap_mode='r'),
        raw=np.load(p['raw'],mmap_mode='r'),SQ3=np.load(p['sq3'],mmap_mode='r'))
    names=('source','U30','raw','SQ3');panels=[seq[k] for k in names];n=p['frames']
    assert all(a.shape==(n,288,512,3) for a in panels)
    out=RUN/'previews';out.mkdir();path=out/MOVIE;positions=[(0,56),(512,56),(0,400),(512,400)]
    base=p['base_bytes']*8/p['seconds']/1000
    titles=['原片（同分辨率采样）',f'U30   {p["total_kbps"]:.2f} kbps',f'原始 GLC   {base:.2f} kbps',f'SQ3   {p["total_kbps"]:.2f} kbps']
    notes=['15 FPS | 82.60 秒 | 512×288','SQ3 + 0.30 温和锐化 | 实验对照','仅基础码流','已保留版本 | 无额外锐化']
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    canvas=Image.new('RGB',(1024,688),(20,29,40));draw=ImageDraw.Draw(canvas)
    for j,(title,note) in enumerate(zip(titles,notes,strict=True)):
        x,y=(j%2)*512,(j//2)*344
        assert draw.textbbox((0,0),title,font=font)[2]<492 and draw.textbbox((0,0),note,font=small)[2]<492
        draw.text((x+10,y+4),title,font=font,fill=(239,246,252));draw.text((x+10,y+31),note,font=small,fill=(169,198,218))
    events=load_json(LONG/'evaluation/summary.json')['source_change_review_points']
    ids=sorted({0,n//2,n-1,*[i for b in range(150,n,150) for i in (b-1,b,b+1)],
        *[i for r in events for i in (r['frame']-1,r['frame'],r['frame']+1) if 0<=i<n]})
    command=[str(FFMPEG),'-hide_banner','-loglevel','error','-nostdin','-f','rawvideo','-pixel_format','rgb24',
        '-video_size','1024x688','-framerate','15','-i','pipe:0','-an',
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
                if i and i%300==0:print('ENCODE FOURWAY',i,'/',n,flush=True)
            process.stdin.close();assert process.wait(timeout=60)==0
        finally:
            if process.poll() is None:process.kill();process.wait()
    stream=json.loads(subprocess.check_output([str(FFMPEG.with_name('ffprobe.exe')),'-v','error','-select_streams','v:0','-count_frames',
        '-show_entries','stream=width,height,avg_frame_rate,nb_read_frames,duration','-of','json',str(path)]))['streams'][0]
    assert int(stream['nb_read_frames'])==n and Fraction(stream['avg_frame_rate'])==15
    assert (stream['width'],stream['height'])==(1024,688) and abs(float(stream['duration'])-p['seconds'])<.001
    errors={};contact=Image.new('RGB',(1024,688*3),(20,29,40))
    with (out/'decode_verify.log').open('wb') as log:
        process=subprocess.Popen([str(FFMPEG),'-v','error','-xerror','-i',str(path),'-vf',
            'scale=in_color_matrix=bt709:in_range=tv,format=rgb24','-f','rawvideo','pipe:1'],stdout=subprocess.PIPE,stderr=log)
        try:
            for i in range(n):
                buf=process.stdout.read(1024*688*3);assert len(buf)==1024*688*3
                frame=np.frombuffer(buf,np.uint8).reshape(688,1024,3)
                if i in ids:
                    e=[float(np.abs(frame[y:y+288,x:x+512].astype(np.float32)-panel[i]).mean()) for panel,(x,y) in zip(panels,positions,strict=True)]
                    assert max(e)<6,(i,e);errors[str(i)]=e
                if i in (0,n//2,n-1):
                    im=Image.fromarray(frame);im.save(out/f'verified_frame_{i:06d}.png');contact.paste(im,(0,(0,n//2,n-1).index(i)*688))
            assert not process.stdout.read(1) and process.wait(timeout=30)==0
        finally:
            if process.poll() is None:process.kill();process.wait()
    contact.save(out/'首中尾四宫格核对.png')
    save_json(out/'verification.json',dict(complete=True,path=str(path),stream=stream,layout=names,positions=positions,
        headings=titles,notes=notes,preencode_tiles_exact=True,all_frames_decoded=True,checked_frames=ids,
        panel_MAE=errors,sha256=file_sha(path),bytes=path.stat().st_size,original_speed_subjective_review='pending',
        source_change_review_points=events,no_extra_display_sharpening=True,
        viewing_encoding='x264 fast CRF12 BT709 YUV420; file size is not communication bitrate'))
    print('FOURWAY VERIFIED',path,flush=True)

def rel(origin,target):return os.path.relpath(target,origin.parent).replace('\\','/')

def publish():
    p=load_json(RUN/'execution_plan.json');check_inputs(p['input_sha256'])
    v=load_json(RUN/'previews/verification.json');assert v['complete'] and load_json(RUN/'visual_inspection.json')['complete']
    assert file_sha(STAGE/'当前采用配置.json')==CONFIG
    VIEW.mkdir();files={}
    for src in sorted((RUN/'previews').iterdir()):
        if src.suffix not in ('.mp4','.png'):continue
        dst=VIEW/src.name;shutil.copy2(src,dst);assert file_sha(dst)==file_sha(src)
        files[dst.name]=dict(sha256=file_sha(dst),bytes=dst.stat().st_size,original=str(src),independent_copy=True)
    save_json(VIEW/'media_manifest.json',dict(complete=True,files=files))
    movie=VIEW/MOVIE
    body=['# U30 完整长片四宫格（2026-11-13）','',
        '**已按用户要求完成完整82.6秒原速四宫格，将右上角H.265替换为U30。**','',
        f'[播放完整四宫格]({rel(REPORT,movie)})','',
        '| 左上 | 右上 |','| --- | --- |','| 原片 | U30（SQ3后加固定0.30亮度锐化） |',
        '| 左下 | 右下 |','| 原始GLC | 已保留的SQ3 |','',
        '四路采用相同1239帧、15FPS、512×288时间线，合成1024×688，82.6秒原速，无音频、慢放或补尾帧。保留上一版中文标题、时间和帧号布局。','',
        'U30沿用上一轮冻结算法：仅在已完成的SQ3像素上做sigma=1、增益0.30的亮度高通增强，共同校正最多±3级；统一量化后加到RGB。不重新计算GLC/Q9/SQ3，不增加光流、历史帧或通信字节，不按场景调参。完整长片U30逐帧重放一致，四短片旧U30也全部逐帧复现。','',
        '本次长片扩展是用户明确要求的观看实验。上一轮U30未通过数值筛查的记录和阈值保留，不能将此次生成对照视为自动通过或正式采用。用户认可的SQ3独立回退版本及正式配置均未改。','',
        f'U30和SQ3均为基础流加辅助流合计{p["total_kbps"]:.4f} kbps；原GLC仅基础流{p["base_bytes"]*8/p["seconds"]/1000:.4f} kbps。H.265不在本次四宫格中。观看文件采用高质量H.264，文件体积不是传输码率。','',
        f'全部1239帧完成解码核验；首中尾、8处处理边界前后和既有源高变化时点共{len(v["checked_frames"])}个时点逐面板核对。源高变化不等于已确认切镜。静帧核对不替代原速观看。','',
        f'[执行计划]({rel(REPORT,RUN/"execution_plan.json")}) · [算法核验]({rel(REPORT,RUN/"verification.json")}) · [媒体核验]({rel(REPORT,RUN/"previews/verification.json")}) · [SQ3保留版本]({rel(REPORT,DEST/"README.md")})','']
    REPORT.write_text('\n'.join(body),encoding='utf-8')
    (VIEW/'README.md').write_text('\n'.join(['# U30 长片四宫格','',f'[播放完整82.6秒四宫格]({MOVIE})','',
        '左上：原片　｜　右上：U30','左下：原始GLC　｜　右下：已保留SQ3','',
        '1239帧，15FPS原速，无音频。U30使用上一轮相同0.30强度；其此前未通过数值筛查，此处按用户要求提供完整长片观看，不自动替换SQ3。','',
        f'[本轮说明]({rel(VIEW/"README.md",REPORT)})','']),encoding='utf-8')
    result=dict(complete=True,date='2026-11-13',request=QUOTE,run=str(RUN),viewing=str(VIEW),movie=str(movie),report=str(REPORT),
        layout=p['layout'],frames=p['frames'],fps=p['fps'],seconds=p['seconds'],profile='SQ3_U30',
        user_requested_extension=True,prior_numeric_failure_preserved=True,subjective_review='pending',selected=None,
        automatic_adoption=False,preferred_fallback='SQ3',formal_config_unchanged=True)
    save_json(RUN/'round_result.json',result)
    sp=REPO/'05_资料与索引/会话收束_20261110/接续状态.json';state=load_json(sp)
    state['previous_latest_experiment_before_u30_long']=state.get('latest_experiment')
    state.update(status='U30_long_fourway_complete',active_experiment_plan=None,latest_experiment=result,
        current_session_work='User-requested full U30 four-way complete; top-right U30, top-left source, bottom-left raw GLC, bottom-right preserved SQ3.',
        new_experiment_report=str(REPORT.relative_to(REPO)),latest_analysis_report=str(REPORT.relative_to(REPO)),
        experiment_execution_gate=dict(status='completed_round',currently_running=False,automatic_next_run=False),
        next_experiment_plan=dict(status='awaiting_U30_long_viewing_feedback',automatic_next_run=False),
        next_priority=['Review U30 versus SQ3 simultaneously in the right column at original speed'])
    state['workspace_paths']['latest_U30_long_viewing']=str(VIEW/'README.md');save_json(sp,state)
    for entry in (REPO/'README.md',REPO/'交接文档.md',REPO/'当前项目状态与接续.md',REPO/'新会话接续.txt',STAGE/'README.md',
            REPO/'05_资料与索引/会话收束_20261110/文件索引.md',WORKSPACE/'05_成果与观看/README.md',WORKSPACE/'02_实验数据/README.md'):
        old=entry.read_text(encoding='utf-8');backup=RUN/'entries_before'/entry.relative_to(WORKSPACE);backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(entry,backup)
        entry.write_text(f'**最新（2026-11-13）：按用户要求，U30完整82.6秒四宫格已完成。** 左上原片、右上U30、左下原GLC、右下SQ3。\n\n[播放完整对照]({rel(entry,movie)}) · [本轮说明]({rel(entry,REPORT)})\n\nSQ3回退版本不变；此前U30数值筛查失败记录保留，本次为明确要求的长片观看。当前无活动实验。\n\n---\n\n'+old,encoding='utf-8')
    evidence=STAGE/'evidence_20261113/u30_long_fourway';evidence.mkdir()
    for src in sorted(RUN.rglob('*')):
        if src.is_file() and src.suffix in ('.json','.py') and 'entries_before' not in str(src.relative_to(RUN)):
            dst=evidence/'run'/src.relative_to(RUN);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    shutil.copy2(REPORT,evidence/REPORT.name)
    manifest={str(f.relative_to(evidence)).replace('\\','/'):dict(sha256=file_sha(f),bytes=f.stat().st_size) for f in sorted(evidence.rglob('*')) if f.is_file()}
    save_json(evidence/'manifest.json',dict(complete=True,files=manifest,self_excluded=True))
    for name,r in manifest.items():assert file_sha(evidence/name)==r['sha256']
    for name,r in load_json(DEST/'manifest.json')['files'].items():assert file_sha(DEST/name)==r['sha256']
    check_inputs(p['input_sha256']);assert file_sha(STAGE/'当前采用配置.json')==CONFIG
    save_json(RUN/'publication.json',dict(complete=True,viewing=str(VIEW),movie=str(movie),report=str(REPORT),
        media_count=len(files),evidence_files=len(manifest),accepted_archive_unchanged=True,formal_config_unchanged=True,no_active_experiment=True))
    print('PUBLISHED',movie,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','receive','finish','publish']);a=parser.parse_args()
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    if a.action=='freeze':freeze()
    elif a.action=='receive':receive()
    elif a.action=='finish':verify();media()
    else:publish()
