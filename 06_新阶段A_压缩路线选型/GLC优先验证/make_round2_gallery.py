"""Scientific comparison sheets and local HTML playback gallery from decoded pixels."""
from pathlib import Path
import argparse
import csv
import hashlib
import html
import json
import shutil
import subprocess
import sys

import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--previous-evidence',type=Path,required=True);p.add_argument('--ffmpeg',type=Path,required=True);p.add_argument('--short-preview-only',action='store_true');p.add_argument('--playback-only',action='store_true')
a=p.parse_args();sys.stdout.reconfigure(encoding='utf-8');a.output.mkdir(parents=True,exist_ok=True);fig=a.output/'figures';fig.mkdir(exist_ok=True)
fontpath='C:/Windows/Fonts/msyh.ttc'
def font(n):return ImageFont.truetype(fontpath,n)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def frame(clip,run,index):
    path=a.run/clip/('source' if run is None else run+'/receiver/frames')/f'im{index+1:05d}.png'
    with Image.open(path) as im:
        im=im.convert('RGB')
        return im.copy() if im.size==(1280,720) else im.resize((1280,720),Image.Resampling.LANCZOS)
def label(clip,run,title):
    if run is None:return title+' | 源图'
    root=a.run/clip/run
    m=read(root/('metrics.json' if (root/'metrics.json').exists() else ('verified.json' if (root/'verified.json').exists() else 'selection.json')))
    return title+f" | {m.get('file_bytes',m.get('bytes'))/1000:.1f} KB"
def sheet(name,title,subtitle,clip,index,panels,box=None):
    w=384; cols=len(panels); header=98; imageh=216;cropheight=216 if box else 0
    canvas=Image.new('RGB',(w*cols,header+imageh+cropheight+70),'#f3f5f7');d=ImageDraw.Draw(canvas)
    d.rectangle((0,0,w*cols,69),fill='#18283a');d.text((16,8),title,font=font(25),fill='white');d.text((16,42),subtitle,font=font(15),fill='#d2e3ee')
    for c,(run,title_) in enumerate(panels):
        x=c*w;im=frame(clip,run,index)
        d.text((x+10,74),label(clip,run,title_),font=font(18),fill='#152b40')
        canvas.paste(im.resize((w,imageh),Image.Resampling.LANCZOS),(x,header))
        if box:
            x0,y0,x1,y1=box
            d.rectangle((x+x0*w/1280,header+y0*imageh/720,x+x1*w/1280,header+y1*imageh/720),outline='#ffb54d',width=2)
            crop=im.crop(box);crop.thumbnail((w-16,cropheight-18),Image.Resampling.NEAREST)
            # Fixed dimensions chosen so this is integer nearest-neighbor magnification.
            factor=min((w-16)//(x1-x0),(cropheight-18)//(y1-y0))
            crop=im.crop(box).resize(((x1-x0)*factor,(y1-y0)*factor),Image.Resampling.NEAREST)
            canvas.paste(crop,(x+(w-crop.width)//2,header+imageh+10))
        if run:
            mp=a.run/clip/run/'metrics.json';y=header+imageh+cropheight+6
            if mp.exists():
                m=read(mp)
                d.text((x+10,y),f"DISTS {m['dists_mean']:.4f} / {len(m['dists_frames'])}帧（越低越好）",font=font(16),fill='#263c50')
                d.text((x+10,y+25),f"20 kbps 文件等待 {m['payload_wait_seconds']['20']:.1f} 秒",font=font(15),fill='#263c50')
            else:d.text((x+10,y),'过程预览：感知指标尚未计算',font=font(16),fill='#263c50')
        else:d.text((x+10,header+imageh+cropheight+9),'局部为同坐标最近邻放大；DISTS为预设帧均值' if box else '统一720p评价；DISTS为预设采样帧均值',font=font(14),fill='#263c50')
    canvas.save(fig/name)

if a.short_preview_only:
    sheet('GLC三档过程预览.png','GLC：45.3 / 27.8 / 16.7 KB 三档重建','video6 源帧90；同一50帧视频；下方为小鱼位置3倍最近邻放大','short',40,[(None,'原图'),('glc_720_g64','GLC 720p'),('glc_540_g64','GLC 540p'),('glc_360_g64','GLC 360p')],(1016,216,1128,280))
    delta=read(a.run/'short/av1_720_for_540_g64/selection.json')['relative_error']
    sheet('同预算540p过程预览.png','约27.8 KB预算：GLC与H.265 / AV1',f'video6 源帧90；原生720p传统编码对照；AV1完整文件预算偏差 {delta:+.2%}','short',40,[(None,'原图'),('glc_540_g64','GLC 540p'),('h265_720_for_540_g64','H.265 720p'),('av1_720_for_540_g64','AV1 720p')],(1016,216,1128,280))
    print('Saved short previews',flush=True)
    sys.exit(0)

summary=[]
if not a.playback_only:
    for clip in ['short','long108','long91','long24']:
        result=read(a.run/clip/'results.json')
        evidence=a.output/clip;evidence.mkdir(exist_ok=True)
        save(evidence/'inputs.json',result['input'])
        save(evidence/'metrics.json',result)
        commands={str(p.relative_to(a.run/clip)).replace('\\','/'):read(p) for p in (a.run/clip).rglob('*.log.json')}
        save(evidence/'commands.json',commands)
        for r in result['results']:
            summary.append({'clip':clip,**{k:v for k,v in r.items() if k not in ['config','per_frame','evaluation']},'codec':r['config']['codec'],'height':r['config']['height'],'target_height':r['config'].get('target_height',r['config']['height']),'gop':r['config']['gop'],'budget_error':r['config'].get('relative_error',0)})
    save(a.output/'summary.json',summary)
    columns=['clip','run','codec','height','target_height','gop','file_bytes','duration_seconds','wait_20_kbps_s','wait_30_kbps_s','wait_40_kbps_s','global_psnr_db','fish_psnr_db','fish_pixels','ms_ssim_mean','dists_mean','dists_sample_frames','temporal_error_delta_mae_mean','budget_error']
    with (a.output/'summary.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
        for row in summary:
            flat={key:row.get(key) for key in columns}
            flat.update({f'wait_{rate}_kbps_s':row['payload_wait_seconds'][str(rate)] for rate in [20,30,40]})
            flat['dists_sample_frames']=len(row['dists_frames']);writer.writerow(flat)
    save(a.output/'perceptual_weight_source.json',read(a.run/'assets/torch/hub/checkpoints/vgg16_source.json'))
    save(a.output/'environment_provenance.json',read(a.run/'environment_provenance.json'))
    save(a.output/'av1_keyframe_correction.json',read(a.run/'av1_keyframe_audit/audit.json'))
    save(a.output/'verification.json',read(a.run/'verification.json'))

    sheet('01_GLC分辨率与细节.png','GLC：文件更小以后，小鱼还能保留多少？','video6 源帧90；同一50帧视频、q0 / GOP64 / reset32；工作分辨率不同','short',40,[(None,'原图'),('glc_720_g64','GLC 720p'),('glc_540_g64','GLC 540p'),('glc_360_g64','GLC 360p')],(1016,216,1128,280))
    for h in [540,360]:
        sheet(f'02_同预算_{h}p与原生对照.png',f'低预算对照：GLC {h}p 与原生720p编码','源帧90；按完整文件大小匹配，具体偏差见报告；下方为相同位置局部','short',40,[(None,'原图'),(f'glc_{h}_g64',f'GLC {h}p'),(f'h265_720_for_{h}_g64','H.265 720p'),(f'av1_720_for_{h}_g64','AV1 720p')],(1016,216,1128,280))
        sheet(f'03_同预算_同分辨率_{h}p.png',f'控制输入分辨率：三种编码均为{h}p','源帧90；使用同一RGB下采样输入，接收端统一还原到720p','short',40,[(None,'原图'),(f'glc_{h}_g64','GLC'),(f'h265_{h}_for_{h}_g64','H.265'),(f'av1_{h}_for_{h}_g64','AV1')],(1016,216,1128,280))

    # Every GT component at every annotated short-clip frame, without outcome selection.
    cm=read(a.previous_evidence/'crop_manifest.json')['frames']
    save(a.output/'crop_manifest.json',read(a.previous_evidence/'crop_manifest.json'))
    for h in [540,360]:
        panels=[(None,'原图'),(f'glc_{h}_g64','GLC'),(f'h265_720_for_{h}_g64','H.265 720p'),(f'av1_720_for_{h}_g64','AV1 720p')]
        allparts=[(int(n),j+1,b) for n,parts in cm.items() for j,b in enumerate(parts)]
        canvas=Image.new('RGB',(1024,76+len(allparts)*206),'white');d=ImageDraw.Draw(canvas)
        d.text((10,7),f'全部GT局部 | GLC {h}p预算 | 最近邻4倍；框为GT范围',font=font(21),fill='black')
        for col,(run,t) in enumerate(panels):d.text((col*256+8,41),t,font=font(19),fill='black')
        for row,(n,j,b) in enumerate(allparts):
            x0,y0,x1,y1=b['bbox_xyxy'];cx=(x0+x1)//2;cy=(y0+y1)//2
            left=max(0,min(1224,cx-28));top=max(0,min(680,cy-20));box=(left,top,left+56,top+40)
            for col,(run,t) in enumerate(panels):
                x=col*256+12;y=76+row*206
                d.text((x,y),f'帧{n} / 目标{j} / {b["mask_pixels"]} px',font=font(14),fill='black')
                canvas.paste(frame('short',run,n-50).crop(box).resize((224,160),Image.Resampling.NEAREST),(x,y+24))
                d.rectangle((x+(x0-left)*4,y+24+(y0-top)*4,x+(x1-left)*4-1,y+24+(y1-top)*4-1),outline='#ffad42',width=1)
        canvas.save(fig/f'04_全部小鱼目标_{h}p预算.png')

    for clip in ['long108','long91','long24']:
        sheet(f'05_{clip}_连续末帧.png',f'{clip}：连续250帧后的重建','固定展示索引249，工作分辨率均540p，GOP64；不能仅凭静态图判断闪烁',clip,249,[(None,'原图'),('glc_540_g64','GLC'),('h265_540_for_540_g64','H.265'),('av1_540_for_540_g64','AV1')])
        info=read(a.run/clip/'inputs.json')
        panels=[(None,'原图'),('glc_540_g64','GLC'),('h265_540_for_540_g64','H.265'),('av1_540_for_540_g64','AV1')]
        times=[0,124,249]
        canvas=Image.new('RGB',(1280,1040),'#f3f5f7');d=ImageDraw.Draw(canvas)
        d.rectangle((0,0,1280,78),fill='#18283a')
        d.text((16,8),f'{clip}：固定首 / 中 / 末帧，同分辨率、相近完整文件预算',font=font(25),fill='white')
        d.text((16,44),'三段均为已有开发资料；静态截图不等于时序验收，完整播放见本地 gallery。',font=font(17),fill='#d2e3ee')
        for row,(run,t) in enumerate(panels):
            y=82+row*238;d.text((10,y+91),t,font=font(18),fill='#152b40')
            for col,n in enumerate(times):
                x=120+col*384
                d.text((x+6,y),f'索引 {n} / {n/info["fps"]:.2f}s',font=font(16),fill='#152b40')
                canvas.paste(frame(clip,run,n).resize((384,216),Image.Resampling.LANCZOS),(x,y+22))
        canvas.save(fig/f'07_{clip}_首中末固定帧.png')

    # Fixed times around a GOP boundary; region set from source geometry, not output quality.
    panels=[(None,'原图'),('glc_540_g64','GLC GOP64'),('glc_540_g128','GLC GOP128'),('h265_540_for_540_g64','H.265 GOP64'),('av1_540_for_540_g64','AV1 GOP64')]
    times=[61,62,63,64,65,66];box=(600,360,984,576)
    canvas=Image.new('RGB',(120+6*256,64+len(panels)*160),'white');d=ImageDraw.Draw(canvas)
    d.text((10,8),'时序局部：video108，跨索引64的I帧边界（GOP64）；横向按时间排列',font=font(23),fill='black')
    for c,n in enumerate(times):d.text((128+c*256,40),f'索引 {n} / {n/25:.2f}s',font=font(16),fill='black')
    for row,(run,t) in enumerate(panels):
        y=64+row*160;d.text((4,y+55),t,font=font(16),fill='black')
        for col,n in enumerate(times):canvas.paste(frame('long108',run,n).crop(box).resize((256,144),Image.Resampling.LANCZOS),(120+col*256,y))
    canvas.save(fig/'06_跨I帧边界时序对照.png')
    save(a.output/'temporal_crop_manifest.json',{'clip':'long108','indices':times,'crop_xyxy':list(box),'source_selection_frame':64,'selection_basis':'Coral structure visible in source frame 64; same coordinates and times for every method.','visual_QA_revision':'Initial water-only crop [240,110,624,326] was replaced with the source coral region during figure review. No metric or frame selection changed.'})

    # Supplementary diagnostic after the long91 fish-region result was observed.
    # Fixed first/middle/last annotated times and largest GT area, never reconstruction quality.
    panels=[(None,'原图'),('glc_540_g64','GLC 540p'),('h265_540_for_540_g64','H.265 540p'),('av1_540_for_540_g64','AV1 540p')]
    info=read(a.run/'long91/inputs.json');crop_records=[]
    canvas=Image.new('RGB',(1536,870),'#f3f5f7');d=ImageDraw.Draw(canvas)
    d.rectangle((0,0,1536,96),fill='#18283a')
    d.text((16,8),'长片 video91：固定标注帧的鱼区细节',font=font(28),fill='white')
    d.text((16,47),'补充诊断：源帧0/120/240各最大GT连通区；边缘目标在源图中已有截断；3倍最近邻放大',font=font(18),fill='#d2e3ee')
    for col,(run,t) in enumerate(panels):d.text((col*384+10,105),label('long91',run,t),font=font(19),fill='#152b40')
    for row,n in enumerate([0,120,240]):
        mask=(np.asarray(Image.open(Path(info['mask_dir'])/f'{n:05d}.png'))>0).astype(np.uint8)
        _,_,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
        bx,by,bw,bh,area=map(int,max(stats[1:],key=lambda s:s[4]))
        left=max(0,min(1280-112,bx+bw//2-56));top=max(0,min(720-64,by+bh//2-32));box=(left,top,left+112,top+64)
        crop_records.append({'source_frame':n,'selection':'largest source-GT connected component','bbox_xyxy':[bx,by,bx+bw,by+bh],'mask_pixels':area,'crop_xyxy':list(box),'touches_source_boundary':bx==0 or by==0 or bx+bw==1280 or by+bh==720})
        for col,(run,t) in enumerate(panels):
            x=col*384+24;y=142+row*239
            d.text((x,y),f'源帧{n} / GT连通区 {area} px',font=font(16),fill='#152b40')
            canvas.paste(frame('long91',run,n).crop(box).resize((336,192),Image.Resampling.NEAREST),(x,y+27))
            d.rectangle((x+(bx-left)*3,y+27+(by-top)*3,x+(bx+bw-left)*3-1,y+27+(by+bh-top)*3-1),outline='#ffb54d',width=1)
    canvas.save(fig/'09_long91_固定GT鱼区补充诊断.png')
    save(a.output/'long91_supplementary_crop_manifest.json',{'post_result_diagnostic':True,'reason':'Investigate the poorer fish-region error despite lower DISTS; does not change primary metrics or predeclared samples.','crops':crop_records})

# Local-only continuous playback; lossless H.264 rendering of displayed RGB.
gallery=a.run/'gallery';gallery.mkdir(exist_ok=True);videos=[]
for clip in ['short','long108','long91','long24']:
    info=read(a.run/clip/'inputs.json');fps=info['fps']
    runs=[None,'glc_540_g64','h265_540_for_540_g64','av1_540_for_540_g64']
    if clip=='short':runs += ['glc_360_g64']
    if clip=='long108':runs += ['glc_540_g128']
    for run_ in runs:
        title='原图' if run_ is None else run_;fn=f'{clip}_{title}.mp4';dst=gallery/fn
        meta=dst.with_suffix('.json')
        if not dst.exists():
            temp=dst.with_name(dst.stem+'.part.mp4')
            cmd=[str(a.ffmpeg),'-hide_banner','-loglevel','error','-nostdin','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(fps),'-i','-','-an','-c:v','libx264rgb','-crf','0','-preset','veryfast','-threads','2','-movflags','+faststart',str(temp)]
            proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
            digest=hashlib.sha256()
            for i in range(info['frames']):
                pixels=frame(clip,run_,i).tobytes();digest.update(pixels);proc.stdin.write(pixels)
            proc.stdin.close()
            if proc.wait():raise RuntimeError(f'Preview failed: {dst}')
            checkcmd=[str(a.ffmpeg),'-v','error','-threads','2','-i',str(temp),'-map','0:v:0','-pix_fmt','rgb24','-c:v','rawvideo','-threads','1','-f','hash','-hash','sha256','-']
            decoded_hash=subprocess.check_output(checkcmd,text=True).strip().split('=')[-1]
            assert decoded_hash==digest.hexdigest(),f'Preview RGB changed: {dst}'
            temp.rename(dst)
            with dst.open('rb') as handle:movie_hash=hashlib.file_digest(handle,'sha256').hexdigest()
            save(meta,{'clip':clip,'run':run_,'frames':info['frames'],'fps':fps,'display_rgb_sha256':digest.hexdigest(),'preview_decoded_rgb_sha256':decoded_hash,'all_rgb_pixels_match':True,'preview_sha256':movie_hash,'preview_bytes':dst.stat().st_size,'encode_command':cmd,'decode_hash_command':checkcmd})
            print(f'PREVIEW {clip}/{title}: {info["frames"]} frames, RGB hash verified',flush=True)
        elif not meta.exists():raise RuntimeError(f'Existing preview has no verification record: {dst}')
        videos.append((clip,title,fn))
groups=[]
for clip in ['short','long108','long91','long24']:
    cards=''.join(f'<section><h3>{html.escape(t)}</h3><video controls preload="metadata" src="{html.escape(fn)}"></video></section>' for c,t,fn in videos if c==clip)
    groups.append(f'<main data-clip="{clip}"'+(' hidden' if clip!='short' else '')+'>'+cards+'</main>')
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>GLC第二轮：连续重建查看</title>
<style>body{font-family:system-ui;margin:24px;background:#edf2f6;color:#173049}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}[hidden]{display:none!important}section{background:white;padding:14px;border-radius:10px}video{width:100%}button,select{padding:10px 18px;margin:5px}</style>
<h1>GLC第二轮：连续重建查看</h1>
<p>按原帧率播放，只播放当前片段组。预览用libx264rgb CRF0保存显示RGB；其文件大小是观看副本，不能用于通信码率。部分浏览器不支持此RGB格式时请用本地播放器；PNG是质量评价依据。</p>
<p>查看目标是否消失、形变、闪烁；未填写人工分数。并排播放若卡顿，请逐个播放核对，避免把播放器掉帧误认为重建闪烁。</p>
<select id="clip" onchange="choose()"><option>short</option><option>long108</option><option>long91</option><option>long24</option></select>
<button onclick="playGroup()">当前组从头播放</button><button onclick="pauseAll()">暂停</button><p id="status"></p>
'''+''.join(groups)+'''
<script>
function pauseAll(){document.querySelectorAll('video').forEach(v=>v.pause());}
function choose(){pauseAll();document.querySelectorAll('main').forEach(m=>m.hidden=m.dataset.clip!==document.getElementById('clip').value);}
async function playGroup(){pauseAll();let failed=false;const active=document.querySelectorAll('main:not([hidden]) video');await Promise.all(Array.from(active,async v=>{try{v.currentTime=0;await v.play();}catch(e){failed=true;}}));document.getElementById('status').textContent=failed?'当前浏览器未能播放全部副本，请使用支持RGB H.264的本地播放器。':'';}
</script></html>'''
(gallery/'index.html').write_text(page,encoding='utf-8')
save(a.output/'preview_verification.json',{'note':'Lossless viewing copies only; these bytes are not transmission payloads.','videos':[read((gallery/fn).with_suffix('.json')) for c,t,fn in videos]})
print(json.dumps({'summary_rows':len(summary),'figures':len(list(fig.glob('*.png'))),'local_playbacks':len(videos),'gallery':str(gallery/'index.html')},ensure_ascii=False),flush=True)
