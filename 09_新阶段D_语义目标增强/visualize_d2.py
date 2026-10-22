"""Make synchronized D2 viewing copies from original and independently decoded PNGs."""
from __future__ import annotations
import argparse
from fractions import Fraction
import random
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from prepare_d2 import read_json, save_json


def run(args):
    root=args.run; evaluation=read_json(root/'evaluation.json')
    point_root=Path(evaluation.get('points_root',root))
    matrix=read_json(root/'matrix.json'); inputs=read_json(root/'inputs.json')
    points={p['id']:p for p in evaluation['points']}
    corpus=Path(read_json(root/'preparation.json')['source_corpus'])
    source_manifest=read_json(corpus/'ready_manifest.json')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    out=root/'viewing'; out.mkdir(exist_ok=False)
    records=[]
    for clip in inputs['clips']:
        sequence=next(s for s in source_manifest['fish_sequences'] if s['video_id']==clip['video_id'])
        # The same originally labeled track is present in both clips. Crop is
        # fixed across time and all methods; no selection on reconstructed quality.
        boxes=[i['bbox'] for f in sequence['frames'] for i in f['instances'] if i['track_id']==38]
        if not boxes: raise ValueError('Registered common fish track is missing')
        x0=max(0,int(min(b[0] for b in boxes))-48); y0=max(0,int(min(b[1] for b in boxes))-48)
        x1=min(1920,int(max(b[0]+b[2]+1 for b in boxes))+48)
        y1=min(1080,int(max(b[1]+b[3]+1 for b in boxes))+48)
        crop=(x0,y0,x1,y1)
        for budget,selection in evaluation['budget_selections'][str(clip['video_id'])].items():
            chosen=selection['selected']
            methods=list('ABCO'); random.Random(20261021+clip['video_id']+(budget=='B10')).shuffle(methods)
            cell_w=384; image_h=round(cell_w*(y1-y0)/(x1-x0)); image_h=min(image_h,640)
            height=(image_h+64+1)//2*2; width=5*cell_w
            target=out/f'video_{clip["video_id"]:03d}_{budget}_blind.mp4'
            argv=[matrix['runtime']['ffmpeg'],'-hide_banner','-loglevel','error','-nostdin',
                  '-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
                  '-r',clip['fps'],'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','14',
                  '-pix_fmt','yuv420p','-movflags','+faststart','-n',str(target)]
            log=target.with_suffix('.log').open('wb')
            process=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
            try:
                for frame in clip['records']:
                    canvas=Image.new('RGB',(width,height),(18,24,32)); draw=ImageDraw.Draw(canvas)
                    for col,method in enumerate([None]+methods):
                        label='Original' if method is None else f'View {col}'
                        point=chosen.get(method)
                        draw.text((col*cell_w+10,8),label,font=font,fill='white')
                        draw.text((col*cell_w+10,height-26),f'Frame {frame["index"]+1}/64',font=font,fill='white')
                        if method is None: path=Path(frame['paths']['A'])
                        elif point: path=point_root/'points'/point/'decoder/receiver/frames'/f'im{frame["index"]+1:05d}.png'
                        else:
                            draw.text((col*cell_w+10,65),'No point within budget',font=font,fill='white')
                            continue
                        panel=Image.open(path).convert('RGB').crop(crop)
                        panel.thumbnail((cell_w-8,image_h),Image.Resampling.LANCZOS)
                        canvas.paste(panel,(col*cell_w+(cell_w-panel.width)//2,36+(image_h-panel.height)//2))
                    process.stdin.write(np.asarray(canvas).tobytes())
                    if frame['index'] in (0,31,63):
                        canvas.save(out/f'video_{clip["video_id"]:03d}_{budget}_frame{frame["index"]:02d}.png')
                process.stdin.close()
                if process.wait()!=0: raise RuntimeError('FFmpeg viewing-copy generation failed')
            except BaseException:
                process.kill(); process.wait()
                raise
            finally:
                log.close()
            records.append({'video_id':clip['video_id'],'budget':budget,'filename':target.name,
                            'crop_xyxy':list(crop),'crop_track_id':38,'crop_method':'all64 source GT bboxes plus48px context',
                            'view_key':{str(i+1):{'method':m,'point':chosen[m]} for i,m in enumerate(methods)},
                            'fps':clip['fps'],'frames':64,'duration_exact':clip['duration_exact'],
                            'all_frames_evaluated_numerically':True,'crop_is_not_all_fish':True,
                            'lossy_viewing_copy_only':True})
            print('Created '+target.name,flush=True)
    save_json(out/'viewing_key.json',records)
    cards=[]
    for item in records:
        levels=evaluation['budget_selections'][str(item['video_id'])]
        same_budget_points=('B0' in levels and 'B10' in levels and levels['B0']['selected']==levels['B10']['selected'])
        budget_note=('<p>本片 B0 与 B10 选中了相同文件，两份视频仅栏目顺序不同；增加 10% 上限没有改变本轮所选点。</p>'
                     if same_budget_points else '')
        cards.append(f'<section><h2>片段 {item["video_id"]:03d} · {item["budget"]}</h2>'
          +budget_note+
          f'<video controls loop playsinline preload="metadata" src="{item["filename"]}"></video>'
          '<div class="controls"><button data-speed="0.5">半速 0.5×</button>'
          '<button data-speed="1">原速 1×</button><button data-step="-1">前一帧</button>'
          '<button data-step="1">后一帧</button><span class="frame">帧 1 / 64</span></div>'
          '<p>Original 为原图；View 1–4 为隐藏方法名的同步对照。固定裁剪覆盖共同鱼轨迹 38，不代表全部鱼。可全屏或暂停逐帧看，判断运动时请使用原速。</p></section>')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>D2 目标画质对照</title><style>body{background:#111923;color:#e7edf5;font:16px system-ui;max-width:1500px;margin:32px auto;padding:0 20px}h1{font-size:28px}video{width:100%;background:#000}section{margin:32px 0}p{line-height:1.7;color:#b9c6d5}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px}button{font:inherit;color:#e7edf5;background:#243346;border:1px solid #52647a;border-radius:6px;padding:7px 12px;cursor:pointer}.frame{font-variant-numeric:tabular-nums}</style>
<h1>D2：原图与四组目标画质对照</h1><p>每段 64 帧、30000/1001 fps，约 2.135 秒，可循环播放并全屏查看。所有栏目使用相同时间、裁剪和显示尺寸。预览为有损观看副本，数值指标来自独立解码的无损 PNG；本页不表示人工观看验收已通过。</p>'''+''.join(cards)+'''<script>
const fps=30000/1001;
document.querySelectorAll('section').forEach(section=>{
 const video=section.querySelector('video'),label=section.querySelector('.frame');
 section.querySelectorAll('[data-speed]').forEach(button=>button.addEventListener('click',()=>{video.playbackRate=Number(button.dataset.speed)}));
 section.querySelectorAll('[data-step]').forEach(button=>button.addEventListener('click',()=>{
  video.pause();const current=Math.min(63,Math.floor(video.currentTime*fps+1e-4));
  video.currentTime=(Math.max(0,Math.min(63,current+Number(button.dataset.step)))+0.01)/fps;
 }));
 video.addEventListener('timeupdate',()=>{label.textContent='帧 '+Math.min(64,Math.floor(video.currentTime*fps+1e-4)+1)+' / 64'});
});
</script></html>'''
    (out/'index.html').write_text(html,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    run(parser.parse_args())
