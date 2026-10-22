"""Labeled D2.2 viewing, budget/decoder evidence and tables, with no quality selection."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from prepare_d2 import read_json,save_json

LABELS={'R':'共同缩小原图','A':'普通 GLC','B':'全图低通 1.2','C':'预测保护：目标原像素',
    'O':'GT保护：目标原像素','W':'全图轻度低通 0.6','P':'预测保护：目标低通 0.6','G':'GT保护：目标低通 0.6'}


def viewing(root,result):
    out=root/'viewing';out.mkdir(exist_ok=False);plan=read_json(root/'plan.json');runtime=read_json(root/'runtime.json')
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    sections=[];manifest=[]
    for clip in read_json(root/'inputs.json')['clips']:
        vid=clip['video_id'];chosen={p['method']:p for p in result['points'] if p['video_id']==vid and p['category']=='selected'}
        crop=clip['fish_crop_original'];crop=(int(crop[0]*512/1920),int(crop[1]*288/1080),int(np.ceil(crop[2]*512/1920)),int(np.ceil(crop[3]*288/1080)))
        modes=[('B_O_G',['B','O','G'],3),('W_G',['W','G'],2),('all',['R','A','B','W','O','G'],3)]
        if vid==21:modes.append(('B_C_P',['B','C','P'],3))
        for mode,methods,cols in modes:
            if any(m!='R' and m not in chosen for m in methods):continue
            dimensions=(cols*512,((len(methods)+cols-1)//cols)*666)
            target=out/f'video_{vid:03d}_{mode}.mp4'
            cmd=[runtime['ffmpeg'],'-hide_banner','-loglevel','error','-nostdin','-f','rawvideo','-pix_fmt','rgb24','-s',f'{dimensions[0]}x{dimensions[1]}',
                '-r',clip['fps'],'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','14','-pix_fmt','yuv420p','-movflags','+faststart','-n',str(target)]
            with target.with_suffix('.log').open('wb') as log:
                proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
                try:
                    for f in clip['frames']:
                        if (root/'STOP').exists():raise RuntimeError('Stopped at viewing frame boundary')
                        canvas=Image.new('RGB',dimensions,'#101820');draw=ImageDraw.Draw(canvas)
                        for i,m in enumerate(methods):
                            x=(i%cols)*512;y=(i//cols)*666
                            if m=='R':path=Path(f['resized']['512x288']['rgb']);subtitle='相同时间和分辨率的原始参考'
                            else:
                                p=chosen[m];path=Path(p['decoder'])/'receiver/frames'/f'im{f["index"]+1:05d}.png'
                                subtitle=f'{p["file_bytes"]:,} B | {p["file_kbps"]:.2f} kbps | q{p["point"]["q"]}'
                            title=LABELS[m];draw.text((x+10,y+5),title,font=font,fill='white')
                            draw.text((x+10,y+33),subtitle,font=small,fill='#c4d5e5')
                            im=Image.open(path).convert('RGB');canvas.paste(im,(x,y+60))
                            draw.text((x+10,y+355),'同一固定鱼区放大'+(' · GT仅离线诊断' if m in ('O','G') else ''),font=small,fill='#c4d5e5')
                            enlarged=ImageOps.contain(im.crop(crop),(496,240),Image.Resampling.NEAREST)
                            canvas.paste(enlarged,(x+(512-enlarged.width)//2,y+386+(240-enlarged.height)//2))
                            draw.text((x+10,y+638),f'帧 {f["index"]+1}/64 · 512×288 · 原速29.97 fps',font=small,fill='white')
                        proc.stdin.write(np.asarray(canvas).tobytes())
                        if mode in ('B_O_G','B_C_P') and f['index'] in plan['evaluation']['fixed_visual_frame_indices']:canvas.save(out/f'v{vid:03d}_{mode}_{f["index"]:04d}.png')
                    proc.stdin.close()
                    if proc.wait()!=0:raise RuntimeError('Viewing encode failed')
                except BaseException:proc.kill();proc.wait();raise
            manifest.append({'video_id':vid,'mode':mode,'path':target.name,'frames':64,'fps':clip['fps'],'size':list(dimensions),'fish_crop':crop,'lossy_viewing_only':True})
            print('VIEW '+target.name,flush=True)
        table=''.join(f'<tr><td>{LABELS[m]}</td><td>{chosen[m]["point"]["q"]}</td><td>{chosen[m]["file_bytes"]:,}</td><td>{chosen[m]["file_kbps"]:.2f}</td><td>{chosen[m]["metrics"]["delivery"]["fish"]["psnr_db"]:.3f}</td></tr>' for m in ('A','B','W','C','P','O','G') if m in chosen)
        if vid==21:extra='<h3>真实预测：原像素保护与目标轻度低通</h3><video controls loop preload="metadata" src="video_021_B_C_P.mp4"></video>'
        else:extra='<p>011预测掩码仍全程触发回退，P与C完全相同，直接复用。这里不把回退改成全图模糊。</p>'
        sections.append(f'<section><h2>片段 {vid:03d}</h2><table><tr><th>方法</th><th>q</th><th>完整文件 B</th><th>kbps</th><th>鱼PSNR dB</th></tr>{table}</table><h3>目标低通是否改善准确保护？</h3><video controls loop preload="metadata" src="video_{vid:03d}_B_O_G.mp4"></video><p>O/G使用同一GT掩码，目标低通强度从0改为0.6；背景仍为1.2。GT方案仅离线诊断。</p>{extra}<details><summary>与全图轻度低通比较</summary><video controls loop preload="metadata" src="video_{vid:03d}_W_G.mp4"></video></details><details><summary>原图与全部机制对照（可全屏）</summary><video controls loop preload="metadata" src="video_{vid:03d}_all.mp4"></video></details></section>')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>D2.2 目标轻度低通</title><style>body{max-width:1500px;margin:30px auto;padding:0 20px;background:#101820;color:#eaf2fa;font:16px system-ui}p{line-height:1.75;color:#c5d4e2}video{width:100%;background:#000}section{margin:44px 0}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:12px;text-align:left;border-bottom:1px solid #516272}summary{padding:18px 0;cursor:pointer}button{padding:8px 15px;margin:8px 7px 8px 0;background:#21374a;border:1px solid #789;color:white;border-radius:6px;font:inherit}</style><h1>D2.2：让目标也接受轻度低通</h1><p>只试预先固定的目标强度0.6，背景强度1.2、掩码和回退保持不变。所有视频保留原帧率和同一时间轴；上部全画面，下部固定鱼区放大。完整发送文件为glcv，MP4只是本地有损观看副本。实际字节差额见表和报告，不能把同预算自动当成等码率。</p><p>GT结果只说明准确定位条件下的机制表现，不能作为可部署成果。连续清晰度仍需结合原图观看判断。</p>'''+''.join(sections)+'''<script>document.querySelectorAll('video').forEach(v=>{const row=document.createElement('div');[['原速',()=>v.playbackRate=1],['半速',()=>v.playbackRate=.5],['前一帧',()=>step(-1)],['后一帧',()=>step(1)]].forEach(([label,fn])=>{const b=document.createElement('button');b.textContent=label;b.onclick=fn;row.appendChild(b)});function step(d){v.pause();const fps=30000/1001;v.currentTime=(Math.max(0,Math.min(63,Math.floor(v.currentTime*fps+1e-4)+d))+.01)/fps}v.after(row)});</script></html>'''
    (out/'index.html').write_text(html,encoding='utf-8');save_json(out/'viewing_manifest.json',manifest)
    checks=[];ffprobe=str(Path(runtime['ffmpeg']).with_name('ffprobe.exe'))
    for item in manifest:
        probe=json.loads(subprocess.check_output([ffprobe,'-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration','-of','json',str(out/item['path'])],text=True))['streams'][0]
        if probe['nb_frames']!='64' or probe['r_frame_rate']!='30000/1001' or probe['avg_frame_rate']!='30000/1001' or [probe['width'],probe['height']]!=item['size']:raise ValueError('Viewing metadata mismatch')
        checks.append({'file':item['path'],**probe})
    save_json(root/'viewing_checks.json',{'all_passed':True,'videos':checks,'subjective_acceptance':False})


def tables(result):
    blocks=[]
    for category in ('selected','q0'):
        rows=[f'## {"选定点" if category=="selected" else "固定q0诊断（非等码率）"}',
            '| 片段 | 方法/q | 文件 B | kbps | 相对S0 | 鱼PSNR | 内边界PSNR | 其余PSNR | 回原尺寸鱼PSNR |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for vid in (11,21):
            group={p['method']:p for p in result['points'] if p['category']==category and p['video_id']==vid}
            for method in ('A','B','W','C','P','O','G'):
                if method not in group:continue
                p=group[method];m=p['metrics']
                rows.append(f'| {vid:03d} | {method}/q{p["point"]["q"]} | {p["file_bytes"]:,} | {p["file_kbps"]:.3f} | {p["growth_vs_S0"]:+.2%} | {m["delivery"]["fish"]["psnr_db"]:.3f} | {m["delivery_boundaries"]["fish_inner_edge"]["psnr_db"]:.3f} | {m["delivery"]["rest"]["psnr_db"]:.3f} | {m["original_upsampled"]["fish"]["psnr_db"]:.3f} |')
        blocks.append('\n'.join(rows))
    rows=['## 新方案比较','| 类别 | 片段 | 比较 | 字节变化 | 鱼PSNR变化 | 其余变化 | 严格近等体积 |','|---|---|---|---:|---:|---:|---|']
    for c in result['comparisons']:
        if c['candidate'] not in ('W','P','G'):continue
        if c['candidate']=='P' and c['video_id']==11:continue
        if c['comparator']=='O' and c['candidate']=='W':continue
        rows.append(f'| {c["category"]} | {c["video_id"]:03d} | {c["candidate"]}/{c["comparator"]} | {c["byte_change_fraction"]:+.3%} | {c["fish_delta_db"]:+.3f} dB | {c["rest_delta_db"]:+.3f} dB | {"是" if c["near_equal_no_larger_2pct"] else "否"} |')
    blocks.append('\n'.join(rows));return '\n\n'.join(blocks)+'\n'


def archive(root,evidence,numeric_output):
    here=Path(__file__).resolve().parent;result=read_json(root/'evaluation.json')
    if not read_json(root/'viewing_checks.json')['all_passed']:raise ValueError('Viewing incomplete')
    (root/'numeric_tables.md').write_text(tables(result),encoding='utf-8')
    numeric_output.write_text(tables(result),encoding='utf-8')
    save_json(root/'postprocess_progress.json',{'state':'completed','unique_evaluated_streams':result['unique_evaluated_streams'],
        'frame_point_metrics':result['frame_point_metrics'],'subjective_acceptance':False})
    evidence.mkdir(exist_ok=False)
    for name in ('plan.json','sources.json','preparation.json','selection.json','new_encodes.json','evaluation.json','decoder_audit.json','progress.json','postprocess_progress.json','viewing_checks.json'):
        shutil.copy2(root/name,evidence/name)
    files=['run_d22.py','finish_d22.py','d22_plan_v1.json','run_d21_codec.py','finish_d21.py','run_d2_lowrate.py','d2_codec_worker.py','prepare_d2.py']
    save_json(evidence/'code_provenance.json',{'run':str(root),'files':[{'path':n,'sha256':hashlib.sha256((here/n).read_bytes()).hexdigest()} for n in files],
        'model_or_dataset_integrity_rescan':False})


if __name__=='__main__':
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--archive-dir',type=Path,default=here/'evidence_d22_20261022')
    p.add_argument('--tables-output',type=Path,default=here/'D22数值附表_20261022.md')
    a=p.parse_args();root=a.run.resolve();evidence=a.archive_dir.resolve();numeric_output=a.tables_output.resolve()
    for target in (root/'viewing',evidence,numeric_output):
        if target.exists():raise FileExistsError(f'Refusing to overwrite completed output: {target}')
        if not target.parent.is_dir():raise FileNotFoundError(f'Output parent does not exist: {target.parent}')
    result=read_json(root/'evaluation.json');viewing(root,result);archive(root,evidence,numeric_output)
    print('D2.2 viewing and evidence complete',flush=True)
