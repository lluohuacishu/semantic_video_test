"""Independent D2.1 metrics and labeled viewing, including a separate q0 diagnostic."""
from __future__ import annotations
import argparse
from fractions import Fraction
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont, ImageOps
from prepare_d2 import read_json, save_json
from finish_d2_lowrate import stats
from evaluate_d2 import region

LABELS={'R':'共同缩小原图','A':'普通 GLC','B':'全图低通','C':'原鱼＋潜水员保护','F':'仅鱼预测保护','O':'GT 鱼保护（仅诊断）'}


def evaluate(root):
    if read_json(root/'progress.json')['state']!='codec_complete':raise ValueError('Codec incomplete')
    selection=read_json(root/'selection.json');clips=read_json(root/'inputs.json')['clips'];gate=read_json(root/'rate_gate.json')
    entries=selection['entries'];unique={e['point']['output']:e for e in entries};audits=[];metrics={};frame_metrics=[]
    started=time.perf_counter()
    for output,e in unique.items():
        encoder=Path(output);decoder=Path(e['decoder']);p=e['point'];enc=read_json(encoder/'encode.json');dec=read_json(decoder/'decode.json')
        check=dec['reconstruction_verification']
        if dec['source_media_reads'] or not dec['independent_decoder'] or not dec['encoder_modules_disabled']:raise ValueError('Receiver isolation failed')
        if len(dec['frames'])!=64 or not dec['all_frame_checksums_match'] or not all(check[k] for k in ('verified','reference_verified','rgb_verified','complete_sequence')):raise ValueError('Independent reconstruction failed')
        if enc['file_sha256']!=dec['input_sha256'] or enc['pid']==dec['pid']:raise ValueError('Sender/receiver provenance failed')
        size=(encoder/'stream.glcv').stat().st_size;key=str(e['video_id']);clip=next(c for c in clips if c['video_id']==e['video_id'])
        duration=Fraction(clip['duration_exact']);anchor=gate['anchors'][key]
        if size!=p['file_bytes'] or size>int(40000*duration/8) or 10*size>11*anchor['S0']:raise ValueError('Complete-file budget failed')
        header=enc['header']
        if [header['width'],header['height']]!=[512,288] or header['frames']!=64 or header['intra_period']!=64 or header['reset_interval']!=32:raise ValueError('Codec geometry/GOP changed')
        if abs(header['fps']-float(Fraction(clip['fps'])))>1e-10:raise ValueError('Frame rate changed')
        metrics[output]={'file_bytes':size,'file_kbps':float(8*size/duration/1000),'growth_vs_S0':size/anchor['S0']-1,
            'encoder_seconds':enc['seconds'],'decoder_seconds':dec['seconds'],'independent_decode':True,
            'rate_cap_passed':True,'growth_cap_passed':True,'side_information_bytes':0,'duration_exact':str(duration)}
        audits.append({'id':p['id'],'encoder':str(encoder),'decoder':str(decoder),'sender_pid':enc['pid'],'receiver_pid':dec['pid'],
            'file_sha256':enc['file_sha256'],'source_media_reads':dec['source_media_reads'],'complete_sequence_verified':True,'actual_file_bytes':size})
    for clip in clips:
        subset={output:e for output,e in unique.items() if e['video_id']==clip['video_id']}
        sums={output:{scale:{r:[0,0] for r in ('fish','rest','all')} for scale in ('delivery','original_upsampled')} for output in subset}
        edges={output:{r:[0,0] for r in ('fish_inner_edge','fish_interior')} for output in subset}
        for f in clip['frames']:
            delivery=np.asarray(Image.open(f['resized']['512x288']['rgb']).convert('RGB'))
            gt=np.asarray(Image.open(f['resized']['512x288']['gt']))!=0
            original=np.asarray(Image.open(f['original_rgb']).convert('RGB'));original_gt=np.asarray(Image.open(f['original_gt']))!=0
            edge=gt & ~ndimage.binary_erosion(gt);interior=gt & ~edge
            for output,e in subset.items():
                im=Image.open(Path(e['decoder'])/'receiver/frames'/f'im{f["index"]+1:05d}.png').convert('RGB')
                if im.size!=(512,288):raise ValueError('Decoded size mismatch')
                decoded=np.asarray(im)
                values={'delivery':stats(delivery,decoded,gt),'original_upsampled':stats(original,np.asarray(im.resize((1920,1080),Image.Resampling.LANCZOS)),original_gt)}
                for scale,regions in values.items():
                    for name,(sse,count) in regions.items():sums[output][scale][name][0]+=sse;sums[output][scale][name][1]+=count
                squared=(decoded.astype(np.int32)-delivery.astype(np.int32))**2
                for name,mask in [('fish_inner_edge',edge),('fish_interior',interior)]:
                    edges[output][name][0]+=int(squared[mask].sum(dtype=np.int64));edges[output][name][1]+=3*int(mask.sum())
                frame_metrics.append({'id':e['point']['id'],'encoder':output,'index':f['index'],
                    'delivery':{name:region(*value) for name,value in values['delivery'].items()}})
        for output in subset:
            metrics[output]['metrics']={scale:{name:region(*v) for name,v in regions.items()} for scale,regions in sums[output].items()}
            metrics[output]['metrics']['delivery_boundaries']={name:region(*v) for name,v in edges[output].items()}
        print(f'EVALUATED {clip["video_id"]:03d} {len(subset)} streams',flush=True)
    points=[]
    for e in entries:
        points.append({**e,**metrics[e['point']['output']]})
    comparisons=[]
    for category in ('selected','q0'):
        for clip in clips:
            group={p['method']:p for p in points if p['category']==category and p['video_id']==clip['video_id']}
            for method in 'CFO':
                if method not in group:continue
                for comparator in ('A','B','C'):
                    if method==comparator:continue
                    p=group[method];base=group[comparator];gap=(base['file_bytes']-p['file_bytes'])/base['file_bytes']
                    comparisons.append({'category':category,'video_id':clip['video_id'],'candidate':method,'comparator':comparator,
                        'byte_change_fraction':p['file_bytes']/base['file_bytes']-1,'near_equal_no_larger_2pct':0<=gap<=.02,
                        'fish_delta_db':p['metrics']['delivery']['fish']['psnr_db']-base['metrics']['delivery']['fish']['psnr_db'],
                        'rest_delta_db':p['metrics']['delivery']['rest']['psnr_db']-base['metrics']['delivery']['rest']['psnr_db'],
                        'original_fish_delta_db':p['metrics']['original_upsampled']['fish']['psnr_db']-base['metrics']['original_upsampled']['fish']['psnr_db']})
    result={'schema':'d21-evaluation-v1','date':'2026-10-22','points':points,'comparisons':comparisons,
        'new_encodes':selection['new_encodes'],'new_independent_decodes':selection['new_decodes'],
        'unique_evaluated_streams':len(unique),'frame_point_metrics':len(frame_metrics),
        'all_evaluated_hard_caps_passed':True,'subjective_acceptance':False,'independent_source_generalization':False,
        'GT_is_deployable':False,'seconds':time.perf_counter()-started,
        'quality_reference':'same resized original RGB; separate original-size reference after upsampling',
        'same_q0_is_equal_rate':False,'semantic_inference':False,'training':False}
    save_json(root/'evaluation.json',result);save_json(root/'frame_metrics.json',frame_metrics);save_json(root/'decoder_audit.json',audits)
    return result


def viewing(root,result):
    out=root/'viewing';out.mkdir(exist_ok=False);clips=read_json(root/'inputs.json')['clips'];runtime=read_json(root/'runtime.json')
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    manifest=[];sections=[]
    for clip in clips:
        vid=clip['video_id'];chosen={p['method']:p for p in result['points'] if p['video_id']==vid and p['category']=='selected'}
        crop=clip['fish_crop_original'];crop=(int(crop[0]*512/1920),int(crop[1]*288/1080),int(np.ceil(crop[2]*512/1920)),int(np.ceil(crop[3]*288/1080)))
        modes=[('all',['R','A','B','C','F','O'],3),('B_O',['B','O'],2)]
        if vid==21:modes.append(('C_F',['C','F'],2))
        for mode,methods,cols in modes:
            if any(m!='R' and m not in chosen for m in methods):continue
            rows=(len(methods)+cols-1)//cols;cell_w=512;cell_h=666;dimensions=(cols*cell_w,rows*cell_h)
            target=out/f'video_{vid:03d}_{mode}.mp4'
            cmd=[runtime['ffmpeg'],'-hide_banner','-loglevel','error','-nostdin','-f','rawvideo','-pix_fmt','rgb24','-s',f'{dimensions[0]}x{dimensions[1]}',
                '-r',clip['fps'],'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','14','-pix_fmt','yuv420p','-movflags','+faststart','-n',str(target)]
            with target.with_suffix('.log').open('wb') as log:
                proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
                try:
                    for f in clip['frames']:
                        canvas=Image.new('RGB',dimensions,'#101820');draw=ImageDraw.Draw(canvas)
                        for i,m in enumerate(methods):
                            x=(i%cols)*cell_w;y=(i//cols)*cell_h
                            if m=='R':path=Path(f['resized']['512x288']['rgb']);subtitle='相同时间和分辨率的参考'
                            else:
                                p=chosen[m];path=Path(p['decoder'])/'receiver/frames'/f'im{f["index"]+1:05d}.png'
                                subtitle=f'{p["file_bytes"]:,} B | {p["file_kbps"]:.2f} kbps | q{p["point"]["q"]}'
                            title=LABELS[m]
                            if vid==11 and m in ('C','F'):title=('原语义方案' if m=='C' else '仅鱼方案')+'（全程回退）'
                            im=Image.open(path).convert('RGB');draw.text((x+10,y+5),title,font=font,fill='white')
                            draw.text((x+10,y+33),subtitle,font=small,fill='#c4d5e5');canvas.paste(im,(x,y+60))
                            draw.text((x+10,y+355),'同一固定鱼区放大',font=small,fill='#c4d5e5')
                            enlarged=ImageOps.contain(im.crop(crop),(496,240),Image.Resampling.NEAREST)
                            canvas.paste(enlarged,(x+(512-enlarged.width)//2,y+386+(240-enlarged.height)//2))
                            draw.text((x+10,y+638),f'帧 {f["index"]+1}/64 · 512×288 · 原速 29.97 fps',font=small,fill='white')
                        proc.stdin.write(np.asarray(canvas).tobytes())
                        if mode=='all' and f['index'] in (0,21,42,63):canvas.save(out/f'v{vid:03d}_{f["index"]:04d}.png')
                        if mode=='B_O' and f['index']==0:canvas.save(out/f'v{vid:03d}_B_O_0000.png')
                    proc.stdin.close()
                    if proc.wait()!=0:raise RuntimeError('Viewing encode failed')
                except BaseException:
                    proc.kill();proc.wait();raise
            manifest.append({'video_id':vid,'mode':mode,'path':target.name,'frames':64,'fps':clip['fps'],'size':list(dimensions),'fish_crop':crop,'lossy_viewing_only':True})
            print('VIEW '+target.name,flush=True)
        table=''.join(f'<tr><td>{LABELS[m]}</td><td>{chosen[m]["point"]["q"]}</td><td>{chosen[m]["file_bytes"]:,}</td><td>{chosen[m]["file_kbps"]:.2f}</td><td>{chosen[m]["metrics"]["delivery"]["fish"]["psnr_db"]:.3f}</td></tr>' for m in 'ABCFO' if m in chosen)
        gallery=''.join(f'<a href="../diagnostics/v{vid:03d}_{i:04d}.png"><img src="../diagnostics/v{vid:03d}_{i:04d}.png" alt="片段{vid:03d}第{i+1}帧保护诊断"></a>' for i in (0,21,42,63))
        extra='<h3>关闭潜水员分支的实际影响</h3><video controls loop preload="metadata" src="video_021_C_F.mp4"></video>' if vid==21 else '<p>011 的仅鱼方案仍全部回退，与原 C 输入完全相同，直接复用码流。</p>'
        sections.append(f'<section><h2>片段 {vid:03d}</h2><table><thead><tr><th>方案</th><th>q</th><th>完整文件 B</th><th>kbps</th><th>鱼 PSNR dB</th></tr></thead><tbody>{table}</tbody></table><h3>全图低通与 GT 保护</h3><video controls loop preload="metadata" src="video_{vid:03d}_B_O.mp4"></video><p>GT 方案使用人工真值，仅用于诊断；实际大小如表所示。</p>{extra}<details><summary>原图与全部五种方案（可全屏）</summary><video controls loop preload="metadata" src="video_{vid:03d}_all.mp4"></video></details><details><summary>查看固定四帧的分支和保护范围</summary><div class="gallery">{gallery}</div></details></section>')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>D2.1 保护机制诊断</title><style>body{max-width:1400px;margin:30px auto;padding:0 20px;background:#101820;color:#eaf2fa;font:16px system-ui}p{line-height:1.75;color:#c5d4e2}video{width:100%;background:#000}section{margin:44px 0}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:12px;text-align:left;border-bottom:1px solid #516272}summary{padding:18px 0;cursor:pointer}.gallery img{width:100%;margin:8px 0}button{padding:8px 15px;margin:8px 7px 8px 0;background:#21374a;border:1px solid #789;color:white;border-radius:6px;font:inherit}a{color:#85caff}</style><h1>D2.1：保护范围与真实画质</h1><p>全部视频保留 512×288、30000/1001 原帧率和同一时间轴。每格先展示全画面，再展示同一固定鱼区放大。表中码率按完整 glcv 计算；MP4 是本地有损观看副本。GT 保护不作为可部署成果，主观清晰度尚未验收。</p><p>011 的仅鱼保护仍触发全程回退；021 的仅鱼保护和 GT 保护已做有限对照。精确质量与大小差额见实验报告。</p>'''+''.join(sections)+'''<script>document.querySelectorAll('video').forEach(v=>{const row=document.createElement('div');[['原速',()=>v.playbackRate=1],['半速',()=>v.playbackRate=.5],['前一帧',()=>step(-1)],['后一帧',()=>step(1)]].forEach(([label,fn])=>{const b=document.createElement('button');b.textContent=label;b.onclick=fn;row.appendChild(b)});function step(d){v.pause();const fps=30000/1001;v.currentTime=(Math.max(0,Math.min(63,Math.floor(v.currentTime*fps+1e-4)+d))+.01)/fps}v.after(row)});</script></html>'''
    (out/'index.html').write_text(html,encoding='utf-8');save_json(out/'viewing_manifest.json',manifest)


def run(args):
    root=args.run.resolve();result=evaluate(root);viewing(root,result)
    save_json(root/'postprocess_progress.json',{'state':'completed','unique_evaluated_streams':result['unique_evaluated_streams'],
        'frame_point_metrics':result['frame_point_metrics'],'subjective_acceptance':False})
    print('D2.1 evaluation and viewing completed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);run(p.parse_args())
