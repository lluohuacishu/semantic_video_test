"""Evaluate compliant independent outputs and create labeled, synchronized viewing."""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from prepare_d2 import read_json, save_json
from evaluate_d2 import region


def stats(reference,restored,gt):
    diff=reference.astype(np.int32)-restored.astype(np.int32);square=diff*diff
    target=int(square[gt].sum(dtype=np.int64));total=int(square.sum(dtype=np.int64))
    count=3*int(gt.sum());all_count=reference.size
    return {'fish':(target,count),'rest':(total-target,all_count-count),'all':(total,all_count)}


def evaluate(root):
    if read_json(root/'progress.json')['state']!='codec_complete': raise ValueError('Independent decode is incomplete')
    gate=read_json(root/'rate_gate.json');selection=read_json(root/'rate_selection.json')
    clips=read_json(root/'inputs.json')['clips'];width,height=gate['size'];points=[];frame_metrics=[];audits=[]
    started=time.perf_counter()
    for clip in clips:
        key=str(clip['video_id']);sums={}
        for method,item in selection[key].items():
            p=dict(item['selected']);folder=root/'points'/p['id']
            enc=read_json(folder/'encoder/encode.json');dec=read_json(folder/'decoder/decode.json')
            verification=dec['reconstruction_verification']
            if dec['source_media_reads'] or not dec['independent_decoder'] or not dec['encoder_modules_disabled']:
                raise ValueError('Receiver source isolation failed')
            if len(dec['frames'])!=len(clip['frames']) or not dec['all_frame_checksums_match'] or not all(verification[k] for k in ('verified','reference_verified','rgb_verified','complete_sequence')):
                raise ValueError('Independent complete reconstruction verification failed')
            if enc['file_sha256']!=dec['input_sha256'] or enc['pid']==dec['pid']:
                raise ValueError('Sender/receiver provenance mismatch')
            actual=(folder/'encoder/stream.glcv').stat().st_size
            duration=Fraction(clip['duration_exact']);anchor=gate['anchors'][key]
            if actual!=p['file_bytes'] or actual>int(40000*duration/8) or 10*actual>11*anchor['S0']:
                raise ValueError('Complete-file hard rate/growth gate failed')
            p.update(file_kbps=float(8*actual/duration/1000),growth_vs_S0=actual/anchor['S0']-1,
                     independent_decode=True,rate_cap_passed=True,growth_cap_passed=True,side_information_bytes=0,
                     encoder_seconds=enc['seconds'],decoder_seconds=dec['seconds'],duration_exact=str(duration))
            points.append(p);sums[method]={scale:{r:[0,0] for r in ('fish','rest','all')} for scale in ('delivery','original_upsampled')}
            audits.append({'id':p['id'],'sender_pid':enc['pid'],'receiver_pid':dec['pid'],'source_media_reads':dec['source_media_reads'],
                           'complete_sequence_verified':True,'actual_file_bytes':actual,'file_sha256':enc['file_sha256']})
        for frame in clip['frames']:
            delivery=np.asarray(Image.open(frame['resized'][f'{width}x{height}']['rgb']).convert('RGB'))
            gt=np.asarray(Image.open(frame['resized'][f'{width}x{height}']['gt']))!=0
            original=np.asarray(Image.open(frame['original_rgb']).convert('RGB'))
            original_gt=np.asarray(Image.open(frame['original_gt']))!=0
            for method,item in selection[key].items():
                path=root/'points'/item['selected']['id']/'decoder/receiver/frames'/f'im{frame["index"]+1:05d}.png'
                decoded=Image.open(path).convert('RGB')
                if decoded.size!=(width,height): raise ValueError('Decoder resolution mismatch')
                values={'delivery':stats(delivery,np.asarray(decoded),gt),
                        'original_upsampled':stats(original,np.asarray(decoded.resize((1920,1080),Image.Resampling.LANCZOS)),original_gt)}
                for scale,regions in values.items():
                    for name,(sse,count) in regions.items():
                        sums[method][scale][name][0]+=sse;sums[method][scale][name][1]+=count
                frame_metrics.append({'id':item['selected']['id'],'index':frame['index'],
                    **{scale:{k:region(*v) for k,v in regs.items()} for scale,regs in values.items()}})
        for p in [p for p in points if p['video_id']==clip['video_id']]:
            p['metrics']={scale:{name:region(*v) for name,v in values.items()} for scale,values in sums[p['method']].items()}
        print(f'Evaluated {clip["video_id"]:03d}, 3 methods x64 frames',flush=True)
    comparisons=[]
    for clip in clips:
        indexed={p['method']:p for p in points if p['video_id']==clip['video_id']}
        for method in ('A','B'):
            c=indexed['C'];base=indexed[method];gap=(base['file_bytes']-c['file_bytes'])/base['file_bytes']
            comparisons.append({'video_id':clip['video_id'],'candidate':'C','comparator':method,
                'byte_change_fraction':c['file_bytes']/base['file_bytes']-1,'near_equal_no_larger_2pct':0<=gap<=.02,
                'delivery_fish_delta_db':c['metrics']['delivery']['fish']['psnr_db']-base['metrics']['delivery']['fish']['psnr_db'],
                'delivery_rest_delta_db':c['metrics']['delivery']['rest']['psnr_db']-base['metrics']['delivery']['rest']['psnr_db'],
                'original_fish_delta_db':c['metrics']['original_upsampled']['fish']['psnr_db']-base['metrics']['original_upsampled']['fish']['psnr_db']})
    controls=read_json(root/'controls.json')['records'];semantic_activity={}
    for clip in clips:
        rows=[r for r in controls if r['video_id']==clip['video_id']]
        semantic_activity[str(clip['video_id'])]={'frames':len(rows),'fallback_counts':dict(Counter(r['fallback'] or 'none' for r in rows)),
            'active_frames':sum(r['fallback'] is None for r in rows),'mean_alpha':sum(r['protected_alpha_mean'] for r in rows)/len(rows)}
    result={'schema':'d2-lowrate-results-v1','date':'2026-10-22','size':[width,height],'points':points,'comparisons':comparisons,
            'semantic_activity':semantic_activity,
            'hard_rate_bps':40000,'frame_point_metrics':len(frame_metrics),'all_selected_rate_and_growth_pass':True,
            'subjective_acceptance':False,'independent_source_generalization':False,'quality_reference':'same resized source before semantic processing; separately original source after upsampling',
            'quality_evaluation_seconds':time.perf_counter()-started}
    save_json(root/'evaluation.json',result);save_json(root/'frame_metrics.json',frame_metrics);save_json(root/'decoder_audit.json',audits)
    return result


def viewing(root,result):
    clips=read_json(root/'inputs.json')['clips'];runtime=read_json(root/'runtime.json');out=root/'viewing';out.mkdir(exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    labels={'R':'缩小后的原图','A':'普通 GLC','B':'全图低通','C':'语义保护'};records=[];cards=[]
    width,height=result['size']
    for clip in clips:
        chosen={p['method']:p for p in result['points'] if p['video_id']==clip['video_id']}
        crop=clip['fish_crop_original'];crop=(int(crop[0]*width/1920),int(crop[1]*height/1080),int(np.ceil(crop[2]*width/1920)),int(np.ceil(crop[3]*height/1080)))
        for mode,methods in [('A_C',['A','C']),('all',['R','A','B','C'])]:
            cell=512;panel_height=666;target=out/f'video_{clip["video_id"]:03d}_{mode}.mp4'
            cmd=[runtime['ffmpeg'],'-hide_banner','-loglevel','error','-nostdin','-f','rawvideo','-pix_fmt','rgb24','-s',f'{cell*len(methods)}x{panel_height}',
                 '-r',clip['fps'],'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','14','-pix_fmt','yuv420p','-movflags','+faststart','-n',str(target)]
            with target.with_suffix('.log').open('wb') as log:
                proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
                try:
                    for f in clip['frames']:
                        canvas=Image.new('RGB',(cell*len(methods),panel_height),'#101820');draw=ImageDraw.Draw(canvas)
                        for col,m in enumerate(methods):
                            x=col*cell
                            if m=='R': path=Path(f['resized'][f'{width}x{height}']['rgb']);subtitle='相同时间与分辨率的参考'
                            else:
                                p=chosen[m];path=root/'points'/p['id']/'decoder/receiver/frames'/f'im{f["index"]+1:05d}.png'
                                subtitle=f'{p["file_bytes"]:,} B | {p["file_kbps"]:.2f} kbps | q{p["q"]}'
                            title=labels[m]
                            if m=='C' and result['semantic_activity'][str(clip['video_id'])]['active_frames']==0:title='语义方案（全部回退原图）'
                            im=Image.open(path).convert('RGB');draw.text((x+12,6),title,font=font,fill='white')
                            draw.text((x+12,34),subtitle,font=small,fill='#b6c6d6');canvas.paste(im,(x,60))
                            draw.text((x+12,356),'同一固定鱼区放大',font=small,fill='#b6c6d6')
                            enlarged=ImageOps.contain(im.crop(crop),(cell-16,240),Image.Resampling.NEAREST)
                            canvas.paste(enlarged,(x+(cell-enlarged.width)//2,386+(240-enlarged.height)//2))
                            draw.text((x+12,638),f'帧 {f["index"]+1}/64 · {width}×{height} · 原速 29.97 fps',font=small,fill='white')
                        proc.stdin.write(np.asarray(canvas).tobytes())
                        if mode=='A_C' and f['index'] in (0,31,63):canvas.save(out/f'video_{clip["video_id"]:03d}_{f["index"]:02d}.png')
                    proc.stdin.close()
                    if proc.wait()!=0:raise RuntimeError('Viewing encoding failed')
                except BaseException:proc.kill();proc.wait();raise
            records.append({'video_id':clip['video_id'],'mode':mode,'file':target.name,'crop_delivery_xyxy':crop,'crop_track':clip['crop_track'],'lossy_local_viewing_only':True})
            print('Created '+target.name,flush=True)
        rows=''.join(f'<tr><td>{labels[m]}</td><td>{chosen[m]["file_bytes"]:,} B</td><td>{chosen[m]["file_kbps"]:.2f} kbps</td><td>{chosen[m]["metrics"]["delivery"]["fish"]["psnr_db"]:.3f} dB</td></tr>' for m in 'ABC')
        activity=result['semantic_activity'][str(clip['video_id'])]
        note=('本段 64 帧保护区过大，现有规则全部回退原图；C 与 A 的差异来自 q，不构成语义收益证据。' if activity['active_frames']==0 else f'本段 {activity["active_frames"]}/64 帧实际启用语义低通保护；仍需按实际字节差额判断。')
        cards.append(f'<section><h2>片段 {clip["video_id"]:03d} · {"海草背景" if clip["video_id"]==11 else "开阔海底"}</h2><p>鱼像素平均占画面 {100*clip["mean_fish_fraction"]:.2f}%；512×288，原帧率 30000/1001，约 2.135 秒。</p><p><strong>{note}</strong></p><table><tr><th>方法</th><th>完整发送文件</th><th>文件码率</th><th>鱼区 PSNR</th></tr>{rows}</table><h3>普通 GLC 与语义方案</h3><video controls loop playsinline src="video_{clip["video_id"]:03d}_A_C.mp4"></video><div><button data-speed="1">原速</button><button data-speed="0.5">半速</button><button data-step="-1">前一帧</button><button data-step="1">后一帧</button></div><details><summary>查看原图、全图低通及三种输出</summary><video controls loop playsinline src="video_{clip["video_id"]:03d}_all.mp4"></video></details></section>')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>D2 大目标低码率对比</title><style>body{font:16px system-ui;background:#101820;color:#e9eff6;max-width:1100px;margin:28px auto;padding:0 20px}p{line-height:1.7;color:#bbcad7}video{width:100%;background:#000}section{margin:38px 0}button{font:inherit;padding:8px 14px;margin:8px 6px 8px 0;background:#26384a;color:white;border:1px solid #65788b;border-radius:6px}table{border-collapse:collapse;margin:14px 0}td,th{padding:9px 18px;text-align:left;border-bottom:1px solid #526172}summary{cursor:pointer;padding:14px 0}</style><h1>D2：大目标，完整文件不超过 40 kbps</h1><p>直接标明方法，不要求盲猜。上半部为全画面，下半部为同一固定鱼区的像素放大。所有对比保持同一原帧率和分辨率。这里的 MP4 仅为本地观看副本，其文件大小不计作通信实验；表中数值来自实际发送的完整 glcv。放大图用于观察已有细节，不补造细节。</p>'''+''.join(cards)+'''<script>document.querySelectorAll('section').forEach(s=>{const v=s.querySelector('video'),fps=30000/1001;s.querySelectorAll('[data-speed]').forEach(b=>b.onclick=()=>v.playbackRate=Number(b.dataset.speed));s.querySelectorAll('[data-step]').forEach(b=>b.onclick=()=>{v.pause();const i=Math.floor(v.currentTime*fps+1e-4);v.currentTime=(Math.max(0,Math.min(63,i+Number(b.dataset.step)))+.01)/fps;});});</script></html>'''
    (out/'index.html').write_text(html,encoding='utf-8');save_json(out/'viewing_manifest.json',records)


def report(root,result):
    points=result['points'];comparison=result['comparisons'];gate=read_json(root/'rate_gate.json')
    rows=['| 片段 | 方法/q | 完整文件 B | kbps | 相对基准增幅 | 鱼 PSNR | 其余画面 PSNR | 回到原尺寸后的鱼 PSNR |','|---|---|---|---|---|---|---|---|']
    for p in points:
        rows.append(f'| {p["video_id"]:03d} | {p["method"]}/q{p["q"]} | {p["file_bytes"]:,} | {p["file_kbps"]:.3f} | {p["growth_vs_S0"]:+.2%} | {p["metrics"]["delivery"]["fish"]["psnr_db"]:.3f} | {p["metrics"]["delivery"]["rest"]["psnr_db"]:.3f} | {p["metrics"]["original_upsampled"]["fish"]["psnr_db"]:.3f} |')
    deltas=['| 片段 | 比较 | 文件变化 | 鱼区变化 dB | 其余画面变化 dB | 不更大且差额≤2% |','|---|---|---|---|---|---|']
    for c in comparison:
        deltas.append(f'| {c["video_id"]:03d} | C vs {c["comparator"]} | {c["byte_change_fraction"]:+.2%} | {c["delivery_fish_delta_db"]:+.3f} | {c["delivery_rest_delta_db"]:+.3f} | {"是" if c["near_equal_no_larger_2pct"] else "否"} |')
    text='''# D2 大目标低码率修订结果

日期：2026-10-22。用户指出旧视频难看出差异、300 多 kbps 文件不可接受，并明确选择保留原帧率、适当降低分辨率。

## 纠正预算和素材

旧 D2 的 1080p 结果只作历史高码率诊断，不是可接受的通信文件。本次把完整 glcv ≤40,000 bit/s 改为硬门槛，并同时保留每片相对冻结普通 GLC 基准最多 +10%。约 2.135 秒的片段对应 10,677 B 绝对上限。没有通过改帧率、延长时长或只统计神经载荷来压低数值。

换用 011 海草背景和 021 开阔海底，两段全部鱼像素平均占画面 3.20%、3.24%；旧 001/002 约为 0.16%、0.20%。窗口只按原标注目标面积和原始画面挑选，没有按模型收益挑选。原先保守预留组中的 011/021 已明确转为开发素材；经过标注筛选或源图查看的序列不能再声称是完全未见测试，007/025 未参与本轮。尚未证明跨来源独立性。

## 先通过码率门槛

640×360、q0：011 为 12,493 B，超限；021 为 10,616 B，达标。共同分辨率因此降为 512×288、原帧率 30000/1001，q0 基准分别为 9,865 和 8,846 B，约 36.96 和 33.14 kbps。两段有效预算分别为 10,677 和 9,730 B，均取绝对码率与 +10% 两者中的较小值。

先锁定上述普通基准与共同分辨率，再生成 B 全图低通和 C 冻结预测保护低通。A/B 都在同一有效预算下按实际字节选点；C 尝试不大于所选 A/B。每方法每片最多四个 q 点，按字节选，未用重建画质选参数；超限探测文件只保留为诊断，不作为交付。

021 的 C@q0=9,372 B 无法小于当前 A=9,094 B 和 B=8,785 B，但符合两项用户硬限制。独立解码前登记 v2 修订：不追加任何编码，保留未配平记录，选择已有 C@q0 展示实际大小/质量取舍；不得标为等字节收益。原失败记录和修订前选择均保留。

## 语义实际启用情况

011 的全部 64 帧因保护区域超过半幅画面触发现有回退，C 的编码输入与 A 相同；两者 q 不同产生的质量差异不得归为语义效果。021 的 64 帧全部实际启用预测保护，平均 alpha 覆盖约 16.10%。本次没有根据结果调整阈值或关闭潜水员分支。

## 实测输出

A=普通 GLC，B=全图低通，C=预测语义保护。

'''+ '\n'.join(rows)+'\n\n'+ '\n'.join(deltas)+'''

主指标先对相同 512×288 原始缩小输入和鱼 GT 计算跨帧汇总 SSE，再计算 PSNR；不把缩小后的 PSNR 与上一轮 1080p PSNR 直接比较。另列解码结果放大回 1920×1080 后对原图/原 GT 的指标，包含缩小造成的信息损失。低通 sigma 固定为编码尺寸的 1.2 像素。没有训练或阈值搜索。

所选六点均通过独立接收重建验证，接收配置不含原图或 GT；所有合规判断使用完整码流和真实播放时长。没有进行旧数据或权重的完整性重扫。新画面在准备时只做时间对应，避免把标注序号误当实际 PTS。

## 观看与限制

默认直接标明普通 GLC/语义保护，提供同步全画面、固定鱼区放大和全部对照，不要求用户盲猜。放大裁剪由原标注鱼轨迹在全部 64 帧中的包围框决定，未按重建效果挑图。MP4 是本地有损观看副本，不参与文件码率或数值质量计算。静帧检查不等于连续观看通过；主观是否更清楚仍待用户判断。

这次达标的是低码率文件约束，并不自动代表语义收益、完整 D 轮或潜水员视频通过。两个开发片段不足以证明泛化；只按本轮真实数值报告，不补造“更清楚”的结论。

本地观看：`'''+str(root/'viewing/index.html')+'''`。
'''
    (root/'result_report.md').write_text(text,encoding='utf-8')


def run(args):
    root=args.run.resolve();result=evaluate(root);report(root,result);viewing(root,result)
    save_json(root/'postprocess_progress.json',{'state':'completed','selected_points':6,'frame_point_metrics':384})
    print('Low-rate evaluation and labeled viewing complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);run(p.parse_args())
