"""Publish the fixed D2.3 cache-only screening, including failed candidates."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
from PIL import Image
from prepare_d2 import read_json, save_json


def result_table(result):
    rows=['| 片段 | 方案 | 平均保护面积 | 最大保护面积 | 原尺寸鱼覆盖 | 512鱼覆盖 | 512鱼内边界覆盖 | 回退帧 | 最终鱼保护率（含回退） |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for vid,clip in result['clips'].items():
        for method,v in clip['methods'].items():
            rows.append(f'| {int(vid):03d} | {method} | {v["mean_original_hard_area"]:.2%} | {v["max_original_hard_area"]:.2%} | {v["original_prefallback_fish_recall"]:.2%} | {v["delivery_prefallback_fish_recall"]:.2%} | {v["delivery_prefallback_edge_recall"]:.2%} | {v["fallback_frames"]}/64 | {v["final_fish_core_recall"]:.2%} |')
    return '\n'.join(rows)


def report(result):
    a=result['clips']['11'];b=result['clips']['21'];fc=a['components']['fish'];removed=a['cleanup']
    tiny=sum(x['components'] for x in fc['bins'] if x['max']<64)
    return f'''# D2.3：缓存掩码有限修复筛选

日期：2026-10-22。用户审阅更新后的D2.md并说“那继续吧”，授权先查缓存、检查021退步、合格后才有限编码。

**结果：两个预登记候选都未通过筛选，没有进入新编码。** 清理极小连通块没有修好011；进一步缩小扩张虽然减少回退，却丢失更多鱼及边界。021也出现局部边界退步。不能把“不再回退”当成“正确保护”。

## 1. 范围和门槛

直接复用011/021各64帧的原尺寸二值鱼/潜水员种子，以及512×288参考和GT。全部128帧完成，两个候选共256帧次检查；原C alpha在128帧全部逐像素复现。CPU分析耗时约{result['seconds']:.1f}秒，未加载语义或GLC模型，新增推理/训练/编码/解码均为0，未重扫数据或权重。

| 方案 | 鱼种子 | 鱼扩张半径 | 潜水员分支 | 历史 |
|---|---|---:|---|---|
| C | 原二值种子 | 16 | 原种子、16px扩张 | 前一帧原种子 |
| K16 | 删除少于64原尺寸像素的鱼连通块 | 16 | 完全保留 | 前一帧清理后种子 |
| K8 | 同K16 | 8 | 完全保留，仍16px扩张 | 同K16 |

使用8邻接。64像素相当于原尺寸8×8小块，约编码尺寸2.13×2.13像素；只是预先固定的小碎块诊断阈值，不代表小块一定不是鱼。不删除大块、不用GT决定保留哪个块、不按片段设不同规则。此诊断明确测试D1原先“保留小块”规则的一项有限例外，正式默认保护规则没有改变。

目标仍保留原像素，背景sigma1.2、外侧软边和回退条件不变。空预测或保护范围超过半幅继续回退原RGB；回退后的100%保留不能拿来提高定位成绩。

预登记沿用D1的必要条件：每片原尺寸鱼覆盖≥95%、平均保护面积≤25%、单帧≤50%。本轮额外要求无回退、编码尺寸鱼覆盖≥95%、鱼内边界覆盖≥90%；021鱼及边界累计覆盖下降≤0.5个百分点、任一帧下降≤2个百分点。两片全部通过才允许新编码，结果出来后不放宽门槛。这不是完整D1实例/潜水员或生产验收。

来源：[预登记计划](d23_cache_repair_plan_v1.json)、[结果证据](evidence_d23_20261022/cache_results.json)。

## 2. 保护面积下降不等于安全修复

表中面积、鱼覆盖和内边界覆盖均来自**回退前**的保护，最后一列单独列出应用回退后的实际保留。覆盖率按像素累计，面积为逐帧平均/最大值；内边界是512×288鱼GT的一像素内边界。

{result_table(result)}

011的K16仍64/64回退。K8减少为21/64回退，但原尺寸鱼覆盖由84.40%降至76.57%；最终包含回退的鱼保留也只有88.02%，低于旧C全图透传的100%。因此不会将K8编码成“修复成功”的新方案。

021的K16平均面积由15.35%降到13.07%，累计鱼覆盖变化很小，但最差一帧的内边界覆盖下降4.32个百分点，超过预登记2个百分点限制。K8累计鱼覆盖下降0.758个百分点、边界下降3.446个百分点，最差一帧边界下降8.80个百分点。它们都未通过本轮筛选，不能替换用户觉得还行的原C。

本轮新增的90%绝对边界筛选比原021 C的84.33%严格，它不意味着此前主观正向评价被否定。即使不考虑这一新增条件，011仍不满足原有95%鱼覆盖和25%面积门槛，021 K16/K8也仍有逐帧边界退步；没有靠新增边界条件独自决定本轮结论。

## 3. 011的大块与真实鱼混在一起

011共有{fc['component_frame_count']:,}个鱼预测“连通块×帧”，其中小于64像素的{tiny:,}个占{tiny/fc['component_frame_count']:.2%}，却只占全部鱼种子像素的{removed['removed_pixels']/fc['total_seed_pixels']:.3%}。删除了{removed['removed_pixels']:,}个种子像素，其中{removed['removed_fish_GT_pixels']:,}个与鱼GT重叠，不能称无损去噪。

每帧最大鱼连通块合计占种子像素的{fc['largest_component_seed_fraction']:.2%}，并包含原始鱼种子已经覆盖的GT鱼像素的{fc['largest_component_GT_overlap_fraction_of_seed_overlap']:.2%}。后一个分母是“原始种子已覆盖的鱼像素”，不是全部真实鱼。直接把最大块当海草删除，也会丢掉大量已检测鱼区域。

GT外区域不一概标为背景或误检；鱼GT并不提供完整潜水员及背景标注。连通块也不等于鱼实例。所有连通块面积分档和GT交叠在结果JSON里完整保留。

## 4. 一个明确但有限的结论

逐像素验证了两候选全部回退前保护都是原C回退前保护的子集。原C在011只能覆盖84.40%的原尺寸GT鱼像素，因此**只删当前种子、提高同一概率图阈值、缩小扩张或减少历史保留，都无法补回原保护以外的漏鱼**，在不新增保护区域的前提下达不到95%。

这只限定“得到原保护子集”的改法，不是说所有后处理、模型输入或新推理都无效。当前二值缓存没有保存概率图，不能反推出高置信度；只做更严格阈值同样不会恢复漏检。后续需要能区分大块非鱼响应并补回漏鱼的新定位信息。

## 5. 诊断观看与验证

本地运行：`C:/Users/lenovo/Documents/ChatGPT/语义通信/02_实验数据/D_语义目标增强/D23_20261022_r1`。

- [诊断页](../../../../02_实验数据/D_语义目标增强/D23_20261022_r1/viewing/index.html)：两片各固定第1/22/43/64帧，上排原图/种子、下排C/K16/K8回退前保护。红色预测，绿色GT。
- `frame_diagnostics.json`包含128帧连通块统计、原尺寸/编码尺寸覆盖、回退、最终保护及连续掩码IoU/XOR。后两者仅为描述性统计，不能当主观闪烁评分。
- `masks/v*/C,K16,K8/*.png`用bit0保存回退前硬保护、bit1保存最终完全保护核。不是概率图或可发送码流。
- `cache_results.json`保留全部失败检查。`progress.json`为`completed_no_candidate_for_codec`。没有新PSNR、实际码率或连续解码视频结果。
- 代码检查、GT不参与清理的合成检查、原C复现、子集验证、图像尺寸与引用检查见[verification.json](evidence_d23_20261022/verification.json)。

## 6. 接续方向

本轮按预登记在缓存筛选阶段结束，0个候选获准编码。021原C和GT原像素保护的“还行”评价继续保留，目标轻度低通仍不优先。没有改生产默认值、阈值、回退或协议预算。

下一步不宜继续只缩当前二值掩码。可考虑另立一个小范围定位诊断：固定少量011/021帧，补存置信度并检查模型看到的图像范围是否影响大块响应和漏鱼；需要同时检查假保护与漏鱼，不能仅提高阈值。该后续尚未执行，本轮不自动加候选、补推理、训练或改GLC。

复现见[D23复现说明](D23复现说明.md)。历史D2.1/D2.2数值、用户反馈和既有输出均保持原位。
'''


def viewing(root,result):
    out=root/'viewing';out.mkdir(exist_ok=False)
    data=[]
    for vid in (11,21):
        for index in (0,21,42,63):
            name=f'v{vid:03d}_{index:04d}.png';p=root/'diagnostics'/name
            with Image.open(p) as im:
                if im.size!=(1536,748):raise ValueError('Unexpected diagnostic panel size')
            data.append({'video':vid,'frame':index+1,'src':'../diagnostics/'+name})
    table=''.join('<tr>'+''.join(f'<td>{html.escape(x)}</td>' for x in (f'{int(vid):03d}',method,f'{s["mean_original_hard_area"]:.2%}',f'{s["original_prefallback_fish_recall"]:.2%}',f'{s["delivery_prefallback_edge_recall"]:.2%}',str(s['fallback_frames'])))+'</tr>' for vid,c in result['clips'].items() for method,s in c['methods'].items())
    source='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>D2.3 缓存修复筛选</title><style>body{max-width:1540px;margin:32px auto;padding:0 20px;background:#101820;color:#edf4fb;font:16px system-ui}p{line-height:1.8;color:#c6d5e3}h1{font-size:30px}img{width:100%;height:auto;display:block;border:1px solid #435467}button{background:#20384e;color:white;border:1px solid #678;padding:10px 18px;margin:5px;font:inherit;border-radius:6px}button[aria-pressed=true]{background:#427084}table{border-collapse:collapse;width:100%;margin:25px 0}td,th{padding:12px;border-bottom:1px solid #435467;text-align:left}a{color:#95cbff}.box{padding:18px;background:#1b2a38;border-left:4px solid #d5aa56}</style><h1>D2.3：少回退，未必保护得更好</h1><p class="box">两个预登记候选均未通过，因此没有新增编码。011缩小扩张后回退从64帧减为21帧，但回退前鱼覆盖从84.40%降到76.57%。021也有边界退步。</p><p>K16：清理小于64原尺寸像素的鱼连通块；K8：再把鱼扩张从16缩到8。两者都保持潜水员分支。下面显示的是回退前保护，不能把最终全图回退的100%保留算成定位成功。</p><table><tr><th>片段</th><th>方法</th><th>平均面积</th><th>原尺寸鱼覆盖</th><th>512鱼内边界覆盖</th><th>回退帧/64</th></tr>'''+table+'''</table><div id="buttons"></div><p id="caption"></p><img id="panel" alt="原图、二值种子与三个回退前保护掩码"><p>固定第1/22/43/64帧；红色为预测或保护，绿色为鱼GT。这些静态掩码不是新的解码视频，也不提供新的主观清晰度结论。</p><p><a href="../../D22_20261022_r1/viewing/index.html">返回D2.2连续视频及已评价方案</a></p><script>const frames='''+json.dumps(data,ensure_ascii=False)+''';const box=document.querySelector('#buttons');const img=document.querySelector('#panel');const caption=document.querySelector('#caption');frames.forEach((item,i)=>{const b=document.createElement('button');b.textContent=`${String(item.video).padStart(3,'0')} · 第${item.frame}帧`;b.setAttribute('aria-pressed','false');b.onclick=()=>{box.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed','false'));b.setAttribute('aria-pressed','true');img.src=item.src;caption.textContent=b.textContent};box.appendChild(b);if(i===0)b.click()});</script></html>'''
    (out/'index.html').write_bytes(source.encode('utf-8'))
    save_json(out/'manifest.json',data)


def finish(args):
    root=args.run.resolve();here=Path(__file__).resolve().parent;result=read_json(root/'cache_results.json')
    evidence=here/'evidence_d23_20261022';target=here/'D23缓存修复筛选结果_20261022.md'
    if evidence.exists() or target.exists() or (root/'viewing').exists():raise FileExistsError('Completed output exists')
    if result['eligible_candidates']:raise ValueError('Passing candidates require the conditional codec phase, not cache-only finalization')
    if read_json(root/'progress.json')['state']!='completed_no_candidate_for_codec':raise ValueError('Cache phase incomplete')
    viewing(root,result);text=report(result)
    target.write_bytes(text.encode('utf-8'));(root/'result_report.md').write_bytes(text.encode('utf-8'))
    evidence.mkdir()
    for name in ('plan.json','sources.json','cache_results.json','progress.json','code_provenance.json'):
        shutil.copy2(root/name,evidence/name)
    save_json(root/'postprocess_progress.json',{'state':'completed','panels':8,'new_videos':0,'new_encodes':0,'new_decodes':0,'eligible_candidates':[]})
    shutil.copy2(root/'postprocess_progress.json',evidence/'postprocess_progress.json')
    save_json(evidence/'postprocess_code_provenance.json',{'files':[{'path':n,'sha256':hashlib.sha256((here/n).read_bytes()).hexdigest()} for n in ('finish_d23_cache.py',)],'model_or_dataset_rescan':False})
    print('D2.3 cache-only report and diagnostics completed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);finish(p.parse_args())
