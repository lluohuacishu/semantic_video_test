"""Package small D2.1 evidence and a numeric report without moving large assets."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from prepare_d2 import read_json,save_json


def run(root):
    here=Path(__file__).resolve().parent;ev=read_json(root/'evaluation.json');diag=read_json(root/'cache_diagnosis.json')
    if read_json(root/'postprocess_progress.json')['state']!='completed':raise ValueError('Postprocessing incomplete')
    runtime=read_json(root/'runtime.json');ffprobe=str(Path(runtime['ffmpeg']).with_name('ffprobe.exe'));checks=[]
    for item in read_json(root/'viewing/viewing_manifest.json'):
        path=root/'viewing'/item['path']
        probe=json.loads(subprocess.check_output([ffprobe,'-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration','-of','json',str(path)],text=True))['streams'][0]
        if probe['nb_frames']!='64' or probe['r_frame_rate']!='30000/1001' or probe['avg_frame_rate']!='30000/1001':raise ValueError('Viewing timeline mismatch')
        if [probe['width'],probe['height']]!=item['size']:raise ValueError('Viewing layout mismatch')
        checks.append({'file':item['path'],**probe})
    save_json(root/'viewing_checks.json',{'videos':checks,'all_passed':True,'visual_review':'Fixed source/overlay/comparison PNGs inspected; continuous subjective acceptance remains pending'})
    def table(category):
        rows=['| 片段 | 方案/q | 完整文件 B | kbps | 相对 S0 | 鱼 PSNR | 鱼边界 PSNR | 其余画面 PSNR | 回原尺寸鱼 PSNR |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for vid in (11,21):
            points={p['method']:p for p in ev['points'] if p['video_id']==vid and p['category']==category}
            for method in 'ABCFO':
                p=points[method];m=p['metrics']
                rows.append(f'| {vid:03d} | {method}/q{p["point"]["q"]} | {p["file_bytes"]:,} | {p["file_kbps"]:.3f} | {p["growth_vs_S0"]:+.2%} | {m["delivery"]["fish"]["psnr_db"]:.3f} | {m["delivery_boundaries"]["fish_inner_edge"]["psnr_db"]:.3f} | {m["delivery"]["rest"]["psnr_db"]:.3f} | {m["original_upsampled"]["fish"]["psnr_db"]:.3f} |')
        return '\n'.join(rows)
    comparisons=['| 片段 | 比较 | 文件变化 | 鱼 PSNR 差值 | 其余画面差值 | 不更大且差额≤2% |','|---|---|---:|---:|---:|---|']
    for c in ev['comparisons']:
        if c['category']=='selected' and (c['candidate']=='O' and c['comparator'] in 'AB' or c['video_id']==21 and c['candidate']=='F'):
            comparisons.append(f'| {c["video_id"]:03d} | {c["candidate"]} vs {c["comparator"]} | {c["byte_change_fraction"]:+.3%} | {c["fish_delta_db"]:+.3f} dB | {c["rest_delta_db"]:+.3f} dB | {"是" if c["near_equal_no_larger_2pct"] else "否"} |')
    report='''# D2.1 缓存诊断与有限消融结果

日期：2026-10-22。用户审阅 D2.md 后授权开始缓存诊断和有限消融。

**结论：定位确有问题，但准确 GT 鱼保护也未达到当前数值验收目标。仅关闭潜水员分支无效；当前“保留目标原像素＋背景低通”的收益不能靠提高分割准确度直接保证。** 这不等于主观画质已经判定，全图低通的 PSNR 优势也不能直接说成所有鱼看起来更清楚。

## 1. 范围、预算与执行

沿用 011/021 各64帧、512×288、原30000/1001 fps、相同GOP/reset。冻结 S0 仍为9865/8846 B，有效上限10677/9730 B，同时要求完整文件≤40,000 bit/s和逐片增幅≤10%。没有训练、重新语义推理、完整性重扫、复制ZIP或重跑旧矩阵；GLC权重、神经计算、熵编码和接收器保持原配置。

实际完成128帧缓存诊断，逐像素重现全部原 C alpha 与编码输入；新增11次编码（上限15），12次独立解码（上限13），评价15个唯一合规码流、960个帧×码流。011仅鱼输入等于原图，直接复用，未重复编码。探测中的超限文件保留证据，不作为交付；表中选定点与q0诊断均通过两项硬限制。

协议：[缓存计划](d21_cache_plan_v1.json)、[有限编码计划](d21_codec_plan_v1.json)、[选点修订](d21_selection_amendment_v2.json)。[完整小型证据](evidence_d21_20261022/evaluation.json)，大文件仍在本地 D21_20261022_r1。

## 2. 011：鱼分支本身已足以触发回退

| 分支 | 原始种子面积 | 仅扩张面积 | 仅加历史面积 | 扩张＋历史面积 | 最终超半幅帧数 |
|---|---:|---:|---:|---:|---:|
| 鱼 | 37.01% | 51.70% | 38.63% | 52.63% | 64/64 |
| 潜水员 | 1.32% | 2.08% | 1.37% | 2.14% | 0/64 |
| 合并 | 38.33% | 53.74% | 39.99% | 54.72% | 64/64 |

鱼分支仅扩张时52/64帧超限，加一帧历史后64/64；合并分支仅扩张就64/64。不能把潜水员分支视为唯一原因，关闭它不能修复011。阶段和分支差额存在重叠，不作为可直接相加的独立因果贡献。

固定示意帧可见鱼预测覆盖了大块海草纹理。鱼GT不提供完整潜水员/背景标注，因此报告使用“非鱼GT区域”，不将所有GT外预测自动计为误检。

## 3. 021：大部分鱼输入已保留，编码后仍没有收益

| 保护 | 平均alpha | 鱼像素完全保护率 | 鱼内边界完全保护率 | 编码前鱼MSE |
|---|---:|---:|---:|---:|
| 原 C：鱼＋潜水员 | 16.10% | 95.38% | 84.33% | 0.01464 |
| F：仅鱼 | 14.47% | 94.28% | 81.13% | 0.05253 |
| O：GT鱼 | 4.56% | 100% | 100% | 0 |

仅鱼方案减少了保护面积，也丢失了原合并分支补充的部分鱼边界；同q0时文件增加125 B（+1.334%），鱼PSNR仅+0.0055 dB，鱼边界PSNR反而下降约0.0985 dB。关闭潜水员分支没有形成有意义的收益。

GT保护在两段均64/64帧启用，完全保留真实鱼输入，平均alpha为5.27%/4.56%。边界指标为编码尺寸GT的一像素内边界，属于诊断，不能代替视频时序或主观清晰度。

## 4. 选定文件与实际质量

A=普通GLC，B=全图低通，C=原鱼＋潜水员预测保护，F=仅鱼预测保护，O=GT鱼保护（只作离线诊断，不能部署，也不是保证最优的上界）。F/O只改变种子来源，沿用16px扩张、一帧历史、原软边、回退及编码尺寸sigma1.2的低通。

'''+table('selected')+'\n\n'+'\n'.join(comparisons)+'''

011的O相对A少7 B、相对B少58 B，均满足严格近等体积；鱼PSNR分别低0.468/2.063 dB。021的O相对A少50 B、鱼PSNR高0.092 dB；相对B多10 B、鱼PSNR低1.240 dB。021 O对B不满足“不更大”规则，不能标为严格等字节收益；实际差额仅0.104%仍必须披露。

021的O@q25=9670 B只比配平目标B=9660 B多10 B；原严格选择保留O@q0=8431 B。为避免把约13%的体积不足误当准确保护机制失败，在读取任何新重建画面或数值质量之前登记v2修订。当时首批独立解码已启动；新增0次编码，只补解码已有O@q25。修订前选择和q0结果均保留，不掩盖匹配失败。

## 5. 固定q0诊断

此表只隔离全局q的变化，**不代表等码率**；实际字节同时列出，重复输入/解码均复用。

'''+table('q0')+'''

两段固定q0时，B仍取得比O更高的鱼PSNR。011的B文件更小；021的O比B少70 B（0.82%），满足O不更大且差额≤2%，O仍较低。因此原结论不完全由不同q造成。编码前O鱼MSE为0而B约10.68/11.46，编码后B却更好：这证明保留目标原始像素不保证重建后的目标保真。它没有证明原因一定是去噪，也没有证明“修改GLC内部比特分配”已是必要方案。

## 6. 观看与接续

本地观看页：`'''+str(root/'viewing/index.html')+'''`。提供两段B/O同步全画面＋固定鱼区放大、021 C/F、原图与全部方案，以及固定第1/22/43/64帧掩码诊断。五个观看视频均核对64帧、原30000/1001帧率。MP4为有损本地观看副本，质量指标来自独立解码PNG，实际发送大小来自完整glcv。

静帧已检查版式与内容，**持续观看清晰度仍待用户判断**。021的一些GT输出边缘较明显，但不能仅凭单帧锐度或PSNR断言主观通过；须结合原图查看边缘、拖影和稳定性。潜水员连续视频与独立来源泛化仍未验收。

本轮到此停止，不自动继续搜q、调阈值、训练或修改GLC。建议先观看B与O，若继续，应围绕“目标也允许轻度低通”与当前硬保留机制做单变量对照，而不是先投入分割微调。该建议尚未实施，也不代表已经证明新机制有效。

复现入口：[复现说明](D21复现说明.md)。原D2与D2.1已完成目录不得重启。
'''
    (root/'result_report.md').write_text(report,encoding='utf-8')
    (here/'D21缓存诊断与消融结果_20261022.md').write_text(report,encoding='utf-8')
    evidence=here/'evidence_d21_20261022';evidence.mkdir(exist_ok=False)
    names=['cache_plan.json','codec_plan.json','cache_diagnosis.json','selection.json','selection_before_amendment.json',
           'selection_amendment_v2.json','evaluation.json','decoder_audit.json','new_encodes.json','progress.json','postprocess_progress.json','viewing_checks.json']
    for name in names:shutil.copy2(root/name,evidence/name)
    files=['analyze_d21_cache.py','run_d21_codec.py','amend_d21_selection.py','finish_d21.py','record_d21_results.py',
           'd1_protection.py','d2_codec_worker.py','d21_cache_plan_v1.json','d21_codec_plan_v1.json','d21_selection_amendment_v2.json']
    save_json(evidence/'code_provenance.json',{'run':str(root),'files':[{ 'path':name,'sha256':hashlib.sha256((here/name).read_bytes()).hexdigest()} for name in files],
        'model_or_dataset_integrity_rescan':False})
    print('Report, video metadata checks and small evidence recorded',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);run(p.parse_args().run.resolve())
