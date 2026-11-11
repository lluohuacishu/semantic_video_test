"""Release the bounded structural-guide experiment and its actual sidecars."""

import argparse
from pathlib import Path
import shutil

from receiver_structure import file_sha,load_json,save_json


def release(workspace,run):
    stage=Path(__file__).resolve().parent
    results={case:load_json(run/case/'evaluation/summary.json') for case in ('video1','video2','video3')
             if (run/case/'evaluation/summary.json').exists()}
    assert 'video1' in results and all(r['complete'] for r in results.values())
    choice=load_json(run/'decision.json')
    assert set(choice['completed_cases'])==set(results)
    view=workspace/'05_成果与观看/GLC_低码率结构引导_20261111';view.mkdir(exist_ok=False)
    evidence=stage/'evidence_20261111/structure_guidance';evidence.mkdir(exist_ok=False)
    copies={};index=1
    for case,result in results.items():
        for name in result['profiles']:
            for suffix,label in (('full','全画面'),('coral','珊瑚放大')):
                source=run/case/'evaluation'/f'K_vs_{name}_{suffix}.mp4'
                if not source.exists():continue
                dest=view/f'{index:02d}_{case}_左K_右{name}_{label}.mp4';index+=1
                shutil.copy2(source,dest);assert file_sha(source)==file_sha(dest)
                copies[dest.name]=dict(source=str(source),sha256=file_sha(dest))
            source=run/case/'evaluation'/f'source_raw_K_{name}.mp4'
            dest=view/f'{index:02d}_{case}_四宫格_原片_原GLC_K_{name}.mp4';index+=1
            shutil.copy2(source,dest);assert file_sha(source)==file_sha(dest)
            copies[dest.name]=dict(source=str(source),sha256=file_sha(dest))
            for suffix,label in (('full','全画面'),('coral','珊瑚放大')):
                source=run/case/'evaluation'/f'raw_vs_{name}_{suffix}.mp4'
                if not source.exists():continue
                dest=view/f'{index:02d}_{case}_左原GLC_右{name}_{label}.mp4';index+=1
                shutil.copy2(source,dest);assert file_sha(source)==file_sha(dest)
                copies[dest.name]=dict(source=str(source),sha256=file_sha(dest))
            target=view/'实际传输文件'/case/name;target.mkdir(parents=True)
            for source in (run.parent/'source_dense_flow_three_clips_20261109_01'/case/'base.glcsnd',
                           run/case/name/'guide.glcguide'):
                dest=target/source.name;shutil.copy2(source,dest);assert file_sha(source)==file_sha(dest)
                copies[str(dest.relative_to(view))]=dict(source=str(source),sha256=file_sha(dest))
    save_json(view/'来源核验.json',dict(complete=True,files=copies))
    inputs=load_json(run/'execution_plan.json')['input_sha256']
    assert all(file_sha(Path(p))==d for p,d in inputs.items())
    snapshot=run/'source_final';snapshot.mkdir(exist_ok=False)
    files=('structure_guide_stream.py','probe_structure_guidance.py','encode_budget_guide.py',
           'receive_structure_guide.py','receive_guide_texture.py','evaluate_structure_guide.py','release_structure_guide.py',
           'receiver_structure.py','run_fullframe_reference.py','evaluate_fullframe_reference.py',
           'run_fullframe_factorial.py','run_compact_support.py','compact_support_stream.py',
           'predictive_support_stream.py','run_p0.py','verify_video1_persistent_structure.py',
           'audit_deflicker_texture.py','probe_reef.py')
    for name in files:shutil.copy2(stage/name,snapshot/name)
    shutil.copy2(stage.parent/'11_优化阶段_纹理稳定性/temporal_filter.py',snapshot/'temporal_filter.py')
    save_json(snapshot/'manifest.json',{p.name:file_sha(p) for p in snapshot.iterdir()})
    save_json(run/'release_verification.json',dict(complete=True,cases=list(results),
        frozen_inputs_unchanged=True,frozen_input_count=len(inputs),
        source_blocked_receivers=True,actual_guide_decodes_and_replays_exact=True,
        old_K_raw_scores_exact=True,preview_and_transmission_copies_exact=True,
        preview_count=index-1,no_H265_equal_total_comparison=True,no_realtime_claim=True))
    rows=[];details=[];dynamics=[];times=[]
    for case,result in results.items():
        scope='coral' if case=='video1' else 'full';scores=result['scores'][scope]
        for name in result['profiles']:
            rate=result['rates'][name];s=scores[name]
            rows.append(f"| {case} / {scope} | {name} | {rate['auxiliary_bytes']:,} | {rate['auxiliary_kbps']:.4f} | {rate['auxiliary_saving_percent']:.2f}% | {scores['K']['coarse_mae']:.6f} → {s['coarse_mae']:.6f} | {scores['K']['fine_ncc']:.6f} → {s['fine_ncc']:.6f} |")
            full=result['scores']['full'];dynamic=result['source_change_diagnostic']['scores']
            details.append(f"| {case} / {name} | {full['K']['psnr_db']:.3f} → {full[name]['psnr_db']:.3f} | {full['K']['fine_amplitude']:.3f} → {full[name]['fine_amplitude']:.3f} | {result['coverage'][name]['full']['active_fraction']*100:.2f}% |")
            dynamics.append(f"| {case} / {name} | {dynamic['K']['RGB_delta_error']:.4f} → {dynamic[name]['RGB_delta_error']:.4f} | {dynamic['K']['change_amplitude_ratio']:.4f} → {dynamic[name]['change_amplitude_ratio']:.4f} |")
            t=result['runtime'][name];times.append(f"| {case} / {name} | {t['flow_seconds']:.2f} | {t['render_seconds']:.2f} | {t['peak_allocated_mib']:.2f} |")
    probe=load_json(run/'probe_summary.json')['records']
    report='''# 低码率结构引导：替代逐帧运动与可靠区传输

2026-11-11。用户原速查看 R0 后明确反馈 video1 与原 GLC 近乎无差距，拒绝该零辅助替代。本轮改测两种信息表示：源运动轨迹参数，以及低分辨率结构引导视频。原基础码流与潜变量参考链保持不变。

## 1. 先筛掉不合算的轨迹表示

固定源运动网格，对每个参考区间、每个运动节点拟合一次或三次 Bernstein 轨迹；新锚点运动固定为零，旧参考过渡允许非零截距，系数单位 1/8px，再恢复 K 使用的半像素网格。安全退出判据未放宽，重新检查拟合后运动的源对应误差与源光度一致性。

一次轨迹的系数＋可靠区载荷为 '''+str(probe['T1']['payload_only_bytes'])+''' B，但只保留 K '''+f"{probe['T1']['K_needed_cells_retained_fraction']*100:.2f}"+'''% 的原有可靠像素帧单元。三次轨迹保留 '''+f"{probe['T3']['K_needed_cells_retained_fraction']*100:.2f}"+'''%，载荷却达 '''+str(probe['T3']['payload_only_bytes'])+''' B，未计头就超过 K 的 24,649 B。这是表示与安全覆盖初筛，没有渲染或宣称其画质失败，也没有把载荷字节当实际新格式码率。该两档不继续开发。

## 2. 让接收端从实际传输的结构引导恢复运动

发送端把源 RGB 降到 128×72 或 256×144，用 H.265 编成小视频。GLCGDE01 完整封装包含基础流 SHA 绑定、尺寸、帧数、帧率、编解码约定、有效载荷长度、HEVC 及 CRC，全部计入辅助字节。

接收端只读原基础包与实际 guide.glcguide，先解出引导画面，三次插值放大后估计双向 RAFT 运动。越界、双向一致性、运动梯度以及两档光度检查也全部从这条接收到的引导计算。被复用的像素仍是历史原始 GLC RGB；引导画面不作为输出纹理，也没有偷偷读取原片或源光流。

结构复用保持当前六个锚点 0/15/45/75/105/135、六帧旧参考过渡、全 RGB 混合、2px 向内羽化与两次取整。没有语义分割、手画编辑区或对珊瑚单独调参数。可靠区逐像素逐帧计算；这是整个部署方案对照，和 K 的网格量化、4×4/五帧可靠区表示同时不同，不能归因成只改变一个因素。

首轮 G256 引导允许过多低纹理水体复用，左侧区域有回归。随后另冻结唯一一档接收端修正 T3：当前与投影后的参考引导，其 sigma3 局部亮度标准差均至少为 3，再与原可靠区取交集，向内羽化且 alpha 只能减少。没有调阈值扫描或手画水体 ROI。T3 为所有片段固定的接收端软件规则，使用与 G256 完全相同的实际辅助文件；没有发送方按片段选择的阈值、掩码或其他未计费信令。

初始固定 CRF 档产生 44.0696/54.7312 kbps，未达到码率要求，在看候选质量前停止。保留初始失败文件后，另冻结两档两遍编码目标：128×72 的 8 kbps、256×144 的 12 kbps。实际完整文件码率如下，目标码率不是额外硬约束或实测值。这次没有按最终候选分数搜索 CRF。

## 3. 实际接收与无损评分

每段各 150 帧、15 FPS、10 秒。评分使用无损 RGB 和与 K 完全相同的外部源相邻对应支持；不以候选可靠区缩小主分母。以下 MAE 越低、NCC 越高越好，百分比是辅助完整文件节省。

| 素材/区域 | 方案 | 辅助 B | 辅助 kbps | 比 K 省 | K → 引导 MAE | K → 引导 NCC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
'''+ '\n'.join(rows)+'''

| 素材/方案 | 整帧 K → 引导 PSNR dB | 细纹幅度 K → 引导 | 实际复用覆盖 |
| --- | ---: | ---: | ---: |
'''+ '\n'.join(details)+'''

## 4. 动态区域检查的范围

额外用原片每对相邻帧 RGB 变化的前 10% 像素，且变化至少 12，检查输出帧差与原片帧差的差异。该区域由源片独立选择，包含鱼、气泡和相机运动导致的背景边缘；它不是语义前景分割，不能独自证明鱼体边界或运动观感通过。

| 素材/方案 | K → 引导 RGB 帧差误差 | 相对源片变化幅度 K → 引导 |
| --- | ---: | ---: |
'''+ '\n'.join(dynamics)+'''

原速全画面、珊瑚放大和源片/原 GLC/K/引导四宫格均已生成。仍需按原速画面判断鱼体、气泡、遮挡退出和珊瑚形变，低分辨率引导可能漏掉细边界。本轮不以时序指标单独记录用户验收。

## 5. 接收边界、核验与计算成本

初始两份接收端读取白名单为基础包、已独立解码的 RGB 及其接收回执、实际 guide.glcguide、共享 RAFT 权重。T3 独立进程复用相同接收端已经本地计算的引导运动缓存，核对原无源回执、运动文件哈希、基础绑定和从实际包再次解码的引导 RGB 一致；这些缓存不传输。源 RGB、源双向运动、源发送元数据、发送端解出的引导数组与 K 辅助包的读取负例均阻断。FFmpeg 只通过标准输入接收实际 HEVC 有效载荷。Python open 审计是实验输入约束，不是操作系统沙箱。

实际包的引导解码与发送端预期逐位一致；原可靠性实现的 CPU 重放 RGB/alpha/共享权重逐位一致，零权重像素原样输出；K/raw 历史分数精确重现。CRC、截断、错误基础绑定、超限与有效 CRC 的错误帧数/帧率/尺寸/长度/编码器/放大方式负例通过。全部预览检查帧数、15 FPS、时长及首中末面板身份。原输入哈希保持不变。

| 素材/方案 | RAFT 秒/10秒视频 | 门控融合秒 | 峰值 PyTorch MiB |
| --- | ---: | ---: | ---: |
'''+ '\n'.join(times)+'''

共享 GLC/RAFT 权重和软件需预装；本地生成的稠密光流、掩码与 RGB 不是传输文件。T3 表中 RAFT 时间继承其原始本地缓存的计时，门控融合为新增 CPU 实测；T3 重新解引导的时间没有单独测量。当前两遍 H.265 与整段推理是批处理实验；机器存在其他 GPU 活动，不能据当次墙钟时间宣称独占性能或实时。传输错误恢复、未见素材和长片未测。

## 6. 当前决定

'''+choice['summary_zh']+'''

默认基线仍由实际观感与后续验证决定，本轮不覆盖此前已认可的 K/U。新的同总字节 H.265 主输出尚未重跑；引导内部使用 H.265 不能替代该公平画质对照。用户要的是有效降低辅助码率，不能把退回原 GLC 的结果算作成功。

## 文件

- 运行：`02_实验数据/优化_GLC后处理/structure_guidance_20261111_01`，初筛、预算调整、实际发送/接收和评分分开留存。
- 观看：`05_成果与观看/GLC_低码率结构引导_20261111`，左右对照均左 K、右标注的候选；四宫格为上排原片/原 GLC，下排 K/候选。
- 实际传输文件：观看目录 `实际传输文件/<素材>/<方案>/base.glcsnd + guide.glcguide`。
- 小证据：[evidence_20261111/structure_guidance](evidence_20261111/structure_guidance/README.md)。
- 代码：`probe_structure_guidance.py`、`encode_budget_guide.py`、`structure_guide_stream.py`、`receive_structure_guide.py`、`evaluate_structure_guide.py`。所有复现另建输出目录，现有基础与实验结果保留。
'''
    (stage/'低码率结构引导视频验证_20261111.md').write_text(report,encoding='utf-8')
    (view/'README.md').write_text('# 低码率结构引导对照\n\n'+choice['summary_zh']+'\n\n左右视频：左 K、右文件名所标候选。四宫格：上排原片/原 GLC，下排 K/候选。均为 10 秒、15 FPS 原速。\n\n实际传输文件包含原 base.glcsnd 与新 guide.glcguide；引导接收器及共享模型需预装。评分用无损 RGB，MP4 仅供观看。\n',encoding='utf-8')
    for path in run.rglob('*'):
        if path.is_file() and path.suffix in ('.json','.csv','.py'):
            dest=evidence/path.relative_to(run);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    (evidence/'README.md').write_text('# 低码率结构引导证据\n\nexecution_plan.json 保存初始冻结计划；budget_guide_plan.json 保存初始超预算后、看候选质量前的两档预算设置。\n各素材发送回执、无源接收回执、评分 CSV/JSON、原速预览核验与 decision.json 分开保存。\nsource_final 是精确代码与 SHA，release_verification.json 核验原输入及观看文件副本。大数组与视频保留在运行和观看目录。\n',encoding='utf-8')
    save_json(evidence/'manifest.json',{str(p.relative_to(evidence)):file_sha(p) for p in evidence.rglob('*') if p.is_file()})
    print('STRUCTURE GUIDE RELEASE COMPLETE',view,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();release(a.workspace.resolve(),a.run.resolve())
