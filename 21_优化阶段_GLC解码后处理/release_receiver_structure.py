"""Archive receiver-only validation, its lossless byte probe and viewing files."""

import argparse
from pathlib import Path
import shutil

from receiver_structure import file_sha, load_json, save_json


def release(workspace, run, resume=False):
    stage = Path(__file__).resolve().parent
    final = load_json(run/'final_summary.json')
    assert final['complete'] and final['frozen_inputs_unchanged']
    evidence = stage/'evidence_20261111/receiver_structure'
    evidence.mkdir(exist_ok=resume)
    view = workspace/'05_成果与观看/GLC_接收端零辅助验证_20261111'
    view.mkdir(exist_ok=resume)
    aliases = [
        ('video1/evaluation/K_vs_R0_full.mp4','01_video1_左K_右零辅助_全画面.mp4'),
        ('video1/evaluation/K_vs_R0_coral.mp4','02_video1_左K_右零辅助_珊瑚.mp4'),
        ('video2/evaluation/K_vs_R0_full.mp4','03_video2_左K_右零辅助_全画面.mp4'),
        ('video3/evaluation/K_vs_R0_full.mp4','04_video3_左K_右零辅助_全画面.mp4'),
        ('video1/evaluation/raw_vs_R0_coral.mp4','05_video1_左原GLC_右零辅助_珊瑚.mp4'),
        ('video1/evaluation/source_raw_K_R0.mp4','06_video1_四宫格_原片_原GLC_K_零辅助.mp4'),
        ('video2/evaluation/source_raw_K_R0.mp4','07_video2_四宫格_原片_原GLC_K_零辅助.mp4'),
        ('video3/evaluation/source_raw_K_R0.mp4','08_video3_四宫格_原片_原GLC_K_零辅助.mp4'),
        ('video1/evaluation/offline_factorial_coral.mp4','09_video1_离线交叉诊断_珊瑚.mp4')]
    copies = {}
    for name,alias in aliases:
        shutil.copy2(run/name,view/alias)
        assert file_sha(run/name) == file_sha(view/alias)
        copies[alias] = dict(source=str(run/name),sha256=file_sha(view/alias))
    for case in ('video1','video2','video3'):
        source = run.parent/'source_dense_flow_three_clips_20261109_01'/case/'base.glcsnd'
        dest = view/'零辅助方案的传输文件'/case/'base.glcsnd'
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
        assert file_sha(source) == file_sha(dest)
        copies[str(dest.relative_to(view))] = dict(source=str(source),sha256=file_sha(dest))
    save_json(view/'来源核验.json',dict(complete=True,files=copies))
    save_json(run/'visual_review.json',dict(status='not_user_reviewed',date='2026-11-11',
        inspected='Static source/raw/K/R0 contact sheets; no assertion of motion acceptance from still images',
        original_speed_previews_verified=True,default='K',
        fish_bubbles_diver_temporal_acceptance='Not certified; full-frame source comparisons provided'))
    probes = {case:load_json(run/case/'prediction_byte_probe/summary.json') for case in final['cases']}
    save_json(run/'prediction_byte_probe_summary.json',dict(complete=True,cases=probes,
        predictor_metadata_included=False,not_a_deployable_packet=True))
    snapshots = run/'source_final'; snapshots.mkdir(exist_ok=resume)
    for name in ('receiver_structure.py','evaluate_receiver_structure.py','run_receiver_structure.py',
        'probe_receiver_prediction.py','release_receiver_structure.py','run_fullframe_reference.py',
        'evaluate_fullframe_reference.py','run_support_factorial.py','verify_video1_persistent_structure.py',
        'compact_support_stream.py','predictive_support_stream.py','fullframe_structure_stream.py',
        'run_p0.py','probe_reef.py','audit_deflicker_texture.py'):
        shutil.copy2(stage/name,snapshots/name)
    shutil.copy2(stage.parent/'11_优化阶段_纹理稳定性/temporal_filter.py',snapshots/'temporal_filter.py')
    save_json(snapshots/'manifest.json',{p.name:file_sha(p) for p in snapshots.iterdir() if p.name!='manifest.json'})
    inputs = load_json(run/'execution_plan.json')['input_sha256']
    assert all(file_sha(Path(p)) == digest for p,digest in inputs.items())
    save_json(run/'release_verification.json',dict(complete=True,frozen_inputs_unchanged=True,
        input_count=len(inputs),preview_copies_exact=True,base_copies_exact=True,
        preview_count=len(aliases),all_three_receiver_input_guards_passed=True,
        all_three_SS_reproduced=True,all_three_RR_replays_exact=True,
        all_three_K_raw_scores_reproduced=True,default_remains_K=True))
    table = []; scores = []; factorial = []; byte_rows = []; timings = []
    for case,result in final['cases'].items():
        rate = result['rates']; scope = 'coral' if case == 'video1' else 'full'
        s = result['scores'][scope]
        table.append(f"| {case} | {rate['base_bytes']:,} | {rate['K_aux_bytes']*8/10000:.4f} | 0 | {rate['K_total_kbps']:.4f} → {rate['R0_total_kbps']:.4f} | {rate['total_bytes_saved_fraction']*100:.2f}% |")
        scores.append(f"| {case} / {'珊瑚' if case=='video1' else '整帧'} | {s['raw']['coarse_mae']:.6f} | {s['K']['coarse_mae']:.6f} | {s['RR']['coarse_mae']:.6f} | {(s['RR']['coarse_mae']/s['K']['coarse_mae']-1)*100:+.2f}% | {s['K']['fine_ncc']:.6f} → {s['RR']['fine_ncc']:.6f} |")
        factorial.append(f"| {case} / {scope} | {s['SS']['coarse_mae']:.6f} | {s['SR']['coarse_mae']:.6f} | {s['RS']['coarse_mae']:.6f} | {s['RR']['coarse_mae']:.6f} |")
        p = probes[case]
        byte_rows.append(f"| {case} | {p['K_bytes']:,} | {p['canonical_K_repack_bytes']:,} | {p['optimistic_surrogate_bytes']:,} | {p['delta_bytes']:+,} | {p['K_motion_payload_bytes']:,} → {p['residual_motion_payload_bytes']:,} |")
        t = result['timing']
        timings.append(f"| {case} | {t['flow_seconds']:.2f} | {t['render_seconds']:.2f} | {t['peak_allocated_mib']:.2f} |")
    coral = {n:v['coral'] for n,v in final['cases']['video1']['coverage'].items()}
    motion = final['cases']['video1']['motion_discrepancy_px']['coral']
    report = '''# 接收端零辅助流验证与运动预测差值字节探针

2026-11-11。按用户“运动、可靠区是否能在解码端计算”的要求，已完成三段独立接收端处理、九组离线交叉对照、无损评分、原速预览及不改变 K 的运动差值字节探针。当前保留 K；零辅助版 R0 没有保住当前稳定收益，不能按省字节直接升级。

## 1. 输入边界与实验设置

三段各前 150 帧、15 FPS、512×288、10 秒。R0 在代码和结果中记为 RR。参考素材全部为历史未经后处理的 GLC 解码 RGB。没有原片像素混入输出，没有递归引用滤波结果。

运动使用同一个预装 RAFT-Things large 权重、FP32、24 次迭代；本轮每段重新从解码 RGB 推理 174 对双向运动。锚点 0/15/45/75/105/135、六帧旧参考过渡、全 RGB 结构复用、2px 向内羽化及两次取整，沿用现有稠密结构参考框架。三段统一参数，没有手画编辑 ROI 或按素材调阈值。

可靠区依次检查越界、双向对应误差 <1.5px、sigma3 色差 <16、sigma1 色差 <24、运动梯度 <0.75，再做 3×3 腐蚀。这是对应关系可靠性检查，**没有语义分割**。R0 的色差也只从解码 RGB 计算。

R0 使用逐像素、逐帧稠密门控与运动，无须量化成待传网格。K 还包含半像素网格、4×4 块与五帧交集；所以 R0 对 K 是部署方案对照，归因另看固定稠密表示的 2×2 实验。

## 2. 码率确实归零，稳定收益没有保住

| 素材 | 本体 B | K 辅助 kbps | R0 辅助 kbps | K → R0 总 kbps | 总字节减少 |
| --- | ---: | ---: | ---: | ---: | ---: |
'''+ '\n'.join(table)+'''

基础文件完整字节计费。R0 不传辅助文件；运行目录中的光流、掩码和 RGB 数组是接收端本地诊断产物。共享代码和约 21.1MB RAFT 权重需预装，此处没有把模型分发算作每段新增码率。当前运行器仍是整段批处理，未做流式部署。

评价在独立进程读取原片，使用与历史 K 相同的外部源对应和有效像素分母，不用候选自己的可靠区筛掉坏样本。评分输入为无损 RGB。

| 素材/区域 | 原 GLC MAE↓ | K MAE↓ | R0 MAE↓ | R0 对 K 误差变化 | K → R0 细纹 NCC↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
'''+ '\n'.join(scores)+'''

这些百分比是指标变化，不能当作肉眼改善幅度或运动主体验收。完整源片/原 GLC/K/R0 四宫格已经生成；鱼、气泡、潜水员的原速感受没有替用户作出通过结论。

## 3. 拆开运动与可靠区判断

S 表示原片，R 表示 GLC 解码 RGB。第一字母是运动来源，第二字母仅是门控中的光度检查来源；双向与梯度几何检查始终随对应运动走。所有输出的参考 RGB 始终来自 GLC。

| 素材/区域 | SS 源运动/源光度 | SR 源运动/接收光度 | RS 接收运动/源光度 | RR 全接收 |
| --- | ---: | ---: | ---: | ---: |
'''+ '\n'.join(factorial)+f'''

SS/SR/RS 只作离线诊断，没有把原片运动或光度判断当免费可用信息。SS 的 RGB、alpha 和旧参考共享权重均逐位复现已存稠密参照，排除了换实现导致基线变化。

video1 珊瑚非锚帧的可复用覆盖率，SS 为 {coral['SS']['non_key_active']*100:.2f}%，SR 为 {coral['SR']['non_key_active']*100:.2f}%，RR 为 {coral['RR']['non_key_active']*100:.2f}%。换为解码画面的光度检查，会把许多正在发生生成形变的位置判为不一致并退出复用；那些位置恰恰可能需要纠正。接收端运动也跟随解码形变：在源对应可靠的非锚帧珊瑚上，两种 RAFT 的运动差异均值 {motion['mean']:.3f}px、P95 {motion['p95']:.3f}px。这是对源 RAFT 的差异，不能称作真实光流误差。

两因素有明显交互，不能把回归简单分摊成运动和可靠区的贡献百分比。本轮结论限定于当前 RAFT 与可靠区判据直接搬到接收端的实现；没有验证所有可能的接收端结构先验。

## 4. 保留 K 的接收端运动预测差值探针

额外冻结一个候选：把接收端稠密运动缩到 K 的同网格、量化到半像素，预测 K 必需节点；只编码差值，保留 K 可靠区。压缩器仍比较已有的空间、时间、速度预测及完整网格布局，没有通过改变画面或支持区换字节。还原后必要运动、支持区及 K RGB/alpha 全部逐位一致。

| 素材 | K 实际 B | K 规范重打包 B | 差值乐观代理 B | 对 K 增量 B | 运动载荷 B |
| --- | ---: | ---: | ---: | ---: | ---: |
'''+ '\n'.join(byte_rows)+'''

“代理 B”使用旧封装测量差值载荷，尚未加入新的预测器版本、参数及校验信令，因此不是可部署的新辅助包。即使这个偏乐观的计数更大，也只能否定本次完整运动差值配合已有编码器的方式，不能概括所有稀疏纠错方案。

## 5. 核验与耗时

三份接收端进程只允许读取基础码流、已独立解码的 RGB 及其接收回执、共享权重；原片、发送端元数据、源双向光流、源掩码、旧 K 辅助包共七类路径的读取负例全部阻断。Python open 审计用于本实验可复现性，不等同操作系统隔离。基础解码复用已核验缓存，本轮未重新执行 GLC 神经解码。

三段 RR 用原可靠性实现作 CPU 重放，RGB/alpha/共享权重逐位一致；三段 SS 逐位复现旧稠密结果；raw/K 历史分数精确重现；56 份冻结输入哈希不变。九段预览检查 150 帧、15 FPS、10 秒及首中末面板身份。零权重像素保持原 GLC。

| 素材 | RAFT 秒/10秒视频 | 门控与融合秒 | 峰值 PyTorch 分配 MiB |
| --- | ---: | ---: | ---: |
'''+ '\n'.join(timings)+'''

本次机器存在其他 GPU 活动；这些是当次批处理墙钟时间，不能据此作独占性能基准。当前远未完成实时验证。配对仅依赖当前及过去参考帧，但整段运行器、显存与输出缓存尚未改成流式。

## 6. 结论与下一步边界

辅助码率仍须降低。本轮将“直接零辅助”落实成了可检查结果，也补测了“不改变 K 的完整运动差值”。当前继续保留 K 的 19.7192/18.4288/15.5296 kbps 辅助，R0 为失败参照，不把降低字节与降低误差混成一项成功。

后续优先检验更低维的稳定结构运动、事件式参考刷新和少量纠错的表示方式：先证明保留有效稳定收益，再按完整文件与 K 比字节。当前生成画面中无法通过直接对应可靠性筛出的结构偏差，需要显式建模；单纯搬运同一光流和同一光度门控已不足。不要继续扩大混合边界以增加辅助码率，也不要重复本轮 2×2 验证。下一设计尚未运行，不能预记码率收益。

本轮没有新的同总字节 H.265、未见素材、长片或实时部署结论。原始基础码流和潜变量参考链未改，当前成果仍未提交或推送。

## 文件与复现

- 运行：`02_实验数据/优化_GLC后处理/receiver_structure_20261111_01`。
- 观看：`05_成果与观看/GLC_接收端零辅助验证_20261111`；01–04 左 K 右 R0；06–08 上排原片/原 GLC，下排 K/R0；09 为离线交叉诊断。
- 小证据：[evidence_20261111/receiver_structure](evidence_20261111/receiver_structure/README.md)。
- 新运行入口：`run_receiver_structure.py --workspace <工作区> --output <新目录>`；逐段接收、控制实验、评分。另运行 `probe_receiver_prediction.py --run <运行目录> --case video1`（video2/3 同理），然后 `release_receiver_structure.py --workspace <工作区> --run <运行目录>`。发布脚本的观看及证据目录要求不存在；完整重跑请在复制的工作区或修改发布目标，原有结果保留。
'''
    (stage/'接收端零辅助验证与字节探针_20261111.md').write_text(report,encoding='utf-8')
    (view/'README.md').write_text('''# 接收端零辅助验证

R0（代码名 RR）只使用 GLC 解码画面估计运动和可靠区，新增辅助 0 B。当前未保住 K 的稳定收益，保留 K。

01–04：左 K，右 R0；02 为 video1 珊瑚放大。05：左原 GLC，右 R0。
06–08：上排原片、原 GLC；下排 K、R0。09 为离线交叉诊断，不是四种可部署候选。
均为 150 帧、15 FPS、10 秒原速。评分使用无损 RGB，MP4 仅供观看。

“零辅助方案的传输文件”只含三份原有 base.glcsnd；接收端需预装 GLC 和 RAFT 模型及处理代码，本地运行生成的光流与掩码不传输。
''',encoding='utf-8')
    for path in run.rglob('*'):
        if path.is_file() and path.suffix in ('.json','.csv','.py'):
            dest=evidence/path.relative_to(run);dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dest)
    (evidence/'README.md').write_text('''# 接收端零辅助验证证据

execution_plan.json 为候选计算前冻结计划和 56 份输入哈希。
各 video*/RR/receipt.json 是无源接收端回执；SS/SR/RS 仅作离线诊断。
各 evaluation/summary.json 与 CSV 保存同分母无损评分、历史基线复现、覆盖率、运动差异及原速预览核验。
prediction_byte_probe 是保持 K 精确输出的乐观差值字节探针，尚非新传输格式。
source_final 保存精确代码与 SHA；release_verification.json 保存最终核验。
大体积 RGB、光流及视频保留在运行和观看目录，未复制入仓库。
''',encoding='utf-8')
    save_json(evidence/'manifest.json',{str(p.relative_to(evidence)):file_sha(p)
        for p in evidence.rglob('*') if p.is_file() and p != evidence/'manifest.json'})
    print('RELEASE COMPLETE',view,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='Resume this same owned release directory after an interrupted report build')
    a=p.parse_args();release(a.workspace.resolve(),a.run.resolve(),a.resume)
