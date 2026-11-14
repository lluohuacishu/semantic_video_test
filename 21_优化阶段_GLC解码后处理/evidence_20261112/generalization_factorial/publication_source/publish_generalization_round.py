"""Publish the verified fixed-parameter generalization round; no adoption."""
import argparse
import json
import os
from pathlib import Path
import shutil

from runtime_speed_common import STAGE, REPO, WORKSPACE, file_sha, load_json, save_json


REPORT_NAME = '泛化改进_引导质量与保守混合_20261112.md'
VARIANTS = ('Q6_T3', 'Q6_F8', 'Q6_C8', 'Q12_T3', 'Q12_F8', 'Q12_C8')


def rel(path, parent):
    return Path(os.path.relpath(path, parent)).as_posix()


def put(path, content):
    path.write_text(content.rstrip() + '\n', encoding='utf-8', newline='\n')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def publish(run):
    verified = load_json(run / 'final_verification.json')
    assert verified['complete'] and verified['verified_variants'] == 24
    summary = load_json(run / 'summary.json')
    extra = load_json(run / 'extra_scene/summary.json')
    plan = load_json(run / 'execution_plan.json')
    assert summary['selected'] is None and not any(v['passed'] for v in summary['numeric_screen'].values())
    report = STAGE / REPORT_NAME
    viewing = WORKSPACE / '05_成果与观看/GLC_泛化改进_20261112'
    evidence = STAGE / 'evidence_20261112/generalization_factorial'
    for path in (report, viewing, evidence, run / 'publication.json'):
        if path.exists():
            raise FileExistsError(path)
    adopted = STAGE / '当前采用配置.json'
    adopted_sha = file_sha(adopted)
    assert plan['input_sha256'][str(adopted)] == adopted_sha
    docs = [
        REPO / '交接文档.md', REPO / '当前项目状态与接续.md', REPO / 'README.md',
        REPO / '项目总结与未来方向.md', STAGE / 'README.md',
        STAGE / '当前方案总结与瓶颈路线_20261111.md',
        STAGE / '下一阶段_海洋素材与长视频泛化计划_20261111.md',
        REPO / '05_资料与索引/会话收束_20261110/文件索引.md',
    ]
    state_path = REPO / '05_资料与索引/会话收束_20261110/接续状态.json'
    continuation = REPO / '新会话接续.txt'
    attributes = REPO / '.gitattributes'
    evidence.mkdir(parents=True)
    viewing.mkdir()
    previous = {}
    for path in [*docs, state_path, continuation, attributes]:
        dst = evidence / 'prior_entries' / path.relative_to(REPO)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        previous[str(path.relative_to(REPO))] = file_sha(path)
    if (run / 'progress.json').exists():
        shutil.copy2(run / 'progress.json', evidence / 'prior_entries/run_progress.json')

    videos = []
    for path in sorted((run / 'previews').glob('*.mp4')):
        videos.append(path)
    extra_eval = run / 'extra_scene/archive82_first10s/evaluation'
    videos += sorted(extra_eval.glob('*.mp4'))
    assert len(videos) == 15
    media = videos + sorted((run / 'previews').glob('*_contact.png')) + [extra_eval / 'archive82_contact.png']
    links = []
    for src in media:
        dst = viewing / src.name
        os.link(src, dst)
        assert file_sha(src) == file_sha(dst) and os.path.samefile(src, dst)
        links.append(dict(source=str(src), destination=str(dst), sha256=file_sha(dst), hardlink=True))

    combined = dict(summary['cases'])
    combined['archive82_first10s'] = extra
    full_rows = []
    for variant in ('raw', 'K', *VARIANTS):
        values = [f"{c['scores']['full'][variant]['coarse_mae']:.6f}"
                  if variant in c['scores']['full'] else '未测' for c in combined.values()]
        full_rows.append([{'raw': '原 GLC', 'K': '旧 K（仅历史参照）'}.get(variant, variant), *values])
    change_rows = []
    for case, c in combined.items():
        a, b = c['scores']['full']['Q6_T3'], c['scores']['full']['Q12_T3']
        reef = c['scores'].get('reef')
        change_rows.append([case, f"{100*(b['coarse_mae']/a['coarse_mae']-1):+.2f}%",
                            f"{100*(reef['Q12_T3']['coarse_mae']/reef['Q6_T3']['coarse_mae']-1):+.2f}%" if reef else '未预登记',
                            f"{b['psnr_db']-a['psnr_db']:+.3f}",
                            f"{c['rates']['Q6_T3']:.4f} → {c['rates']['Q12_T3']:.4f}"])
    rate_rows = []
    for case, c in combined.items():
        parent = run / 'extra_scene' if case == 'archive82_first10s' else run
        spec = load_json(run / 'extra_scene/receiver_input_registry.json')[case] if case == 'archive82_first10s' else plan['cases'][case]
        base_bytes = Path(spec['base']).stat().st_size
        s6 = load_json(parent / case / 'Q6/send.json')
        s12 = load_json(parent / case / 'Q12/send.json')
        rate_rows.append([case, base_bytes, s6['bytes'], s12['bytes'],
                          f"{s6['total_kbps']:.4f}", f"{s12['total_kbps']:.4f}"])
    screening_rows = []
    for variant, record in summary['numeric_screen'].items():
        failures = []
        for case, scores in record['cases'].items():
            if not scores['pass']:
                failures.append(case)
        screening_rows.append([variant, '未通过', '、'.join(failures)])
    extra_regions = [[region, f"{s['Q6_T3']['coarse_mae']:.6f}", f"{s['Q12_T3']['coarse_mae']:.6f}",
                      f"{100*(s['Q12_T3']['coarse_mae']/s['Q6_T3']['coarse_mae']-1):+.2f}%"]
                     for region, s in extra['scores'].items()]
    v2 = summary['cases']['video2']
    boundary_reduction = 100 * (1 - summary['numeric_screen']['Q12_T3']['cases']['video2']['boundary_ratio'])
    report_text = f"""# 泛化改进：引导质量与保守混合实验

日期：2026-11-12。已完成 **4 个历史片段 × 6 档 = 24 组**，每片 150 帧、15 FPS、512×288、10 秒。三片主矩阵为 18 组，另一个预登记历史场景为 6 组。

**结论：提高引导质量明显改善 video2 的时序指标，但本轮没有统一候选通过全部预设条件。Q12_T3 仅作为诊断和观看候选，未替换当前采用的 S8I + CURRENT_I12。**

[原速观看目录]({rel(viewing / 'README.md', STAGE)}) · [video2 当前/候选双列]({rel(viewing / 'video2_current_vs_Q12_T3.mp4', STAGE)}) · [最终核验]({rel(run / 'final_verification.json', STAGE)}) · [小证据清单](evidence_20261112/generalization_factorial/manifest.json)

## 用户反馈与研究边界

2026-11-11 用户反馈原文保留于[上一轮初筛报告](当前方案跨素材初筛_video2_video3_20261111.md)：video2 整体退化，观感较差、边界感严重、跳变多；video3 效果不错。鱼身斑点密集导致困难是用户判断，本轮不将其写成已验证的唯一原因。

2026-11-12 用户要求“好的，那开始进一步实验吧，整一整泛用性”，已恢复实验。前三片仍是已知开发素材。第四段为旧 82.6 秒海洋原片的**前 10 秒**，在主矩阵评分前登记，含礁石、潜水员、气泡和小鱼；它也属于历史素材，不是独立留出集。

本轮没有验证当前后处理的长视频、网络实时或任意时长接口；没有做当前新预算下的同码率 H.265 比较，也没有进行完整接收链路提速计时。不能把指标改善解释为编码效率或实时性已经改善。

## 固定实验协议

- 引导因素 Q6 / Q12：内部目标 6 / 12 kbps，256×144、15 FPS、YUV420、x265 slower two-pass；实际完整包字节单独计费。Q6 主矩阵复用当前已发送包；Q12 新编码。第四场景两档都实际编码。
- 混合因素 T3：当前 2px 羽化；F8：相同有效支持内改成 8px 羽化；C8：F8 再乘局部引导流形变置信度。梯度经 sigma=2 平滑，≤0.10 为 1，≥0.50 为 0，中间 cubic smoothstep。F8/C8 权重不超过原 T3。
- 固定六锚点 0/15/45/75/105/135、RAFT Things FP32 12 次、无 TF32、6 帧旧参考渐退与两次取整。Q12 指引导目标码率，**不是改成 24 次 RAFT**。
- 所有参考像素仍来自原始 GLC RGB，不回灌基础参考链；没有逐片规则、逐片阈值或源掩码。接收端只读基础包、独立基础解码及其回执、引导包和预置权重。
- 第四场景仅在隔离 worker 中注入显式输入路径 registry，冻结控制器文件哈希不变，六档算法参数完全相同。

主计划与输入/代码 SHA 在 [execution_plan.json]({rel(run / 'execution_plan.json', STAGE)})；第四场景登记在 [registration.json]({rel(run / 'registered_extra_source/registration.json', STAGE)})。没有按结果修改筛选阈值。

## 主指标和范围

表中“时序 MAE”是既有评价器的 **coarse_mae：源运动对齐、公共有效支持上的低频时序变化误差**，越小越好。它不等于人眼跳变次数，也不等于所有像素的普通帧差。细频 MAE、完整频带 MAE、NCC、PSNR、四象限、逐帧数据均保留在 JSON/CSV。

前三片另使用历史固定珊瑚区域。边界代理指标取固定 Q6_T3 支持边缘上 candidate−source RGB 残差的空间梯度；只作诊断，不能证明用户感受到的边界已消失。动态指标是在源高变化像素上的 RGB 帧差误差，并附变化幅度比，避免把抹平运动误判为改善。

### 全帧时序 MAE

{table(['方案', 'video1', 'video2', 'video3', '历史原片前10秒'], full_rows)}

旧 K 是历史参照，包预算不同，不属于本轮六档，也不是公平同码率胜负判断。第四场景没有补跑旧 K。

### Q12_T3 相对当前 Q6_T3

{table(['场景', '全帧时序误差变化', '历史珊瑚误差变化', 'PSNR变化/dB', '实际总码率/kbps'], change_rows)}

video2 的全帧时序误差由 2.261521 降至 1.630273（降低 27.91%），珊瑚区域降低 30.67%；全帧逐帧误差 P95 从 {v2['full_frame_p95']['Q6_T3']:.4f} 降至 {v2['full_frame_p95']['Q12_T3']:.4f}。边界代理只降低 **{boundary_reduction:.2f}%**，未达到预设的 5%。PSNR 虽增加 0.185 dB，仍低于原 GLC 的 21.085 dB，因此不能宣称保真与边界问题一起解决。

video3 全帧时序误差上升 2.48%，历史珊瑚区域上升 **10.12%**。提高引导质量对所有片段均更好这一假设不成立。

Q6_F8 / Q6_C8 不增加传输字节，对 video2 全帧仅改善约 4.0% / 4.2%，边界代理约改善 5.5% / 5.4%；同时 video1 珊瑚误差恶化约 13.0% / 12.8%，video3 珊瑚恶化约 8.9%。C8 相比 F8 未显示有意义的额外收益。Q12 配合 F8/C8 也未能通过跨片约束。

### 额外历史场景

{table(['区域', '当前Q6_T3', 'Q12_T3', '误差变化'], extra_regions)}

该场景 Q12_T3 全帧降低 5.14%，PSNR 增加 0.083 dB；左上象限小幅回退 0.69%。源高变化像素的变化幅度比从 0.6369 增至 0.6478，仍低于理想值 1；不能据此断言动态目标已经恢复。本场景没有事后补选“珊瑚框”或套用主实验边界阈值，也没有用于推翻主矩阵未通过的结论。

## 实际传输成本

{table(['场景', '基础包/B', 'Q6引导包/B', 'Q12引导包/B', 'Q6合计/kbps', 'Q12合计/kbps'], rate_rows)}

计算为完整必要包字节 × 8 ÷ 10 秒 ÷ 1000。模型为预置资产，flow/alpha/NPY 和观看 MP4 是本地资产，不计为传输包。各路仍低于各自 40 kbps 上限；本轮总码率也恰好低于 40，但约束本身是每路各 40。Q12 的改善伴随真实辅助字节增加，并非免费收益。

## 预设筛选与决定

筛选仅用于候选排序，不代替主观验收：video2 全帧误差至少降低 10%，边界代理至少降低 5%，动态误差不恶化超过 1%，PSNR 不下降超过 0.05 dB；其他两片全帧和历史珊瑚误差不恶化超过 5%，PSNR 不下降超过 0.1 dB，动态误差不恶化超过 3%。

{table(['候选', '整体筛选', '未通过场景'], screening_rows)}

Q12_T3 在 video2 边界代理和 video3 珊瑚回退两项上失败。机器结果 selected=null、best_video2_diagnostic=Q12_T3。当前采用配置保持原字节；Q12_T3 和零新增字节的 Q6_C8 均保留原速观看，主观验收 pending。

## 核验与观看

- 24 组 RGB/alpha/shared 的 SHA、形状、类型、有限值、权重范围、零权重像素不变及支持集约束通过。
- 8 份引导包重新解码一致，包绑定、实际字节、光流哈希和接收读取白名单/拒绝测试通过。
- 三片 Q6_T3 与既有当前结果逐位一致；第四场景通过原无缓存 I12 单独回放，RGB/alpha/shared 也逐位一致。
- 输入、预置权重、冻结代码及当前采用配置 SHA 不变。第四场景源 chunk 哈希和独立基础解码参考一致性通过。
- 15 个 MP4 均为 150 帧、15 FPS、10 秒，首/中/末帧面板回读与哈希核验通过；观看目录为原文件硬链接。MP4 仅用于观看，指标来自无损数组。
- 静态抽帧图只用于检查布局与明显图像问题，不代表已完成原速主观验收；没有把肉眼结论写为已通过。

双列左为当前 Q6_T3，右为候选；四宫格左上原片、右上原 GLC、左下当前、右下候选。video2 六宫格按行优先为 Q6_T3、Q6_F8、Q6_C8、Q12_T3、Q12_F8、Q12_C8，完整保留失败候选。

## 后续基调

本轮证据支持把“引导质量是否足够”和“何时可靠地复用/退出历史纹理”分开研究。**不建议直接把所有片段统一升到 Q12 或统一加宽羽化。** 下一轮可用 Q12_T3 作高质量引导诊断参照，优先研究接收端可计算的对应可靠性、遮挡与退出规则，目标是改善 video2 同时保护 video1/3；这是下一步假设，尚未验证。

继续保留固定参数跨片协议与负面样本，先完成本轮原速观感记录；随后再登记真正独立海洋素材。长视频另需解决固定 150 帧封装、有界缓存、跨块参考与码率开销，再运行完整长片。当前实验不外推这些能力。

## 文件与复现边界

- 权威结果目录：{run.as_posix()}
- 主流程：run_generalization_factorial.py；核验/预览：finish_generalization_factorial.py。
- 额外场景：run_generalization_extra.py；最终核验：verify_generalization_round.py；本次文档和索引发布：publish_generalization_round.py。
- 冻结执行源码、计划、摘要、接收回执与逐帧 CSV 复制到 evidence_20261112/generalization_factorial，manifest 记录字节 SHA。整理前入口保存在 prior_entries。
- 任何重跑必须使用新运行目录，禁止覆盖本目录、历史冻结输入或旧采用结果。发布程序遇到已有目标即拒绝执行。
- 本轮代码、报告和入口修改留在工作树，未提交或推送；原有无关修改保留。部署包和网页默认未改。
"""
    put(report, report_text)

    viewing_text = f"""# GLC 泛化改进原速对照（2026-11-12）

已完成四段历史片段 × 六档 = 24 组。**尚无统一候选通过全部预设条件，当前采用配置未改。**

建议先看 video2，再看 video3 回退，最后对照 video1 和额外场景。双列均为**左：当前 Q6_T3；右：标题所示候选**。四宫格为左上源、右上原 GLC、左下当前、右下候选。所有视频 10 秒、15 FPS。

| 观看目的 | 双列 | 四宫格 |
| --- | --- | --- |
| video2：Q12 时序指标明显改善，边界仍待看 | [当前 vs Q12_T3](video2_current_vs_Q12_T3.mp4) | [源/原GLC/当前/候选](video2_source_raw_current_Q12_T3.mp4) |
| video3：检查珊瑚回退 | [当前 vs Q12_T3](video3_current_vs_Q12_T3.mp4) | [四宫格](video3_source_raw_current_Q12_T3.mp4) |
| video1：检查原成功素材 | [当前 vs Q12_T3](video1_current_vs_Q12_T3.mp4) | [四宫格](video1_source_raw_current_Q12_T3.mp4) |
| 历史82.6秒原片的前10秒：礁石/潜水员/气泡 | [当前 vs Q12_T3](archive82_current_vs_candidate.mp4) | [四宫格](archive82_source_raw_current_candidate.mp4) |
| video2：不增加字节的 C8 对照 | [当前 vs Q6_C8](video2_current_vs_Q6_C8.mp4) | [四宫格](video2_source_raw_current_Q6_C8.mp4) |
| video1：C8 的稳定性代价 | [当前 vs Q6_C8](video1_current_vs_Q6_C8.mp4) | [四宫格](video1_source_raw_current_Q6_C8.mp4) |
| video3：C8 的稳定性代价 | [当前 vs Q6_C8](video3_current_vs_Q6_C8.mp4) | [四宫格](video3_source_raw_current_Q6_C8.mp4) |

[video2 完整六宫格](video2_full_2x3_matrix.mp4)：按行优先为 Q6_T3、Q6_F8、Q6_C8、Q12_T3、Q12_F8、Q12_C8。

Q12_T3 相对当前：video2 全帧时序误差 −27.91%，总码率 30.9904→39.1096 kbps，但边界代理仅改善 1.09%；video3 珊瑚误差 +10.12%。额外场景全帧误差 −5.14%。这些是固定评价指标，原速主观反馈仍待记录。

请关注斑点鱼体与背景交界、纹理形状、换锚附近跳变，以及 video3 原本稳定的珊瑚。四段均为历史素材的短段，不代表独立留出或长片已验证。Q12 指引导目标码率，RAFT 仍为 12 次。

[完整实验报告]({rel(report, viewing)}) · [结果与核验目录]({rel(run, viewing)}) · [观看文件SHA清单](manifest.json)

所有 MP4/PNG 为已有结果的硬链接，没有重新编码。不要原地编辑这些文件，以免同时改变权威结果；需剪辑时先另存副本。指标来自无损数组，观看视频体积不表示传输码率。未自动采用、未提交/推送。
"""
    put(viewing / 'README.md', viewing_text)
    save_json(viewing / 'manifest.json', dict(date='2026-11-12', files=links, previews=15, source_run=str(run)))

    current_text = """已完成固定 2×3 的主矩阵和额外历史场景，共 24 组；没有候选通过全部预设条件。Q12_T3 使 video2 全帧时序误差降低 27.91%，实际总码率 30.9904→39.1096 kbps；边界代理仅改善 1.09%，video3 珊瑚误差回退 10.12%。额外历史片段前 10 秒改善 5.14%。保留 Q12_T3 供观看，尚未采用。"""
    for path in docs:
        old = path.read_text(encoding='utf-8-sig')
        title, body = old.split('\n', 1)
        latest = f"""**最新进展（2026-11-12）：泛化改进本轮已完成。** {current_text}

用户已恢复实验，11 月 11 日“仅记录、暂停”的指令是历史状态。既有反馈继续保留：video2 整体退化、边界严重且跳变多；video3 效果不错；密集斑点的成因解释仍待验证。新候选主观验收 pending，当前采用配置及部署默认未改。

[本轮报告]({rel(report, path.parent)}) · [原速观看]({rel(viewing / 'README.md', path.parent)}) · [24组最终核验]({rel(run / 'final_verification.json', path.parent)}) · [证据清单]({rel(evidence / 'manifest.json', path.parent)})

后续基调：先对照本轮原速画面，重点研究可跨素材工作的对应可靠性与纹理复用退出；不把统一升码率或加宽羽化当作已成功方案。独立留出、当前后处理长视频和实时仍未验证。所有本轮修改尚未提交/推送。
"""
        if path.name == '当前项目状态与接续.md':
            latest += """
| 事项 | 当前状态 |
| --- | --- |
| 用户授权 | 已恢复进一步实验；本轮完成 |
| 结果 | 4×6=24组；无统一候选通过；Q12_T3仅诊断候选 |
| 核验 | 24组、8份引导、4个逐位基线、15个预览通过 |
| 采用 | S8I+CURRENT_I12保持原配置，保留video2负面反馈 |
| 待验证 | 候选原速观感、独立留出、长视频、有界缓存、实时 |
| 版本 | codex/glc-temporal-stability；本轮未提交/推送 |
"""
        put(path, title + '\n\n' + latest + '\n## 历史入口正文（截至2026-11-11）\n\n以下保留旧轮原文，其中“当前/本轮/暂停”属于当时状态；当前状态以页首 2026-11-12 更新为准。\n\n' + body.lstrip())

    old_continuation = continuation.read_text(encoding='utf-8-sig')
    put(continuation, f"""GLC 接续，2026-11-12：泛化改进固定实验已完成，非运行中。
先读：{report}
原速观看：{viewing / 'README.md'}
权威结果：{run}
本轮24组（video1/2/3及历史82.6秒素材前10秒，各六档）。引导Q6/Q12 × T3/F8/C8；RAFT均为12次、FP32、六锚点，未逐片调参。
{current_text}
核验24组/8份引导/4个逐位基线/15个视频通过；冻结输入、推理代码与采用配置不变。
新候选主观反馈pending；保留11月11日video2负面、video3正面反馈。用户已恢复实验，不再处于旧暂停状态。
下轮方向：对应可靠性/遮挡/复用退出，固定规则跨片；独立留出与长片另行验证，不能拿前10秒代替完整长视频。
未提交/推送，未修改部署默认。不要覆盖冻结结果或重跑指向已有目录的发布程序。

----- 2026-11-11 接续原文（历史状态，暂停已被新指示解除） -----
{old_continuation}""")

    state = load_json(state_path)
    state['date'] = '2026-11-12'
    state['status'] = 'generalization_factorial_complete_no_universal_candidate'
    state['current_session_work'] = 'Completed 24 fixed-condition outputs on four historical short clips; verification and publication complete; visual review pending.'
    state['new_experiment_report'] = str(report.relative_to(REPO))
    state['latest_analysis_report'] = str(report.relative_to(REPO))
    state['new_experiments_uncommitted'] = True
    state['experiment_execution_gate'] = dict(status='authorized_round_complete', date='2026-11-12',
        authorization='好的，那开始进一步实验吧，整一整泛用性', previous_pause_superseded=True)
    state['active_experiment_plan'] = None
    state['completed_generalization_plan'] = str(run / 'execution_plan.json')
    state['previous_experiment_plan_before_generalization'] = state.get('latest_experiment_plan')
    state['latest_experiment_plan'] = dict(status='completed', path=str(run / 'execution_plan.json'), outputs=24, extra_scene_plan=str(run / 'extra_scene/execution_plan.json'))
    state['next_priority'] = [
        'Review the new original-speed comparisons and retain negative results.',
        'Study receiver-side correspondence reliability, occlusion and texture-reuse exit with fixed rules across clips; not yet tested.',
        'Register genuinely unseen marine clips after freezing the next protocol.',
        'Implement and verify bounded caches, chunk boundaries and guide accounting before current-profile full-length validation.']
    state['previous_next_experiment_plan_before_generalization'] = state.get('next_experiment_plan')
    state['next_experiment_plan'] = dict(status='proposed_not_executed', priority=state['next_priority'],
        no_per_clip_tuning=True, current_profile_unchanged=True, long_video_tested=False)
    state['workspace_paths']['generalization_factorial_run'] = str(run.relative_to(WORKSPACE))
    state['workspace_paths']['generalization_factorial_viewing'] = str((viewing / 'README.md').relative_to(WORKSPACE))
    state['latest_generalization_experiment'] = dict(complete=True, cases=list(combined), variants=list(VARIANTS),
        trial_count=24, selected=None, diagnostic_candidate='Q12_T3', numeric_screen_passed=False,
        current_profile_unchanged=True, subjective_review='pending', independently_heldout=False,
        long_video_tested=False, realtime_tested=False, source_run=str(run), report=str(report),
        viewing=str(viewing), verification=str(run / 'final_verification.json'),
        video2_full_mae_change_percent=-27.91257481465503, video2_boundary_proxy_change_percent=-boundary_reduction,
        video3_reef_mae_change_percent=10.123879458520912,
        extra_full_mae_change_percent=100*(extra['scores']['full']['Q12_T3']['coarse_mae']/extra['scores']['full']['Q6_T3']['coarse_mae']-1))
    state['publication'] = dict(base_commit='ae46c91c6820ffc49012714c44659ead0e1fc162',
        current_changes_committed=False, pushed=False, scope='2026-11-12 generalization round plus preserved prior uncommitted work')
    save_json(state_path, state)
    rule = '21_优化阶段_GLC解码后处理/evidence_20261112/** -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol'
    attr_text = attributes.read_text(encoding='utf-8-sig')
    if rule not in attr_text:
        put(attributes, attr_text.rstrip() + '\n# Preserve 2026-11-12 frozen evidence byte hashes.\n' + rule)
    assert file_sha(adopted) == adopted_sha
    save_json(run / 'progress.json', dict(stage='complete', trial_count=24, selected=None, viewing_candidate='Q12_T3', subjective_review='pending'))
    save_json(run / 'publication.json', dict(complete=True, date='2026-11-12', report=str(report), viewing=str(viewing),
        evidence=str(evidence), media=links, prior_entry_sha256=previous, adopted_configuration_sha256=adopted_sha,
        adoption_changed=False, committed=False, pushed=False, publication_script_sha256=file_sha(__file__)))

    for path in sorted(run.rglob('*')):
        if path.is_file() and (path.suffix in ('.json', '.csv') or 'source_snapshot' in path.parts):
            dst = evidence / 'run' / path.relative_to(run)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
    for name in ('run_generalization_factorial.py', 'finish_generalization_factorial.py',
                 'run_generalization_extra.py', 'verify_generalization_round.py', 'publish_generalization_round.py'):
        dst = evidence / 'publication_source' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(STAGE / name, dst)
    files = [dict(path=str(p.relative_to(evidence)), bytes=p.stat().st_size, sha256=file_sha(p))
             for p in sorted(evidence.rglob('*')) if p.is_file()]
    save_json(evidence / 'manifest.json', dict(date='2026-11-12', files=files, source_run=str(run),
        scope='Exact-byte small records, source snapshots and prior entries; large media/arrays remain local'))
    for record in files:
        assert file_sha(evidence / record['path']) == record['sha256']
    publication_checks = dict(complete=True, evidence_files=len(files), hardlinked_media=len(links), videos=len(videos),
        evidence_manifest_sha256=file_sha(evidence / 'manifest.json'), report_sha256=file_sha(report),
        adopted_profile_unchanged=file_sha(adopted)==adopted_sha,
        entry_sha256={str(p.relative_to(REPO)):file_sha(p) for p in [*docs,state_path,continuation,attributes]})
    save_json(run / 'publication_verification.json', publication_checks)
    print(json.dumps(dict(report=str(report), viewing=str(viewing), evidence_files=len(files),
                          hardlinked_media=len(links), videos=len(videos), selected=None), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    publish(args.run.resolve())
