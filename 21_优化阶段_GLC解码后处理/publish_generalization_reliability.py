"""Publish the mild-reliability round while retaining the user's clarification."""
import argparse
import json
import os
from pathlib import Path
import shutil

from runtime_speed_common import STAGE, REPO, WORKSPACE, file_sha, load_json, save_json
from run_generalization_reliability import PROFILES


REPORT = '泛化改进第二轮_温和可靠性与中档引导_20261112.md'


def relative(path, parent):
    return Path(os.path.relpath(path, parent)).as_posix()


def put(path, value):
    path.write_text(value.rstrip() + '\n', encoding='utf-8', newline='\n')


def mdtable(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str,row)) + ' |' for row in rows])


def publish(run):
    plan, summary = load_json(run/'execution_plan.json'), load_json(run/'summary.json')
    verification, previews = load_json(run/'verification.json'), load_json(run/'preview_verification.json')
    assert verification['complete'] and previews['complete']
    selected = summary['selected']
    chosen = previews['selected_for_viewing']
    other = previews['additional_candidate']
    cases = summary['cases']
    report = STAGE/REPORT
    viewing = WORKSPACE/'05_成果与观看/GLC_泛化第二轮_20261112'
    evidence = STAGE/'evidence_20261112/generalization_reliability'
    for path in (report, viewing, evidence, run/'publication.json'):
        if path.exists():
            raise FileExistsError(path)
    config = STAGE/'当前采用配置.json'
    config_sha = file_sha(config)
    assert config_sha == plan['input_sha256'][str(config)]
    viewing.mkdir()
    evidence.mkdir(parents=True)
    media = []
    for path in sorted((run/'previews').iterdir()):
        if path.suffix not in ('.mp4','.png'):
            continue
        dest = viewing/path.name
        os.link(path,dest)
        assert os.path.samefile(path,dest) and file_sha(path)==file_sha(dest)
        if path.suffix=='.mp4':
            assert file_sha(dest)==previews['videos'][path.name]['sha256']
        media.append(dict(source=str(path), destination=str(dest),sha256=file_sha(dest),hardlink=True))
    assert sum(Path(x['destination']).suffix=='.mp4' for x in media)==17

    rows = []
    for profile in ('raw', *PROFILES, 'Q12_T3'):
        rows.append([{'raw':'原GLC','Q6_T3':'当前Q6_T3','Q12_T3':'上一轮Q12_T3（参考）'}.get(profile,profile),
                     *[f"{c['scores']['full'][profile]['coarse_mae']:.6f}" for c in cases.values()]])
    delta_rows = []
    for name, record in summary['numeric_screen'].items():
        values = record['cases']
        delta_rows.append([name,*[f"{100*(d['full_mae_ratio']-1):+.2f}%" for d in values.values()],
                           '通过' if record['passed'] else '未通过'])
    chosen_rows = []
    compact = {}
    for case,c in cases.items():
        a,b=c['scores']['full']['Q6_T3'],c['scores']['full'][chosen]
        reef=c['scores'].get('reef')
        diff=100*(b['coarse_mae']/a['coarse_mae']-1)
        reef_diff=100*(reef[chosen]['coarse_mae']/reef['Q6_T3']['coarse_mae']-1) if reef else None
        border=100*(c['boundary_residual_gradient'][chosen]/c['boundary_residual_gradient']['Q6_T3']-1)
        dynamic=100*(c['dynamic'][chosen]['RGB_delta_error']/c['dynamic']['Q6_T3']['RGB_delta_error']-1)
        chosen_rows.append([case,f'{diff:+.2f}%',f'{reef_diff:+.2f}%' if reef_diff is not None else '未设框',
                            f"{b['psnr_db']-a['psnr_db']:+.3f}",f'{border:+.2f}%',f'{dynamic:+.2f}%',
                            f"{c['rates']['Q6_T3']:.4f}→{c['rates'][chosen]:.4f}"])
        compact[case]=dict(full_mae_change_percent=diff,reef_mae_change_percent=reef_diff,
                           psnr_change_db=b['psnr_db']-a['psnr_db'],boundary_proxy_change_percent=border,
                           dynamic_error_change_percent=dynamic,current_kbps=c['rates']['Q6_T3'],candidate_kbps=c['rates'][chosen])
    rate_rows=[]
    for case,spec in plan['cases'].items():
        s6=load_json(run/case/'Q6/send.json')
        s9=load_json(run/case/'Q9/send.json')
        rate_rows.append([case,Path(spec['base']).stat().st_size,s6['bytes'],s9['bytes'],
                          f"{s6['total_kbps']:.4f}",f"{s9['total_kbps']:.4f}",f"{cases[case]['rates']['Q12_T3']:.4f}"])
    screening_rows=[]
    for name,record in summary['numeric_screen'].items():
        failures=[case for case,d in record['cases'].items() if not d['pass']]
        screening_rows.append([name,'通过' if record['passed'] else '未通过','、'.join(failures) or '无'])
    decision = (f'**{selected} 通过本轮较温和的预设数值筛选，作为统一观看候选。**' if selected else
                f'**没有候选通过全部预设数值筛选；{chosen} 是更符合适度改善目标的平衡观看候选，尚未验收。**')
    feedback=load_json(run/'user_feedback_20261112.json')
    v2=compact['video2']
    latest=f"固定四段素材共新生成24组，另引用上一轮4组Q12_T3参照。{chosen}相对当前video2全帧时序误差变化{v2['full_mae_change_percent']:+.2f}%，实际总码率{v2['current_kbps']:.4f}→{v2['candidate_kbps']:.4f}kbps。"
    if selected:
        latest+=f'{selected}通过本轮数值筛选，主观验收尚待记录。'
    else:
        latest+='没有统一候选通过全部数值筛选，主观验收尚待记录。'

    text=f"""# 泛化改进第二轮：温和可靠性与中档引导

日期：2026-11-12。四段历史素材、同样的六套固定配置，**24 组新输出 + 4 组上一轮 Q12_T3 参照**。{decision} 当前采用配置保持原字节，未自动采用。

[原速观看目录]({relative(viewing/'README.md',STAGE)}) · [video2 双列]({relative(viewing/f'video2_current_vs_{chosen}.mp4',STAGE)}) · [核验]({relative(run/'verification.json',STAGE)}) · [小证据](evidence_20261112/generalization_reliability/manifest.json)

## 用户最新口径

> {feedback['verbatim_user_feedback']}

按这次澄清，video2 的目标是**适度改善**，当前说明不把它描述为“特别难看”或必须彻底修复。历史原话保留，最新理解以本段为准。这不是用户已采用上一轮 Q12_T3，也不代表完成本轮原速验收。

上一轮 video2 提高引导质量后指标改善，但 video3 珊瑚有回退；宽羽化则削弱稳定纹理。本轮降低改动强度，区分引导码率与可靠性衰减，不逐片调参。

## 冻结设计

| 因素 | 内容 |
| --- | --- |
| Q6 | 复用当前真实 6kbps 内部目标引导包和已核验的接收本地光流 |
| Q9 | 新编码固定 9kbps 内部目标，256×144/15FPS/YUV420，x265 slower two-pass；没有按结果搜索 |
| T3 | 当前门控，2px羽化保持 |
| R25 | 在原有循环一致性、引导光度差、流梯度接近拒绝阈值时平滑降权；最多减25% |
| D25 | R25 加独立基础解码的低频RGB差异风险，用两种风险的最大值；总衰减仍最多25% |

风险函数 z²(3−2z)，z 为截断至[0,1]的归一化残差。固定起止区间：cycle 0.75–1.5px、guide sigma3差 8–16、guide sigma1差 12–24、flow梯度0.375–0.75；D25额外的原GLC sigma3 RGB最大通道差为12–32。alpha'=alpha×(1−0.25×risk)。

**有效支持保持一致**，不扩大处理区域、不改变2px羽化；完全可靠的像素保留原权重。不可靠区仍沿用原硬拒绝规则，没有在遮挡外强行延续旧纹理。该改法是门限附近的连续降权，**没有实现跨帧门控记忆，也不保证消除跳变**。

六锚点0/15/45/75/105/135、RAFT FP32/12次/无TF32、六帧过渡、原始GLC参考和两次取整均保持。接收端不读取源RGB、源掩码、源光流或历史优化输出。D25只比较接收端已经得到的原始GLC图像；这可能抑制大幅错误改写，也可能抑制有益稳定，因此必须跨片核查。

四段均为既有历史素材前10秒、150帧、512×288。额外段含礁石、潜水员、气泡和小鱼，来自82.6秒原片，仍不算独立留出或完整长视频验证。

## 统一结果

时序指标沿用源运动对齐、公共有效支持上的低频变化误差 coarse_mae。越小越好，但不等同肉眼跳变次数。原片只在发送和独立评价读取。每片同帧PSNR、四象限、历史珊瑚区、动态误差、逐帧P95和边界代理均保留。

{mdtable(['方案','video1','video2','video3','额外历史10秒'],rows)}

Q12_T3只引用上一轮已完成结果，本轮没有重新运行该档，也不计入24组新输出。

### 各新候选相对当前的全帧误差变化

{mdtable(['候选','video1','video2','video3','额外历史10秒','数值筛选'],delta_rows)}

### 主要观看候选 {chosen}

{mdtable(['场景','全帧误差','珊瑚误差','PSNR/dB','边界代理','动态误差','总码率/kbps'],chosen_rows)}

边界代理仍是固定当前T3支持边缘上的candidate−source残差空间梯度，只用于观察。动态误差是源高变化像素上的RGB帧差误差；变化幅度比另见JSON，不能把抹平运动当作稳定成功。所有百分比相对于当前Q6_T3。

video2 的逐帧P95：{cases['video2']['full_frame_p95']['Q6_T3']:.4f} → {cases['video2']['full_frame_p95'][chosen]:.4f}。上述数字不是主观“更好看”的替代结论，请结合原速对照。

### 如何看待这轮的小幅进展

Q6_R25不增加传输字节，四片全帧误差分别降低约0.89%、1.10%、0.10%、1.06%；video3历史珊瑚误差仅增加0.14%，全帧PSNR和动态误差均小幅改善。这说明温和降权在这四段上较均衡，但收益很小、可能肉眼不明显，不能写成泛化已解决。

Q9_R25是另一观看参照：video2误差降低25.38%，总码率30.9904→37.9488kbps，video3珊瑚误差增加5.37%。相较上一轮Q12_T3约10.12%的局部回退有所减轻，但仍超过本轮5%限制。

D25引入原GLC差异约束后，Q6下video2改善4.26%，却让video3珊瑚回退12.03%；Q9_D25该局部回退为16.70%。本轮不优先保留这条更强约束。更低的像素误差与更稳定的纹理仍存在取舍。

观看排序在全部评分后记录于review_selection.json：优先Q6_R25、次看Q9_R25，用同一候选对照所有片段。自动数值筛选selected仍为null，单看video2误差最低的Q9_D25仍保留在完整矩阵。没有事后降低2%门槛，也没有逐片选不同配置。

## 实际字节成本

{mdtable(['素材','基础包/B','Q6引导/B','Q9引导/B','Q6总kbps','Q9总kbps','旧Q12总kbps'],rate_rows)}

完整必要包字节×8÷10秒÷1000，不把引导目标值当实际码率。每路各40kbps上限仍满足；本地数组、光流缓存、模型预置资产和观看MP4不计成传输包。Q6的R25/D25不增加字节，Q9有真实字节成本。没有新增同码率H.265对照。

## 本轮筛选含义

这次按“适度改善”目标，在运行前将排序条件设为：video2全帧时序误差至少下降2%；其他片全帧和预登记珊瑚区域不恶化超过5%；各片PSNR不下降超过0.1dB、动态误差不恶化超过3%。**不再强制边界代理改善5%。** 这些是实验排序阈值，不是把用户“改善一些”机械等同2%，也不是主观验收标准。

{mdtable(['候选','结果','未通过场景'],screening_rows)}

通过者先优先保留原辅助字节，再比较video2时序误差。{decision} 上一轮筛选规则和结果未改，不能因本轮目标调整而改写上一轮“未通过”的事实。

## 验证与代价

- 24组输出的SHA、形状、dtype、有限值和范围通过；四段Q6_T3均与上一轮当前基线逐位一致。
- R25/D25的alpha及shared在对应T3的75%–100%之间，非零支持逐位相同，零alpha像素仍为原GLC。
- 8份引导条件重新解码一致。Q6沿用4份可追溯接收本地流；Q9新运行4份RAFT，各174对。没有把接收缓存当发送辅助信息。
- 四段Q9_T3使用原无缓存T3门控再次渲染，RGB/alpha/shared逐位一致；这只核验同光流的渲染实现，不是另一套RAFT独立复现。
- 源读拒绝、读白名单、包绑定、冻结输入/代码与采用配置SHA通过。17个原速视频均核验150帧/15FPS/10秒及首中末面板。
- R25/D25增加局部统计计算。Q6复用光流，本轮**不做端到端速度比较**，也不据局部render时间宣称实时。

计算代价也需要保留：本次四片各一次的Q6局部render，T3为5.20–5.39秒，R25为8.38–8.55秒，D25为9.29–9.64秒／10秒片段；这些是单次局部记录，不含光流和基础GLC解码，不能当作重复测速结论。R25约1%的指标改善是否值得额外计算，需要结合原速观感判断。

所有失败候选保留，video2有完整六宫格。双列左当前Q6_T3、右标题候选；四宫格源/原GLC/当前/候选按行优先。静态抽帧图用于布局检查，候选主观反馈pending。

## 当前决定与后续

{latest} 当前采用配置和部署默认未改，本轮没有提交或推送。

后续先结合这两组原速对照确认是否达到“改善一些、其他片保持”的体验目标。保持统一参数，继续记录正负结果；只有在规则确定后才登记独立海洋留出素材。长片需另行处理有界缓存、固定150帧封装、跨块参考和码率统计，本轮不提前作结论。

## 文件

- 权威运行：{run.as_posix()}
- 推理候选：mild_reliability_gate.py；控制器：run_generalization_reliability.py。
- 核验/观看：finish_generalization_reliability.py；发布：publish_generalization_reliability.py。
- 冻结计划、源码、接收回执、评分CSV/JSON、用户澄清和旧入口快照保存在本轮小证据目录；manifest记录准确字节。
- 核验执行后仅调整了预览的候选排序；verification_source_note.json与SHA一致的执行时源码快照记录该差异，核验函数和推理算法未变。
- 重跑必须选新目录，不覆盖当前或历史输出。现有采用和无关未提交修改均保留。
"""
    put(report,text)

    view_rows=[]
    for case in cases:
        view_rows.append([case,f'[当前 vs {chosen}]({case}_current_vs_{chosen}.mp4)',
                          f'[四宫格]({case}_source_raw_current_{chosen}.mp4)',
                          f'[当前 vs {other}]({case}_current_vs_{other}.mp4)'])
    put(viewing/'README.md',f"""# GLC 泛化第二轮原速观看

2026-11-12。{decision} 主观反馈待记录，尚未采用。

最新目标：video2适度改善即可，用户澄清并不是特别难看。请以此为观看基调。

本轮Q6/Q9 × T3/R25/D25，四段共24组新输出，统一参数。R25对门限附近的可疑复用最多降权25%；D25再考虑原GLC低频差异，仍最多25%。Q9有额外字节成本。

**双列左当前Q6_T3、右标题候选**；四宫格左上源、右上原GLC、左下当前、右下候选。每条10秒/15FPS。

{mdtable(['素材','主要双列','主要四宫格','另一候选双列'],view_rows)}

{latest}

[video2六档完整矩阵](video2_all_six_profiles.mp4)：按行优先Q6_T3、Q6_R25、Q6_D25、Q9_T3、Q9_R25、Q9_D25。

目录也保留另一候选的各片四宫格和四张抽帧图。对照时注意鱼体/背景交界、纹理替换、换锚跳变及video1/3原本稳定的背景；边界无需达到“完全消失”才算有进展。

[完整报告]({relative(report,viewing)}) · [核验]({relative(run/'verification.json',viewing)}) · [SHA清单](manifest.json)

历史短段开发实验，未验证独立留出/长视频/实时。MP4只供观看，指标来自无损数组。文件为原结果硬链接，剪辑请先另存副本；未提交/推送。
""")
    save_json(viewing/'manifest.json',dict(date='2026-11-12',source_run=str(run),files=media,video_count=17))

    entries=['交接文档.md','当前项目状态与接续.md','README.md','项目总结与未来方向.md',
             '21_优化阶段_GLC解码后处理/README.md','21_优化阶段_GLC解码后处理/当前方案总结与瓶颈路线_20261111.md',
             '21_优化阶段_GLC解码后处理/下一阶段_海洋素材与长视频泛化计划_20261111.md',
             '05_资料与索引/会话收束_20261110/文件索引.md']
    for name in entries:
        path=REPO/name
        old=path.read_text(encoding='utf-8-sig')
        assert (run/'prior_entries'/name).read_bytes()==path.read_bytes(),name
        title,body=old.split('\n',1)
        update=f"""**最新：泛化第二轮完成（2026-11-12）。** {latest}

**用户最新澄清：video2改善一些即可，并非特别难看。** 当前叙述以此为准，历史评价保留但不能继续夸大严重程度。新候选主观验收pending，采用配置未改。

[第二轮报告]({relative(report,path.parent)}) · [原速观看]({relative(viewing/'README.md',path.parent)}) · [核验]({relative(run/'verification.json',path.parent)})

本轮已完成，无实验继续运行。下一步根据候选原速观感确定是否保留，再进行独立素材验证；长视频、实时尚未验证。本轮未提交/推送。
"""
        put(path,title+'\n\n'+update+'\n## 前轮记录（以页首最新状态为准）\n\n'+body.lstrip())
    continuation=REPO/'新会话接续.txt'
    old=continuation.read_text(encoding='utf-8-sig')
    assert old==(run/'prior_entries/新会话接续.txt').read_text(encoding='utf-8-sig')
    put(continuation,f"""GLC接续，2026-11-12：泛化第二轮完成。
用户澄清：video2改善一些即可，并非特别难看；历史原话保留，但当前不得夸大严重性。
先读：{report}
观看：{viewing/'README.md'}
权威运行：{run}
{latest}
24组新输出，4组Q12历史参照；四段Q6基线逐位一致，Q9原门控回放逐位一致。R25/D25最大降权25%，有效区域不变。
数值选中：{selected}；主要观看：{chosen}；主观pending；未自动采用，未改部署默认。
无独立留出/完整长片/实时验证。实验已结束，无活动实验。未提交/推送，保留原有修改。

----- 前轮接续（历史） -----
{old}""")
    state_path=REPO/'05_资料与索引/会话收束_20261110/接续状态.json'
    state=load_json(state_path)
    state['date']='2026-11-12'
    state['status']='generalization_reliability_complete_candidate_pending_visual' if selected else 'generalization_reliability_complete_no_universal_candidate'
    state['active_experiment_plan']=None
    state['current_session_work']='Completed mild confidence and moderate guide round, 24 new outputs, 4 historical references; visual acceptance pending.'
    state['new_experiment_report']=str(report.relative_to(REPO))
    state['latest_analysis_report']=str(report.relative_to(REPO))
    state['latest_user_perceptual_feedback']=feedback
    state['user_latest_decision']=feedback['verbatim_user_feedback']
    state['experiment_execution_gate']=dict(status='authorized_round_complete',date='2026-11-12',authorization=feedback['verbatim_user_feedback'])
    state['previous_generalization_experiment']=state.get('latest_generalization_experiment')
    state['latest_generalization_experiment']=dict(complete=True,round='mild_reliability',new_outputs=24,historical_reference_conditions=4,
        selected=selected,viewing_candidate=chosen,subjective_review='pending',automatic_adoption=False,current_profile_unchanged=True,
        independent_holdout=False,long_video=False,realtime=False,run=str(run),report=str(report),viewing=str(viewing),
        verification=str(run/'verification.json'),candidate_changes=compact,objective='Modest video2 improvement; do not overstate severity')
    state['latest_experiment_plan']=dict(status='completed',path=str(run/'execution_plan.json'),new_outputs=24)
    state['next_priority']=['Review the new candidates at original speed against the modest-improvement objective.',
                            'Retain one fixed rule only after numerical and visual assessment; then register independent marine clips.',
                            'Handle bounded caches, chunk/reference rules and guide accounting before full-length validation.']
    state['next_experiment_plan']=dict(status='proposed_not_executed',priority=state['next_priority'],current_profile_unchanged=True)
    state['workspace_paths']['generalization_reliability_run']=str(run.relative_to(WORKSPACE))
    state['workspace_paths']['generalization_reliability_viewing']=str((viewing/'README.md').relative_to(WORKSPACE))
    state['publication']=dict(current_changes_committed=False,pushed=False,scope='Mild reliability round and preserved prior uncommitted changes')
    state['new_experiments_uncommitted']=True
    save_json(state_path,state)
    assert file_sha(config)==config_sha
    save_json(run/'progress.json',dict(stage='complete',selected=selected,viewing_candidate=chosen,subjective_review='pending',new_outputs=24))
    save_json(run/'publication.json',dict(complete=True,report=str(report),viewing=str(viewing),evidence=str(evidence),
        adopted_configuration_sha256=config_sha,automatic_adoption=False,committed=False,pushed=False,
        media=media,candidate_changes=compact,publisher_sha256=file_sha(__file__)))
    for path in sorted(run.rglob('*')):
        if path.is_file() and (path.suffix in ('.json','.csv') or 'source_snapshot' in path.parts or 'prior_entries' in path.parts):
            dest=evidence/'run'/path.relative_to(run)
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dest)
    for name in ('finish_generalization_reliability.py','publish_generalization_reliability.py'):
        dest=evidence/'publication_source'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(STAGE/name,dest)
    manifest=[dict(path=str(p.relative_to(evidence)),bytes=p.stat().st_size,sha256=file_sha(p))
              for p in sorted(evidence.rglob('*')) if p.is_file()]
    save_json(evidence/'manifest.json',dict(date='2026-11-12',files=manifest,source_run=str(run)))
    for item in manifest:
        assert file_sha(evidence/item['path'])==item['sha256']
    save_json(run/'publication_verification.json',dict(complete=True,evidence_files=len(manifest),media_hardlinks=len(media),
        report_sha256=file_sha(report),evidence_manifest_sha256=file_sha(evidence/'manifest.json'),
        entry_sha256={name:file_sha(REPO/name) for name in [*entries,'新会话接续.txt','05_资料与索引/会话收束_20261110/接续状态.json']},
        adopted_configuration_unchanged=file_sha(config)==config_sha))
    print(json.dumps(dict(report=str(report),viewing=str(viewing),selected=selected,viewing_candidate=chosen,
                          candidate_changes=compact,evidence_files=len(manifest)),ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    publish(args.run)
