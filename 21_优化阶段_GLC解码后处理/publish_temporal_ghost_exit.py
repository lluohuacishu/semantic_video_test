"""Publish the causal temporal-exit round, preserving history and adoption."""
import argparse
import copy
import os
from pathlib import Path
import re
import shutil
from publish_q9_ghost_exit import sha,read,put,save,relative,link,table,delta,STAGE,REPO,WORKSPACE,EXPECTED_CONFIG
from temporal_conflict_exit import PROFILES,SETTINGS

TITLE='Q9残像连续帧退出与低频修正实验_20261112.md'
NAMES={'video1':'video1','video2':'video2','video3':'video3',
       'reef_spotted_fish':'礁石斑点鱼','seagrass_spotted_fish':'海草斑点鱼'}

def publish(run):
    plan=read(run/'execution_plan.json');summary=read(run/'summary.json')
    verification=read(run/'verification.json');media=read(run/'previews/verification.json')
    inspection=read(run/'visual_inspection.json')
    assert summary['complete'] and verification['complete'] and media['complete']
    assert inspection['images_actually_viewed'] and summary['new_outputs']==15
    assert sha(STAGE/'当前采用配置.json')==EXPECTED_CONFIG
    assert sha(run/'case_registration.json')==plan['registration_sha256']
    for name,h in plan['code_sha256'].items():assert sha(STAGE/name)==h==sha(run/'source_snapshot'/name)
    parent=Path(plan['parent']);old=read(parent/'summary.json')
    report=STAGE/TITLE;view=WORKSPACE/'05_成果与观看/GLC_Q9连续帧退出_20261112'
    evidence=STAGE/'evidence_20261112/q9_temporal_exit'
    statepath=REPO/'05_资料与索引/会话收束_20261110/接续状态.json'
    entries=[REPO/n for n in ['交接文档.md','当前项目状态与接续.md','README.md','项目总结与未来方向.md',
                              '新会话接续.txt','05_资料与索引/会话收束_20261110/文件索引.md']]
    entries += [STAGE/n for n in ['README.md','当前方案总结与瓶颈路线_20261111.md',
                                  '下一阶段_海洋素材与长视频泛化计划_20261111.md']]
    entries += [WORKSPACE/'02_实验数据/README.md',WORKSPACE/'05_成果与观看/README.md']
    for p in [report,view,evidence,run/'publication.json']:
        if p.exists():raise FileExistsError(p)
    before=run/'publication_before';before.mkdir(exist_ok=False)
    old_manifest_hashes={str(p):sha(p) for p in (STAGE/'evidence_20261112').rglob('manifest.json')}
    snapshots={}
    for p in entries+[statepath]:
        dest=before/relative(p,WORKSPACE);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        snapshots[str(p)]=sha(p)
    save(before/'snapshot_manifest.json',dict(files=snapshots,existing_evidence_manifests=old_manifest_hashes))

    changes={};temporal=[];psnr=[];roi=[];motion=[];rates=[];eventrows=[]
    for case,s in summary['cases'].items():
        b=s['scores']['full']['Q9_R25'];changes[case]={}
        for n in PROFILES:
            v=s['scores']['full'][n]
            changes[case][n]=dict(coarse_temporal_change_percent=delta(v['coarse_mae'],b['coarse_mae']),
                                   psnr_change_db=v['psnr_db']-b['psnr_db'])
        temporal.append([NAMES[case]]+[f"{changes[case][n]['coarse_temporal_change_percent']:+.2f}%" for n in PROFILES]+[
            f"{delta(old['cases'][case]['scores']['full']['EL12']['coarse_mae'],b['coarse_mae']):+.2f}%"])
        psnr.append([NAMES[case]]+[f"{changes[case][n]['psnr_change_db']:+.4f}" for n in PROFILES])
        if 'reef' in s['scores']:
            q=s['scores']['reef']['Q9_R25']['coarse_mae']
            roi.append([NAMES[case]]+[f"{delta(s['scores']['reef'][n]['coarse_mae'],q):+.2f}%" for n in PROFILES])
        motion.append([NAMES[case]]+[f"{s['dynamic'][n]['RGB_delta_error']:.4f}" for n in ('Q9_R25',*PROFILES)])
        rates.append([NAMES[case],f"{s['total_kbps']:.4f}",0])
        for label,vs in s['user_event_crops'].items():
            q=vs['Q9_R25']['mean_rgb_mae']
            eventrows.append([NAMES[case]+'/'+label,f"{q:.6f}"]+[
                f"{vs[n]['mean_rgb_mae']:.6f}（{delta(vs[n]['mean_rgb_mae'],q):+.2f}%）" for n in PROFILES])
    events={}
    for case in ('reef_spotted_fish','video2'):
        events[case]={}
        for label,vs in summary['cases'][case]['user_event_crops'].items():
            events[case][label]={n:dict(mean_rgb_mae=vs[n]['mean_rgb_mae'],
                 mean_rgb_mae_change_percent=delta(vs[n]['mean_rgb_mae'],vs['Q9_R25']['mean_rgb_mae']),
                 exact_frame61_rgb_mae=vs[n]['exact_frame61']['rgb_mae']) for n in ('Q9_R25',*PROFILES)}
    result=dict(date='2026-11-12',complete=True,reference='Q9_R25',profiles=list(PROFILES),new_outputs=15,new_candidate_outputs=15,
                cases=list(NAMES),changes=changes,user_events=events,selected=None,automatic_adoption=False,
                user_subjective_review='pending',prior_Q9_positive_feedback_preserved=True,
                current_profile_sha256=EXPECTED_CONFIG,current_profile_unchanged=True,
                extra_transmission_bytes=0,long_video_rerun=False,independent_holdout=False,
                conclusion_zh='连续帧确认和运动代理仍不足以解决残像；低频有界修正降低部分时序代价，但各片仍有相对Q9的回退。保持研究对照，不自动采用。')
    save(run/'round_result.json',result)
    def rl(label,path):return link(label,path,STAGE)
    report_text=f"""# Q9 残像连续帧退出与低频修正实验（2026-11-12）

本轮固定三候选P3/M3/ML3，在原五段10秒短片上完成15组输出。研究问题是：给退出条件加入连续帧证据、限制退出/恢复速度，能否减少上一轮直接退回原GLC引入的波动。ML3另测试只修正较平滑的错误部分。

**三候选均未采用。** ML3相对本轮全RGB退出的时序代价较小，但相对Q9_R25仍有回退；video2鱼尾改善有限。Q9_R25仍是可追溯研究参照，正式S8I + CURRENT_I12配置未改。原三段Q9正面反馈和video2适度改善目标保持。用户指出的礁石4.07秒、video2鱼尾仍是重点；本轮未重跑长片。

## 1. 范围与预登记

五片为video1/video2/video3/礁石斑点鱼/海草斑点鱼，每片150帧、15 FPS、512×288。复用上一轮已独立重渲染且逐位核验的Q9_R25与已收到的原GLC、原RAFT流缓存；**没有新增码包、没有新增光流模型推理**，不是独立留出验证。

三候选在产生本轮输出前一并冻结，用统一参数跑全五片。本轮没有观察结果后追加候选，也没有按片调参。阈值沿用上轮EL12，并预设时间与运动代理参数，这些是研究设定，不是已获验证的通用阈值。

{rl('冻结计划',run/'execution_plan.json')} · {rl('接收算法',STAGE/'temporal_conflict_exit.py')} · {rl('执行程序',STAGE/'run_temporal_ghost_exit.py')}

## 2. 机制与对照关系

原Q9的历史纹理在低码率guide中缺乏可靠运动对应时，可能把鱼形留在旧位置。上轮EL12能减轻局部旧影，却也将原GLC的变化重新带入。此次不只平滑一张开关掩码，而是先确认连续帧的差异方向，并分离“什么时候退出”和“退出时改哪些像素成分”。

| 候选 | 本轮变化 | 用途 |
| --- | --- | --- |
| P3 | 连续三帧有方向一致的raw−Q9冲突；门控每帧最多上升1/3、下降1/6 | 确认与缓慢恢复的基础对照 |
| M3 | P3再乘以运动迹象记忆 | 检查运动证据能否减少错误退出 |
| ML3 | 与M3相同门控，只应用三帧平均sigma1残差，并限制在当前raw与Q9之间 | 检查不完全放回原GLC细纹理能否保住稳定 |

历史两帧通过接收端已有光流复合到当前坐标：当前→锚帧，再锚帧→历史帧。只用过去与当前帧；保持两帧模糊图及门控状态，无未来帧读取。遇原锚点0/15/45/75/105/135重置状态，前两帧不启用退出。此行为会带来确认延迟、锚点附近漏检或重新出现的风险，尚未加入切镜检测。

冲突按sigma3的6→12与sigma1的12→24计算，再乘连续三帧带符号残差的一致性（0.75→1）。运动代理比较过去两帧间隔的“原GLC变化幅度是否超过Q9”，阈值2→6、1像素膨胀；冲突持续时保持证据，冲突很低时记忆每帧减半。它**不是鱼分割或真实物体运动的保证**，压缩波动也可能触发。

ML3的修正逐通道限制在当前Q9→当前原GLC的线段内，强度不超过M3。它降低高频纹理回流，也可能把高频鱼轮廓留住；不能仅凭更稳定就判断已正确消除残像。源码与计划记录精确公式。

## 3. 用户事件局部结果

沿用冻结框：礁石旧位置A=[416,112,452,144]、B=[488,112,512,144]，评价53–74帧；video2鱼尾宽框=[140,145,285,288]，评价45–74帧。用户只指定礁石4.07秒（零起算61帧），没有指定video2时间。ROI只用于独立评价和裁剪，从未作为接收输入。

下表是固定框内源RGB平均像素误差，括号相对Q9_R25。**不是“残像减少百分比”或鱼数量**。

{table(['位置','Q9_R25','P3','M3','ML3'],eventrows)}

第61帧数值和逐帧评价见{rl('结构化结果',run/'round_result.json')}及各片evaluation/user_event_crops.csv。三帧确认、运动代理和有界低频修正均没有获得用户原速验收。

静图实查记录：

{chr(10).join('- '+x for x in inspection['observations_zh'])}

{rl('礁石3–5秒放大对照',view/'reef_spotted_fish_user_event_crop_all.mp4')} · {rl('video2鱼尾3–5秒放大对照',view/'video2_user_event_crop_all.mp4')} · {rl('礁石4.07秒细节',view/'reef_exact4p07_footprints.png')}

## 4. 稳定性、空间误差与运动取舍

全帧源对齐低频时序误差变化如下。所有百分比都相对同一个Q9_R25，正数表示回退。上轮EL12用于跨轮参照；基线评分已核对逐项一致。这个指标不是可见闪烁次数，也不能替代真实物体的去留。

{table(['片段','P3','M3','ML3','上轮EL12'],temporal)}

全RGB退出的P3/M3即使有时间约束仍有回退，说明仅靠保持开关状态不足以保护原Q9效果。ML3降低了这部分代价，但不能以牺牲鱼轮廓修正为代价只追稳定数值。

历史珊瑚ROI（不是本轮用户框）的时序变化：

{table(['片段','P3','M3','ML3'],roi)}

全帧PSNR变化（dB，相对Q9_R25）：

{table(['片段','P3','M3','ML3'],psnr)}

运动区域RGB变化误差（绝对值，越低越好）：

{table(['片段','Q9_R25','P3','M3','ML3'],motion)}

PSNR改善不等于消除拖尾。运动区域由原评价流程确定，仅评价端使用；运动幅度比和全部分区指标保存在各片summary/CSV，不能只选有利指标报告。既有Q9_R25 selected=null及video3历史珊瑚+5.37%记录继续保持，旧指标的比较基线与本轮不同。

## 5. 字节、实现检查与复现

{table(['片段','原包实际总kbps','新增传输字节'],rates)}

5片输入/基线SHA、接收读白名单、15组数组、逐像素有界性、零门控不变、原alpha=0不变、M3权重不超过P3、ML3偏离Q9不超过M3均已检查。用户两片0–75帧从接收输入重放，与保存结果逐位一致。验证还覆盖无冲突恒等、持续变化能触发、冲突结束能恢复、锚点状态重置。

评分CSV及用户ROI误差独立重算一致，源/raw/Q9各指标与前轮相同。读边界是Python可复现性检查，不是操作系统安全沙箱。源图与ROI只供独立评分；接收不读取源图、发送收据、用户事件/评价框文件。

输出用内存映射逐帧写入；算法状态保留两帧。但本轮复用整段本地数组/流缓存，没有证明整个收发管线固定RAM或实时。receipt里的耗时仅三候选共同后处理与输出写入，不包括基础解码、RAFT推理或评价。

{rl('完整核验',run/'verification.json')} · {rl('预览核验',run/'previews/verification.json')} · {rl('小证据清单',evidence/'manifest.json')}

采用配置SHA256仍为：{EXPECTED_CONFIG}。冻结推理程序、旧结果、旧证据均不改；未提交/推送。

## 6. 观看与后续方向

{rl('全部原速观看入口',view/'README.md')}。五个10秒六宫格、两条10秒Q9/ML3双列、两条原视频3–5秒放大六宫格；六宫格按行：原片/原GLC、Q9_R25/P3、M3/ML3。所有预览15 FPS，MP4是观看转码，评分用无损NPY。

本轮已结束，无活动实验。ML3提供了“减少原GLC细纹理回流”的有用对照，但还不能作为修复版。下一步应先检查剩余旧鱼轮廓和运动代理误触发位置，再决定是否需要更可靠的对应/遮挡证据；不再仅扩大时间平滑或继续扫阈值。该下一步尚未实施。

继续保留用户对Q9三段的正面评价，不因本轮局部问题否定整体效果；长片切镜与复杂场景暂作失效观察，不追求优秀全片评分。
"""
    put(report,report_text)
    view.mkdir()
    files=sorted(p for p in (run/'previews').iterdir() if p.suffix in ('.mp4','.png'))
    assert len(files)==12
    media_records={}
    for src in files:
        dst=view/src.name;os.link(src,dst);assert os.path.samefile(src,dst)
        media_records[src.name]=dict(source=str(src),bytes=src.stat().st_size,sha256=sha(src),hardlink=True)
        if src.suffix=='.mp4':assert sha(src)==media['videos'][src.name]['sha256']
    save(view/'media_manifest.json',dict(complete=True,files=media_records))
    vl=lambda label,p:link(label,p,view)
    put(view/'README.md',f"""# Q9 连续帧退出诊断观看（2026-11-12）

P3/M3/ML3为本轮诊断候选，均未采用。ML3保留更多原Q9稳定性，但残像与鱼尾问题尚未解决，用户原速验收待记录。

## 优先观看

- {vl('礁石3–5秒六宫格放大',view/'reef_spotted_fish_user_event_crop_all.mp4')}：原片4.07秒对应裁剪约1.07秒。
- {vl('video2鱼尾3–5秒六宫格放大',view/'video2_user_event_crop_all.mp4')}：此时间范围由本轮选定，用户只指出鱼尾。
- {vl('礁石完整10秒：左Q9_R25 / 右ML3',view/'reef_spotted_fish_Q9_vs_ML3.mp4')}。
- {vl('video2完整10秒：左Q9_R25 / 右ML3',view/'video2_Q9_vs_ML3.mp4')}。

六宫格每行左→右：原视频/原GLC；Q9_R25/P3；M3/ML3。15 FPS原速；局部只放大空间，不改变速度。

## 五片完整六宫格

{chr(10).join('- '+vl(NAMES[c],view/(c+'_all_six.mp4')) for c in NAMES)}

## 静图

{chr(10).join('- '+vl(p.name,view/p.name) for p in files if p.suffix=='.png')}

{vl('完整报告',report)} · {vl('本轮结构化结果',run/'round_result.json')} · {vl('核验',run/'verification.json')} · {vl('媒体SHA',view/'media_manifest.json')}

媒体为运行目录硬链接，编辑前先另存副本；不得用MP4体积计算语义通信码率。没有重跑长片，没有替换正式采用配置。
""")
    for p in entries:
        oldtext=p.read_text(encoding='utf-8-sig')
        prefix=f"""**最新（2026-11-12）：连续帧退出与低频修正本轮完成，五片15组。** P3测试三帧冲突确认和缓慢恢复，M3增加运动代理记忆，ML3只施加有界低频残差。ML3减少部分时序回退，但相对Q9仍有代价，video2鱼尾改善有限；三候选均未采用。

{link('本轮报告',report,p.parent)} · {link('原速与放大对照',view/'README.md',p.parent)} · {link('核验',run/'verification.json',p.parent)} · {link('证据',evidence/'manifest.json',p.parent)}

礁石4.07秒和video2鱼尾仍是重点，候选原速反馈待记录；原三段Q9正面评价、Q9_R25研究参照、正式采用配置保持。本轮复用既有码包/光流，无新增传输字节，没有重跑长片。实验已收束，无活动实验，未提交/推送。

下一步先检查剩余旧轮廓和运动代理误触发，再决定对应/遮挡证据方案；尚未运行，不继续盲扫时间平滑参数。长片暂不追求优秀全片评分。

## 既往记录（以下为当时状态，以页首本轮结果为准）
"""
        if p.suffix=='.txt':
            prefix=f"""GLC接续，2026-11-12：Q9连续帧退出本轮已完成，五片15组，无活动实验。
P3三帧确认与缓慢恢复；M3加运动代理；ML3用有界低频修正。ML3减少部分时序代价，但残像/鱼尾未解决，均未采用。
保持Q9_R25研究参照、原三段正面反馈、正式S8I+CURRENT_I12配置。旧selected=null和video3珊瑚+5.37%记录保留。
本轮无新增传输字节，复用旧RAFT流，没有重跑长片。下一步先查剩余旧轮廓和运动代理误触发，尚未运行。
报告：{report.as_posix()}
观看：{(view/'README.md').as_posix()}
运行：{run.as_posix()}
核验：{(run/'verification.json').as_posix()}
未提交/推送。

此前接续正文（当时状态，以页首为准）：
"""
            put(p,prefix+'\n'+oldtext)
        else:
            title,body=oldtext.split('\n',1)
            put(p,title+'\n\n'+prefix+'\n'+body.lstrip('\n'))

    state=read(statepath)
    state['previous_latest_experiment_before_temporal_exit']=copy.deepcopy(state.get('latest_experiment'))
    state['previous_generalization_before_temporal_exit']=copy.deepcopy(state.get('latest_generalization_experiment'))
    state['previous_next_plan_before_temporal_exit']=copy.deepcopy(state.get('next_experiment_plan'))
    state['previous_latest_diagnosis_before_temporal_exit']=copy.deepcopy(state.get('latest_diagnosis'))
    state['latest_diagnosis']=dict(report=relative(report,REPO),
        prior_reef_frame61_attribution_preserved=str(parent/'diagnosis/attribution.json'),
        temporal_confirmation_alone_insufficient=True,
        low_frequency_correction_reduces_temporal_cost_but_leaves_tradeoffs=True,
        video2_tail_not_resolved=True,not_a_universal_ghosting_fix=True)
    latest=copy.deepcopy(result);latest.update(run=str(run),report=str(report),viewing=str(view),
                                              verification=str(run/'verification.json'),evidence=str(evidence/'manifest.json'))
    state.update(status='Q9_temporal_exit_round_complete',active_experiment_plan=None,
                 current_session_work='Completed 15 frozen P3/M3/ML3 outputs, verification, previews and handoff; none adopted.',
                 latest_experiment=latest,latest_generalization_experiment=copy.deepcopy(latest),
                 latest_analysis_report=relative(report,REPO),new_experiment_report=relative(report,REPO),
                 latest_experiment_plan=dict(status='completed',path=str(run/'execution_plan.json'),new_outputs=15),
                 user_latest_decision='那继续吧',new_experiments_uncommitted=True,
                 next_priority=['Review original-speed user-event views, preserving prior Q9 positive assessment.',
                                'Inspect remaining old contours and motion-proxy false activations before choosing new correspondence/occlusion evidence.',
                                'Defer whole-long-video score optimization.'],
                 next_experiment_plan=dict(status='proposed_not_run',reference='Q9_R25',
                    focus=['remaining stale contours','motion proxy false activation','reliable correspondence/occlusion evidence'],
                    parameters_not_chosen=True,long_video_quality_priority='deferred_per_user'),
                 experiment_execution_gate=dict(status='completed_round',currently_running=False,new_run_started=True,
                                                automatic_next_run=False,date='2026-11-12'),
                 publication=dict(current_changes_committed=False,pushed=False,scope='Q9 temporal exit: 15 outputs, report, evidence and previews',
                                  receipt=str(run/'publication.json')))
    state['latest_user_perceptual_feedback']['new_candidate_user_review']='pending'
    state['preferred_research_reference']['latest_experiment_report']=str(report)
    state['preferred_research_reference']['current_focus']='Inspect residual fish contours and imperfect motion proxy after temporal-exit round; Q9_R25 retained.'
    save(statepath,state)
    assert sha(STAGE/'当前采用配置.json')==EXPECTED_CONFIG

    save(run/'publication.json',dict(complete=True,report=str(report),viewing=str(view),evidence=str(evidence),
          new_outputs=15,videos=9,pngs=3,current_profile_unchanged=True,report_sha256=sha(report),
          updated_entry_sha256={str(p):sha(p) for p in entries+[statepath]},
          publisher_sha256=sha(__file__),committed=False,pushed=False))
    evidence.mkdir(parents=True)
    for src in [p for p in run.rglob('*') if p.is_file() and p.suffix.lower() in ('.json','.csv','.py','.md','.txt')]:
        dst=evidence/'run'/relative(src,run);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for src in [report,Path(__file__),STAGE/'publish_q9_ghost_exit.py']:
        dst=evidence/('report' if src==report else 'publication_source')/src.name
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for src in files:
        if src.suffix=='.png':
            dst=evidence/'visuals'/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for src in [view/'README.md',view/'media_manifest.json']:
        dst=evidence/'viewing'/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for src in entries+[statepath]:
        dst=evidence/'entries_after'/relative(src,WORKSPACE);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    links=0
    for p in [report,view/'README.md']+entries:
        text=p.read_text(encoding='utf-8-sig')
        if p in entries:text=text.split('## 既往记录（以下为当时状态，以页首本轮结果为准）')[0]
        for href in re.findall(r'\]\(([^)]+)\)',text):
            target=(p.parent/href.split('#',1)[0]).resolve()
            assert target.exists() or target==evidence/'manifest.json',(str(p),href)
            links+=1
    for p,h in old_manifest_hashes.items():assert sha(p)==h
    for p,h in snapshots.items():assert sha(before/relative(p,WORKSPACE))==h
    for src in files:assert sha(src)==sha(view/src.name)==media_records[src.name]['sha256']
    for p in evidence.rglob('*.json'):read(p)
    assert read(statepath)['active_experiment_plan'] is None
    qa=dict(complete=True,entry_links_checked=links,evidence_history_unchanged=True,
            input_config_unchanged=True,media_sha_match=True,strict_json_checked=True,
            state_complete_no_active_plan=True,current_config_sha256=sha(STAGE/'当前采用配置.json'))
    save(run/'publication_verification.json',qa);shutil.copy2(run/'publication_verification.json',evidence/'run/publication_verification.json')
    manifest={}
    for p in sorted(evidence.rglob('*')):
        if p.is_file():
            assert p.suffix not in ('.mp4','.npy')
            manifest[relative(p,evidence)]=dict(bytes=p.stat().st_size,sha256=sha(p))
    save(evidence/'manifest.json',dict(complete=True,files=manifest,source_run=str(run),
                                      large_arrays_and_videos_excluded=True,self_excluded=True))
    for p,d in manifest.items():assert sha(evidence/p)==d['sha256']
    print('PUBLISHED',len(manifest),'evidence files;',links,'links checked',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    publish(p.parse_args().run.resolve())

