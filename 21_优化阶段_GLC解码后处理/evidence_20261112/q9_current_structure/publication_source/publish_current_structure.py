"""Publish four-clip current-structure research and the user's revised priority."""
import argparse
import copy
import os
from pathlib import Path
import re
import shutil
from publish_q9_ghost_exit import sha,read,put,save,relative,link,table,delta,STAGE,REPO,WORKSPACE,EXPECTED_CONFIG

TITLE='Q9当前轮廓保留与运动范围限制_20261112.md'
NAMES={'video1':'video1','video2':'video2','video3':'video3','reef_spotted_fish':'礁石斑点鱼'}
METHODS=('CS50','CS100','CSM50')

def publish(run):
    plan=read(run/'execution_plan.json');main=read(run/'summary.json')
    ep=read(run/'motion_extension/execution_plan.json');extra=read(run/'motion_extension/summary.json')
    verify=read(run/'verification.json');ev=read(run/'motion_extension/verification.json')
    media=read(run/'previews/verification.json');visual=read(run/'visual_inspection.json')
    feedback=read(run/'user_feedback_20261112.json')
    assert all(x['complete'] for x in (main,extra,verify,ev,media))
    assert main['new_outputs']==8 and extra['new_outputs']==4 and media['video_count']==10
    assert visual['images_actually_viewed'] and sha(STAGE/'当前采用配置.json')==EXPECTED_CONFIG
    for folder,p in [(run,plan),(run/'motion_extension',ep)]:
        for name,h in p['code_sha256'].items():assert sha(STAGE/name)==h==sha(folder/'source_snapshot'/name)
    assert sha(run/'summary.json')==ep['first_phase_summary_sha256']
    assert sha(run/'execution_plan.json')==ep['first_phase_plan_sha256']
    report=STAGE/TITLE;view=WORKSPACE/'05_成果与观看/GLC_Q9当前轮廓保留_20261112'
    evidence=STAGE/'evidence_20261112/q9_current_structure'
    statepath=REPO/'05_资料与索引/会话收束_20261110/接续状态.json'
    entries=[REPO/n for n in ['交接文档.md','当前项目状态与接续.md','README.md','项目总结与未来方向.md',
                              '新会话接续.txt','05_资料与索引/会话收束_20261110/文件索引.md']]
    entries += [STAGE/n for n in ['README.md','当前方案总结与瓶颈路线_20261111.md',
                                  '下一阶段_海洋素材与长视频泛化计划_20261111.md']]
    entries += [WORKSPACE/'02_实验数据/README.md',WORKSPACE/'05_成果与观看/README.md']
    for p in [report,view,evidence,run/'publication.json']:
        if p.exists():raise FileExistsError(p)
    before=run/'publication_before';before.mkdir(exist_ok=False)
    old_manifests={str(p):sha(p) for p in (STAGE/'evidence_20261112').rglob('manifest.json')}
    snapshots={}
    for p in entries+[statepath]:
        dst=before/relative(p,WORKSPACE);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
        snapshots[str(p)]=sha(p)
    save(before/'snapshot_manifest.json',dict(files=snapshots,old_manifests=old_manifests))

    def result(case,method):
        return (extra if method=='CSM50' else main)['cases'][case]
    changes={};times=[];spatial=[];reefs=[];regions=[];dyn=[];rate_rows=[];interventions=[]
    for case in NAMES:
        b=main['cases'][case]['scores']['full']['Q9_R25'];changes[case]={}
        for method in METHODS:
            s=result(case,method);v=s['scores']['full'][method]
            changes[case][method]=dict(full_coarse_temporal_change_percent=delta(v['coarse_mae'],b['coarse_mae']),
                                      full_psnr_change_db=v['psnr_db']-b['psnr_db'])
        times.append([NAMES[case]]+[f"{changes[case][m]['full_coarse_temporal_change_percent']:+.2f}%" for m in METHODS])
        spatial.append([NAMES[case]]+[f"{changes[case][m]['full_psnr_change_db']:+.4f}" for m in METHODS])
        s=extra['cases'][case]
        if 'reef' in s['scores']:
            q=s['scores']['reef']['Q9_R25']['coarse_mae']
            reefs.append([NAMES[case]]+[f"{delta(result(case,m)['scores']['reef'][m]['coarse_mae'],q):+.2f}%" for m in METHODS])
        for label,values in main['cases'][case]['current_regions'].items():
            q=values['Q9_R25']['mean_rgb_mae']
            regions.append([NAMES[case]+'/'+label,f'{q:.6f}']+[
                f"{result(case,m)['current_regions'][label][m]['mean_rgb_mae']:.6f}（{delta(result(case,m)['current_regions'][label][m]['mean_rgb_mae'],q):+.2f}%）" for m in METHODS])
        d=s['dynamic'];dyn.append([NAMES[case],f"{d['Q9_R25']['RGB_delta_error']:.4f}",f"{d['CSM50']['RGB_delta_error']:.4f}",
                                 f"{d['Q9_R25']['change_amplitude_ratio']:.4f}",f"{d['CSM50']['change_amplitude_ratio']:.4f}"])
        rate_rows.append([NAMES[case],f"{s['total_kbps']:.4f}",0])
        stat=ev['cases'][case];interventions.append([NAMES[case],f"{100*stat['active_fraction']:.2f}%",f"{stat['mean_motion_weight']:.4f}"])
    result_json=dict(date='2026-11-12',complete=True,new_outputs=12,new_candidate_outputs=12,
        first_phase_outputs=8,adaptive_extension_outputs=4,cases=list(NAMES),profiles=list(METHODS),
        reference='Q9_R25',prior_ML3_reference_only=True,selected=None,viewing_candidate='CSM50',
        automatic_adoption=False,current_profile_unchanged=True,current_profile_sha256=EXPECTED_CONFIG,
        changes=changes,user_subjective_review='pending',previous_Q9_positive_feedback_preserved=True,
        priority='Current visible fish and overall watchability; reef ghost footprint optimization deprioritized.',
        long_video_not_rerun=True,video4_dense_coral_deferred=True,seagrass_not_run=True,
        independent_holdout=False,new_flow_model_inference=False,extra_transmission_bytes=0,
        conclusion_zh='全画面当前轮廓修正有明显跨片取舍；运动范围限制后的CSM50保住接近Q9的全帧时序，但鱼区改善较小、礁石小鱼基本未恢复。仅作为温和观看候选。')
    save(run/'round_result.json',result_json)
    rl=lambda label,p:link(label,p,STAGE)
    text=f"""# Q9 当前轮廓保留与运动范围限制（2026-11-12）

本轮按用户新优先级继续研究video1/2/3和礁石斑点鱼，共12组新输出：第一阶段CS50/CS100四片8组，观察到背景回退后登记CSM50补充四片4组。**CSM50保留为温和观看候选，未采用；没有证明鱼消失问题已解决。** Q9_R25仍为研究参照，正式S8I + CURRENT_I12配置不变。

## 1. 用户最新要求与本轮范围

原话：

> {feedback['verbatim']}

本轮按最后一句明确的素材列表执行：video1/2/3为主要观看目标，礁石片仍保留对照，但不继续纠结旧位置残像。优先看当前鱼是否还在、形状和运动是否自然、整段是否能看，不再用清除旧鱼影的误差代替成功。

长视频仍是后续主要应用目标。用户报告大面积珊瑚时画面难看，计划另行裁成video4；**本轮不定位、不裁切、不假定该片段时间，也不运行video4**。未重跑完整长片或海草片。礁石“鱼都快消失”的反馈没有单独指定具体候选档位，不把它写成对所有Q9子档的评价。此前Q9三段正面反馈、video2适度改善目标继续保留。

{rl('结构化最新反馈',run/'user_feedback_20261112.json')}

## 2. 先区分当前信息与旧影

本轮检查了四片原视频、独立原GLC、Q9和接收引导图的第45/61/90帧。礁石小鱼在原GLC和低码率引导中已经有明显的形状/细节损失，后处理只退出旧纹理无法重建本来就没有保存的信息。主鱼仍在，但与原片的形状和纹理有差异。video1/2/3也保留原GLC已有重建偏差。

这是这些采样帧的观察，不代表所有时间点或所有小鱼的统一结论。此轮没有鱼检测器或鱼数量统计，不能将像素误差改善写成“恢复了多少鱼”。

{rl('诊断静图目录中的礁石对照',run/'diagnosis/reef_spotted_fish_current_inputs.png')} · {rl('video2输入对照',run/'diagnosis/video2_current_inputs.png')}

## 3. 第一阶段：将部分当前结构放回Q9

CS50和CS100在生成本轮结果前统一冻结。两者均计算当前原GLC与Q9的RGB残差，做sigma1高斯平滑，逐通道限制在Q9→原GLC的方向和范围内，再分别施加50%或100%修正：

    d = current_raw − Q9
    correction = clamp(Gaussian_sigma1(d), min(d, 0), max(d, 0))
    CS50  = round(Q9 + 0.5 × correction)
    CS100 = round(Q9 + 1.0 × correction)

这是**有界的当前结构修正**，不是精确的频率分割，也不保证语义轮廓正确。Q9细纹理仍被保留，因此旧细节可能留下；原GLC波动也可能重新进入。没有加入鱼掩码、原片信息、人工ROI或新码流。

两种强度用于辨别修正范围和代价，所有四片共用参数，没有按片调整。CS100是较强机制对照，CS50是温和对照；均未自动采用。

{rl('第一阶段冻结计划',run/'execution_plan.json')} · {rl('修正公式代码',STAGE/'current_structure_preservation.py')}

## 4. 透明登记的补充：运动范围限制CSM50

第一阶段显示，CS50在video1/2全帧时序上接近Q9，但video3/礁石仍有较大回退。因此在第一阶段结果完成后，才冻结四组CSM50，不能称它与前两候选一同预登记。补充计划保存第一阶段plan/summary SHA，原输出不改。

CSM50对CS50的修正再乘接收端运动权重。利用已收到引导产生的当前→锚点RAFT流，先估计整幅主导仿射运动，再找偏离该运动的局部区域：

- 在原alpha>0.5的8像素网格采样，最少64点；三次确定性的截尾最小二乘拟合，保留70分位以内或1.5像素内的点。
- 局部流偏差0.75→3像素产生权重，sigma2空间平滑；alpha=0处不改。
- 原始guide流和Q9重放基线均复用，**没有新光流模型推理、没有增加传输字节**。
- CSM50是“局部运动迹象”约束，**不是鱼分割**；视差、流错误、占画面很大的移动物体都可能影响拟合。小鱼若在guide中消失，也可能完全不触发。
- 不新增时间门控或切镜机制，锚点变化处仍可能改变修正范围。该轮未验证全片场景切换。

{rl('补充冻结计划',run/'motion_extension/execution_plan.json')} · {rl('运动限制源码',STAGE/'moving_structure_preservation.py')}

## 5. 跨片结果

所有百分比相对同一Q9_R25。全帧源对齐低频时序误差越低越好；正数表示回退，**不是肉眼闪烁次数或可看程度评分**。

{table(['素材','CS50','CS100','CSM50'],times)}

CSM50取消了全画面修正的大部分背景代价，四片全帧指标接近Q9。它也显著缩小了改动范围，因此不能把“接近Q9”当作已经恢复鱼形。

历史珊瑚ROI的低频时序误差（并非用户未来video4）：

{table(['历史ROI所属素材','CS50','CS100','CSM50'],reefs)}

CSM50仍有video1/2历史珊瑚ROI约+4.10%/+3.07%的代价，不能只报告全帧。video3珊瑚几乎不变。这个新比较基线是Q9，旧Q9数值筛选selected=null及video3珊瑚+5.37%的历史记录使用不同基线，仍原样保留。

全帧PSNR变化，单位dB：

{table(['素材','CS50','CS100','CSM50'],spatial)}

固定当前主体观察区的RGB像素误差，区间为整段0–149帧。宽框包含鱼和背景，不是语义掩码；括号为相对Q9的误差变化，不能称为鱼恢复比例：

{table(['观察区','Q9_R25','CS50','CS100','CSM50'],regions)}

CSM50的video1鱼/水区、video2主鱼和鱼尾误差仅小幅下降；video3当前前景区变化很小，礁石小鱼区基本没改。**礁石小鱼未恢复，video2鱼形问题未解决**，不再围绕这段旧影追加试验。

运动区域诊断：

{table(['素材','Q9动态RGB误差','CSM50动态RGB误差','Q9变化幅度比','CSM50变化幅度比'],dyn)}

动态误差降低，但变化幅度比也略降，所以也不能声称真实运动恢复得更多。该表与PSNR/时序/主体区域一起看，原速主观验收仍待用户反馈。

运动权重覆盖统计，仅用于解释改动规模：

{table(['素材','权重>0像素比例','平均运动权重'],interventions)}

## 6. 视觉检查与观看入口

已查看四片候选的整幅静帧表，帧号30/61/90/120（2.0/4.07/6.0/8.0秒），没有只看旧位置裁剪。

{chr(10).join('- '+x for x in visual['observations_zh'])}

优先看video1/2/3的Q9与CSM50原速双列，左Q9_R25、右CSM50；礁石作为补充。六宫格含原片、原GLC和全部本轮候选，可用于辨认已有重建损失。

{rl('全部观看入口',view/'README.md')} · {rl('video1双列',view/'video1_Q9_vs_CSM50.mp4')} · {rl('video2双列',view/'video2_Q9_vs_CSM50.mp4')} · {rl('video3双列',view/'video3_Q9_vs_CSM50.mp4')}

## 7. 字节与复现核验

{table(['素材','原包实际总kbps','新增传输字节'],rate_rows)}

共12组候选数组完成核验。第一阶段逐帧独立公式复算；补充逐帧重放及纯仿射/局部运动合成检查；当前raw/Q9逐通道界限、alpha=0不变、CSM50改变量不大于CS50均成立。源图、发送收据、用户事件和历史ML3输出不属于接收允许读取的数据；评分另行读取源图。Python读边界是可复现性检查，不是操作系统安全沙箱。

所有Q9和仅作评价对照的ML3指标与前轮一致；CSV汇总、主体宽框误差直接重算一致，所有码包/基线和冻结源码SHA未变。10个10秒/15 FPS观看视频完成尺寸、帧数、时长和抽样面板核验。评分使用无损NPY；MP4为观看转码，不用它的体积计算码率。

{rl('第一阶段核验',run/'verification.json')} · {rl('补充核验',run/'motion_extension/verification.json')} · {rl('媒体核验',run/'previews/verification.json')} · {rl('小证据',evidence/'manifest.json')}

正式采用配置SHA256仍为{EXPECTED_CONFIG}。未提交/推送，未替换部署默认。

## 8. 当前决定与接续

保留CSM50供video1/2/3原速观看，CS50/CS100供机制对照。**selected=null，尚未采用**，不因平均指标接近Q9就代替主观认可。当前输出目标是整体可看、主体保留和运动自然，残像专题降为次要。

下一步先结合这三段的原速观感决定是否保留运动范围限制，再研究当前原GLC重建本身的形状/细节损失；不能把原GLC已经丢失的信息无限寄望于后处理恢复。此处是后续方向，尚未启动新一轮。

长视频整体可看性保留为后续目标；大面积珊瑚问题等用户单独提供video4后再登记实验。未假定切片位置，未提前处理。礁石继续留作普通参考，不再主攻其旧影。本轮已收束，无活动实验。
"""
    put(report,text)
    view.mkdir()
    mediafiles=sorted(p for p in (run/'previews').iterdir() if p.suffix in ('.mp4','.png'))
    assert len(mediafiles)==14
    media_records={}
    for src in mediafiles:
        dst=view/src.name;os.link(src,dst);assert os.path.samefile(src,dst)
        h=sha(src);media_records[src.name]=dict(source=str(src),bytes=src.stat().st_size,sha256=h,hardlink=True)
        if src.suffix=='.mp4':assert h==media['videos'][src.name]['sha256']
    save(view/'media_manifest.json',dict(complete=True,files=media_records))
    vl=lambda label,p:link(label,p,view)
    put(view/'README.md',f"""# Q9 当前轮廓保留：原速观看（2026-11-12）

本轮重心是video1/2/3整体可看和当前主体保留，礁石只作补充，不主攻旧影。CSM50为温和观看候选，**未采用**。长视频大面积珊瑚等用户提供video4后再做。

## 优先看video1/2/3

{chr(10).join('- '+vl(c+'：左Q9_R25 / 右CSM50',view/(c+'_Q9_vs_CSM50.mp4')) for c in ('video1','video2','video3'))}

{vl('礁石补充双列',view/'reef_spotted_fish_Q9_vs_CSM50.mp4')}。以上都是完整10秒、15 FPS原速。先看鱼是否仍在、轮廓与运动是否自然、背景是否更跳，不必只找旧位置残像。

## 原片与候选六宫格

{chr(10).join('- '+vl(NAMES[c],view/(c+'_all_six.mp4')) for c in NAMES)}

布局每行左→右：原视频/原GLC；Q9_R25/CS50；CS100/CSM50。CS50/CS100是全画面当前结构修正；CSM50增加运动范围限制。较强CS100跨片时序回退明显，保留用于解释取舍。

## 主体放大辅助

- {vl('video2当前主鱼，全10秒',view/'video2_current_fish_crop.mp4')}
- {vl('礁石当前主鱼，全10秒',view/'reef_spotted_fish_current_fish_crop.mp4')}

放大四宫格为原视频/原GLC；Q9/CSM50。它们不是旧鱼位置裁剪。

## 静帧

{chr(10).join('- '+vl(NAMES[c],view/(c+'_whole_scene_contact.png')) for c in NAMES)}

CSM50全帧时序接近Q9，但局部改善较小，礁石小鱼未恢复，鱼形问题不能称已解决。静图不是原速验收。

{vl('完整报告',report)} · {vl('结构化结果',run/'round_result.json')} · {vl('第一阶段核验',run/'verification.json')} · {vl('补充核验',run/'motion_extension/verification.json')}

媒体为硬链接，修改前另存副本；不可用观看MP4体积表示语义通信码率。正式采用配置未改。
""")
    for p in entries:
        old=p.read_text(encoding='utf-8-sig')
        top=f"""**最新（2026-11-12）：按用户新要求，以video1/2/3整体可看和当前鱼形保留为主；礁石旧影不再主攻。** 当前研究素材按用户最后一句为video1/2/3与礁石。长视频大面积珊瑚问题已记录，等用户单独提供video4后再实验；本轮未裁片、未跑长片或海草。

本轮当前结构修正8组，加透明登记的运动范围限制4组，共12组完成。CSM50全帧时序接近Q9，但局部改善小、礁石小鱼未恢复，保留为温和观看候选，尚未采用。原Q9三段正面反馈和正式采用配置保持。

{link('本轮报告',report,p.parent)} · {link('video1/2/3观看入口',view/'README.md',p.parent)} · {link('最新用户反馈',run/'user_feedback_20261112.json',p.parent)} · {link('小证据',evidence/'manifest.json',p.parent)}

本轮已收束，无活动实验。后续先看三段整体观感，再决定运动范围限制是否保留；鱼形/细节损失需区分原GLC与后处理，不能只追旧影指标。未提交/推送。

## 既往记录（以下保留当时状态，以页首新优先级为准）
"""
        if p.suffix=='.txt':
            top=f"""GLC接续，2026-11-12：当前轮廓保留本轮完成，四片12组，无活动实验。
最新用户优先级：video1/2/3与长视频整体能看；本轮按最后一句跑video1/2/3与礁石，礁石残像降优先级。
用户反馈礁石鱼快消失，不能只优化旧影。长视频大面积珊瑚待用户另取video4，本轮不处理、不假定时间。
CS50/CS100共8组，运动范围限制CSM50自适应补充4组；CSM50仅温和观看候选，未采用，礁石小鱼未恢复。
保持Q9_R25研究参照及三段正面反馈，正式S8I+CURRENT_I12不改；历史selected=null、video3珊瑚+5.37%不改。
报告：{report.as_posix()}
观看：{(view/'README.md').as_posix()}
运行：{run.as_posix()}
后续先看video1/2/3原速整体，再决定是否保留运动限制；不再主攻礁石旧影。未提交/推送。

此前记录（当时状态，以页首为准）：
"""
            put(p,top+'\n'+old)
        else:
            title,body=old.split('\n',1);put(p,title+'\n\n'+top+'\n'+body.lstrip('\n'))

    state=read(statepath)
    for name in ['latest_experiment','latest_generalization_experiment','latest_diagnosis',
                 'next_experiment_plan','latest_user_perceptual_feedback','next_priority']:
        state['previous_'+name+'_before_current_structure']=copy.deepcopy(state.get(name))
    latest=copy.deepcopy(result_json)
    latest.update(run=str(run),report=str(report),viewing=str(view),verification=str(run/'verification.json'),
                  extension_verification=str(run/'motion_extension/verification.json'),evidence=str(evidence/'manifest.json'))
    state.update(status='Q9_current_structure_round_complete',active_experiment_plan=None,
        latest_experiment=latest,latest_generalization_experiment=copy.deepcopy(latest),
        latest_analysis_report=relative(report,REPO),new_experiment_report=relative(report,REPO),
        latest_experiment_plan=dict(status='completed',path=str(run/'execution_plan.json'),
                                    extension=str(run/'motion_extension/execution_plan.json'),new_outputs=12),
        latest_diagnosis=dict(report=str(report),fish_presence_priority=True,reef_ghost_optimization_deprioritized=True,
             local_motion_constraint_reduces_global_regression=True,small_fish_not_restored=True,
             base_GLC_loss_not_recoverable_by_simple_exit=True),
        latest_user_perceptual_feedback=dict(feedback,type='user_priority_change',
             feedback_path=str(run/'user_feedback_20261112.json'),new_candidate_review='pending'),
        current_session_work='Completed 8 current-structure and 4 adaptive local-motion outputs; preserve watchability on video1/2/3, reef supplemental; video4 deferred.',
        user_latest_decision=feedback['verbatim'],new_experiments_uncommitted=True,
        next_priority=['Review video1/2/3 original-speed Q9/CSM50 for whole-scene watchability and current fish presence.',
                       'Keep reef as secondary reference; do not continue ghost-footprint optimization.',
                       'Separate original GLC reconstruction loss from postprocessing changes.',
                       'Wait for user-supplied video4 dense-coral segment; long-video watchability remains future goal.'],
        next_experiment_plan=dict(status='proposed_not_run',reference='Q9_R25',viewing_candidate='CSM50',
            focus=['video1/2/3 watchability','current fish shape and motion'],reef_ghost_priority='deferred',
            video4=dict(status='awaiting_user_clip',source_path=None,time_interval=None),
            full_long_video_rerun='not_started',parameters_not_chosen=True),
        experiment_execution_gate=dict(status='completed_round',currently_running=False,new_run_started=True,
            automatic_next_run=False,date='2026-11-12'),
        publication=dict(current_changes_committed=False,pushed=False,receipt=str(run/'publication.json'),
            scope='Current-structure four-clip round and revised user priority'))
    state['preferred_research_reference'].update(latest_experiment_report=str(report),
        latest_user_priority_path=str(run/'user_feedback_20261112.json'),
        current_focus='Video1/2/3 watchability and current fish presence; reef ghost work deprioritized; future video4 deferred.',
        viewing_candidate='CSM50',viewing_candidate_not_adopted=True)
    save(statepath,state)
    assert sha(STAGE/'当前采用配置.json')==EXPECTED_CONFIG
    save(run/'publication.json',dict(complete=True,date='2026-11-12',report=str(report),viewing=str(view),evidence=str(evidence),
        new_outputs=12,videos=10,pngs=4,report_sha256=sha(report),publisher_sha256=sha(__file__),
        updated_entry_sha256={str(p):sha(p) for p in entries+[statepath]},current_profile_unchanged=True,committed=False,pushed=False))
    evidence.mkdir(parents=True)
    for src in [p for p in run.rglob('*') if p.is_file() and p.suffix.lower() in ('.json','.csv','.py','.md','.txt')]:
        dst=evidence/'run'/relative(src,run);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for src in [report,Path(__file__),STAGE/'publish_q9_ghost_exit.py']:
        dst=evidence/('report' if src==report else 'publication_source')/src.name
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for src in [p for p in mediafiles if p.suffix=='.png']:
        dst=evidence/'visuals'/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for src in (run/'diagnosis').glob('*.png'):
        dst=evidence/'diagnosis'/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for src in [view/'README.md',view/'media_manifest.json']:
        dst=evidence/'viewing'/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for src in entries+[statepath]:
        dst=evidence/'entries_after'/relative(src,WORKSPACE);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    checked=0
    for p in [report,view/'README.md']+entries:
        text=p.read_text(encoding='utf-8-sig')
        if p in entries:text=text.split('## 既往记录（以下保留当时状态，以页首新优先级为准）')[0]
        for href in re.findall(r'\]\(([^)]+)\)',text):
            target=(p.parent/href.split('#',1)[0]).resolve()
            assert target.exists() or target==evidence/'manifest.json',(str(p),href)
            checked+=1
    for p,h in old_manifests.items():assert sha(p)==h
    for p,h in snapshots.items():assert sha(before/relative(p,WORKSPACE))==h
    for src in mediafiles:assert sha(src)==sha(view/src.name)==media_records[src.name]['sha256']
    for p in evidence.rglob('*.json'):read(p)
    state=read(statepath)
    assert state['active_experiment_plan'] is None and state['latest_experiment']['new_outputs']==12
    assert state['latest_user_perceptual_feedback']['verbatim']==feedback['verbatim']
    qa=dict(complete=True,new_links_checked=checked,old_evidence_unchanged=True,
        media_sha_match=True,strict_json_checked=True,latest_user_priority_preserved=True,
        state_complete_no_active_plan=True,current_config_sha256=sha(STAGE/'当前采用配置.json'))
    save(run/'publication_verification.json',qa)
    shutil.copy2(run/'publication_verification.json',evidence/'run/publication_verification.json')
    files={}
    for p in sorted(evidence.rglob('*')):
        if p.is_file():
            assert p.suffix not in ('.mp4','.npy')
            files[relative(p,evidence)]=dict(bytes=p.stat().st_size,sha256=sha(p))
    save(evidence/'manifest.json',dict(complete=True,date='2026-11-12',files=files,source_run=str(run),
         large_arrays_and_videos_excluded=True,self_excluded=True))
    for name,v in files.items():assert sha(evidence/name)==v['sha256']
    print('PUBLISHED',len(files),'evidence files;',checked,'links checked',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    publish(p.parse_args().run.resolve())

