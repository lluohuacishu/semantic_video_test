"""Publish fixed video4 transfer diagnosis, preserve all previous accepted results."""
import argparse
import copy
import os
from pathlib import Path
import re
import shutil
from collections import Counter
from publish_q9_ghost_exit import sha,read,put,save,relative,link,table,delta,STAGE,REPO,WORKSPACE,EXPECTED_CONFIG

TITLE='video4密集珊瑚_Q9与GS25固定参数诊断_20261113.md'
NAMES=('raw','guide','Q9_R25','GS25')
HISTORY='## 此前记录（保留当时状态，以页首video4结果为准）'

def publish(run):
    plan=read(run/'execution_plan.json');summary=read(run/'summary.json')
    reg=read(run/'case_registration.json');verify=read(run/'verification.json')
    media=read(run/'previews/verification.json');visual=read(run/'visual_inspection.json')
    imported=read(run/'source_import_receipt.json');ep=read(run/'evaluation_plan.json')
    assert summary['complete'] and verify['complete'] and media['complete']
    assert media['video_count']==5 and len(visual['images_actually_viewed'])==5
    assert reg['frames']==104 and sha(STAGE/'当前采用配置.json')==EXPECTED_CONFIG
    assert sha(imported['archived_source_path'])==reg['source_video_sha256']
    for p in (plan,ep):
        for n,h in p['code_sha256'].items():assert sha(STAGE/n)==h
    for n,h in plan['input_sha256'].items():assert sha(n)==h
    report=STAGE/TITLE
    view=WORKSPACE/'05_成果与观看/GLC_video4密集珊瑚_20261113'
    evidence=STAGE/'evidence_20261113/video4_fixed_diagnosis'
    statepath=REPO/'05_资料与索引/会话收束_20261110/接续状态.json'
    entries=[REPO/n for n in ('README.md','交接文档.md','当前项目状态与接续.md','项目总结与未来方向.md',
        '新会话接续.txt','05_资料与索引/会话收束_20261110/文件索引.md')]
    entries += [STAGE/n for n in ('README.md','当前方案总结与瓶颈路线_20261111.md','下一阶段_海洋素材与长视频泛化计划_20261111.md')]
    entries += [WORKSPACE/'02_实验数据/README.md',WORKSPACE/'05_成果与观看/README.md',WORKSPACE/'01_项目/GLC后处理/README.md']
    for p in (report,view,evidence,run/'publication.json'):assert not p.exists(),p
    before=run/'publication_before';before.mkdir()
    old_manifests={str(p):sha(p) for folder in STAGE.glob('evidence_*') for p in folder.rglob('manifest.json')}
    before_hash={}
    for p in entries+[statepath]:
        dst=before/relative(p,WORKSPACE);dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dst);before_hash[str(p)]=sha(p)
    save(before/'snapshot_manifest.json',dict(files=before_hash,old_manifests=old_manifests))

    scores=summary['scores'];full=scores['full'];coral=scores['coral_lower'];turtle=scores['turtle_wide']
    changes=dict(Q9_full_temporal_vs_raw_percent=delta(full['Q9_R25']['coarse_mae'],full['raw']['coarse_mae']),
        GS25_full_temporal_vs_Q9_percent=delta(full['GS25']['coarse_mae'],full['Q9_R25']['coarse_mae']),
        GS25_coral_temporal_vs_Q9_percent=delta(coral['GS25']['coarse_mae'],coral['Q9_R25']['coarse_mae']),
        GS25_coral_RGB_MAE_vs_Q9_percent=delta(coral['GS25']['rgb_mae'],coral['Q9_R25']['rgb_mae']),
        Q9_coral_coarse_spatial_vs_raw_percent=delta(coral['Q9_R25']['coarse_spatial_mae'],coral['raw']['coarse_spatial_mae']),
        GS25_turtle_wide_RGB_MAE_vs_Q9_percent=delta(turtle['GS25']['rgb_mae'],turtle['Q9_R25']['rgb_mae']),
        GS25_full_PSNR_minus_Q9_db=full['GS25']['psnr_db']-full['Q9_R25']['psnr_db'],
        GS25_dynamic_amplitude_vs_Q9_percent=delta(summary['dynamic']['GS25']['change_amplitude_ratio'],
                                                 summary['dynamic']['Q9_R25']['change_amplitude_ratio']))
    result=dict(date='2026-11-13',complete=True,case='video4',frames=104,fps=15,
        source_video_seconds=6.9,billing_seconds=104/15,new_post_outputs=2,
        reference='Q9_R25',viewing_candidate='GS25',selected=None,automatic_adoption=False,
        fixed_transfer_not_new_parameter_search=True,development_failure_clip=True,independent_holdout=False,
        base_kbps=summary['rates']['base_kbps'],guide_kbps=summary['rates']['guide_kbps'],
        total_kbps=summary['rates']['total_kbps'],extra_GS25_transmission_bytes=0,
        source_archive=imported['archived_source_path'],source_sha256=reg['source_video_sha256'],
        changes=changes,source_reconstruction_shape_error_precedes_Q9=True,
        Q9_temporal_gain_not_same_frame_shape_recovery=True,guide_texture_insufficient=True,
        GS25_motion_amplitude_attenuation=True,subjective_review='pending',
        no_video4_matched_blur_control=True,previous_three_clips_regression_exact=True,
        formal_adopted_profile_unchanged=True,full_long_video_not_rerun=True,
        conclusion_zh='video4原GLC已有珊瑚空间形态和主体细节偏差，Q9降低时序误差但未恢复形状；GS25有限修正、仍有动态幅度代价，尚未解决密集珊瑚观感。',
        next_focus_zh='在三片回归约束下区分基础重建空间失真、guide细节不足和历史复用新增形态误差；再检验一个局部结构可靠性假设，不能只加大平滑。')
    save(run/'round_result.json',result)
    num=lambda x:f'{x:.6f}' if x is not None else '缺测'
    full_rows=[[n,num(full[n]['coarse_mae']),f"{full[n]['psnr_db']:.4f}",num(full[n]['rgb_mae']),
                num(summary['dynamic'][n]['RGB_delta_error']),f"{summary['dynamic'][n]['change_amplitude_ratio']:.4f}"] for n in NAMES]
    coral_rows=[[n,num(coral[n]['coarse_mae']),num(coral[n]['rgb_mae']),num(coral[n]['coarse_spatial_mae']),
                 num(coral[n]['fine_spatial_mae']),num(coral[n]['fine_amplitude'])] for n in ('source',)+NAMES]
    turtle_rows=[[n,num(turtle[n]['rgb_mae']),num(turtle[n]['coarse_spatial_mae']),num(turtle[n]['fine_spatial_mae'])] for n in NAMES]
    missing=Counter(x['scope'] for x in summary['low_support'])
    event_rows=[[k,','.join(map(str,v['raw']['frames']))]+[num(v[n]['coarse_mae']) for n in ('raw','Q9_R25','GS25')]
                for k,v in summary['events'].items()]
    rl=lambda label,p:link(label,p,STAGE)
    put(report,f"""# video4 密集珊瑚：固定Q9与GS25诊断（2026-11-13）

**本轮已完成整段video4的固定参数对照；没有为video4另调阈值。** 新片的原GLC中已存在珊瑚轮廓与局部纹理偏差，Q9压低时序误差但未恢复真实形状；GS25进一步改善部分空间/时序指标，同时压低动态幅度。保留GS25供观看，selected=null，尚未采用或宣称密集珊瑚问题解决。

{rl('原速观看',view/'README.md')} · {rl('结构化结果',run/'round_result.json')} · {rl('核验',run/'verification.json')} · {rl('小证据',evidence/'manifest.json')}

## 1. 素材登记与长度处理

用户授权：“好了，我把video4整过去了，你可以开始下一轮了”。

文件实际位于桌面临时目录，搜索后找到，路径询问已解决。原文件保留，并按字节复制到项目素材目录：

{rl('项目内video4原片',Path(imported['archived_source_path']))}

源文件SHA256：{reg['source_video_sha256']}，24,696,220 B，1280×720，H.264，30 FPS，207视频帧，视频流6.900秒；容器含音频时长6.911896秒。本轮仅处理视频。

整段按时间戳重采样为15 FPS并缩至512×288，实际104帧；按输出帧数计时为6.933333秒，末帧时间戳6.866667秒。与源视频流时长差1/30秒属于固定帧率端点取整，**没有补到150帧、没有重复尾帧凑10秒、没有改变播放帧率来慢放**。104帧均来自整段抽帧，103个相邻帧均不同。

这是用户主动提供的密集珊瑚失败案例，含海龟主体；参与诊断后属于开发素材，不称为随机独立留出。

## 2. 固定方法与短片适配

基础GLC继续q=0、intra_period=64、reset_interval=32、独立FP32发送/接收，参考校验不变。只把原150帧文件接口的有效长度改为104；新基础流重新编码，不是从旧长片重建输出截取。

引导仍256×144/15 FPS、x265 slower两遍、内部目标9 kbps、keyint=150、scenecut=0。使用已有GLCGDE02变长包，实际104帧、起点0、总帧数104，绑定本次完整基础包SHA并带CRC。只有一个自描述包，接收解码不需要额外传输索引。

Q9保持R25门控、RAFT FP32/12次、无TF32、2像素羽化、6帧旧参考过渡及原两次取整。原锚计划按有效长度截为0/15/45/75，未人为减少有效范围内锚点。未来105/135锚点不生成；18个有效过渡事件按实际长度存储。

GS25保持Q9 + 0.25 × Gaussian_sigma1(guide − Q9)，末端裁剪取整。guide RGB直接参与GS25输出，这是上一轮已经登记的研究像素来源变化；Q9仍只复用原GLC像素。源图不进入两者接收输出。

本轮未加入新的局部权重、切镜检测、时间滤波或video4专属参数，也未重跑完整82.6秒长片。

## 3. 首先定位到基础空间重建问题

已查看源片首/中/尾、0/26/52/78/103帧全图，以及26/52/78帧珊瑚和海龟宽框。

- 第0帧原GLC已出现与原片不同的珊瑚团块轮廓、局部细部和海龟鳍部形态；此时Q9没有可复用的更早鱼/珊瑚参考，说明这些静态偏差并非全由Q9历史复用产生。
- Q9后续仍沿用很多基础重建中的错误细部；静帧中不能看到完整形态恢复。
- guide单独明显模糊，局部有块状/条带失真，后段尤其可见。它能够补充部分低频信息，但并没有足够细节直接恢复原片珊瑚。
- GS25改动温和，仍保留Q9大量细节与形状偏差。静图没有代替原速观感验收。

这些是本片抽样观察与下述指标共同支持的定位，不代表已经证明基础网络内哪一模块造成失真，也不能据此归纳所有珊瑚视频的统一成因。

## 4. 全帧结果

{table(['方法','低频时序MAE','PSNR dB','同帧RGB MAE','动态RGB误差','动态幅度/源'],full_rows)}

Q9相对原GLC低频时序误差变化{changes['Q9_full_temporal_vs_raw_percent']:+.2f}%，但PSNR几乎不变、RGB空间误差略升。GS25相对Q9低频时序误差变化{changes['GS25_full_temporal_vs_Q9_percent']:+.2f}%，PSNR增加{changes['GS25_full_PSNR_minus_Q9_db']:.4f} dB。

动态幅度比从Q9的{summary['dynamic']['Q9_R25']['change_amplitude_ratio']:.4f}降至GS25的{summary['dynamic']['GS25']['change_amplitude_ratio']:.4f}（相对变化{changes['GS25_dynamic_amplitude_vs_Q9_percent']:+.2f}%）。此比值是源高变化像素上的像素变化总量比，不是海龟游动速度。不能把幅度更小直接写成运动更自然。

guide单独PSNR更高，但明显模糊、动态幅度低，不能按此直接替换Q9。

## 5. 密集珊瑚与海龟区域

珊瑚宽框在推理前按源图登记为x=0..512、y=128..288，包含海龟和部分背景，不是语义分割。

{table(['方法','低频时序MAE','同帧RGB MAE','低频空间MAE','细纹理空间MAE','细纹理幅度'],coral_rows)}

这里的“时序”比较前后帧经源光流对齐后的变化；“空间”比较同一时间的输出与原片。二者不可混用。

Q9的珊瑚低频空间误差相对原GLC上升{changes['Q9_coral_coarse_spatial_vs_raw_percent']:.2f}%，同时它的时序误差下降，体现“稳定并不等于形状正确”。GS25相对Q9的珊瑚低频时序误差下降{-changes['GS25_coral_temporal_vs_Q9_percent']:.2f}%，RGB误差下降{-changes['GS25_coral_RGB_MAE_vs_Q9_percent']:.2f}%。

源珊瑚细纹理幅度5.7909，原GLC为5.6460，幅度数量接近，但原GLC细纹理空间误差仍为7.9206。该结果与静帧说明后续需要检查细节是否对应真实结构，不能仅用细纹理幅度判断清晰度或正确性。guide幅度仅1.5023，也不能承担真实高频补全。

海龟所在宽框x=50..410、y=150..288（仍包含大量珊瑚背景）：

{table(['方法','RGB MAE','低频空间MAE','细纹理空间MAE'],turtle_rows)}

GS25该宽框RGB误差相对Q9下降{-changes['GS25_turtle_wide_RGB_MAE_vs_Q9_percent']:.2f}%，不是海龟恢复比例；鳍部和主体形态仍有偏差。本轮没有目标检测、分割或形状恢复计数。

## 6. 时间事件与缺测

{table(['事件组','目标帧','原GLC低频时序','Q9','GS25'],event_rows)}

原GLC最大全帧峰值位于64帧/4.27秒，恰与基础I帧及重置事件重合；Q9/GS25仍有峰值。这里只记录事件关联，不断言唯一成因。Q9后段94/98帧等也有较高误差，不能只修某一个锚点。

全帧、珊瑚和海龟宽框103个帧对全部有效。低纹理水域water_upper有52个帧对公共支持不足64，top_left有14个；均对所有方法同步排除，记录为缺测，未用0代替。所有同帧空间指标仍使用全部104帧和区域像素。

本片少于10秒、仅一个guide包，**本轮没有跨guide分块边界，也没有验证长片切镜策略**。旧长片基础输入采用TRT FP16，而本次源片重新用FP32编码且参考时间线从0开始；不能把本次短片与旧长片视觉差别直接归因于GS25。

## 7. 实际码率与核验

基础23,556 B /27.1800 kbps；辅助9,619 B /11.0988 kbps；合计33,175 B /38.2788 kbps。所有基础包头、辅助头和CRC计入，按104/15秒计费，两路分别小于40 kbps。GS25相对同一Q9传输新增0字节。观看MP4体积不是通信码率。

为避免短片接口改动带来算法漂移，已完成：

1. 旧video1完整150帧重新基础编码/独立解码，与冻结原基础包及原RGB逐位一致。
2. 旧video1前104帧重新计算RAFT特征与光流，与旧缓存前缀逐位一致。
3. video1/2/3各前104帧重放Q9和GS25，RGB/alpha/shared及GS25输出与前轮对应前缀逐位一致。
4. 新变长评分器在旧video1完整150帧上，对全部既有范围与五种方法的原有空间/时序指标和动态指标逐项完全一致。
5. video4源重新抽帧一致、基础参考校验、guide重新解码、接收禁读、Q9重放、GS25独立公式及所有范围CSV再汇总通过。
6. 五个观看视频核对104帧、15 FPS、6.933秒、尺寸与抽样面板；静图已查看。

未测本轮端到端速度；没有为video4重跑等细节模糊控制。因此不能把上一轮三片的模糊控制结果直接当作video4已排除平滑贡献。原video1/2/3的Q9正面反馈保持，静态核验不是用户原速验收。

{rl('完整核验',run/'verification.json')} · {rl('长度适配回归',run/'regression/prefix_regression.json')} · {rl('媒体核验',run/'previews/verification.json')} · {rl('静图记录',run/'visual_inspection.json')}

## 8. 本轮结论与下一步

video4固定参数验证支持GS25有有限的跨素材指标收益，仍未解决密集珊瑚的真实形态与观感。**不将两段小数改善直接变成采用结论；正式配置保持，GS25仅供观看。**

后续重点应转向“错误纹理是否被持续保留、当帧结构何处可信”。先在固定Q9/GS25参照下区分基础空间重建失真、guide高频不足和历史复用新增误差，再检验一个局部结构可靠性假设，并回归video1/2/3。若关键结构在两路输入均缺失，再进入辅助表示或基础重建分支；单纯加大融合强度不能保证恢复。

完整长片、切镜及跨包连续性仍按路线单独验证；礁石旧影不升级为本轮主攻。当前已收束，无活动实验，未提交/推送。

正式配置SHA256：{EXPECTED_CONFIG}。历史selected=null和video3珊瑚+5.37%保留原比较口径；本轮GS25百分比以Q9为基线。
""")

    view.mkdir()
    files=sorted([p for p in (run/'previews').iterdir() if p.suffix in ('.mp4','.png')]+
                 [run/'source_contact.png',run/'diagnostic_frame52.png'])
    assert len(files)==10
    manifests={}
    for src in files:
        dst=view/src.name;os.link(src,dst);assert os.path.samefile(src,dst)
        h=sha(src)
        if src.suffix=='.mp4':assert h==media['videos'][src.name]['sha256']
        manifests[src.name]=dict(source=str(src),bytes=src.stat().st_size,sha256=h,hardlink=True)
    save(view/'media_manifest.json',dict(complete=True,files=manifests,video_count=5,png_count=5))
    vl=lambda label,p:link(label,p,view)
    put(view/'README.md',f"""# video4 密集珊瑚：原速观看

完整处理用户提供的6.9秒海龟/珊瑚片段，30→15 FPS后104帧，观看6.933秒；未补到10秒。固定Q9与GS25，没有video4专属调参。总码率38.2788 kbps，GS25无额外传输字节，尚未采用。

- {vl('优先：左Q9 / 右GS25',view/'video4_Q9_vs_GS25.mp4')}
- {vl('全画面四宫格：原片 / 原GLC；Q9 / GS25',view/'video4_source_raw_Q9_GS25.mp4')}
- {vl('密集珊瑚放大四宫格',view/'video4_coral_crop.mp4')}
- {vl('海龟所在宽框放大四宫格',view/'video4_turtle_crop.mp4')}
- {vl('原片 / 收到的引导视频',view/'video4_source_vs_guide.mp4')}

放大四宫格布局同全画面：上原片/原GLC，下Q9/GS25。宽框包含背景，不是语义分割。

GS25相对Q9珊瑚区低频时序误差-18.23%、RGB误差-8.17%，但动态幅度继续降低。原GLC已经存在珊瑚形态偏差，GS25没有完整恢复；重点看细部形状、海龟鳍部及运动，不仅看稳定程度。用户原速反馈待记录。

{vl('完整报告',report)} · {vl('五列全图静帧',view/'video4_whole_scene_contact.png')} · {vl('珊瑚静帧',view/'video4_coral_contact.png')} · {vl('海龟静帧',view/'video4_turtle_contact.png')} · {vl('核验',run/'verification.json')}

全部对照共享同一新编码基础流。本次短片不等同旧82.6秒长片的同位置重建结果，未验证长片分块/切镜。视频为硬链接，编辑前另存副本；MP4大小不能作为通信码率。
""")

    for p in entries:
        old=p.read_text(encoding='utf-8-sig')
        top=f"""**最新（2026-11-13）：video4密集珊瑚固定参数诊断完成。** 整段6.9秒源片按15 FPS处理为104帧，未补到10秒；Q9与GS25两组输出及旧三片长度适配回归均完成。

原GLC已出现珊瑚形态/纹理偏差；Q9降低时序误差但未恢复形状。GS25相对Q9珊瑚低频时序误差-18.23%、RGB误差-8.17%，同时动态幅度降低，仍仅作观看候选。总38.2788 kbps，GS25无新增传输字节。

{link('video4报告',report,p.parent)} · {link('原速与放大对照',view/'README.md',p.parent)} · {link('项目内源片',Path(imported['archived_source_path']),p.parent)} · {link('小证据',evidence/'manifest.json',p.parent)}

video4已收到并登记，本轮已收束，无活动实验；它是开发失败案例，不是独立留出。接下来区分基础重建空间失真、guide细节不足与历史复用新增误差，再检验局部结构可靠性。video1/2/3原Q9正面反馈、正式采用配置和旧数值筛选均保留；未重跑长片、未提交/推送。

{HISTORY}
"""
        if p.suffix=='.txt':
            top=f"""GLC接续，2026-11-13：video4固定Q9与GS25诊断完成，无活动实验。
整段源6.9秒，30→15 FPS为104帧，按6.933秒计费；未补帧凑10秒。video4已收到。
原GLC已有珊瑚形态偏差；Q9稳定收益不等于恢复形状。GS25珊瑚时序-18.23%、RGB误差-8.17%，动态幅度也降。
Q9与GS25传输总38.2788 kbps；GS25仅观看候选，未采用。旧三片Q9/GS25前104帧回归逐位一致。
优先研究空间结构与可靠纹理复用；视频4属开发失败案例。长片本轮未重跑。
正式S8I+CURRENT_I12及三段Q9正面反馈保持；历史selected=null、video3珊瑚+5.37%不改。
报告：{report.as_posix()}
观看：{(view/'README.md').as_posix()}
源片：{Path(imported['archived_source_path']).as_posix()}
运行：{run.as_posix()}
未提交/推送。

此前记录（以下保留当时状态）：
"""
            put(p,top+'\n'+old)
        else:
            title,body=old.split('\n',1);put(p,title+'\n\n'+top+'\n'+body.lstrip('\n'))

    state=read(statepath)
    for key in ('latest_experiment','latest_generalization_experiment','latest_diagnosis',
                'next_experiment_plan','next_priority','user_latest_decision'):
        state['previous_'+key+'_before_video4']=copy.deepcopy(state.get(key))
    latest=dict(result,run=str(run),report=str(report),viewing=str(view),
                verification=str(run/'verification.json'),evidence=str(evidence/'manifest.json'))
    state.update(status='Q9_video4_fixed_diagnosis_complete',active_experiment_plan=None,
        latest_experiment=latest,latest_generalization_experiment=copy.deepcopy(latest),
        latest_analysis_report=relative(report,REPO),new_experiment_report=relative(report,REPO),
        latest_experiment_plan=dict(status='completed',path=str(run/'execution_plan.json'),
            new_post_outputs=2,short_length_regression_completed=True),
        latest_diagnosis=dict(report=str(report),dense_coral_base_shape_error=True,
            Q9_temporal_gain_not_shape_recovery=True,GS25_limited_gain_and_motion_tradeoff=True,
            no_generalization_success_claim=True),
        video4_registration=dict(status='fixed_Q9_GS25_diagnosis_complete',source_path=imported['archived_source_path'],
            original_path=reg['source_video'],sha256=reg['source_video_sha256'],frames=104,
            source_seconds=6.9,billing_seconds=104/15,report=str(report),development_clip=True),
        current_session_work='Completed whole-video4 fixed Q9/GS25 diagnosis, short-length regressions and 5 original-speed comparisons.',
        user_latest_decision=plan['authorization'],new_experiments_uncommitted=True,
        next_priority=['Review video4 original-speed coral and turtle comparisons alongside existing video1/2/3.',
            'Separate base spatial reconstruction loss, guide detail loss, and historical texture reuse errors.',
            'Test one receiver-local structure reliability hypothesis with video1/2/3 regression.',
            'After freezing quality profile, separately check long-video cuts, block continuity and independent heldout clips.',
            'Bound CPU memory and remeasure end-to-end speed after quality direction stabilizes.'],
        next_experiment_plan=dict(status='proposed_not_run',reference='Q9_R25',viewing_candidate='GS25',
            focus=['dense coral spatial fidelity','current subject shape','reliable detail reuse','motion preservation'],
            video4=dict(status='registered_first_diagnosis_complete',source_path=imported['archived_source_path'],
                source_sha256=reg['source_video_sha256'],source_interval='entire 6.9s clip',development_case=True),
            video123_regression_required=True,parameters_not_chosen=True,full_long_video_rerun='not_started'),
        experiment_execution_gate=dict(status='completed_round',currently_running=False,new_run_started=True,
            automatic_next_run=False,date='2026-11-13'),
        publication=dict(current_changes_committed=False,pushed=False,receipt=str(run/'publication.json'),
            scope='Video4 fixed transfer diagnosis and variable-length regression'))
    state['preferred_research_reference'].update(latest_experiment_report=str(report),
        current_focus='Video1/2/3 and video4 watchability; base spatial errors and current-structure reliability.',
        viewing_candidate='GS25',viewing_candidate_not_adopted=True,
        video4_diagnosis_report=str(report),video4_user_original_speed_feedback='pending')
    if 'latest_strategy_summary' in state:
        state['latest_strategy_summary']['progress_after_summary']=dict(video4_received=True,
            fixed_diagnosis_completed=True,latest_report=str(report))
    save(statepath,state)
    save(run/'publication.json',dict(complete=True,date='2026-11-13',report=str(report),viewing=str(view),
        evidence=str(evidence),source_archive=imported['archived_source_path'],videos=5,pngs=5,new_post_outputs=2,
        report_sha256=sha(report),publisher_sha256=sha(__file__),
        updated_entry_sha256={str(p):sha(p) for p in entries+[statepath]},
        current_profile_unchanged=True,committed=False,pushed=False))
    evidence.mkdir(parents=True)
    for src in [p for p in run.rglob('*') if p.is_file() and p.suffix.lower() in ('.json','.csv','.py','.md','.txt')]:
        dst=evidence/'run'/relative(src,run);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for src in (report,Path(__file__),STAGE/'publish_q9_ghost_exit.py'):
        dst=evidence/('report' if src==report else 'publication_source')/src.name
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for src in [p for p in files if p.suffix=='.png']:
        dst=evidence/'visuals'/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for src in (view/'README.md',view/'media_manifest.json'):
        dst=evidence/'viewing'/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for src in entries+[statepath]:
        dst=evidence/'entries_after'/relative(src,WORKSPACE);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    links=0
    for p in [report,view/'README.md']+entries:
        text=p.read_text(encoding='utf-8-sig')
        if p in entries:text=text.split(HISTORY)[0].split('此前记录（以下保留当时状态）：')[0]
        for href in re.findall(r'\]\(([^)]+)\)',text):
            target=(p.parent/href.split('#',1)[0]).resolve()
            assert target.exists() or target==evidence/'manifest.json',(p,href)
            links+=1
    for p,h in old_manifests.items():assert sha(p)==h
    for p,h in before_hash.items():assert sha(before/relative(p,WORKSPACE))==h
    for src in files:assert sha(src)==sha(view/src.name)==manifests[src.name]['sha256']
    for p in evidence.rglob('*.json'):read(p)
    final=read(statepath);old=read(run/'state_before_run.json')
    assert final['active_experiment_plan'] is None and not final['experiment_execution_gate']['currently_running']
    assert final['latest_user_perceptual_feedback']==old['latest_user_perceptual_feedback']
    assert final['video4_registration']['sha256']==reg['source_video_sha256']
    assert sha(STAGE/'当前采用配置.json')==EXPECTED_CONFIG
    qa=dict(complete=True,links_checked=links,old_manifests_unchanged=True,
        media_hashes_match=True,source_archive_bit_exact=True,prior_subjective_feedback_preserved=True,
        video4_registered_and_state_complete=True,no_active_experiment=True,current_config_sha256=EXPECTED_CONFIG)
    save(run/'publication_verification.json',qa);shutil.copy2(run/'publication_verification.json',evidence/'run/publication_verification.json')
    manifest={}
    for p in sorted(evidence.rglob('*')):
        if p.is_file():
            assert p.suffix not in ('.npy','.npz','.mp4','.glcsnd','.glcg2')
            manifest[relative(p,evidence)]=dict(bytes=p.stat().st_size,sha256=sha(p))
    save(evidence/'manifest.json',dict(complete=True,date='2026-11-13',source_run=str(run),
        files=manifest,large_arrays_and_videos_excluded=True,self_excluded=True))
    for n,v in manifest.items():assert sha(evidence/n)==v['sha256']
    print('VIDEO4 PUBLISHED',len(manifest),'evidence files;',links,'links;',len(files),'media files',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    publish(p.parse_args().run.resolve())

