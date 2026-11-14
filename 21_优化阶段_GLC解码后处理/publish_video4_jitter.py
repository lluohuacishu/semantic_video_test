"""Publish checked video4 jitter candidates; preserve all earlier experiments."""
import argparse
import csv
import os
from pathlib import Path
import re
import shutil
from runtime_speed_common import STAGE,REPO,DATA,file_sha,load_json,save_json,check_inputs
from run_video4_jitter import CONFIG

TITLE='video4抖动_三帧短参考实验_20261113.md'


def pct(value,reference):return (value/reference-1)*100
def link(origin,target):return os.path.relpath(target,origin.parent).replace('\\','/')


def publish(run):
    assert load_json(run/'verification.json')['complete']
    assert load_json(run/'previews/verification.json')['complete']
    assert load_json(run/'visual_inspection.json')['complete']
    plan=load_json(run/'execution_plan.json');check_inputs(plan['input_sha256'])
    assert file_sha(STAGE/'当前采用配置.json')==CONFIG
    all_results=load_json(run/'summary.json')['cases'];s=all_results['video4'];reg=load_json(run/'case_registration.json')['cases']
    root=REPO.parents[2];view=root/'05_成果与观看/GLC_video4抖动三帧对照_20261113'
    view.mkdir(exist_ok=False);report=STAGE/TITLE
    evidence=STAGE/'evidence_20261113/video4_short_history_jitter';evidence.mkdir(exist_ok=False)
    media={}
    for src in sorted((run/'previews').iterdir()):
        if src.suffix.lower() not in ('.mp4','.png'):continue
        dst=view/src.name;shutil.copy2(src,dst)
        assert file_sha(src)==file_sha(dst)
        media[src.name]=dict(source=str(src),bytes=dst.stat().st_size,sha256=file_sha(dst),hardlink=False)
    save_json(view/'media_manifest.json',dict(complete=True,files=media))
    rates={}
    for case,r in reg.items():
        seconds=r['frames']/r['fps'];b=Path(r['base']).stat().st_size;g=Path(r['packet']).stat().st_size
        rates[case]=dict(base_bytes=b,guide_bytes=g,seconds=seconds,base_kbps=b*8/seconds/1000,guide_kbps=g*8/seconds/1000,total_kbps=(b+g)*8/seconds/1000,extra_bytes=0)
    changes={}
    for case,r in all_results.items():
        changes[case]={}
        for name in ('RG25','SR3','SQ3','BLUR_SQ3'):
            a=r['scores']['full'][name];b=r['scores']['full']['GS25'];j=r['jitter']['full'];d=r['dynamic']
            changes[case][name]=dict(coarse_percent=pct(a['coarse_mae'],b['coarse_mae']),fine_percent=pct(a['fine_mae'],b['fine_mae']),
                abrupt_error_percent=pct(j[name]['full_accel_mae'],j['GS25']['full_accel_mae']),
                texture_amplitude_percent=pct(a['fine_amplitude'],b['fine_amplitude']),
                dynamic_amplitude_percent=pct(d[name]['change_amplitude_ratio'],d['GS25']['change_amplitude_ratio']),
                rgb_percent=pct(a['rgb_mae'],b['rgb_mae']),psnr_change_db=a['psnr_db']-b['psnr_db'])
    result=dict(complete=True,date='2026-11-13',cases=list(all_results),reference='GS25',profiles=['RG25','SR3','SQ3'],
        viewing_candidates=['SQ3','SR3'],selected=None,automatic_adoption=False,current_config_sha256=CONFIG,
        primary_viewing_candidate='SQ3',SR3_universal_replacement_rejected_by_video3_regression=True,
        changes=changes,rates=rates,new_outputs=16,includes_ablation_and_matched_blur=True,
        user_video4_previous_result='Frame-wide jitter unacceptable; worse perceptual type than raw GLC wobble.',
        visual_acceptance='pending',new_jitter_proxy_not_human_rating=True,
        conclusion='Finite aligned three-frame averaging reduces multiple temporal error proxies but removes texture and dynamic amplitude; two watch candidates, not a validated fix.',
        run=str(run),report=str(report),viewing=str(view),evidence=str(evidence/'manifest.json'))
    save_json(run/'round_result.json',result)
    lines=[
      '# video4 抖动：三帧短参考实验（2026-11-13）','',
      '**本轮已完成video4及video1/2/3固定参数回归。SQ3、SR3仅供观看，尚未采用。**',
      '',f'[原速观看]({link(report,view/"README.md")}) · [结构化结果]({link(report,run/"round_result.json")}) · [核验]({link(report,run/"verification.json")})','',
      '## 1. 用户反馈改变验收重点','',
      '> 想想办法处理呢，video4因为珊瑚面积大，效果太差了，整个画面是真的在抖动，不像原始glc那样晃动','',
      '将此前video4结果记录为原速观感未通过。用户没有指定Q9还是GS25，故两者均保留为参照。此前平均时序分数改善不等于消除了这种抖动，不能用分数否定反馈。video1/2/3历史Q9正面反馈不变。','',
      '## 2. 机制线索与本轮假设','',
      'Q9逐帧在当前GLC和较旧的变形参考之间切换。guide相邻帧对应可信的珊瑚宽框内，历史权重跨越0.5的像素占比平均约9.41%；这些位置Q9与当前raw的RGB绝对差均值约6.60/255。旧锚点光流与相邻光流复合路径也存在差异。这些是可能的纹理跳换线索，不证明抖动全由门控或光流造成，也不是9.41%的整幅画面被判定错误。','',
      '本轮检验：避免只加大全局guide混合，改为先按收到的guide对齐最近帧，再进行有限长度融合。保留关闭旧锚点的消融，防止把基础GLC自身变化误认为只有Q9造成。','',
      '| 名称 | 处理方式 | 历史范围 |','| --- | --- | --- |',
      '| RG25 | 当前原GLC加原25% guide低频修正 | 无旧锚点；消融控制 |',
      '| SR3 | 对齐当前及最近两帧原GLC，融合后加25% guide低频修正 | 最长2帧，约0.133秒 |',
      '| SQ3 | 对齐当前及最近两帧已有GS25输出，再融合 | 新增融合仅2帧；输入仍含原Q9长参考 |','',
      '三帧名义权重0.5/0.3/0.2；依据guide双向光流一致性、光度差和边缘距离连续衰减历史权重，再归一化。当帧占比至少50%；使用未滤波的历史输入，不递归回灌输出。四片参数完全相同，无手工珊瑚mask或逐片调参。','',
      '两种新候选仍可能模糊细节或形成短尾；有限窗口不等于已证明没有残像。未增加切镜处理，SQ3也没有消除其输入中已有的长参考残像。','',
      '## 3. video4全画面结果','',
      '| 方法 | 低频时序MAE | 细纹理时序MAE | 突变误差代理 | RGB MAE | 细纹理幅度 | 动态幅度/源 |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in ('raw','Q9_R25','GS25','RG25','SR3','SQ3','BLUR_SQ3'):
        a=s['scores']['full'][name];j=s['jitter']['full'][name];d=s['dynamic'][name]
        lines.append(f"| {name} | {a['coarse_mae']:.4f} | {a['fine_mae']:.4f} | {j['full_accel_mae']:.4f} | {a['rgb_mae']:.4f} | {a['fine_amplitude']:.4f} | {d['change_amplitude_ratio']:.4f} |")
    c=changes['video4']['SQ3'];b=s['scores']['full']['BLUR_SQ3'];a=s['scores']['full']['SQ3'];j=s['jitter']['full']
    lines += ['',f"SQ3相对GS25：低频时序{c['coarse_percent']:+.2f}%，细纹理时序{c['fine_percent']:+.2f}%，突变代理{c['abrupt_error_percent']:+.2f}%。代价是细纹理幅度{c['texture_amplitude_percent']:+.2f}%，动态幅度{c['dynamic_amplitude_percent']:+.2f}%。不能把这些直接写成肉眼抖动改善比例。",'',
      '新增“突变误差代理”为源光流对齐后的输出变化误差再作相邻差分；大变化比例使用灰度变化误差绝对值>16的共同有效像素。它不是摄像机抖动位移，也不是经过主观校准的质量分。其基准仍把Q9/GS25排在raw之前，说明它不能单独解释用户反馈的抖动类型。','',
      f"相同全帧平均细纹理幅度的空间模糊对照BLUR_SQ3中，SQ3仍有低频时序{pct(a['coarse_mae'],b['coarse_mae']):+.2f}%、细纹理时序{pct(a['fine_mae'],b['fine_mae']):+.2f}%、突变代理{pct(j['SQ3']['full_accel_mae'],j['BLUR_SQ3']['full_accel_mae']):+.2f}%的差异。这支持收益不全等同于本次高斯模糊控制；没有匹配完整频谱、逐像素纹理或动态幅度，不能排除所有平滑贡献。",'',
      'RG25的时序误差高于GS25：仅去掉长锚点不足以改善这些指标。SR3比SQ3的同帧RGB误差低，但时序误差更高、细节损失也更大；两版并列保留供原速观看，不依据单个数字自动选胜者。','',
      '## 4. 四片回归与代价','',
      '以下全部为SQ3相对同片GS25，负数表示该量下降；纹理与动态下降是代价，不能当作越低越好。','',
      '| 片段 | 低频时序 | 细纹理时序 | 突变代理 | 细纹理幅度 | 动态幅度 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for case in ('video1','video2','video3','video4'):
        c=changes[case]['SQ3'];lines.append('| '+case+' | '+' | '.join(f'{c[k]:+.2f}%' for k in ('coarse_percent','fine_percent','abrupt_error_percent','texture_amplitude_percent','dynamic_amplitude_percent'))+' |')
    sr3=changes['video3']['SR3']
    lines += ['', f"**SR3不能作为统一替代：** video3相对GS25的低频时序{sr3['coarse_percent']:+.2f}%、细纹理时序{sr3['fine_percent']:+.2f}%、突变代理{sr3['abrupt_error_percent']:+.2f}%，发生回归退化。因此主观看候选是SQ3，SR3保留作video4机制/取舍对照，不自动推广到三片。",'',
      '三片完成的是固定参数数值回归和输出核验，不是新的原速主观验收。原Q9输出未改；不能据此宣称三片观感无退化。', '',
      '## 5. 码率、成本和核验','',
      f"video4保持同一基础包和辅助包，总{rates['video4']['total_kbps']:.4f} kbps；所有候选新增传输0字节。源片仍完整104帧/15 FPS，无补帧或慢放，旧视频123各150帧。",'',
      '新增接收端相邻1/2帧RAFT12 FP32对应：video4为205个双向帧对，三片各297对。SQ3还依赖已有Q9链路，增加计算；本轮仅记录组件时间，不声称实时或端到端提速。读取整段和存储光流的研究脚本也不是内存有界部署实现。','',
      '全部帧重放、源文件禁读负例、guide重新解码一致、输入哈希、输出尺寸/类型、当前帧权重下限、过去两帧之外及未来输入扰动不影响融合输出、模糊幅度匹配、原基准指标复现、CSV重汇总、视频帧率/时长/面板检查均完成。源码、数据与媒体清单已归档。静图与时间条检查不代替原速观看。','',
      '## 6. 本轮结论与下一步','',
      '优先看SQ3是否确实减轻用户描述的整片抖动，并检查海龟鳍部与珊瑚细节是否被糊掉；SR3作为对照保留。若稳定收益只来自可见细节损失，不能直接升级采用；后续再研究限制局部非刚性变形和历史权重跳换。SR3在video3已有回归退化，不能仅凭video4的同帧指标推广。','',
      '本轮不重跑完整长片，不宣称密集珊瑚泛化解决，不修改正式采用配置；selected=null。video4仍是参与调试的开发素材。旧video3 +5.37%和原筛选记录保留，不与本轮以GS25为基线的百分比混算。','']
    report.write_text('\n'.join(lines),encoding='utf-8')
    readme=[ '# video4 抖动：原速对照', '',
      '用户反馈此前video4整片抖动明显。本轮两种三帧候选均未采用；重点看抖动、海龟鳍部、珊瑚变软和短尾。', '',
      '- [优先：旧GS25 / SQ3](video4_GS25_vs_SQ3.mp4)',
      '- [原GLC / 旧GS25；SR3 / SQ3](video4_raw_GS25_SR3_SQ3.mp4)',
      '- [原片 / 原GLC；旧GS25 / SQ3](video4_source_raw_GS25_SQ3.mp4)',
      '- [珊瑚放大](video4_coral.mp4) · [海龟放大](video4_turtle.mp4)',
      '- [Q9 / 关闭旧锚点RG25；SR3 / SQ3](video4_Q9_RG25_SR3_SQ3.mp4)',
      '- [同细节模糊 / SQ3](video4_blur_vs_SQ3.mp4)', '',
      'SQ3在既有GS25之后融合当帧和前两帧，保留原Q9链路；SR3从原GLC出发，不用长锚点。两者新增融合不递归回灌、当前帧至少50%。仍可能变软或短尾。', '',
      'SR3在video3时序回归退化，不作为统一替代；主观看候选为SQ3，仍待原速验收。', '',
      'video1/2/3回归：']
    for case in ('video1','video2','video3'):
        readme.append(f'- [{case}：旧GS25 / SQ3]({case}_GS25_vs_SQ3.mp4) · [原片 / Q9；SR3 / SQ3]({case}_source_Q9_SR3_SQ3.mp4)')
    readme += ['', f'[完整报告]({link(view/"README.md",report)}) · [核验]({link(view/"README.md",run/"verification.json")})', '',
      '视频均原速15 FPS，video4完整104帧约6.933秒；总通信码率38.28 kbps，与上一轮相同。媒体文件体积不是通信码率。观看目录使用独立副本。', '']
    (view/'README.md').write_text('\n'.join(readme),encoding='utf-8')
    statepath=REPO/'05_资料与索引/会话收束_20261110/接续状态.json';state=load_json(statepath)
    for key in ('latest_experiment','latest_diagnosis','next_experiment_plan','next_priority','latest_user_perceptual_feedback'):
        state['previous_'+key+'_before_video4_jitter']=state.get(key)
    state.update(status='Q9_video4_jitter_complete',active_experiment_plan=None,
        latest_experiment=result,new_experiment_report=str(report.relative_to(REPO)),latest_analysis_report=str(report.relative_to(REPO)),
        latest_user_perceptual_feedback=dict(video4=load_json(run/'user_feedback.json'),video123_Q9_positive_feedback_preserved=True),
        user_latest_decision=plan['user_feedback']['quote'],
        current_session_work='Completed fixed three-frame jitter experiment on video4 and regression video123; SQ3/SR3 pending original-speed review, not adopted.')
    state['latest_diagnosis']=dict(report=str(report),jitter_visual_failure_recorded=True,finite_history_candidates_ready=True,
        scalar_metrics_not_perceptual_acceptance=True,texture_and_dynamic_loss=True)
    state['latest_user_perceptual_feedback']['video4'].update(status='recorded_and_followup_experiment_complete',new_candidate_review='pending')
    state['experiment_execution_gate']=dict(status='completed_round',currently_running=False,new_run_started=True,automatic_next_run=False,date='2026-11-13')
    state['next_experiment_plan']=dict(status='proposed_not_run',focus='Review SQ3/SR3 original-speed jitter versus detail loss; constrain local flow deformation/support switching if needed.',long_video='not_started',automatic_selection=None)
    state['next_priority']=['Original-speed video4 SQ3/SR3 jitter, turtle shape and blur review','Keep video123 Q9 appearance as regression reference','Investigate local correspondence/support instability before long-video scaling']
    state['workspace_paths']['latest_jitter_run']=str(run);state['workspace_paths']['latest_jitter_viewing']=str(view/'README.md')
    save_json(statepath,state)
    entries=[REPO/'README.md',REPO/'交接文档.md',REPO/'当前项目状态与接续.md',REPO/'项目总结与未来方向.md',REPO/'新会话接续.txt',
        STAGE/'README.md',REPO/'05_资料与索引/会话收束_20261110/文件索引.md',root/'02_实验数据/README.md',root/'05_成果与观看/README.md']
    for p in entries:
        old=p.read_text(encoding='utf-8');before=run/'entries_before'/p.relative_to(root);before.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,before)
        update=(f'**最新（2026-11-13）：video4整片抖动反馈后的三帧短参考实验已完成。** 用户未认可上一轮video4观感；本轮SQ3/SR3及video1/2/3固定参数回归已跑完。SQ3时序突变代理下降，但细节与动态幅度也下降，只供观看，尚未采用。\n\n'
            f'[本轮报告]({link(p,report)}) · [原速对照]({link(p,view/"README.md")}) · [核验]({link(p,run/"verification.json")})\n\n'
            '当前无活动实验；Q9三片历史反馈与正式配置不变。下方为此前记录，旧“最新”或“待观看”状态以本段及本轮报告为准。\n\n---\n\n')
        if old.startswith('# '):head,sep,body=old.partition('\n');new=head+'\n\n'+update+body.lstrip('\n')
        else:new=update+old
        p.write_text(new,encoding='utf-8')
        after=run/'entries_after'/p.relative_to(root);after.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,after)
    statecopy=run/'entries_after'/statepath.relative_to(root);statecopy.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(statepath,statecopy)
    for src in run.rglob('*'):
        if src.is_file() and src.suffix.lower() in ('.json','.csv','.py','.md','.txt','.png'):
            dst=evidence/'run'/src.relative_to(run);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for src in (report,Path(__file__)):
        dst=evidence/src.name;shutil.copy2(src,dst)
    files={str(p.relative_to(evidence)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in sorted(evidence.rglob('*')) if p.is_file()}
    save_json(evidence/'manifest.json',dict(complete=True,files=files,source_run=str(run),large_arrays_and_videos_excluded=True,self_excluded=True))
    links=0
    for p in [report,view/'README.md',*entries]:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            # Only the new publication's targets are a requirement here; preserve old links.
            if p not in (report,view/'README.md') and not any(x in target for x in ('video4抖动','q9_video4_jitter')):continue
            resolved=(p.parent/target.split('#')[0]).resolve()
            assert resolved.exists(),(p,target);links+=1
    check_inputs(plan['input_sha256']);assert file_sha(STAGE/'当前采用配置.json')==CONFIG
    save_json(run/'publication.json',dict(complete=True,viewing=str(view),report=str(report),evidence=str(evidence/'manifest.json'),
        media_files=len(media),evidence_files=len(files),links_checked=links,current_config_unchanged=True,no_active_experiment=True))
    print('PUBLISHED',len(media),'media;',len(files),'evidence files;',links,'links',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();publish(a.run)
