"""Publish the requested original/H.265/GLC/SQ3 full-length four-way video."""
import argparse
import os
from pathlib import Path
import re
import shutil
from runtime_speed_common import STAGE,REPO,WORKSPACE,file_sha,load_json,save_json,check_inputs
from run_sq3_long import CONFIG

TITLE='SQ3长片四宫格_原片_H265_GLC_20261113.md'

def rel(origin,target):return os.path.relpath(target,origin.parent).replace('\\','/')

def publish(run):
    plan=load_json(run/'execution_plan.json');verification=load_json(run/'verification.json')
    media=load_json(run/'previews/verification.json');visual=load_json(run/'visual_inspection.json')
    assert verification['complete'] and media['complete'] and visual['complete']
    check_inputs(plan['input_sha256']);assert file_sha(STAGE/'当前采用配置.json')==CONFIG
    rec=load_json(run/'receive/receipt.json');h=load_json(run/'h265/receipt.json');evaluation=load_json(run/'evaluation/summary.json')
    view=WORKSPACE/'05_成果与观看/GLC_SQ3长片四宫格_20261113';view.mkdir()
    evidence=STAGE/'evidence_20261113/sq3_long_fourway';evidence.mkdir()
    report=STAGE/TITLE;files={}
    for src in sorted((run/'previews').iterdir()):
        if src.suffix not in ('.mp4','.png'):continue
        dst=view/src.name;shutil.copy2(src,dst);assert file_sha(dst)==file_sha(src)
        files[dst.name]=dict(source=str(src),bytes=dst.stat().st_size,sha256=file_sha(dst),hardlink=False)
    save_json(view/'media_manifest.json',dict(complete=True,files=files))
    movie=view/'82秒_四宫格_原片_H265_原GLC_SQ3.mp4';base=plan['base_bytes']*8/82.6/1000;aux=plan['auxiliary_bytes']*8/82.6/1000
    rows=[ '# SQ3 长片四宫格（2026-11-13）', '',
      '**已完成完整82.6秒、1239帧的SQ3输出和原速四宫格。** 用户认可上一轮短片SQ3观感但指出锐度下降；本轮只将同一设置扩展到既有长片，长片主观验收待反馈，正式采用配置不变。', '',
      f'[打开完整四宫格]({rel(report,movie)}) · [观看目录]({rel(report,view/"README.md")}) · [输出核验]({rel(report,run/"verification.json")}) · [视频核验]({rel(report,run/"previews/verification.json")})', '',
      '## 观看布局', '',
      '| 左上 | 右上 |','| --- | --- |',f'| 原片，同分辨率采样 | H.265，{h["kbps"]:.4f} kbps |',
      '| 左下 | 右下 |',f'| 原GLC，{base:.4f} kbps | SQ3，合计{plan["total_kbps"]:.4f} kbps |', '',
      '四路采用相同1239帧源时间线，单路512×288、15 FPS；合成画面1024×688，完整82.6秒，带时间/帧号。没有选取较好片段、补尾帧、慢放或增加锐化。按视频通道比较，不含音频。', '',
      '## SQ3执行方式', '',
      '复用已独立核验的完整基础GLC码流与Q9跨块连续输出，重新解码9个收到的guide包，逐帧生成GS25，再运行上一轮同参数SQ3。基础流来自历史TRT FP16链路，不是video4独立短片FP32重新编码。', '',
      '名义权重为当帧/前一帧/前两帧0.5/0.3/0.2，仍按guide双向一致性与光度差衰减历史权重；当前GS25权重至少50%。历史输入使用原GS25，不递归使用SQ3输出。处理窗口150帧，跨窗口保留2帧，尾块39帧；不会在10秒处理边界额外清空这两帧历史。', '',
      '没有改变Q9全局锚点或guide独立分包方式，没有增加场景切换检测。SQ3的两帧窗口不等于其Q9输入没有旧参考残像；复杂场景、切镜和guide边界仍需看完整输出。没有重调阈值或换成SR3。', '',
      '## H.265比较口径', '',
      '| 项目 | 必要字节 | 实际kbps |','| --- | ---: | ---: |',
      f'| GLC基础流 | {plan["base_bytes"]:,} | {base:.4f} |',
      f'| guide包与必要索引 | {plan["auxiliary_bytes"]:,} | {aux:.4f} |',
      f'| SQ3完整通信预算 | {plan["total_bytes"]:,} | {plan["total_kbps"]:.4f} |',
      f'| H.265完整MP4 | {h["bytes"]:,} | {h["kbps"]:.4f} |','',
      f'H.265采用既有medium控制设置、GOP上限64、scenecut=0、BT709 limited YUV420，CRF仅按完整文件字节搜索，不按质量分挑选。本次CRF为{h["crf"]}，比SQ3总预算少{h["relative_shortfall"]*100:.3f}%，在预先登记的2%容差内。没有填充字节。此为该固定编码设置下的对照，不代表所有H.265设置的最优质量。', '',
      'SQ3新增传输0字节，但原GLC左下只包含基础流，码率低于右下SQ3和右上H.265。四宫格观看文件使用高质量H.264编码，其文件大小不是通信码率。', '',
      'H.265匹配的是整段平均完整文件预算，不是每秒或每个10秒窗口的相同瞬时码率。', '',
      '## 核验与记录', '',
      '- 将旧video4、video1分别改为32帧窗口、2帧重叠运行，所有相邻光流和SQ3 RGB与旧整段输出逐位一致。',
      '- 完整长片1239帧逐帧重放GS25/SQ3公式一致，9个guide包重新解码一致，8处处理边界确实包含前两帧。',
      '- 冻结输入哈希不变；接收源图和发送元数据禁读负例通过。首次路径校验在推理前停止，补充历史目录元数据排除包装后重跑；原数值代码和原冻结计划未改，补充文件单独归档。',
      '- 四宫格全部帧解码检查，帧数、帧率、时长一致；首中尾、8处块边界前后及5个源高变化时点逐面板对齐核验。源高变化时点仅供观察，不等于已确认切镜。',
      '- 静帧检查不代替原速主观验收。未依据PSNR或平均误差宣称长片观感通过。', '',
      '本轮基础同帧指标（全部帧；与之前源光流时序误差不是同一种量）：', '',
      '| 方法 | PSNR/dB | RGB MAE | 细纹理幅度 |','| --- | ---: | ---: | ---: |']
    for name in ('H265','raw','Q9','GS25','SQ3'):
        s=evaluation['scores']['full'][name];rows.append(f'| {name} | {s["psnr_db"]:.4f} | {s["rgb_mae"]:.4f} | {s["fine_amplitude"]:.4f} |')
    rows += ['', f'新增SQ3接收阶段记录耗时{rec["seconds"]:.1f}秒，包含guide解码、相邻光流、融合、哈希和I/O；不含已复用的基础GLC和Q9计算，不是完整端到端测速。', '',
      '后续以原速观看为准：关注密集珊瑚是否仍整片抖动、切换场景时有无短暂旧影、海龟/鱼鳍是否变软，以及10/20/…/80秒分包边界。当前完成并收束，不自动启动下一轮，不修改正式配置。', '']
    report.write_text('\n'.join(rows),encoding='utf-8')
    readme='\n'.join(['# SQ3 完整长片四宫格','',f'[播放82.6秒完整对照]({movie.name})','',
      f'左上：原片　｜　右上：H.265（{h["kbps"]:.2f} kbps）',
      f'左下：原GLC（{base:.2f} kbps）　｜　右下：SQ3（{plan["total_kbps"]:.2f} kbps）','',
      '同时间线1239帧、15 FPS；每路512×288，原速且无音频。保持上一轮SQ3设置，没有加锐化或针对场景调参。',
      f'H.265按SQ3完整通信预算匹配，实际少{h["relative_shortfall"]*100:.3f}%。原GLC单独显示基础码率；观看MP4大小不计作通信成本。','',
      '请关注珊瑚抖动与细节变软、运动边缘、场景切换及10秒整数分包边界。长片未做新的切镜处理，尚待原速反馈。','',
      f'[完整报告]({rel(view/"README.md",report)}) · [核验记录]({rel(view/"README.md",run/"verification.json")})','',
      '[首帧](verified_frame_000000.png) · [中间](verified_frame_000619.png) · [尾帧](verified_frame_001238.png)',''])
    (view/'README.md').write_text(readme,encoding='utf-8')
    result=dict(complete=True,date='2026-11-13',profile='SQ3',frames=1239,fps=15,seconds=82.6,
        total_kbps=plan['total_kbps'],h265_kbps=h['kbps'],h265_shortfall=h['relative_shortfall'],
        requested_layout=['source','H265','raw_GLC','SQ3'],run=str(run),report=str(report),viewing=str(view),movie=str(movie),
        short_subjective_feedback=plan['request'],long_subjective_feedback='pending',selected=None,automatic_adoption=False,
        same_SQ3_settings=True,window_history_verified=True,scene_cut_detector_added=False,
        formal_profile_unchanged=True,current_config_sha256=CONFIG,evidence=str(evidence/'manifest.json'))
    save_json(run/'round_result.json',result)
    statepath=REPO/'05_资料与索引/会话收束_20261110/接续状态.json';state=load_json(statepath)
    for k in ('latest_experiment','latest_diagnosis','next_experiment_plan','latest_user_perceptual_feedback','next_priority'):
        state['previous_'+k+'_before_sq3_long']=state.get(k)
    state.update(status='SQ3_long_fourway_complete',active_experiment_plan=None,latest_experiment=result,
        new_experiment_report=str(report.relative_to(REPO)),latest_analysis_report=str(report.relative_to(REPO)),
        current_session_work='Completed user-requested 82.6-second source/H265/rawGLC/SQ3 four-way; long-video original-speed review pending.',
        user_latest_decision=plan['request']['quote'])
    state['latest_user_perceptual_feedback']=dict(SQ3_short='User says looks acceptable but less sharp.',long_SQ3='not yet reviewed',video123_Q9_feedback_preserved=True)
    state['latest_diagnosis']=dict(report=str(report),complete_long_SQ3_output=True,matched_h265=True,original_speed_review='pending',no_scene_cut_fix_claim=True)
    state['experiment_execution_gate']=dict(status='completed_round',currently_running=False,new_run_started=True,automatic_next_run=False,date='2026-11-13')
    state['next_experiment_plan']=dict(status='awaiting_long_video_feedback',current_output=str(movie),no_parameters_chosen=True,automatic_next_run=False)
    state['next_priority']=['Review requested long four-way at original speed','Record dense coral, fish/turtle boundaries and scene/block transition failures before next hypothesis']
    state['workspace_paths']['latest_SQ3_long_run']=str(run);state['workspace_paths']['latest_SQ3_long_viewing']=str(view/'README.md')
    save_json(statepath,state)
    entries=[REPO/'README.md',REPO/'交接文档.md',REPO/'当前项目状态与接续.md',REPO/'项目总结与未来方向.md',REPO/'新会话接续.txt',
        STAGE/'README.md',REPO/'05_资料与索引/会话收束_20261110/文件索引.md',WORKSPACE/'02_实验数据/README.md',WORKSPACE/'05_成果与观看/README.md']
    for path in entries:
        old=path.read_text(encoding='utf-8');before=run/'entries_before'/path.relative_to(WORKSPACE);before.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,before)
        msg=(f'**最新（2026-11-13）：SQ3完整82.6秒四宫格已完成。** 用户反馈短片SQ3可看但不够锐利；本轮保持同参数，输出原片/H.265/原GLC/SQ3。H.265实际{h["kbps"]:.2f} kbps，SQ3总{plan["total_kbps"]:.2f} kbps，相差{h["relative_shortfall"]*100:.3f}%。\n\n'
            f'[完整四宫格]({rel(path,movie)}) · [本轮报告]({rel(path,report)}) · [核验]({rel(path,run/"verification.json")})\n\n'
            '当前无活动实验；长片原速反馈待记录，正式采用配置不变。下方保留此前状态，以本段和本轮报告为准。\n\n---\n\n')
        if old.startswith('# '):head,sep,body=old.partition('\n');text=head+'\n\n'+msg+body.lstrip('\n')
        else:text=msg+old
        path.write_text(text,encoding='utf-8');after=run/'entries_after'/path.relative_to(WORKSPACE);after.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,after)
    target=run/'entries_after'/statepath.relative_to(WORKSPACE);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(statepath,target)
    for src in run.rglob('*'):
        if src.is_file() and src.suffix.lower() in ('.json','.csv','.py','.md','.txt','.png'):
            dest=evidence/'run'/src.relative_to(run);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    for src in (report,Path(__file__)):shutil.copy2(src,evidence/src.name)
    manifest={str(p.relative_to(evidence)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in sorted(evidence.rglob('*')) if p.is_file()}
    save_json(evidence/'manifest.json',dict(complete=True,source_run=str(run),files=manifest,self_excluded=True,large_arrays_and_videos_excluded=True))
    count=0
    for path in (report,view/'README.md'):
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            assert (path.parent/target).resolve().exists(),(path,target);count+=1
    check_inputs(plan['input_sha256']);assert file_sha(STAGE/'当前采用配置.json')==CONFIG
    save_json(run/'publication.json',dict(complete=True,report=str(report),viewing=str(view),movie=str(movie),
        evidence=str(evidence/'manifest.json'),media_files=len(files),evidence_files=len(manifest),links_checked=count,
        formal_profile_unchanged=True,no_active_experiment=True))
    print('PUBLISHED FULL FOURWAY',movie,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();publish(a.run)
