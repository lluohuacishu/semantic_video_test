"""Original-speed comparisons and publication for the bounded SQ3 probe."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from runtime_speed_common import STAGE,REPO,WORKSPACE,FFMPEG,file_sha,load_json,save_json,check_inputs
from run_sq3_detail import RUN,PROFILES
from preserve_sq3_accepted import SHORT,LONG,DEST,CONFIG

VIEW=WORKSPACE/'05_成果与观看/GLC_SQ3保留与温和锐度实验_20261113'
REPORT=STAGE/'SQ3保留与温和锐度实验_20261113.md'

def sequences(case,names):
    s=load_json(RUN/'case_registration.json')['cases'][case]
    out={n:np.load(s[k],mmap_mode='r') for n,k in [('source','source'),('raw','raw'),('SQ3','sq3')]}
    for n in names:out[n]=np.load(RUN/case/'receive'/n/'rgb.npy',mmap_mode='r')
    if case=='long':out['H265']=np.load(LONG/'h265/rgb.npy',mmap_mode='r')
    return out

def media():
    from run_p0 import preview
    from finish_q9_ghost_exit import inspect
    assert load_json(RUN/'verification.json')['complete']
    result=load_json(RUN/'summary.json');out=RUN/'previews';out.mkdir(exist_ok=False)
    checked={};font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    for case in ('video4','video1','video2','video3'):
        seq=sequences(case,PROFILES);names=['source','SQ3',*PROFILES];panels=[seq[n] for n in names]
        path=out/f'{case}_原片_SQ3_U15_U30.mp4'
        preview(FFMPEG,path,panels,15,0,['Source','SQ3 accepted','U15 mild (experiment)','U30 stronger (experiment)'])
        checked[path.name]=inspect(path,panels)
        pair=out/f'{case}_SQ3_vs_U15.mp4';panels=[seq[n] for n in ('SQ3','SQ3_U15')]
        preview(FFMPEG,pair,panels,15,0,['SQ3 accepted','U15 mild (experiment)']);checked[pair.name]=inspect(pair,panels)
        ids=[len(seq['source'])//4,len(seq['source'])//2,3*len(seq['source'])//4]
        contact=Image.new('RGB',(512*4,314*3),(24,35,48));draw=ImageDraw.Draw(contact)
        for r,i in enumerate(ids):
            for j,n in enumerate(names):
                draw.text((j*512+6,r*314+3),f'{n}  {i/15:.2f}s  f{i}',font=font,fill='white')
                contact.paste(Image.fromarray(seq[n][i]),(j*512,r*314+26))
        contact.save(out/f'{case}_三个时点.png')
        if case in ('video4','video2'):
            crop=np.s_[128:288,0:512] if case=='video4' else np.s_[145:288,140:285]
            path=out/f'{case}_局部_SQ3_vs_U15.mp4';panels=[seq[n] for n in ('SQ3','SQ3_U15')]
            preview(FFMPEG,path,panels,15,0,['SQ3 accepted','U15 mild (experiment)'],crop=crop)
            checked[path.name]=inspect(path,panels,crop)
        print('MEDIA VERIFIED',case,flush=True)
    name=result['long_viewing_candidate']
    if name:
        seq=sequences('long',[name]);panels=[seq[n] for n in ('SQ3',name)]
        path=out/f'82秒_SQ3_vs_{name}.mp4'
        preview(FFMPEG,path,panels,15,0,['SQ3 accepted | 37.78 kbps',f'{name} experiment | 37.78 kbps']);checked[path.name]=inspect(path,panels)
        path=out/f'82秒_四宫格_原片_H265_GLC_{name}.mp4';panels=[seq[n] for n in ('source','H265','raw',name)]
        preview(FFMPEG,path,panels,15,0,['Source','H265 37.43 kbps','GLC base 26.22 kbps',f'{name} total 37.78 kbps']);checked[path.name]=inspect(path,panels)
        # Native frames, including processing boundaries and high source-change
        # points already chosen in the previous full-length experiment.
        ids=[0,149,150,600,619,1200,1238]
        old=load_json(LONG/'evaluation/summary.json')
        ids=sorted(set(ids)|{p['frame'] for p in old['source_change_review_points']})
        contact=Image.new('RGB',(1024,314*len(ids)),(24,35,48));d=ImageDraw.Draw(contact)
        for row,i in enumerate(ids):
            for col,n in enumerate(('SQ3',name)):
                d.text((col*512+8,row*314+3),f'{n}  {i/15:.2f}s  f{i}',font=font,fill='white')
                contact.paste(Image.fromarray(seq[n][i]),(col*512,row*314+26))
        contact.save(out/'长片边界与变化时点.png')
        print('MEDIA VERIFIED full 82.6-second long video',flush=True)
    save_json(out/'verification.json',dict(complete=True,files=checked,fps=15,
        frame_count_checked_by_full_decode=True,panel_alignment_checks='Five evenly spaced frames per video',
        note='No slow motion, padding, or extra display sharpening. Full-size 512x288 panels; local crops are 2x nearest-neighbor.',
        original_speed_subjective_review='pending'))

def relative(origin,target):return os.path.relpath(target,origin.parent).replace('\\','/')

def publish():
    summary=load_json(RUN/'summary.json');short=load_json(RUN/'short_screen.json')
    verification=load_json(RUN/'previews/verification.json');visual=load_json(RUN/'visual_inspection.json')
    assert verification['complete'] and visual['complete']
    plan=load_json(RUN/'execution_plan.json');check_inputs(plan['input_sha256']);assert file_sha(STAGE/'当前采用配置.json')==CONFIG
    VIEW.mkdir(exist_ok=False);mediafiles={}
    for src in sorted((RUN/'previews').iterdir()):
        if src.suffix not in ('.mp4','.png'):continue
        dst=VIEW/src.name;shutil.copy2(src,dst);assert file_sha(src)==file_sha(dst)
        mediafiles[dst.name]=dict(bytes=dst.stat().st_size,sha256=file_sha(dst),original=str(src),independent_copy=True)
    save_json(VIEW/'media_manifest.json',dict(complete=True,files=mediafiles))
    selected=summary['long_viewing_candidate'];longgate=load_json(RUN/'long_screen.json') if selected else None
    lines=['# SQ3 保留与温和锐度实验（2026-11-13）','',
        '**SQ3 已按用户要求独立保留为首选回退版本。** 用户认可完整长片的稳定性，接受锐度损失；本轮继续尝试找回少量边缘对比度，未自动替换 SQ3，也未修改正式部署配置。','',
        f'[SQ3 保留版本]({relative(REPORT,DEST/"README.md")}) · [原话与回退配置]({relative(REPORT,DEST/"回退配置.json")}) · [本轮观看目录]({relative(REPORT,VIEW/"README.md")})','',
        '## 固定方案','',
        '仅处理已接受的 SQ3 像素：Y=0.299R+0.587G+0.114B，取 Y−Gaussian(Y, sigma=1)，分别乘 0.15/0.30（U15/U30），校正截断为±3个8位亮度级，再按三通道共同余量限幅。最后统一量化校正量，同一整数加到RGB，保持通道差不变。','',
        '没有新光流、未来帧或递归历史；所有素材用相同参数，新增传输为0。它仅增强已有对比度，不等于找回真实丢失细节，也不能修复鱼消失、基础结构错误或切镜残影。','',
        '先冻结两个强度、工程容差和四片原有ROI。长片候选按固定顺序U15再U30选第一个全过者，选择的是观看候选，不是采用决定。预设容差：低频时序≤+1%、全频时序≤+3%、细纹理时序≤+5%、全频误差二阶差分≤+5%、RGB/细纹理空间误差≤+1%、PSNR≥−0.1dB，且全帧细纹理幅度至少+2%。这些是工程筛查值，不是人眼可察觉阈值。','',
        '## 四片结果（全部相对 SQ3）','',
        '| 片段 | 候选 | 细纹理幅度 | 低频时序误差 | 细纹理时序误差 | 突变代理 | PSNR差/dB | 全ROI筛查 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for case in ('video1','video2','video3','video4'):
        for name,r in short['screen'].items():
            c=r['changes'][case]['full'];ok=not any(f['case']==case for f in r['failures'])
            lines.append(f'| {case} | {name} | {c["fine_amplitude"]:+.2f}% | {c["coarse_mae"]:+.2f}% | {c["fine_mae"]:+.2f}% | {c["full_accel_mae"]:+.2f}% | {c["psnr_db"]:+.4f} | {"通过" if ok else "未通过"} |')
    lines+=['','细纹理幅度增加只表示对比度增强；时序/突变列是源对齐误差代理，正数为回归，不是人眼抖动百分比。源图仅进入评价与展示，所有接收输出都只读取 SQ3。四片沿用此前评价区域，SQ3的空间、时序、动态及突变指标与旧结果逐项一致。','']
    for name,r in short['screen'].items():
        lines.append(f'- {name}：'+('四片全部预设区域通过数值筛查。' if r['pass_all'] else f'共有{len(r["failures"])}项超出预先容差；详见 short_screen.json。'))
    if selected:
        lg=longgate['screen'][selected];c=lg['changes']['long']['full']
        lines+=['','## 完整长片','',f'按冻结顺序扩展 {selected} 到全部1239帧/15 FPS/82.6秒，未截取片段或补帧。输出SQ3并排对照，以及原片/H.265/原GLC/候选四宫格。',
            f'全帧细纹理幅度{c["fine_amplitude"]:+.2f}%，低频时序误差{c["coarse_mae"]:+.2f}%，细纹理时序误差{c["fine_mae"]:+.2f}%，突变代理{c["full_accel_mae"]:+.2f}%，PSNR{c["psnr_db"]:+.4f}dB；长片全帧及四象限筛查'+('通过。' if lg['pass_all'] else f'存在{len(lg["failures"])}项超限，不能据此升级。'),
            '长片采用源对齐时序评价，包含全序列；低于64像素公共有效支持的帧对按原评价规则排除，明细保留。源高变化时点仅供核对，不等于已确认切镜。','',
            '复用上一轮同码率H.265：37.4263 kbps；SQ3及本轮候选完整通信预算均37.7766 kbps，原GLC基础26.2222 kbps。匹配按整段完整文件字节，H.265比总预算少0.927%，并非逐秒相同码率；观看MP4体积不是通信码率。','',
            f'[82秒 SQ3 与 {selected} 并排]({relative(REPORT,VIEW/f"82秒_SQ3_vs_{selected}.mp4")}) · [候选四宫格]({relative(REPORT,VIEW/f"82秒_四宫格_原片_H265_GLC_{selected}.mp4")})']
    else:lines+=['','两档均未通过所有短片ROI预设约束，本轮未扩展完整长片。保留SQ3，不调整阈值来制造“升级”。']
    lines+=['','## 核验及限制','',
        '- SQ3独立归档包含四短片、长片数组、完整四宫格、源码/计划/输入输出哈希；光流缓存通过旧路径引用。',
        '- 首次试跑在接收校验发现半整数舍入导致RGB通道差偏移1级，评分前停止；旧计划/源码/部分无效输出保留在 `_01`。修正为统一量化校正量后另开 `_02`，参数和筛查规则不变。',
        '- 接收端负例确认源图与发送元数据禁读；所有输出逐帧重放一致、强度0恒等、每像素三通道校正相同且最多3级。',
        '- 冻结输入、SQ3回退副本与正式配置哈希不变。源图仅离线评价/展示使用。',
        '- 所有预览按原速编码并核对帧数/时长，五个分布时点核对面板来源。静帧检查不能替代用户原速主观判断。',
        '- 这是既有开发素材上的受控小试，不是独立测试集泛化验证；没有解决形态、遮挡和切镜问题的结论。','',
        '目前仍以用户认可的SQ3为首选回退方案。新候选需结合原速视频判断锐度改善是否值得承担细纹理变化增加；若看着不如SQ3，直接使用保留版本。','',
        f'[执行计划]({relative(REPORT,RUN/"execution_plan.json")}) · [数值筛查]({relative(REPORT,RUN/"short_screen.json")}) · [核验]({relative(REPORT,RUN/"verification.json")})','']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')
    readme=['# SQ3 保留与温和锐度实验','', 'SQ3已经独立保留。本目录的新候选尚待原速观看，不自动替换已认可的SQ3。','']
    if selected:
        readme += [f'[先看：82秒 SQ3 与 {selected} 并排](82秒_SQ3_vs_{selected}.mp4)', '',
            f'[82秒四宫格：原片/H.265/原GLC/{selected}](82秒_四宫格_原片_H265_GLC_{selected}.mp4)','']
    for c in ('video4','video1','video2','video3'):
        readme += [f'- [{c}：SQ3与U15并排]({c}_SQ3_vs_U15.mp4)；[四宫格：原片/SQ3/U15/U30]({c}_原片_SQ3_U15_U30.mp4)']
    readme+=['','短片四宫格：左上原片、右上SQ3、左下U15、右下U30。15FPS原速，video1/2/3各10秒，video4为104帧约6.933秒。局部对照是最近邻放大，不额外锐化。','',
        f'[完整报告]({relative(VIEW/"README.md",REPORT)}) · [SQ3回退版本]({relative(VIEW/"README.md",DEST/"README.md")})','']
    (VIEW/'README.md').write_text('\n'.join(readme),encoding='utf-8')
    result=dict(complete=True,date='2026-11-13',run=str(RUN),report=str(REPORT),viewing=str(VIEW),
        baseline='SQ3',preferred_fallback_archive=str(DEST),profiles=list(PROFILES),long_viewing_candidate=selected,
        numeric_pass={n:r['pass_all'] for n,r in short['screen'].items()},long_numeric_pass=None if not selected else longgate['screen'][selected]['pass_all'],
        subjective_review='pending',selected=None,automatic_adoption=False,formal_profile_unchanged=True,
        next_focus='Original-speed compare small contrast improvement against temporal cost; use accepted SQ3 if not better.')
    save_json(RUN/'round_result.json',result)
    sp=REPO/'05_资料与索引/会话收束_20261110/接续状态.json';state=load_json(sp)
    state['previous_latest_experiment_before_sq3_detail']=state.get('latest_experiment')
    state.update(date='2026-11-13',status='SQ3_preserved_detail_round_complete',active_experiment_plan=None,
        latest_experiment=result,new_experiment_report=str(REPORT.relative_to(REPO)),latest_analysis_report=str(REPORT.relative_to(REPO)),
        current_session_work='SQ3 user-accepted fallback preserved. Bounded detail experiment completed; new comparison pending subjective review.',
        experiment_execution_gate=dict(status='completed_round',currently_running=False,automatic_next_run=False),
        next_experiment_plan=dict(status='awaiting_optional_candidate_feedback',preferred_fallback='SQ3',automatic_next_run=False),
        next_priority=['Compare candidate versus accepted SQ3 at original speed','Keep SQ3 unless the user prefers a tested candidate'])
    state['latest_diagnosis']=dict(report=str(REPORT),SQ3_user_accepted=True,new_candidate_review='pending',no_true_detail_recovery_claim=True)
    state['workspace_paths']['latest_detail_run']=str(RUN);state['workspace_paths']['latest_detail_viewing']=str(VIEW/'README.md')
    save_json(sp,state)
    entries=[REPO/'README.md',REPO/'交接文档.md',REPO/'当前项目状态与接续.md',REPO/'项目总结与未来方向.md',REPO/'新会话接续.txt',
        STAGE/'README.md',REPO/'05_资料与索引/会话收束_20261110/文件索引.md',WORKSPACE/'02_实验数据/README.md',WORKSPACE/'05_成果与观看/README.md']
    for p in entries:
        old=p.read_text(encoding='utf-8');backup=RUN/'entries_before'/p.relative_to(WORKSPACE);backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,backup)
        msg=(f'**最新（2026-11-13）：用户认可的 SQ3 已独立保留，温和锐度实验完成。** 两档固定强度已回归四片；'+(f'{selected}已扩展完整82.6秒供对照。' if selected else '未通过全约束，未扩展长片。')+
            f'新候选尚待观看；首选回退仍为SQ3，正式配置未改。\n\n[本轮报告]({relative(p,REPORT)}) · [观看入口]({relative(p,VIEW/"README.md")}) · [SQ3回退版本]({relative(p,DEST/"README.md")})\n\n当前无活动实验。下方旧状态作为历史保留。\n\n---\n\n')
        p.write_text(msg+old,encoding='utf-8')
    evidence=STAGE/'evidence_20261113/sq3_detail';evidence.mkdir(exist_ok=False)
    for src in sorted(RUN.rglob('*')):
        if src.is_file() and src.suffix in ('.json','.csv','.py','.md') and 'entries_before' not in str(src.relative_to(RUN)):
            dst=evidence/'run'/src.relative_to(RUN);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    shutil.copy2(REPORT,evidence/REPORT.name);shutil.copy2(Path(__file__),evidence/Path(__file__).name)
    shutil.copy2(RUN.parent/'sq3_detail_20261113_01/failure.json',evidence/'first_attempt_failure.json')
    manifest={str(p.relative_to(evidence)).replace('\\','/'):dict(sha256=file_sha(p),bytes=p.stat().st_size) for p in sorted(evidence.rglob('*')) if p.is_file()}
    save_json(evidence/'manifest.json',dict(complete=True,files=manifest,self_excluded=True))
    for n,r in manifest.items():assert file_sha(evidence/n)==r['sha256']
    for n,r in load_json(DEST/'manifest.json')['files'].items():assert file_sha(DEST/n)==r['sha256']
    check_inputs(plan['input_sha256']);assert file_sha(STAGE/'当前采用配置.json')==CONFIG
    save_json(RUN/'publication.json',dict(complete=True,viewing=str(VIEW),report=str(REPORT),
        evidence=str(evidence/'manifest.json'),media_count=len(mediafiles),evidence_count=len(manifest),
        accepted_archive_unchanged=True,formal_config_unchanged=True,no_active_experiment=True))
    print('PUBLISHED',VIEW,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['media','publish']);a=p.parse_args()
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    if a.action=='media':media()
    else:publish()
