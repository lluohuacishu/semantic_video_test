"""Publish verified known-video transfer evidence without changing the adopted codec."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil

from runtime_speed_common import STAGE, REPO, WORKSPACE, load_json, file_sha, check_inputs

REPORT_NAME='当前方案跨素材初筛_video2_video3_20261111.md'


def write(path,text):
    with Path(path).open('w',encoding='utf-8',newline='\n') as f:f.write(text)


def dump(path,value):write(path,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def relative(path,folder):return Path(os.path.relpath(path,folder)).as_posix()


def chart(run,out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig,axes=plt.subplots(2,1,figsize=(11,6.8),sharex=True,constrained_layout=True)
    for ax,case in zip(axes,('video2','video3')):
        with (run/case/'evaluation/temporal.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        for method,label,color in [('raw','Original GLC','#7a8795'),('K','Historical K (higher rate)','#a578b7'),
            ('current','Current S8I + I12','#087f72'),('H265','Matched H.265','#377fcb')]:
            values=[r for r in rows if r['scope']=='full' and r['method']==method]
            ax.plot([int(r['frame'])/15 for r in values],[float(r['coarse_sum'])/int(r['pixels']) for r in values],
                label=label,color=color,lw=1.35)
        for frame in (64,128):ax.axvline(frame/15,color='#c57a45',ls='--',lw=.8,alpha=.6)
        ax.set_title(case+' | full-frame source-aligned temporal error (lower is better)',loc='left',fontsize=10)
        ax.set_ylabel('Coarse MAE');ax.grid(alpha=.18);ax.set_xlim(0,10)
    axes[0].legend(ncol=2,fontsize=8,frameon=False);axes[1].set_xlabel('Time in 10-second clip (s)')
    path=out/'temporal_comparison.png';fig.savefig(path,dpi=160);plt.close(fig);return path


def publish(run):
    if (run/'publication.json').exists():raise FileExistsError('Already published')
    plan=load_json(run/'execution_plan.json');verified=load_json(run/'verification/summary.json')
    assert verified['complete'];check_inputs(plan['input_sha256'])
    for name,digest in plan['code_sha256'].items():assert file_sha(STAGE/name)==digest
    summaries={case:load_json(run/case/'evaluation/summary.json') for case in ('video2','video3')}
    chains={case:load_json(run/case/'chain_result.json') for case in summaries}
    viewing=WORKSPACE/'05_成果与观看/GLC_跨素材初筛_20261111'
    evidence=STAGE/'evidence_20261111/known_video_transfer'
    report=STAGE/REPORT_NAME
    for path in (viewing,evidence,report):
        if path.exists():raise FileExistsError(path)
    viewing.mkdir();evidence.mkdir();links={}
    specs=[]
    for n,case in enumerate(summaries,1):
        specs.append((case,f'{case}_四宫格_原片_H265_原GLC_当前I12.mp4',f'{n:02}_{case}_四宫格_原片_H265_原GLC_当前I12.mp4'))
    for first,tag in ((3,'原GLC_当前I12'),(5,'旧K_当前I12')):
        for n,case in enumerate(summaries,first):specs.append((case,f'{case}_{tag}.mp4',f'{n:02}_{case}_{tag}.mp4'))
    for case,source_name,name in specs:
        source=run/case/'evaluation'/source_name;target=viewing/name
        try:os.link(source,target)
        except OSError:shutil.copy2(source,target)
        assert file_sha(source)==file_sha(target)
        links[name]=dict(source=str(source),sha256=file_sha(target))
    figure=chart(run,viewing)
    summary=dict(complete=True,kind='known_development_clips_fixed_parameter_transfer',
        profile='S8I + CURRENT_I12 + exact runtime caches',new_clips=['video2','video3'],
        conclusions=dict(video2='Weak average benefit with spatial and high-change regression; not a successful universal transfer',
            video3='Strong texture stability benefit, slight PSNR cost and reduced change amplitude; perceptual acceptance pending'),
        heldout_tested=False,long_video_tested=False,user_visual_acceptance='pending',
        run=str(run),report=str(report),viewing=str(viewing),verification=str(run/'verification/summary.json'),
        code_and_adopted_profile_unchanged=True,new_work_committed_or_pushed=False,cases={})
    for case,d in summaries.items():
        raw,cur,k=d['scores']['full']['raw'],d['scores']['full']['current'],d['scores']['full']['K']
        summary['cases'][case]=dict(rates_kbps=d['rates_kbps'],
            full_coarse_mae=dict(raw=raw['coarse_mae'],current=cur['coarse_mae'],K=k['coarse_mae']),
            coarse_mae_reduction_percent=(1-cur['coarse_mae']/raw['coarse_mae'])*100,
            current_vs_K_error_change_percent=(cur['coarse_mae']/k['coarse_mae']-1)*100,
            psnr_db=dict(raw=raw['psnr_db'],current=cur['psnr_db']),
            cached_uncached_exact=d['exact_against_uncached_I12'],cold_single_seconds=d['full_chain_cold_seconds'])
    dump(run/'summary.json',summary)
    table=[];details=[]
    for case,d in summaries.items():
        raw,cur=d['scores']['full']['raw'],d['scores']['full']['current']
        reef=d['scores']['historical_reef'];chain=chains[case]
        table.append(f'| {case} | {d["rates_kbps"]["current"]:.4f} | {raw["coarse_mae"]:.4f} → {cur["coarse_mae"]:.4f} | {(1-cur["coarse_mae"]/raw["coarse_mae"])*100:.2f}% | {raw["fine_ncc"]:.4f} → {cur["fine_ncc"]:.4f} | {raw["psnr_db"]:.3f} → {cur["psnr_db"]:.3f} |')
        details.append(f'| {case} | {chain["base_bytes"]:,} | {chain["guide_bytes"]:,} | {chain["total_bytes"]:,} | {d["rates_kbps"]["K"]:.4f} | {d["rates_kbps"]["h265"]:.4f} | {d["matched_hevc"]["relative_shortfall"]*100:.3f}% | {chain["full_chain_cold_seconds"]:.2f} |')
    report_text='''# 当前方案跨素材初筛：video2 / video3

2026-11-11。用户要求找回以前用过的视频，用当前流程初步检查泛用性。本轮已完成两个旧开发素材的完整重跑、固定参数评价和观看文件；**结果明显依赖场景，不能判定已具备普适泛化能力。**

## 1. 试了哪些视频，哪些规则没有变

找回 `01_项目/GLC后处理/素材/20261109_三个片段/` 中的 video1、video2、video3。video1 已是当前方案开发/采用素材，本轮新增测试 video2、video3。源文件 SHA 与历史登记一致。两片原始 1280×720 /30 FPS，时长分别 10.5 秒、10.233333 秒；仍按旧轮次截取前 10 秒，重采样为 512×288 /15 FPS /150 帧，源数组逐位等于历史输入。

算法固定 S8I+CURRENT_I12：六锚点 0/15/45/75/105/135，RAFT FP32/12 次/无 TF32，原 T3 纹理阈值、几何/光度门控、2px 羽化、六帧过渡和两次取整均未更改。S8I 仍为 256×144/15FPS、x265 slower two-pass、内部目标 6 kbps。没有逐片调阈值或挑选更优参数。

实际链路是原视频抽帧 → 新 GLC 基础编码 → 新 S8I 编码封装 → 独立基础解码 → 当前缓存版后处理。随后另跑原 I12 无缓存回放；两个素材的完整 RGB/alpha/shared 均逐位一致。新增代码只是实验控制、评价和发布，旧推理实现与采用配置均保持原字节。

## 2. 主要结果

时序 MAE 为沿用历史协议、在源图像决定的公共支持上对齐并扣除真实变化后的粗尺度误差，越低越好。NCC 衡量细纹理连续性，越高通常越稳定；PSNR 衡量同帧像素保真，越高越好。时序稳定不等于形状或动态目标真实还原。

| 素材 | 当前总码率 kbps | 全帧时序 MAE：原GLC→当前 | MAE 降低 | 细纹理 NCC：原GLC→当前 | RGB PSNR dB：原GLC→当前 |
| --- | ---: | ---: | ---: | ---: | ---: |
'''+ '\n'.join(table)+'''

### video2：只获得局部收益，整体迁移不理想

该片包含近景大鱼、密集细纹理和遮挡。全帧 MAE 只改善 **0.95%**，PSNR 下降 **0.301 dB**。历史预登记的右下珊瑚区域 MAE 改善 **15.76%**，但左上、右上两个象限分别恶化 **6.53% /5.76%**，不能用珊瑚局部收益代表整幅画面。

最坏帧误差从 7.066 降到 4.785，说明部分剧烈跳变被压低；但帧误差 P95 几乎未改善（3.454→3.417），平均收益仍弱。高变化像素的 RGB 帧差误差从 43.279 升至 44.152（约 +2.02%），需要继续关注运动和边界。

相对旧 K，全帧时序 MAE 高 **40.04%**（2.2615 vs 1.6149），当前总码率则低 **29.42%**。这是更低预算下的明显质量取舍，不是全面替代旧 K。初筛结论：保留为当前方案的薄弱/反例素材，不自动验收。

### video3：固定参数迁移有明显正面结果，但还有运动保真代价

该片为较稳定的礁石/珊瑚结构，画面包含水体和小鱼。全帧 MAE 改善 **46.67%**，历史珊瑚区域改善 **61.63%**，细纹理 NCC 从 0.722 升至 0.911；全帧最坏帧误差从 8.518 降到 1.910，P95 从 3.277 降到 1.738。相对旧 K，全帧误差还低 **18.27%**，总码率低 **17.94%**。

收益集中在礁石和珊瑚区域。左上水体的应用覆盖仅约 **0.39%**，相应指标几乎不变，未把低纹理水体平均分当作修复成功。整体 PSNR 仍下降 **0.166 dB**。高变化像素的帧差误差略降（27.679→27.168），但变化幅度相对源从 0.796 降至 0.675；这可能包含不希望的运动变化压低，不能仅凭更稳定就声称动态鱼类已经通过验收。

这些区域/场景描述是抽帧观察与指标的合并解释；两片不足以证明失败由某一场景因素导致。仍需用户原速观看确认，主观验收状态保持 pending。

## 3. 完整字节、传统对照和运行时间

| 素材 | 本体 B | 引导 B | 合计 B | 旧 K 总码率 kbps | H.265 总码率 kbps | H.265 字节短缺 | 当前完整冷启动秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
'''+ '\n'.join(details)+'''

S8I 内部目标相同不代表实际包字节相同；本轮辅助实际分别 5.5112 /8.2960 kbps。所有必要包头已计入，观看 MP4 体积不用于神经压缩码率。H.265 沿用 medium/GOP64 的完整 MP4 字节匹配协议，只按大小选择 CRF，允许低于预算不超过 2%，无填充和观感选档。

H.265 在这两片的全帧 PSNR（24.124 /27.655 dB）及时序 MAE（1.3545 /0.6743）均优于当前 I12。抽帧可看到部分鱼体/珊瑚细纹理更平滑，因此尚不把这组指标等同于整体主观喜好；同时不能宣称当前方案总体优于 H.265。

完整链路每片仅一次冷启动，用时约 66–68 秒/10 秒片段。它是描述性单次测量，不能与上一轮三次中位数作严谨速度比较，且不含网络、浏览器或水声传输。常驻、长片和实时能力本轮未测。

## 4. 核验、入口与后续

- 新抽帧、基础码流和基础解码均与对应历史素材逐位一致；接收端读取实际包和独立解码产物。
- 当前缓存/无缓存 I12 完整 RGB/alpha/shared 一致，零权重像素不变，数组类型/形状/有限值、引导 CRC/基础绑定和帧数均核对。
- 两个素材、两个接收角色共 28 次实际源文件/源光流/发送端收据读取拒绝探测全部通过；这是 Python 审计边界，不是操作系统安全沙箱。
- 12 个视频文件已核对 150 帧、15 FPS、10 秒，四宫格/双列对照已按声明的数组回读检查像素面板。
- 原 GLC 和旧 K 的历史指标精确复现，源文件、权重、采用配置及冻结推理代码哈希未变。

优先研究 video2 的引导信息损失、运动/遮挡估计和错误复用；具体原因尚未通过消融证明。本轮没有试调参数。后续应预先冻结诊断方案，保留 video3 正例与 video2 反例，再考虑独立留出及长片。固定 150 帧封装、锚调度、有界特征/光流缓存和分块延迟仍是长视频前置问题。

这是 **已知开发素材迁移初筛**，不是全新留出集、长视频或普适性验收。当前已采用的 video1 方案和部署默认不自动升级。新增代码/报告/入口修改尚未提交或推送。

'''
    report_text+=f'[原速观看目录]({relative(viewing/"README.md",STAGE)}) · [冻结计划与小证据](evidence_20261111/known_video_transfer/README.md) · [完整原始运行]({relative(run,STAGE)})\n\n'
    if figure:report_text+=f'![全帧逐帧时序误差]({relative(figure,STAGE)})\n\n'
    report_text+='复现入口为 `run_known_video_transfer.py prepare/run/evaluate --run <全新目录>`；核验为 `verify_known_video_transfer.py --run <运行目录>`。`prepare` 必须先执行并冻结输入，严禁覆盖本轮结果。`known_video_transfer_20261111_01` 仅为准备阶段公共模块路径错误的未推理尝试；修正控制脚本后，`_02` 是权威运行。\n'
    write(report,report_text)
    view_text='# video2 / video3 当前方案迁移初筛\n\n2026-11-11。固定 S8I+I12 参数，每片前10秒、150帧、15FPS。先看四宫格，再看原GLC/当前和旧K/当前的双列原速对照。\n\n'
    for name in links:view_text+=f'- [{name}]({name})\n'
    view_text+='\n四宫格顺序：左上原片、右上近同总码率 H.265、左下原GLC、右下当前I12。双列顺序按文件名：原GLC或旧K在左，当前I12在右。\n\nvideo2 总码率30.9904 kbps，整体时序MAE只改善0.95%；video3总码率33.0920 kbps，改善46.67%。已有开发素材，未做新留出或长视频验证；主观验收待用户观看。\n\n'
    view_text+=f'[完整报告]({relative(report,viewing)})\n'
    if figure:view_text+='\n![时序误差](temporal_comparison.png)\n'
    write(viewing/'README.md',view_text);dump(viewing/'manifest.json',links)
    copied=['execution_plan.json','summary.json','verification/summary.json']
    for case in summaries:
        copied += [f'{case}/chain_result.json',f'{case}/evaluation/summary.json',f'{case}/evaluation/temporal.csv',
            f'{case}/evaluation/spatial.csv',f'{case}/evaluation/source_change_diagnostic.csv',f'{case}/evaluation/matched_hevc/receipt.json']
    for name in copied:
        target=evidence/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(run/name,target)
    write(evidence/'README.md',f'# 已知素材迁移初筛证据\n\n冻结计划、两个素材的完整链路/评价/逐帧CSV、H.265匹配记录和独立核验。原始大数组和视频保留本地运行目录。\n\n[报告](../../{REPORT_NAME}) · [原始运行]({relative(run,evidence)})\n\n仅已知video2/3开发素材，不是新留出或长视频验收；当前用户主观反馈 pending。\n')
    dump(evidence/'manifest.json',{name:file_sha(evidence/name) for name in copied})
    entry_paths=[REPO/'README.md',REPO/'交接文档.md',REPO/'当前项目状态与接续.md',REPO/'新会话接续.txt',
        STAGE/'README.md',STAGE/'当前方案总结与瓶颈路线_20261111.md',STAGE/'下一阶段_海洋素材与长视频泛化计划_20261111.md',
        REPO/'05_资料与索引/会话收束_20261110/文件索引.md',REPO/'05_资料与索引/会话收束_20261110/接续状态.json',
        WORKSPACE/'05_成果与观看/README.md']
    snapshots=run/'entry_snapshots';snapshots.mkdir();snapshot_manifest={}
    for index,path in enumerate(entry_paths):
        snapshot=snapshots/f'{index:02}_{path.name}';shutil.copy2(path,snapshot)
        snapshot_manifest[str(path)]=dict(snapshot=snapshot.name,sha256=file_sha(path))
    dump(snapshots/'manifest.json',snapshot_manifest)
    for path in entry_paths:
        if path.suffix!='.md':continue
        old=path.read_text(encoding='utf-8')
        old=old.replace('**2026-11-11 本轮精确缓存与完整文件链路已完成：**','**上一轮记录：2026-11-11 精确缓存与完整文件链路已完成：**')
        note=f'**最新：video2/3 固定参数迁移初筛完成（2026-11-11）。** video2 全帧时序MAE仅改善0.95%，局部有退化；video3改善46.67%。码率分别30.9904/33.0920kbps；缓存版与原I12逐位一致。旧开发素材初筛，主观验收待看；新留出与长视频仍未验证。[初筛报告]({relative(report,path.parent)}) · [原速对照]({relative(viewing/"README.md",path.parent)})。本轮未调参数、未提交/推送。下方旧轮记录保留其当时状态。'
        first,sep,rest=old.partition('\n');new=first+'\n\n'+note+'\n'+rest
        if path.name=='当前项目状态与接续.md':
            new=new.replace('| 泛化 | 当前组合的 video2/3、长片、留出未验证 |','| 泛化 | video2/3初筛完成：video2弱、video3正面；用户待看，长片/留出未验证 |')
            new=new.replace('| 下一步 | 固定方案的其他海洋素材与长视频泛化；本轮暂不扩展素材 |','| 下一步 | 保留video2反例，冻结诊断方案；随后独立素材与长片 |')
        write(path,new)
    text_path=REPO/'新会话接续.txt'
    prior=text_path.read_text(encoding='utf-8')
    text=f'''GLC 接续，2026-11-11，已完成 video2/3 当前方案迁移初筛。
先读：{report}
原速观看：{viewing/'README.md'}
权威运行：{run}
固定 S8I+CURRENT_I12、六锚点/12次/FP32/T3，没有逐片调参。两片缓存/原I12逐位一致，基础码流和解码复现历史。
video2：30.9904kbps，全帧时序MAE仅下降0.95%，PSNR下降0.301dB，相对旧K误差高40.04%；是弱例，勿宣称普适成功。
video3：33.0920kbps，MAE下降46.67%，历史珊瑚区域下降61.63%；PSNR下降0.166dB，动态变化幅度下降须关注。
均为既有开发素材前10秒，主观验收待用户观看；独立留出/长视频未运行，既有部署默认未改。
下一步先冻结video2失败机制诊断，保留video3正例；长片需修改固定150帧协议/锚调度/有界缓存。
本轮新增代码、报告、小证据和入口修改未提交/推送。不要覆盖冻结输入、原始参考链或重跑旧release脚本。

----- 上一轮接续原文（历史状态） -----
'''
    write(text_path,text+prior)
    state_path=REPO/'05_资料与索引/会话收束_20261110/接续状态.json';state=load_json(state_path)
    state['status']='accepted_I12_known_video_transfer_mixed_long_video_pending'
    state['user_latest_decision']='找回之前使用过的视频，用当前流程初步检查泛用性。'
    state['current_session_work']='Completed fixed-parameter video2/video3 full chains, cached/uncached equality, quality evaluation and matched HEVC views; mixed transfer results.'
    state['new_experiment_report']=relative(report,REPO)
    state['workspace_paths']['latest_viewing']=relative(viewing/'README.md',WORKSPACE)
    state['workspace_paths']['current_report']=relative(report,WORKSPACE)
    state['workspace_paths']['known_video_transfer_run']=relative(run,WORKSPACE)
    state['latest_known_video_transfer']=summary
    state['next_priority']=['Freeze a diagnosis of video2 transfer weakness; retain video3 as a positive regression case',
        'Independent held-out marine clips; do not tune separately for each clip',
        'Generalize 150-frame guide/anchor protocol and bound caches before long-video trials']
    state['new_experiments_uncommitted']=True;dump(state_path,state)
    failure=run.with_name('known_video_transfer_20261111_01')
    if failure.exists() and not any(failure.iterdir()):
        dump(failure/'failure.json',dict(stage='prepare',GPU_inference_started=False,
            cause='Controller assumed temporal_filter.py was in stage21; actual shared module is imported from its original stage',
            resolution='Corrected the new controller before freezing/running _02; inference code unchanged'))
    check_inputs(plan['input_sha256'])
    record=dict(complete=True,report=str(report),viewing=str(viewing),evidence=str(evidence),
        viewing_sha256=links,entry_snapshots=str(snapshots),entry_paths=[str(p) for p in entry_paths],
        publisher_sha256=file_sha(__file__),source_snapshot=str(run/'source_snapshot'))
    dump(run/'publication.json',record)
    print(json.dumps(dict(complete=True,report=str(report),viewing=str(viewing),videos=len(links)),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    publish(p.parse_args().run.resolve())
