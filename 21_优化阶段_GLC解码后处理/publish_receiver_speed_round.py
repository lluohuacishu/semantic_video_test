"""Publish only a fully verified timing round, preserving previous entry snapshots."""
import argparse
import csv
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
from runtime_speed_common import *


REPORT='接收链路计时与精确缓存提速_20261111.md'
FUTURE='下一阶段_海洋素材与长视频泛化计划_20261111.md'


def publish(folder):
    t3=load_json(folder/'t3/summary.json');raft=load_json(folder/'raft/summary.json')
    chain=load_json(folder/'complete_chain/summary.json');verification=load_json(folder/'verification/summary.json')
    if not all(v['complete'] for v in (t3,raft,chain,verification)):raise ValueError('Round is incomplete')
    trials=load_json(folder/'complete_chain/trials.json');rt=load_json(folder/'raft/trials.json')
    if len(trials)!=12 or not verification['all_artifacts_match']:raise ValueError('Verification incomplete')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    summary=dict(date='2026-11-11',complete=True,base_commit=head,
        profile='S8I + CURRENT_I12 with exact T3/RAFT computation caches',
        fixed_keys=list(KEYS),iterations=12,source_free_receivers=True,
        base_bytes=31963,guide_bytes=8865,total_kbps=32.6624,
        bit_exact_to_accepted_I12=True,bit_exact_to_I24=False,
        t3=t3,raft=raft,complete_chain=chain,
        original_raft_peak_mib=max(r['peak_gpu_allocated_mib'] for r in rt if r['mode']=='original'),
        selected_raft_peak_mib=max(r['peak_gpu_allocated_mib'] for r in rt if r['mode']=='feature_cache'),
        next_direction='Fixed-parameter generalization to other marine clips and long videos; not run in this round',
        existing_deployment_defaults_changed=False,new_work_committed_or_pushed=False)
    save_json(folder/'summary.json',summary)
    with (folder/'complete_chain/timings.csv').open('x',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['temperature','index','mode','external_seconds','fps','first_post_rgb_seconds',
            'source_seconds','sender_seconds','guide_encode_seconds','base_decode_seconds','flow_seconds','T3_seconds','post_total_seconds'])
        writer.writeheader()
        for r in trials:
            s=r['stages'];writer.writerow(dict(temperature=r['temperature'],index=r['index'],mode=r['mode'],
                external_seconds=r['external_task_wall_seconds'],fps=r['complete_task_fps'],
                first_post_rgb_seconds=r['first_frame_seconds']['first_post_rgb'],source_seconds=s['source']['seconds'],
                sender_seconds=s['sender']['request_wall_seconds'],guide_encode_seconds=s['guide_sender']['seconds'],
                base_decode_seconds=s['receiver']['request_wall_seconds'],flow_seconds=s['post']['flow']['seconds'],
                T3_seconds=s['post']['render_seconds'],post_total_seconds=s['post']['request_wall_seconds']))
    evidence=STAGE/'evidence_20261111/receiver_speed';evidence.mkdir(exist_ok=False)
    files=['execution_plan.json','summary.json','t3/summary.json','t3/trials.json','raft/plan.json','raft/summary.json',
        'raft/trials.json','raft/diagnostic.json','complete_chain/plan.json','complete_chain/summary.json',
        'complete_chain/trials.json','complete_chain/timings.csv','verification/summary.json','fast_entry_smoke/receipt.json',
        'fast_entry_smoke/entry_receipt.json']
    for name in files:
        target=evidence/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(folder/name,target)
    def med(temp,mode,path):
        vals=[]
        for r in trials:
            if r['temperature']==temp and r['mode']==mode:
                v=r
                for part in path:v=v[part]
                vals.append(v)
        return float(np.median(vals))
    table=[]
    for temp,label in (('cold','进程冷启动'),('warm','模型常驻')):
        a,b=chain['timing'][temp]['original'],chain['timing'][temp]['optimized']
        table.append(f'| {label} | {a["median"]:.3f}（{a["minimum"]:.3f}–{a["maximum"]:.3f}） | {b["median"]:.3f}（{b["minimum"]:.3f}–{b["maximum"]:.3f}） | {chain["timing"][temp]["median_time_reduction_percent"]:.2f}% | {150/b["median"]:.3f} |')
    parts=[('源视频抽帧/数组保存',['stages','source','seconds']),('GLC 发送请求',['stages','sender','request_wall_seconds']),
        ('S8I 引导两遍编码/封装',['stages','guide_sender','seconds']),('基础 GLC 独立接收请求',['stages','receiver','request_wall_seconds']),
        ('引导解码',['stages','post','guide_decode_seconds']),('RAFT（含特征缓存构建/回传）',['stages','post','flow','seconds']),
        ('T3 门控与融合',['stages','post','render_seconds']),('无损 RGB/alpha/shared 写盘',['stages','post','output_save_seconds']),
        ('观看 MP4 编码',['stages','post','viewing_encode_seconds']),('后处理完整请求（覆盖上述后处理分项）',['stages','post','request_wall_seconds'])]
    stage_rows=[f'| {name} | {med("warm","original",path):.3f} | {med("warm","optimized",path):.3f} |' for name,path in parts]
    cold=chain['timing']['cold'];warm=chain['timing']['warm']
    report=f'''# 接收链路计时与精确缓存提速

2026-11-11。本轮按用户要求完成计时、T3 缓存和 RAFT 独立提速；下一阶段转向其他海洋素材与长视频泛化，本轮未运行泛化实验。

## 结果与采用边界

- 算法仍是 S8I + CURRENT_I12：六锚点 0/15/45/75/105/135、FP32/12 次、无 TF32，原 T3 阈值、六帧过渡和两次取整不变。
- 新执行路径与已采用 I12 的完整 RGB、alpha、shared **逐位相同**；不是宣称 I12 与 I24 逐位相同。画质与残余跳变没有因此改善或恶化。
- 所有完整链路都重新从原视频抽帧、实际 GLC 编码及 S8I 两遍编码，基础包仍 31,963 B，引导仍 8,865 B，合计 **32.6624 kbps**，且字节哈希与原包相同。
- 新增显式研究入口 [receive_current_i12_fast.py](receive_current_i12_fast.py)，默认使用精确特征缓存及 T3 缓存；现有网页/独立部署包默认未修改。当前仍固定 150 帧和当前引导格式。
- 完成 6 次 T3 重放、9 次 RAFT 计时、12 次完整链路计时（每温度/配置 3 次），另有一次常驻完整暖机和一次显式入口核验。所有结果和失败检查均保留。

## 1. 完整本地文件链路

计时从原视频抽帧开始，经过 GLC 基础编码、S8I 编码/封装、独立基础解码、引导解码、RAFT、T3、无损数组写盘及观看 MP4 编码。外层计时包含实际子进程启动和运行；基准驱动器最初的 Python 导入、末尾独立回归核验及最后一个 worker 退出不在任务计时内。没有网络传输、浏览器播放或水声链路。

冷指每次重新创建三个独立进程，不是清空操作系统文件缓存或机器冷开机；热指三个角色模型常驻，先执行一次完整暖机，暖机不入表。任务间释放临时 CUDA 缓存，每段重新计算图像特征。发送仍 eager FP32/fill-on，与冻结基础编码一致；基础接收仍 eager FP32/fill-off，未夹带 TensorRT 切换。

以下为每段 150 帧／10 秒内容的秒数，中位数（最小–最大）。原 I12 与缓存版采用预先冻结的平衡顺序，每档三次。

| 条件 | 原 I12 完整任务秒 | 精确缓存版完整任务秒 | 中位耗时减少 | 缓存版任务吞吐 FPS |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

这仍远低于 15 FPS，不是实时。热模型任务吞吐也不等于流式稳态播放 FPS。

冷/热缓存版第一次后处理 RGB 算得时刻中位数分别为 **{med('cold','optimized',['first_frame_seconds','first_post_rgb']):.3f}/{med('warm','optimized',['first_frame_seconds','first_post_rgb']):.3f} 秒**；这只是进程通知的首张已计算 RGB，不是已发布的可播放帧。当前离线实现先完成整段光流，再开始渲染；最终可观看文件在任务完成后才完整可用。

### 常驻模型的分阶段时间

| 阶段 | 原 I12 中位秒 | 精确缓存版中位秒 |
| --- | ---: | ---: |
{chr(10).join(stage_rows)}

分项为各自三次中位数，不能直接相加当成另一条实测总时间；后处理完整请求包含其分项，不能重复计入。基础编解码与启动成本稀释局部提速。历史 25.8169 秒只是不含基础解码的另一轮后处理组件和，不能与本表直接比较或拼接。

## 2. T3 精确缓存

缓存 sigma3/sigma1 RGB 模糊和坐标网格，采用四项 LRU，只保存当前/新锚/旧锚附近的数据；额外 RGB 模糊缓存上限约 13.5 MiB。纹理标准差沿用原来的预计算，不重复宣称新优化。

完整 150 帧重放、6 次交替：原实现中位 **{t3['times']['original']['median']:.4f} 秒**，缓存版 **{t3['times']['cached']['median']:.4f} 秒**，减少 **{t3['median_time_reduction_percent']:.2f}%**。计入缓存构建、纹理门控、融合、输出数组分配及零权重核验，排除输入/输出文件操作。六次 RGB/alpha/shared 均逐位等于采用档。该局部比例不作为整条链路加速比例。

## 3. RAFT 独立实验与显存代价

| 路径 | 三次中位秒 | 全部 174 双向调用的光流逐位相同 | 峰值 allocated MiB |
| --- | ---: | --- | ---: |
| 原始 forward | {raft['times']['original']['median']:.4f} | 是 | {summary['original_raft_peak_mib']:.1f} |
| 仅末次上采样 | {raft['times']['final_only']['median']:.4f} | 是 | {max(r['peak_gpu_allocated_mib'] for r in rt if r['mode']=='final_only'):.1f} |
| 末次上采样 + 特征/上下文复用 | {raft['times']['feature_cache']['median']:.4f} | 是 | {summary['selected_raft_peak_mib']:.1f} |

特征缓存的同口径中位耗时减少 **{raft['median_time_reduction_percent']:.2f}%**，包含每次缓存构建、图像上卡、光流回传和本地 RAM 数组写入；没有把预计算藏在计时外，也没有跨任务偷用上一段特征。维持 fnet batch=4、cnet batch=2，使用相同权重和递归算术；实际验证全部 348 个方向一致。独立渲染也逐位一致。

仅减少中间上采样没有稳定中位收益，因此不单独宣称加速成功。另对 7 个代表性双向调用做了 CUDA 事件分段：更新循环仍占主要部分，特征与上下文是其次的一部分。事件诊断会扰动执行，不能把累计事件时间当作排他 GPU 内核时间，也不能乘调用数当完整耗时。

特征缓存使本次峰值 GPU allocated 从约 545 MiB 增至约 1,645 MiB，增加约 **1,100 MiB**。这是明确的速度/显存取舍，缓存当前整段特征，不能未经改造延伸到任意长视频。显存较紧时可显式选择 `--raft-mode original`，保留 T3 缓存；本轮未改变正式部署默认。

## 4. 数据边界、资源和限制

独立基础接收和后处理进程均实施读取审计，并主动验证 7 个源/发送端输入被拒绝。后处理读取本轮接收到的两份包及独立基础解码结果，不读取源图像、源运动、源掩码或发送端收据。输出不回灌 GLC 潜变量参考链。

逐次记录各 worker 的 PyTorch GPU 峰值 allocated/reserved，以及 Windows 进程工作集和进程生命周期峰值。后者在常驻模式会包含此前任务，不是每段独立峰值；没有把不同时刻、不同进程的峰值相加伪装成全系统同步峰值，也没有包含 FFmpeg 子进程的 CPU 峰值。原本地三份光流约 405 MiB；本轮完整链路把光流存放 RAM，两配置采用同一输出策略，不写历史诊断光流大文件。

只有一台 RTX 4060 Laptop、一个开发片段、每配置三次。GPU 温度/负载快照已记录，未锁频、未排除所有桌面负载、没有置信区间；冷启动首次加载和系统文件缓存会造成波动。没有 Orin、跨 GPU、网络/FEC、跨素材、长片或流式验收。

## 5. 复现与接续

原始运行：`{folder}`。小证据：[完整目录](evidence_20261111/receiver_speed/README.md)。原始 JSON/CSV、过程日志、源码快照、13 次完整链路产物和显式入口产物均在新目录；原码流、数组、权重和历史源码保持不变。执行代码 SHA 已核验，没有运行后改写推理实现。

局部与完整计时的入口依次为 `run_t3_cache_benchmark.py`、`run_raft_speed_benchmark.py`、`run_complete_runtime_benchmark.py`，均使用同一个**全新** `--run` 目录（第一步创建，后两步扩展）；完整链路会实际重跑 GPU。`verify_receiver_speed_round.py` 在计时完成后做独立入口和保存文件核验；发布脚本拒绝已有发布目录，避免把验收写回 pending。

单次精确缓存后处理入口为 `receive_current_i12_fast.py --base <base.glcsnd> --base-receipt <独立解码回执> --raw <decoded_rgb.npy> --guide <guide.glcguide> --output <全新目录>`。此入口固定当前 150 帧协议；它不包含基础 GLC 解码，也不是通用部署或完整链路命令。

下一阶段按用户要求转向 **其他海洋素材和长视频泛化**，详见[后续计划]({FUTURE})。先解决固定 150 帧/封装/锚调度与有界缓存接口，再冻结规则测试 video2/3、独立留出和长片；本轮不提前给泛化成立的结论。

工程起点为 `{head[:7]}`，分支 `codex/glc-temporal-stability`。本轮新增代码、报告、小证据和接续修改保留在工作区，尚未提交或推送；此前四宫格推送回执属于前一轮。
'''
    (STAGE/REPORT).write_text(report,encoding='utf-8')
    (evidence/'README.md').write_text('# 精确缓存与完整文件链路证据\n\n所有计时与验收均完成。算法和画面保持已采用 I12；泛化未运行。\n\n'+
        '\n'.join(f'- [{name}]({name})' for name in files)+'\n\n大视频/数组留在本机，NPY 与观看 MP4 不当作传输包。\n',encoding='utf-8')
    targets=[REPO/'README.md',REPO/'交接文档.md',REPO/'当前项目状态与接续.md',REPO/'项目总结与未来方向.md',REPO/'新会话接续.txt',
        STAGE/'README.md',STAGE/'当前方案总结与瓶颈路线_20261111.md',STAGE/'当前采用配置.json',
        REPO/'05_资料与索引/会话收束_20261110/文件索引.md',REPO/'05_资料与索引/会话收束_20261110/接续状态.json']
    snapshots=folder/'entry_snapshots';snapshots.mkdir()
    prior={}
    for path in targets:
        name=path.relative_to(REPO);target=snapshots/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);prior[str(name)]=file_sha(path)
    save_json(snapshots/'manifest.json',prior)
    note=(f'**2026-11-11 本轮精确缓存与完整文件链路已完成：** T3/RAFT 分别减少 {t3["median_time_reduction_percent"]:.2f}%/{raft["median_time_reduction_percent"]:.2f}% 的局部中位耗时；'
        f'完整任务冷启动 {cold["original"]["median"]:.2f}→{cold["optimized"]["median"]:.2f} 秒，常驻模型 {warm["original"]["median"]:.2f}→{warm["optimized"]["median"]:.2f} 秒／10 秒片段。'
        '码流及 I12 的 RGB/alpha/shared 逐位不变，仍非实时；RAFT 峰值显存约增加 1.1 GiB。下一阶段为其他海洋素材和长视频泛化，本轮未运行。')
    for path in [p for p in targets if p.suffix=='.md']:
        text=path.read_text(encoding='utf-8');lines=text.splitlines()
        link=os.path.relpath(STAGE/REPORT,path.parent).replace('\\','/')
        future=os.path.relpath(STAGE/FUTURE,path.parent).replace('\\','/')
        lines[1:1]=['',note+f' [本轮报告]({link}) · [泛化计划]({future})。本轮修改尚未提交/推送；下文旧计时与交付记录保留其原口径。','']
        text='\n'.join(lines)+'\n'
        if path.name=='交接文档.md':
            text=text.replace('本轮任务为整理收束，未追加优化实验。','本轮已完成精确缓存及冷/热完整文件链路计时，优先读页首新报告。')
            text=text.replace('下一轮先补完整链路计时，再做 T3 缓存逐位验证、RAFT 独立提速。固定六锚点、当前 S8I 包与 T3 作为起点。',
                '计时、T3 缓存逐位验证及 RAFT 独立提速本轮已完成。下一阶段转向其他海洋素材与长片泛化；先处理固定帧数与有界缓存，当前六锚点/S8I/T3 仍为算法基线。')
        if path==STAGE/'README.md':
            text=text.replace('先补可信的端到端/冷热重复计时，再优化 T3 重复计算和 RAFT 主耗时。锚点数不减少，辅助预算继续以完整文件计费。视觉优化、留出/长片、当前总码率 H.265 和流式时延分开验证，尚未运行新一轮。',
                '本轮计时与精确缓存已完成。下一阶段以固定参数的其他海洋素材和长视频泛化为主；先处理 150 帧硬编码、封装/锚调度和有界缓存。完整字节预算、动态边界、长片失败和流式时延分别验证。')
        if path.name=='当前项目状态与接续.md':
            text=re.sub(r'\| 时间 \|.*',f'| 时间 | 已完成每档 3 次冷/热完整文件链路；常驻中位 {warm["original"]["median"]:.2f}→{warm["optimized"]["median"]:.2f} 秒／10 秒片段 |',text)
            text=re.sub(r'\| 下一步 \|.*','| 下一步 | 固定方案的其他海洋素材与长视频泛化；本轮暂不扩展素材 |',text)
            text=re.sub(r'\| 版本 \|.*',f'| 版本 | codex/glc-temporal-stability，起点 {head[:7]}；本轮修改未提交/推送 |',text)
        if path.name=='当前方案总结与瓶颈路线_20261111.md':
            text=re.sub(r'## 7\..*?(?=## 8\.)',f'## 7. 本轮完成及下一阶段\n\n原 P0/P1/P2 的本轮计时、T3 精确缓存和 RAFT 独立验证已完成，详见[本轮报告]({REPORT})。完整本地文件链路仍非实时，特征缓存有显存代价，不能把局部百分比相乘或叠加历史时间。\n\n用户已指定下一阶段以其他海洋视频和长视频泛化为主。先处理当前固定 150 帧、引导封装、锚规则和有界缓存，再冻结参数测迁移/留出/长片。本轮没有运行泛化，详细完成条件见[后续计划]({FUTURE})。\n\n',text,flags=re.S)
        path.write_text(text,encoding='utf-8')
    cfg=load_json(STAGE/'当前采用配置.json')
    cfg['runtime_acceleration']=dict(date='2026-11-11',status='verified_bit_exact_research_entry_video1_only',
        entrypoint='receive_current_i12_fast.py',raft_mode='feature_cache',T3_cache='four-entry RGB blur LRU + coordinates',
        bit_exact_to_CURRENT_I12=True,transmitted_bytes_unchanged=True,run=str(folder),report=REPORT,
        complete_file_chain=chain['timing'],peak_raft_allocated_mib=summary['selected_raft_peak_mib'],
        memory_tradeoff='About +1100 MiB peak allocated; all-frame features require bounded redesign for long video',
        original_measurements_preserved=True,deployment_defaults_changed=False)
    cfg['next_priority']=['Fixed-parameter generalization to other marine clips',
        'Generalize 150-frame protocol and anchor schedule; bound feature/flow memory before long videos',
        'Known video2/3 transfer, then independent held-out scenes and long sequences',
        'Evaluate dynamic boundaries, jumps, complete byte budget, latency and failures']
    save_json(STAGE/'当前采用配置.json',cfg)
    path=REPO/'05_资料与索引/会话收束_20261110/接续状态.json';state=load_json(path)
    state.update(status='accepted_I12_exact_runtime_optimized_generalization_next',new_experiments_uncommitted=True,
        new_experiment_report='21_优化阶段_GLC解码后处理/'+REPORT,last_experiment_commit=head,
        user_latest_decision='先完成本轮计时与精确缓存提速；下一阶段为其他海洋视频与长视频泛化，本轮不急着开展泛化。',
        current_session_work='Completed exact T3/RAFT caches and 12 cold/warm complete local file chains; generalization deferred.')
    state['latest_runtime_optimization']=cfg['runtime_acceleration']
    state['previous_publication']=state.get('publication',{})
    state['publication']=dict(base_commit=head,current_changes_committed=False,pushed=False,
        scope='This timing/cache round remains in the working tree; prior four-way publication is archived separately')
    state['next_priority']=cfg['next_priority']
    state['workspace_paths']['latest_receiver_speed_run']=str(folder.relative_to(WORKSPACE))
    state['workspace_paths']['current_report']='01_项目/代码/semantic_video_test/21_优化阶段_GLC解码后处理/'+REPORT
    save_json(path,state)
    (REPO/'新会话接续.txt').write_text(f'''GLC 接续，2026-11-11。\n当前分支 codex/glc-temporal-stability；本轮起点 {head}，本轮代码/报告/小证据/入口修改未提交或推送。\n先读：{STAGE/REPORT}\n后续主线：{STAGE/FUTURE}\n本轮已完成：T3 六次全序列重放、RAFT 九次独立计时、12 次冷/热完整本地文件链路、完整暖机和显式入口核验。\n{note}\n算法仍为 S8I+CURRENT_I12，六锚点不减、FP32/12次/无TF32、T3/过渡/取整不变。新缓存与 I12 逐位相同，不代表 I12 与 I24 相同。\n当前单次精确缓存入口 receive_current_i12_fast.py，仅后处理，固定当前 150 帧引导格式；旧 run_anchor_time_experiment.py 保留回归。网页/部署包默认未整合。\n权威本轮运行：{folder}\n当前历史采用 RGB 路径继续保留，新结果哈希相同；现有03/04/07观看片仍代表相同画面。\n完整链路为原视频→实际两路编码→独立基础解码→后处理→无损数组/观看视频，不含网络/浏览器，仍非实时。冷/热定义、分项、内存及计时边界见报告。\n本轮无新素材、长片、留出、Orin、网络或流式验收；下一阶段先处理固定帧数、封装/锚规则与有界缓存，再冻结配置验证泛化。\n所有原始基础流、源数组、参考链和权重不覆盖；失败/原入口快照保留。旧 release_anchor_time_experiment.py 仍会写回 pending，不可盲目重跑。\n第19阶段及GLVC/UVID文献为无关未跟踪资产，保持原状。\n''',encoding='utf-8')
    manifest={str(p.relative_to(evidence)):dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in evidence.rglob('*') if p.is_file()}
    save_json(evidence/'manifest.json',manifest)
    save_json(folder/'publication.json',dict(complete=True,report=str(STAGE/REPORT),evidence=str(evidence),
        entry_snapshots=str(snapshots),updated_entries=[str(p) for p in targets],committed=False,pushed=False))
    print('PUBLICATION COMPLETE',STAGE/REPORT,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();publish(a.run.resolve())
