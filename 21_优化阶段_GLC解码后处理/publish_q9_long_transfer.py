"""Publish Q9 long/archived transfer evidence without changing adoption."""
import argparse
import csv
import math
import os
from pathlib import Path
import shutil

from runtime_speed_common import STAGE,REPO,WORKSPACE,file_sha,load_json,save_json,check_inputs

REPORT='Q9跨素材与82秒长片验证_20261112.md'
NAMES={'reef_spotted_fish':'礁石斑点鱼','seagrass_spotted_fish':'海草斑点鱼'}


def relative(path,parent):return Path(os.path.relpath(path,parent)).as_posix()
def put(path,text):path.write_text(text.rstrip()+'\n',encoding='utf-8',newline='\n')
def change(a,b):return 100*(a/b-1)
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def aggregation_check(run,summary,archive):
    checked=[]
    for label,folder,scores,dynamic,long in [('long',run/'evaluation',summary['scores'],summary['scores']['full'],True)]+[
        (case,run/'archive_transfer'/case/value['evaluation_dir'],value['scores'],value['dynamic'],False) for case,value in archive['cases'].items()]:
        def read(name):
            with (folder/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
        rows=read('temporal.csv');spatial=read('spatial.csv');dr=read('dynamic.csv')
        for scope,methods in scores.items():
            support={}
            for method,value in methods.items():
                rr=[r for r in rows if r['scope']==scope and r['method']==method]
                n=sum(int(r['pixels']) for r in rr)
                assert n==value['valid_pixels'] and len(rr)==value['valid_pairs' if long else 'pairs']
                support[method]=[(int(r['frame']),int(r['pixels'])) for r in rr]
                for out,column in [('coarse_mae','coarse_sum'),('fine_mae','fine_sum'),('full_mae','full_sum')]:
                    expected=sum(float(r[column]) for r in rr)/n if n else None
                    assert expected is None and value[out] is None or math.isclose(expected,value[out],rel_tol=1e-12,abs_tol=1e-12)
                ss=[r for r in spatial if r['scope']==scope and r['method']==method]
                mse=sum(float(r['rgb_sse']) for r in ss)/(3*sum(int(r['pixels']) for r in ss))
                psnr=10*math.log10(255**2/mse) if mse else None
                assert psnr is None and value['psnr_db'] is None or math.isclose(psnr,value['psnr_db'],abs_tol=1e-10)
            assert all(v==support['raw'] for v in support.values())
        for method,value in dynamic.items():
            rr=[r for r in dr if r['method']==method];n=sum(int(r['pixels']) for r in rr)
            error=sum(float(r['delta_error_sum']) for r in rr)/(3*n)
            amp=sum(float(r['change_sum']) for r in rr)/sum(float(r['source_change_sum']) for r in rr)
            assert math.isclose(error,value['dynamic_error' if long else 'RGB_delta_error'],abs_tol=1e-10)
            assert math.isclose(amp,value['change_amplitude_ratio'],abs_tol=1e-10)
        if long:
            groups={'boundary_exact':lambda i:i in range(150,1239,150),
                    'boundary_next20':lambda i:any(b<=i<=b+20 for b in range(150,1239,150))}
            for group,predicate in groups.items():
                for method,value in summary[group].items():
                    rr=[r for r in rows if r['scope']=='full' and r['method']==method and predicate(int(r['frame']))]
                    assert len(rr)==value['valid_pairs']
                    assert math.isclose(sum(float(r['coarse_sum']) for r in rr)/sum(int(r['pixels']) for r in rr),value['coarse_mae'],abs_tol=1e-12)
        checked.append(label)
    save_json(run/'aggregation_verification.json',dict(complete=True,cases=checked,
        common_support_equal_across_methods=True,csv_temporal_spatial_dynamic_reaggregation_matches=True,
        boundary_exact_and_next20_reaggregation_matches=True,verifier_sha256=file_sha(__file__)))


def publish(run):
    plan=load_json(run/'execution_plan.json');summary=load_json(run/'summary.json')
    archive=load_json(run/'archive_transfer/summary.json');verify=load_json(run/'verification.json')
    assert summary['complete'] and archive['complete'] and verify['complete']
    assert load_json(run/'archive_transfer/verification.json')['complete']
    media_checks=load_json(run/'previews/verification.json');assert media_checks['complete']
    check_inputs(plan['input_sha256'])
    for n,h in plan['code_sha256'].items():assert file_sha(STAGE/n)==h
    config=STAGE/'当前采用配置.json';config_sha=file_sha(config)
    assert config_sha==plan['input_sha256'][str(config)]
    report=STAGE/REPORT;view=WORKSPACE/'05_成果与观看/GLC_Q9跨素材与长片_20261112'
    evidence=STAGE/'evidence_20261112/q9_long_transfer'
    for path in (report,view,evidence,run/'publication.json'):
        if path.exists():raise FileExistsError(path)
    aggregation_check(run,summary,archive)
    save_json(run/'visual_inspection.json',dict(date='2026-11-12',method='Viewed contact images and inspected decoded sampled preview panels',
        long_frames=[0,619,1238,150,300,450,600,750,900,1050,1200],archive_frames=[20,64,80,128,149],
        observed='Panel ordering and image content are consistent. Existing GLC texture/color losses remain; static frames do not establish temporal flicker quality.',
        user_subjective_review='pending for new long and two archived cases; prior positive Q9 video1/2/3 feedback preserved'))
    save_json(run/'verification_source/source_note.json',dict(
        long_verifier_snapshot='finish_q9_long_transfer_v1.py',long_verifier_sha256=verify['verifier_sha256'],
        archive_verifier_current_sha256=file_sha(STAGE/'finish_q9_long_transfer.py'),
        difference='After long verification/media completed, archive verification was extended to evaluation_valid_support and explicit excluded-pair counts. No inference source changed.',
        resume='Original archive evaluator stopped on low support; frozen inference preserved. New resume script reuses completed first case and runs same registered second case.'))
    assert file_sha(run/'verification_source/finish_q9_long_transfer_v1.py')==verify['verifier_sha256']==media_checks['verifier_sha256']
    view.mkdir();evidence.mkdir(parents=True)
    files=[]
    mappings=[(run/'previews/long_raw_vs_Q9.mp4','01_82秒_原GLC与Q9连续.mp4'),
              (run/'previews/long_source_raw_continuous_reset.mp4','02_82秒_原片原GLC连续重置四宫格.mp4'),
              (run/'previews/all8_boundaries_continuous_vs_reset.mp4','03_八处块边界_连续与重置.mp4')]
    for i,(case,c) in enumerate(archive['cases'].items()):
        folder=run/'archive_transfer'/case/c['evaluation_dir']
        mappings.extend([(folder/'raw_vs_Q9.mp4',f'{4+2*i:02d}_{NAMES[case]}_原GLC与Q9.mp4'),
                         (folder/'source_raw_Q9.mp4',f'{5+2*i:02d}_{NAMES[case]}_原片原GLCQ9.mp4'),
                         (folder/'contact.png',case+'_contact.png')])
    mappings.extend((p,p.name) for p in (run/'previews').glob('*.png'))
    mappings.append((run/'source_overlap_contact.png','source_overlap_contact.png'))
    for src,name in mappings:
        dst=view/name;os.link(src,dst);h=file_sha(src);assert file_sha(dst)==h
        files.append(dict(source=str(src),destination=str(dst),sha256=h,bytes=src.stat().st_size,hardlink=True))
        if src.suffix=='.mp4':
            if src.parent==run/'previews':expected=media_checks['videos'][src.name]['sha256']
            else:expected=archive['cases'][src.parent.parent.name]['previews'][src.name]['sha256']
            assert h==expected
    assert sum(Path(r['source']).suffix=='.mp4' for r in files)==7
    save_json(view/'manifest.json',dict(complete=True,source_run=str(run),files=files,video_count=7))
    full=summary['scores']['full'];rates=summary['rates'];boundary=summary['boundary_exact'];next20=summary['boundary_next20']
    full_pct=change(full['continuous']['coarse_mae'],full['raw']['coarse_mae'])
    reset_pct=change(full['reset']['coarse_mae'],full['raw']['coarse_mae'])
    boundary_pct=change(boundary['continuous']['coarse_mae'],boundary['raw']['coarse_mae'])
    cont_vs_reset=change(boundary['continuous']['coarse_mae'],boundary['reset']['coarse_mae'])
    long_rows=[]
    for n,title in [('raw','原GLC'),('continuous','Q9_R25跨块连续'),('reset','Q9_R25每块重置')]:
        v=full[n]
        long_rows.append([title,f"{v['coarse_mae']:.6f}",f"{change(v['coarse_mae'],full['raw']['coarse_mae']):+.2f}%",
                          f"{v['psnr_db']:.4f}",f"{v['dynamic_error']:.4f}",f"{v['change_amplitude_ratio']:.4f}"])
    edge_rows=[]
    for title,group in [('八个边界瞬间 b−1→b',boundary),('边界至其后20帧，共168对',next20),('全片1238对',full)]:
        edge_rows.append([title,*[f"{group[n]['coarse_mae']:.6f}" for n in ('raw','continuous','reset')],
                          f"{change(group['continuous']['coarse_mae'],group['reset']['coarse_mae']):+.2f}%"])
    short_rows=[];cost_rows=[];compact={}
    for case,v in archive['cases'].items():
        a=v['scores']['full']['raw'];b=v['scores']['full']['Q9_R25'];da=v['dynamic']['raw'];db=v['dynamic']['Q9_R25']
        compact[case]=dict(full_mae_change_percent=change(b['coarse_mae'],a['coarse_mae']),psnr_change_db=b['psnr_db']-a['psnr_db'],
            dynamic_error_change_percent=change(db['RGB_delta_error'],da['RGB_delta_error']),
            change_amplitude_ratio_raw=da['change_amplitude_ratio'],change_amplitude_ratio_Q9=db['change_amplitude_ratio'],total_kbps=v['total_kbps'])
        c=compact[case]
        short_rows.append([NAMES[case],f"{a['coarse_mae']:.6f}→{b['coarse_mae']:.6f}",f"{c['full_mae_change_percent']:+.2f}%",
            f"{c['psnr_change_db']:+.4f}",f"{c['dynamic_error_change_percent']:+.2f}%",f"{da['change_amplitude_ratio']:.4f}→{db['change_amplitude_ratio']:.4f}"])
        cost_rows.append([NAMES[case],'10.0',v['base_bytes'],v['guide_bytes'],0,f"{v['base_kbps']:.4f}",f"{v['guide_kbps']:.4f}",f"{v['total_kbps']:.4f}"])
    cost_rows.insert(0,['完整长片','82.6',rates['base_bytes'],rates['packet_bytes'],rates['index_bytes'],f"{rates['base_bytes']*8/82.6/1000:.4f}",f"{rates['auxiliary_kbps']:.4f}",f"{rates['total_kbps']:.4f}"])
    quadrant_rows=[]
    for case,scores,method in [('82.6秒',summary['scores'],'continuous')]+[(NAMES[c],v['scores'],'Q9_R25') for c,v in archive['cases'].items()]:
        quadrant_rows.append([case,*[f"{change(scores[s][method]['coarse_mae'],scores[s]['raw']['coarse_mae']):+.2f}%" for s in ('top_left','top_right','bottom_left','bottom_right')]])
    segment_rows=[]
    for b,group in summary['segments'].items():
        start=int(b);segment_rows.append([f'{start/15:.1f}–{min(start+150,1239)/15:.1f}s',
            *[f"{change(group[m]['coarse_mae'],group['raw']['coarse_mae']):+.2f}%" for m in ('continuous','reset')]])
    rec=load_json(run/'receive/receipt.json');memory=verify['memory']
    result=dict(complete=True,run=str(run),report=str(report),viewing=str(view),profile='Q9_R25',
        long=dict(frames=1239,duration_seconds=82.6,full_mae_change_percent=full_pct,reset_full_mae_change_percent=reset_pct,
                  boundary_exact_change_vs_raw_percent=boundary_pct,boundary_exact_change_vs_reset_percent=cont_vs_reset,
                  total_kbps=rates['total_kbps'],gpu_cache_max_frames=180,constant_process_RAM_proven=False),
        archive=compact,independent_holdout=False,realtime=False,subjective_review='pending_new_cases',
        previous_Q9_video1_video2_video3_positive_feedback_preserved=True,automatic_adoption=False,current_profile_sha256=config_sha)
    save_json(run/'decision.json',result)
    text=f'''# Q9跨素材与82秒长片验证

2026-11-12；状态：**本轮全部完成，无实验继续运行**。固定Q9_R25跑完1239帧/15FPS/82.6秒历史原片，并完成两段额外历史海洋原片的前10秒。本轮比较对象是各自的**原始GLC解码**，不是上一轮Q6_T3；百分比不可混用。

**结论：Q9_R25可以扩展到这条完整长片，两段不同场景也有时序指标收益；长片块边界与海草场景的运动保真仍是明确问题，尚不能宣布泛化已解决。**

[原速观看]({relative(view/'README.md',STAGE)}) · [长片核验]({relative(run/'verification.json',STAGE)}) · [两片核验]({relative(run/'archive_transfer/verification.json',STAGE)}) · [评分再汇总]({relative(run/'aggregation_verification.json',STAGE)}) · [小证据](evidence_20261112/q9_long_transfer/manifest.json)

## 与用户反馈的关系

用户已认可Q9三段视频，video2在Q9下很好；用户没有逐个评价Q9全部子档。上一轮单独展示的是Q9_R25，因此以它作为可追溯参照。Q6_D25的video2闪烁和跳变仍存在；video2目标仍是“改善一些即可”，不夸大严重性。

此次没有重写上一轮`selected=null`，也保留Q9_R25在video3珊瑚区域约+5.37%的数值记录。**既有三段正面主观反馈保持；本轮长片和两段新增素材的原速主观反馈待记录。** 当前采用配置和部署默认未切换。

## 固定实验设计

- 长片复用已经独立解码并核验reference checksum的真实完整GLC基础流；源共1239帧。历史基础流输入编码器是TRT FP16，不是上一轮短片新编码的FP32输入链。首10秒新旧基础RGB差异不可归因于Q9。
- 引导固定256×144/15FPS、HEVC slower two-pass、内部目标9kbps，按150帧窗口编码；最后39帧按实际长度编码，没有复制填满。
- 新GLCGDE02包携带完整基础流SHA、起点/帧数/总帧数、尺寸和CRC；9包按时序连续覆盖。索引manifest字节也计入辅助传输。
- continuous跨窗口保留最后30帧参考，锚点为全局0、15、45、75……；reset每150帧重新按0/15/45/75/105/135组织参考。两者使用同一基础流和同一批引导包。
- 两者均保持RAFT FP32/12次/无TF32、原R25门控、2px羽化、六帧旧参考渐退、原始GLC历史RGB参考和两次取整。没有场景切换检测，也未调素材专属阈值。
- 两段历史源是礁石斑点鱼与海草斑点鱼，运行前登记文件SHA和前10秒。29.97FPS按时间戳重采样至15FPS/150帧；没有通过改播放FPS人为延长时长。它们与既有三片的抽帧画面不同，仍不是严格独立留出集。

接收端只读基础流、独立基础解码和回执、收到的guide/index及预置权重；源图、源元数据、发送回执和旧优化RGB拒绝测试通过。Python audit属于可复现读入边界检查，不是操作系统安全沙箱。

## 完整长片结果

主指标为源运动对齐、公共有效支持上的低频变化误差coarse_mae；越低越好，**不等同肉眼闪烁或跳变次数**。PSNR另看同帧保真，动态误差与变化幅度比用于避免把抹平运动直接当作稳定成功。

{table(['输出','时序误差','相对原GLC','PSNR/dB','动态RGB误差','变化幅度/源'],long_rows)}

continuous的PSNR变化{full['continuous']['psnr_db']-full['raw']['psnr_db']:+.4f}dB，动态RGB误差变化{change(full['continuous']['dynamic_error'],full['raw']['dynamic_error']):+.2f}%。九个时段的平均时序误差都低于原GLC，最后39帧也完成；这不代表每帧、每个物体都改善。

{table(['时间段','continuous相对原GLC','reset相对原GLC'],segment_rows)}

### 块边界是当前主要问题

{table(['统计范围','原GLC','continuous','reset','continuous相对reset'],edge_rows)}

连续参考使八处边界瞬间的误差比重置降低{abs(cont_vs_reset):.2f}%，但仍比原GLC高{boundary_pct:.2f}%。重置对照的全片平均和边界后20帧平均反而更低，不能只凭边界瞬间一个数字认定continuous整体胜出。两模式仅在每个新块前21帧的RGB/alpha/shared有差异，其他帧逐位一致；紧随差异区的帧对仍可能受上一帧影响。

每个引导块独立编码也可能引入跨块外观或门控变化，这是待验证假设，本轮不能将全部峰值归因于重置参考。原基础流I帧等已有问题也仍存在。下一轮应在固定Q9下单独检验引导跨块连续性和支持变化，不宜直接选一个模式作为最终长片方案。

## 两段额外历史素材

{table(['素材','原GLC→Q9时序误差','误差变化','PSNR变化/dB','动态误差变化','变化幅度/源'],short_rows)}

礁石片的时序收益较大，PSNR小幅下降。海草片虽然时序误差降低，PSNR下降约0.22dB，源高变化像素上的变化幅度比从0.523降到0.326；动态误差指标改善也不能排除运动被压弱。需原速关注海草摆动、鱼鳍和细纹理，不能只按平均时序误差宣布它成功。

这两段不是旧video2的再次验收；本轮没有推翻用户对旧video2/Q9的正面反馈。新增素材较缓慢的变化也会使时序指标更易改善。

### 四象限相对原GLC的时序误差变化

{table(['素材','左上','右上','左下','右下'],quadrant_rows)}

长片左上有14个帧对的公共支持不足64像素；礁石片左上有10个，最低为0。对应scope/frame对所有方法一并标为缺测并排除，保留像素数，未用零误差代替。长片全帧1238对、两短片全帧各149对均有效；PSNR使用全部帧和像素。未为低支持区增加人工ROI。

## 实际传输成本

{table(['素材','秒','基础/B','引导包/B','索引/B','基础kbps','辅助kbps','合计kbps'],cost_rows)}

每路40kbps上限均满足；长片每个guide块也单独检查。Q9是编码器内部目标，不能把9直接当作实际辅助码率。基础/guide/index都按完整必要字节与真实时长计费。预置模型、接收本地光流/alpha、NPY和观看MP4不当作传输包；本轮没有新做同码率H.265对照。

## 工程可行性与限制

已证明：该82.6秒片完成9个窗口，GPU/输入缓存最大180帧，所记峰值GPU分配{memory['peak_gpu_allocated_mib']:.1f}MiB；首窗口与原150帧实现的RGB/alpha/shared逐位一致。tail39帧、包SHA/CRC/起点/基础绑定及输出shape/dtype/finite均核验。

**没有证明总内存常数有界。** 释放缓存后的进程Working Set从{memory['working_set_after_release_mib'][0]:.1f}MiB增长至{memory['working_set_after_release_mib'][-1]:.1f}MiB；全长mmap输出驻留页仍累积，后续需要分块关闭/重映射或分块输出并另测。不能据180帧GPU缓存声称任意长视频都可运行。

本次双模式诊断后处理墙钟{rec['postprocess_wall_seconds']:.2f}秒，包含模型加载、首窗原实现重放、两个输出、hash和IO，不是单模式公平测速；没有计入基础GLC编解码和guide发送端完整链路。整块two-pass、有窗口等待且绑定完整基础文件SHA，**仅离线文件处理，未验证网络流式或端到端实时**。

## 失败与恢复记录

1. 准备源数组时，历史尾chunk物理分配64帧、元数据只有23个有效帧。初次复制形状不匹配，尚未推理；保留失败JSON、当时源码和不完整数组。改为按有效帧数复制后才冻结正式计划并运行。
2. 礁石片首版评价在左上第51帧公共支持不足64时停止；该片推理已完成，全部输出保留。新评价目录`evaluation_valid_support`按长片已采用的缺测原则记录并跳过，64像素阈值、分数定义和推理参数未改；复用第一片输出，再运行原已登记的第二片。原失败目录和冻结代码保留。
3. 长片核验/观看执行源码v1保存在verification_source；之后仅扩展额外素材核验以读取恢复目录及缺测计数。执行版本SHA有明确对应，不将当前文件冒称早先执行字节。

## 核验与观看

两片guide重新解码与缓存逐位一致；使用已保存接收本地flow和原GLC独立重渲染，RGB/alpha/shared逐位一致。零alpha不改原GLC，shared不超过对应alpha。所有冻结输入和列出的推理源码SHA、码率字节及CSV重新汇总通过。

7个MP4均核验帧数、15FPS、时长、尺寸和采样面板。长片核验含首/中/尾及全部八处边界附近；抽帧方式不一次性物化完整长片四宫格。已查看静态接触表，静图只验证内容/布局，不代替用户的原速观感验收。

边界剪辑每段取[b−15,b+30)，共8×45帧/24秒，标注原视频时间和窗口编号。窗口之间是人为拼接，不要把剪辑跳转算作模型闪烁。

## 接续方向

保留Q9_R25统一研究参照；先看本轮原速画面，重点是长片边界瞬间和海草运动是否被压弱。之后在固定码率下隔离引导块连续性与门控变化，再解决输出驻留内存。独立留出集、更多长片、网络延迟和实时仍待专门验证。

正式采用配置/部署默认原字节保留；未提交或推送；无后台实验残留。权威运行：`{run.as_posix()}`。大媒体观看文件是硬链接，编辑前另存副本；推理/评分/核验/失败记录和入口旧版快照见小证据manifest。
'''
    put(report,text)
    view_rows=[]
    for _,name in mappings:
        if name.endswith('.mp4'):view_rows.append([name,f'[播放]({name})'])
    put(view/'README.md',f'''# Q9跨素材与82秒长片原速观看

2026-11-12；本轮完成，固定Q9_R25，无逐片调参。既有video1/2/3的Q9正面观感保持；**本目录的新素材反馈待记录**。

{table(['文件','入口'],view_rows)}

建议先看01完整长片，再看04礁石片和06海草片；03专看八处块边界。双列左原GLC、右Q9；03左continuous、右reset。02四宫格按行：原片/原GLC/continuous/reset。05和07按行是原片/原GLC/Q9，右下为明确留空，不是缺帧。

长片82.6秒/1239帧，两短片各10秒/150帧；全部15FPS原速、无音轨。03为八段各3秒的诊断剪辑，窗口间跳转来自剪辑，标题保留原片时间。

长片continuous平均时序误差相对原GLC{full_pct:+.2f}%，实际合计{rates['total_kbps']:.4f}kbps；边界瞬间仍比原GLC差。礁石/海草片分别{compact['reef_spotted_fish']['full_mae_change_percent']:+.2f}%/{compact['seagrass_spotted_fish']['full_mae_change_percent']:+.2f}%，但海草PSNR−0.22dB、运动幅度比0.523→0.326，重点观察真实运动是否被压弱。

历史素材扩展验证，不是严格独立留出；已完成一条完整长片，不代表任意长视频、恒定总内存或实时。当前采用配置未切换。

[完整报告]({relative(report,view)}) · [原始统计]({relative(run/'summary.json',view)}) · [额外两片]({relative(run/'archive_transfer/summary.json',view)}) · [SHA清单](manifest.json)

媒体为原结果硬链接，剪辑请先复制。MP4只供观看，评分来自无损数组，压缩码率来自实际传输包。未提交/推送。
''')
    repo_entries=['交接文档.md','当前项目状态与接续.md','README.md','项目总结与未来方向.md',
        '21_优化阶段_GLC解码后处理/README.md','21_优化阶段_GLC解码后处理/当前方案总结与瓶颈路线_20261111.md',
        '21_优化阶段_GLC解码后处理/下一阶段_海洋素材与长视频泛化计划_20261111.md',
        '05_资料与索引/会话收束_20261110/文件索引.md']
    entries=[REPO/n for n in repo_entries]+[run.parent/'README.md',WORKSPACE/'05_成果与观看/README.md']
    state_path=REPO/'05_资料与索引/会话收束_20261110/接续状态.json';continuation=REPO/'新会话接续.txt'
    prior=run/'publication_prior_entries';prior.mkdir();snap=[]
    for p in [*entries,state_path,continuation]:
        rel=p.relative_to(WORKSPACE);dest=prior/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        snap.append(dict(original=str(p),snapshot=str(dest),sha256=file_sha(p)))
    save_json(prior/'manifest.json',dict(files=snap))
    latest=f'固定Q9_R25完成82.6秒/1239帧全片及两段额外历史10秒。长片平均时序误差{full_pct:+.2f}%，总{rates["total_kbps"]:.4f}kbps；边界瞬间仍有峰值。礁石/海草片误差分别{compact["reef_spotted_fish"]["full_mae_change_percent"]:+.2f}%/{compact["seagrass_spotted_fish"]["full_mae_change_percent"]:+.2f}%，海草PSNR和运动幅度有回退。'
    for p in entries:
        old=p.read_text(encoding='utf-8-sig');title,body=old.split('\n',1)
        update=f'''**最新：Q9跨素材与完整长片验证完成（2026-11-12）。** {latest}

用户对既有三段Q9的正面反馈保持，Q9_R25为可追溯研究参照；新长片和两段额外片原速观感待记录。旧video2仅需适度改善，Q6_D25的闪烁未解决。上一轮selected=null及video3珊瑚+5.37%记录不改。

[本轮报告]({relative(report,p.parent)}) · [7个原速对照]({relative(view/'README.md',p.parent)}) · [完整证据]({relative(evidence/'manifest.json',p.parent)})

本轮已完成，无实验运行。下一重点为块边界连续性、海草运动保真及长片输出内存；GPU缓存有界不等于总RAM有界。已验证这一条完整长片，独立留出/任意长片/实时仍未验证。正式采用配置及部署默认未切换，未提交/推送。
'''
        put(p,title+'\n\n'+update+'\n## 之前记录（以下为当时状态，以页首为准）\n\n'+body.lstrip())
    old=continuation.read_text(encoding='utf-8-sig')
    put(continuation,f'''GLC接续，2026-11-12：Q9跨素材与82.6秒长片本轮已完成，无实验运行。
{latest}
完整报告：{report.as_posix()}
原速观看：{(view/'README.md').as_posix()}
权威运行：{run.as_posix()}
保持用户既有三段Q9正面反馈，新长片/两片观感pending。Q9_R25研究参照；原采用配置及部署未切换。
下一轮优先固定Q9下的块边界/海草运动保真；总进程RAM尚不恒定。不要重跑旧矩阵，不覆盖现有运行。
本轮没有提交/推送，保留全部其他工作区修改。

以下为历史接续正文，以本段为准：

{old}''')
    state=load_json(state_path)
    state['previous_latest_experiment_before_Q9_long']=state.get('latest_experiment')
    state['previous_generalization_before_Q9_long']=state.get('latest_generalization_experiment')
    state.update(status='Q9_full_length_and_archive_transfer_complete',active_experiment_plan=None,
        current_session_work='Completed fixed Q9_R25 full 82.6s continuous/reset test plus two registered historical 10s clips; all verified and published.',
        latest_experiment=result,latest_generalization_experiment=result,new_experiments_uncommitted=True,
        new_experiment_report=relative(report,REPO),latest_analysis_report=relative(report,REPO),
        latest_experiment_plan=dict(status='completed',path=str(run/'execution_plan.json'),frames=1239,archive_cases=2),
        experiment_execution_gate=dict(status='authorized_round_complete',date='2026-11-12',new_run_started=True,currently_running=False),
        next_priority=['Review new long and archive clips at original speed; preserve prior Q9 acceptance.',
                       'Isolate guide-block continuity and support changes at fixed Q9; examine reduced seagrass motion.',
                       'Bound total process RAM through chunked output mapping; independently register heldout footage later.'],
        next_experiment_plan=dict(status='proposed_not_run',reference='Q9_R25',focus=['guide block continuity','motion preservation','bounded output residency'],deployment_unchanged=True),
        publication=dict(current_changes_committed=False,pushed=False,scope='Q9 long/archive transfer evidence and preserved earlier uncommitted work'))
    state['workspace_paths'].update(q9_long_transfer_run=relative(run,WORKSPACE),q9_long_transfer_viewing=relative(view/'README.md',WORKSPACE),
        current_report=relative(report,WORKSPACE),latest_viewing=relative(view/'README.md',WORKSPACE))
    save_json(state_path,state)
    assert file_sha(config)==config_sha
    scripts=['run_q9_long_transfer.py','q9_chunk_protocol.py','run_q9_archive_transfer.py','resume_q9_archive_transfer.py',
             'finish_q9_long_transfer.py','publish_q9_long_transfer.py']
    for name in scripts:
        dest=evidence/'published_source'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(STAGE/name,dest)
    for path in sorted(run.rglob('*')):
        if not path.is_file() or path.suffix not in ('.json','.csv','.py','.md','.txt'):continue
        assert path.stat().st_size<10*1024**2
        dest=evidence/'run'/path.relative_to(run);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    shutil.copy2(report,evidence/REPORT)
    save_json(run/'publication.json',dict(complete=True,report=str(report),viewing=str(view),evidence=str(evidence),
        current_profile_unchanged=True,current_profile_sha256=config_sha,video_count=7,no_active_run=True,
        committed=False,pushed=False,source_sha256=file_sha(__file__)))
    shutil.copy2(run/'publication.json',evidence/'run/publication.json')
    evidence_files=[dict(path=p.relative_to(evidence).as_posix(),bytes=p.stat().st_size,sha256=file_sha(p))
                    for p in sorted(evidence.rglob('*')) if p.is_file()]
    save_json(evidence/'manifest.json',dict(complete=True,source_run=str(run),files=evidence_files,
        scope='Small evidence only; full RGB/flow arrays, packets and large media stay in their original local run.',
        current_profile_sha256=config_sha))
    print('PUBLISHED',len(evidence_files),'evidence files and 7 videos',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    publish(p.parse_args().run)
