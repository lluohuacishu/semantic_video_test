"""Publish the completed Q9 ghost-exit diagnostic round; never change adoption."""
import argparse
import copy
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

STAGE = Path(__file__).resolve().parent
REPO = STAGE.parent
WORKSPACE = REPO.parents[2]
REPORT_NAME = 'Q9残像定位与局部退出实验_20261112.md'
EXPECTED_CONFIG = 'b5eeb844d0adcdfe97648351a0654a2575a98942e1627c98896ba302555f7b04'
NAMES = {'video1':'video1', 'video2':'video2', 'video3':'video3',
         'reef_spotted_fish':'礁石斑点鱼', 'seagrass_spotted_fish':'海草斑点鱼'}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    def bad(value): raise ValueError(value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=bad)

def put(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip()+'\n', encoding='utf-8', newline='\n')

def save(path, data):
    put(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))

def relative(path, parent):
    return Path(os.path.relpath(path, parent)).as_posix()

def link(label, path, parent):
    return '['+label+']('+relative(path, parent)+')'

def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])

def delta(value, base):
    return 100*(value/base-1)

def publish(run):
    run = run.resolve()
    first = read(run/'summary.json')
    extra = read(run/'compact_extension/summary.json')
    plan = read(run/'execution_plan.json')
    extplan = read(run/'compact_extension/execution_plan.json')
    verification = read(run/'verification.json')
    media = read(run/'previews/verification.json')
    assert all(x['complete'] for x in (first, extra, verification, media))
    assert verification['first_phase_new_outputs']==10
    assert verification['adaptive_extension_new_outputs']==5
    assert media['video_count']==9
    assert sha(run/'summary.json')==extplan['parent_summary_sha256']
    assert sha(run/'execution_plan.json')==extplan['parent_plan_sha256']
    assert sha(run/'case_registration.json')==plan['registration_sha256']
    assert sha(run/'user_event_registration.json')==plan['user_events_sha256']
    config = STAGE/'当前采用配置.json'
    assert sha(config)==EXPECTED_CONFIG
    for folder, p in [(run, plan), (run/'compact_extension', extplan)]:
        for name, value in p['code_sha256'].items():
            assert sha(STAGE/name)==value==sha(folder/'source_snapshot'/name)
    assert sha(STAGE/'finish_q9_ghost_exit.py')==verification['verifier_sha256']==media['verifier_sha256']

    report = STAGE/REPORT_NAME
    view = WORKSPACE/'05_成果与观看/GLC_Q9残像退出_20261112'
    evidence = STAGE/'evidence_20261112/q9_ghost_exit'
    state_path = REPO/'05_资料与索引/会话收束_20261110/接续状态.json'
    repo_entries = [REPO/n for n in ['交接文档.md','当前项目状态与接续.md','README.md','项目总结与未来方向.md',
                                    '新会话接续.txt','05_资料与索引/会话收束_20261110/文件索引.md']]
    stage_entries = [STAGE/n for n in ['README.md','当前方案总结与瓶颈路线_20261111.md',
                                       '下一阶段_海洋素材与长视频泛化计划_20261111.md',
                                       'Q9残像反馈与后续重点_20261112.md']]
    entries = repo_entries+stage_entries+[WORKSPACE/'02_实验数据/README.md', WORKSPACE/'05_成果与观看/README.md']
    for path in (report, view, evidence, run/'publication.json'):
        if path.exists():
            raise FileExistsError(path)
    # Snapshot current entries byte-for-byte before modifying their newest status.
    before = run/'publication_before'
    before.mkdir(exist_ok=False)
    before_hashes = {}
    for path in entries+[state_path]:
        target=before/relative(path, WORKSPACE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path,target)
        before_hashes[str(path)]=sha(path)
    historical_manifests = {str(p):sha(p) for p in (STAGE/'evidence_20261112').rglob('manifest.json')}
    save(before/'snapshot_manifest.json', {'files':before_hashes, 'existing_evidence_manifests':historical_manifests})

    save(run/'visual_inspection.json', {
        'date':'2026-11-12',
        'method':'Assistant inspected the three saved candidate contact sheets as static images.',
        'files':['previews/reef_exact4p07_footprints.png','previews/reef_spotted_fish_user_event_contact.png',
                 'previews/video2_user_event_contact.png'],
        'observations_zh':[
            '礁石第61帧旧位置的深色鱼形在EL12/C32中减弱；E32仍明显保留。原GLC的鱼形失真仍在。',
            'video2尾部三张静帧的候选差异较小，不能认定尾部残像已解决。',
            '静图观察不等于原速闪烁/拖尾验收；候选原速用户反馈仍待记录。'],
        'original_speed_user_review':'pending', 'automatic_adoption':False})

    changes = {}
    for case, d in first['cases'].items():
        changes[case]={}
        base = d['scores']['full']['Q9_R25']
        for method in ('E32','EL12','C32'):
            current = extra['cases'][case] if method=='C32' else d
            s = current['scores']['full'][method]
            changes[case][method]={
                'full_coarse_temporal_mae_change_percent':delta(s['coarse_mae'],base['coarse_mae']),
                'psnr_change_db':s['psnr_db']-base['psnr_db'],
                'total_kbps':d['total_kbps'],
                'user_event_crops':current['user_event_crops'],
            }
    result = {
        'date':'2026-11-12', 'complete':True, 'reference':'Q9_R25',
        'first_phase_candidates':['E32','EL12'], 'first_phase_new_outputs':10,
        'adaptive_extension_candidates':['C32'], 'adaptive_extension_new_outputs':5,
        'new_candidate_outputs':15, 'cases':list(first['cases']), 'changes':changes,
        'selected':None, 'automatic_adoption':False, 'current_profile_unchanged':True,
        'current_profile_sha256':EXPECTED_CONFIG, 'user_subjective_review':'pending',
        'previous_Q9_positive_feedback_preserved':True, 'independent_holdout':False,
        'long_video_rerun':False, 'new_packet_or_flow_model_inference':False,
        'extra_transmission_bytes':0,
        'decision_zh':'局部旧鱼位置错误纹理减轻，但三候选均有全帧时序回退；video2尾部改善有限。不替换Q9_R25研究参照，不改变正式采用配置。',
        'next_direction_zh':'下一轮先检查有时间连续性的运动证据和退出/恢复机制；同镜头用户区间优先，保留背景稳定回归检查。尚未运行。',
    }
    save(run/'round_result.json',result)

    temporal_rows=[];psnr_rows=[];rate_rows=[];reef_rows=[]
    for case,d in first['cases'].items():
        temporal_rows.append([NAMES[case]]+[f"{changes[case][m]['full_coarse_temporal_mae_change_percent']:+.2f}%" for m in ('E32','EL12','C32')])
        psnr_rows.append([NAMES[case]]+[f"{changes[case][m]['psnr_change_db']:+.4f}" for m in ('E32','EL12','C32')])
        rate_rows.append([NAMES[case], f"{d['total_kbps']:.4f}", f"{d['guide_kbps']:.4f}",d['guide_bytes'],0])
        if 'reef' in d['scores']:
            q=d['scores']['reef']['Q9_R25']['coarse_mae']
            reef_rows.append([NAMES[case]]+[f"{delta((extra['cases'][case] if m=='C32' else d)['scores']['reef'][m]['coarse_mae'],q):+.2f}%" for m in ('E32','EL12','C32')])
    event_rows=[]
    exact_rows=[]
    for case,key,label,window in [
        ('reef_spotted_fish','reef_old_fish_A','礁石旧位置A','53–74'),
        ('reef_spotted_fish','reef_old_fish_B','礁石旧位置B','53–74'),
        ('video2','tail','video2鱼尾宽框','45–74')]:
        values=first['cases'][case]['user_event_crops'][key]
        q=values['Q9_R25']['mean_rgb_mae']
        row=[label,window,f'{q:.6f}']
        exact=[label,f"{values['raw']['exact_frame61']['rgb_mae']:.6f}",f"{values['Q9_R25']['exact_frame61']['rgb_mae']:.6f}"]
        for method in ('E32','EL12','C32'):
            v=(extra['cases'][case]['user_event_crops'][key] if method=='C32' else values)[method]
            row.append(f"{v['mean_rgb_mae']:.6f}（{delta(v['mean_rgb_mae'],q):+.2f}%）")
            exact.append(f"{v['exact_frame61']['rgb_mae']:.6f}")
        event_rows.append(row);exact_rows.append(exact)
    attribution=read(run/'diagnosis/attribution.json')
    attribution_rows=[]
    for key,v in attribution['regions'].items():
        attribution_rows.append([key,str(v['box']),f"{v['raw_source_MAE']:.6f}",f"{v['Q9_source_MAE']:.6f}",
                                 f"{v['mean_alpha']:.3f}",f"{v['median_flow_pixels']:.3f}"])
    rlink=lambda label,path:link(label,path,STAGE)
    report_text=f"""# Q9 残像定位与局部退出实验（2026-11-12）

本轮已完成五段历史短片上的15组候选输出与核验。用户指出的礁石斑点鱼4.07秒附近，确认有原始锚帧鱼形在旧位置被继续复用；局部退出可减轻该处旧影，但会损伤原Q9的时序稳定。video2鱼尾的局部改善有限，尚不能说残像已解决。

**结论：E32、EL12、C32均保留为诊断对照，未自动采用。Q9_R25继续作为研究参照，正式S8I + CURRENT_I12配置不变。** 已有Q9三段正面反馈保持；新发现的具体残像不撤销此前整体观感。用户关于video2“改善一些即可”和长片暂不追求优秀全片评价的要求继续有效。

## 1. 用户事件与证据边界

用户原话：“04礁石斑点鱼4.07s，然后video2也有残像问题，鱼尾巴那里”。

- “04”对应此前礁石斑点鱼原GLC/Q9_R25对照，起点0秒、15 FPS。4.07秒对齐零起算第61帧（61/15≈4.067秒）。
- video2用户只指定了鱼尾，没有指定时间。本轮自行选45–74帧（3.0–4.93秒）作固定检查区间，不将它写成用户指定时间。
- 礁石旧位置A/B固定检查53–74帧（3.53–4.93秒）；展示裁剪覆盖3.0–5.0秒。ROI坐标仅供诊断和独立评价，**不送入接收算法**，也不是鱼分割。
- 自动48×48误差搜索倾向选到背景，因此后续按用户明确位置定位，没有把自动最高误差块当作鱼残像标注。
- 这五片均为既有素材，每片150帧、512×288、15 FPS、10秒。不是独立留出，本轮没有重跑82.6秒长片。

{rlink('用户事件登记',run/'user_event_registration.json')} · {rlink('评价框登记',run/'diagnosis/user_review_boxes.json')} · {rlink('基线归因',run/'diagnosis/attribution.json')}

## 2. 礁石第61帧的归因

该帧处在同一镜头内，不是换锚渐退帧。使用第45帧原GLC锚帧重放原Q9公式，与保存的第61帧逐位一致。小鱼已移动，但引导图里小鱼信息很弱，局部光流几乎静止；门控仍给旧纹理较大权重，因而把旧鱼形写回当前画面。

{table(['区域','评价框[x0,y0,x1,y1]','原GLC对源RGB MAE','Q9对源RGB MAE','平均alpha','光流中位数/像素'],attribution_rows)}

两框中Q9与变换后原锚帧的RGB MAE分别仅为1.673和2.035。这支持**该处错误历史纹理保留**的解释。原GLC自身也有鱼形和细节失真；不能把此机制泛化为所有片段、鱼尾或切镜残像的唯一原因。

{rlink('原锚帧归因图',run/'diagnosis/reef_f61_old_fish_attribution.png')} · {rlink('第61帧六方案细节',view/'reef_exact4p07_footprints.png')}

## 3. 候选与真实实验顺序

第一阶段先在基线诊断后冻结两套统一参数，再跑五片，共10组。阈值不是独立留出确定的；同一组参数用于所有片，没有按片调参。固定既有Q9码包、原GLC解码、接收本地RAFT缓存；没有新码流或光流模型推理。

两候选均先完整重渲染原Q9_R25，再以当前原GLC与最终Q9的RGB冲突计算退出权重w：

    output = round(Q9 × (1 − w) + current_raw_GLC × w)

这是显式新增的一次末级取整混合，原先两阶段渲染逻辑未替换。w=0保持原Q9，w=1可完全退出到当前原GLC。

| 候选 | 冲突判断 | 退出范围 |
| --- | --- | --- |
| E32 | RGB最大通道差经sigma3平滑，smooth risk区间12→32 | 可完全退回当前原GLC；沿用旧D25差异区间，但不只减权25% |
| EL12 | sigma3差6→12与sigma1差12→24的risk相乘 | 再3×3膨胀1像素，可完全退出 |
| C32 | 在EL12风险中保留support≥0.1、峰值≥0.8的8连通区域 | 面积8–1024像素；逐帧计算，无时间迟滞 |

**C32是看完第一阶段结果后登记的自适应补充**：发现局部改善伴随较大全局时序回退，才冻结“限制连通面积”方案并再跑五片，共5组。其计划保存第一阶段summary/plan SHA，不伪装成三候选同时预登记。C32并非语义分割；区域逐帧合并/分裂可能引起开关变化，此处只是机制风险，未另做因果验证。

{rlink('第一阶段计划',run/'execution_plan.json')} · {rlink('自适应补充计划',run/'compact_extension/execution_plan.json')} · {rlink('末级退出实现',STAGE/'raw_conflict_exit.py')} · {rlink('C32实现',STAGE/'run_compact_ghost_exit.py')}

## 4. 用户指出位置的结果

下表是固定评价框内对源RGB的平均像素误差，越低越好；百分比相对Q9_R25。它**不是残像数量或残像减少百分比**，框内也包含背景和真实运动。

{table(['评价区域','帧区间（含端点）','Q9_R25','E32','EL12','C32'],event_rows)}

第61帧的同口径像素误差（只有礁石第61帧由用户精确时间指定）：

{table(['区域','原GLC','Q9_R25','E32','EL12','C32'],exact_rows)}

已查看三张候选静图：礁石旧鱼位置中EL12/C32的深色残留减轻，E32保留更明显；video2尾部静图差异有限。原速闪烁/拖尾仍应以观看对照判断，候选用户验收待记录，不能将静图检查当作已获用户认可。

{rlink('礁石3–5秒放大六宫格',view/'reef_spotted_fish_user_event_crop_all.mp4')} · {rlink('video2鱼尾3–5秒放大六宫格',view/'video2_user_event_crop_all.mp4')} · {rlink('静图审阅记录',run/'visual_inspection.json')}

## 5. 跨片回归与取舍

全帧源对齐低频时序误差的变化如下。沿用既有源侧评价运动/有效支持，所有方法使用相同支持；低支持帧按已有规则记录在summary。下表**全部相对Q9_R25**，正值表示回退，不是闪烁次数或人的观感打分。

{table(['片段','E32','EL12','C32'],temporal_rows)}

三候选在五片均有回退。C32缩小了部分干预范围，但video2、礁石和海草的时序代价仍明显。不能仅凭旧位置的误差下降宣布方案可用。

历史珊瑚评价框（不是本轮用户框）的低频时序误差变化，同样相对Q9：

{table(['历史ROI所属片','E32','EL12','C32'],reef_rows)}

全帧PSNR的变化，单位dB，同样相对Q9：

{table(['片段','E32','EL12','C32'],psnr_rows)}

PSNR略有改善并不抵消时序退化。海草的变化幅度比由Q9的0.3256上升到EL12的0.4434、C32的0.4226，但EL12动态RGB误差14.2248高于Q9的14.0603，不能把“动得更多”直接写成运动恢复成功。

已有Q9_R25数值筛选selected=null、video3历史珊瑚相对旧基线+5.37%的记录继续保留。该旧比较基线与本节“相对Q9_R25”的比较基线不同，不混用。

## 6. 字节与核验

{table(['片段','实际总kbps','实际辅助kbps','辅助包字节','新增传输字节'],rate_rows)}

上述码率均沿用原包，含既有协议口径；NPY、光流缓存和观看MP4不是传输包。正式采用配置SHA256：

    {EXPECTED_CONFIG}

核验已完成：五片guide重新解码一致，Q9基线RGB/alpha/shared与历史逐位一致；15组候选shape/dtype/finite正确；w=0/w=1端点、alpha=0不变和逐像素凸组合范围正确；C32权重不超过EL12。接收读白名单拒绝源图、用户ROI及不应读取的历史/发送数据，评分CSV和用户ROI误差独立重算一致，传输字节不变。原始完整核验项目以JSON为准。

9个预览视频已核对帧数、15 FPS、时长、尺寸及抽样面板。MP4为观看转码，像素评分使用无损NPY。五个六宫格及两个双列是10秒，两个放大裁剪是2秒；仅做抽样解码核验，不声称逐帧视频编码无损。

{rlink('完整核验',run/'verification.json')} · {rlink('媒体核验',run/'previews/verification.json')} · {rlink('结构化本轮结论',run/'round_result.json')} · {rlink('小证据清单',evidence/'manifest.json')}

## 7. 观看与接续

优先看{rlink('礁石Q9/EL12双列',view/'reef_spotted_fish_Q9_vs_EL12.mp4')}和{rlink('video2 Q9/EL12双列',view/'video2_Q9_vs_EL12.mp4')}，左为原Q9_R25，右为EL12诊断候选，未采用。全部预览见{rlink('观看入口',view/'README.md')}。六宫格按行从左到右为：原视频 / 原GLC / Q9_R25 / E32 / EL12 / C32。

本轮已收束，无继续运行的实验。下一轮建议以用户指出的同镜头区间为起点，研究带时间连续性的运动证据与退出/恢复条件，同时保护原Q9的背景稳定。先确认鱼离开后的旧影是否消失、鱼尾是否跟随真实运动，再看局部和全帧指标；暂不继续扫描静态差异阈值，也不追求长片总评分。

这只是下一轮方向，尚未实现或验证。未改正式采用配置、未改部署默认、未提交/推送；原历史结果与证据保留。
"""
    put(report,report_text)

    view.mkdir(parents=True)
    media_files=sorted([p for p in (run/'previews').iterdir() if p.suffix in ('.mp4','.png')])
    assert len(media_files)==12
    media_records={}
    for src in media_files:
        dst=view/src.name
        os.link(src,dst)
        assert os.path.samefile(src,dst)
        value=sha(src)
        if src.suffix=='.mp4':
            assert value==media['videos'][src.name]['sha256']
        media_records[src.name]={'source':str(src),'sha256':value,'bytes':src.stat().st_size,'hardlink':True}
    save(view/'media_manifest.json',{'complete':True,'files':media_records,'note':'Hard links share bytes. Copy to a new file before editing; never overwrite these views.'})
    vlink=lambda label,path:link(label,path,view)
    view_text=f"""# Q9 残像退出诊断观看（2026-11-12）

本轮候选仅用于观察退出旧纹理的收益与代价，**E32 / EL12 / C32均未采用**。Q9_R25仍是研究参照，正式采用配置未改。用户原先对三段Q9的正面评价保持。

## 优先看这四条

1. {vlink('礁石3–5秒放大六宫格',view/'reef_spotted_fish_user_event_crop_all.mp4')}：用户4.07秒对应本裁剪播放约1.07秒，重点看两条小鱼离开后的旧位置。
2. {vlink('video2鱼尾3–5秒放大六宫格',view/'video2_user_event_crop_all.mp4')}：这是本轮选择的检查区间，用户没有指定video2时间。
3. {vlink('礁石完整10秒，左Q9_R25 / 右EL12',view/'reef_spotted_fish_Q9_vs_EL12.mp4')}。
4. {vlink('video2完整10秒，左Q9_R25 / 右EL12',view/'video2_Q9_vs_EL12.mp4')}。

六宫格布局（从上到下，每行左→右）：原视频 / 原GLC；Q9_R25 / E32；EL12 / C32。所有视频15 FPS原速播放，裁剪只放大空间，不改变播放速度。右侧EL12有诊断/未采用标签。

## 五片完整六宫格

{chr(10).join('- '+vlink(NAMES[case]+'：10秒',view/(case+'_all_six.mp4')) for case in NAMES)}

## 静帧辅助

- {vlink('礁石4.07秒旧位置细节',view/'reef_exact4p07_footprints.png')}
- {vlink('礁石3.53 / 4.07 / 4.60秒',view/'reef_spotted_fish_user_event_contact.png')}
- {vlink('video2鱼尾3.33 / 4.07 / 4.67秒',view/'video2_user_event_contact.png')}

本轮已完成15组候选。礁石固定旧位置误差下降，但五片均有时序误差回退；video2尾部改善很小，不能说已经解决。静图不替代原速验收，候选用户反馈仍待记录。

{vlink('完整报告',report)} · {vlink('结果和无损输出目录',run/'round_result.json')} · {vlink('媒体核验',run/'previews/verification.json')} · {vlink('媒体SHA',view/'media_manifest.json')}

本目录媒体是运行目录的硬链接；如需编辑请先另存副本，避免改变原实验观看文件。MP4体积不能用作语义通信码率。
"""
    put(view/'README.md',view_text)

    # Append the newest interpretation above historical text without rewriting prior evidence.
    for path in entries:
        prefix=f"""**最新（2026-11-12）：Q9残像定位与局部退出实验已完成，五片共15组候选。** 用户已定位到“04礁石斑点鱼4.07s”和video2鱼尾。礁石该处属于同镜头旧锚帧鱼形保留，非切镜/换锚渐退帧；不能推广为所有残像的唯一成因。

EL12/C32减轻了礁石旧位置错误纹理，但video2尾部改善有限，三候选五片均出现源对齐低频时序误差回退。候选原速验收待记录；不替换Q9_R25研究参照、不改正式采用配置。原Q9三段正面反馈及video2适度改善目标保持，本轮未重跑长片。

{link('本轮报告',report,path.parent)} · {link('原速与放大对照',view/'README.md',path.parent)} · {link('15组核验',run/'verification.json',path.parent)} · {link('小证据',evidence/'manifest.json',path.parent)}

本轮已收束，无活动实验。第一阶段两候选10组；观察结果后透明登记C32补充5组。下一方向为有时间连续性的运动证据和退出/恢复，不再仅调静态阈值；长片暂不追求优秀全片评分。未提交/推送。

## 既往记录（以下保留当时状态，以页首本轮结果为准）
"""
        old=path.read_text(encoding='utf-8-sig')
        if path.suffix=='.txt':
            prefix=f"""GLC接续，2026-11-12：Q9残像退出本轮已完成，五片15组，无活动实验。
用户位置：04礁石斑点鱼4.07s；video2鱼尾（无精确时间）。前者已定位到同镜头旧锚帧鱼形保留。
局部退出让礁石旧位置误差下降，video2尾部改善有限；三候选五片均有时序回退，未采用。
保持Q9_R25研究参照、原三段正面反馈、正式S8I+CURRENT_I12配置；长片暂不追求全片好评分，本轮未重跑。
下一方向：有时间连续性的运动证据和退出/恢复机制，先做用户指出短区间；尚未实现/运行。
报告：{report.as_posix()}
观看：{(view/'README.md').as_posix()}
运行：{run.as_posix()}
核验：{(run/'verification.json').as_posix()}
证据：{(evidence/'manifest.json').as_posix()}
未提交/推送。旧selected=null、video3珊瑚+5.37%历史口径保持。

此前接续正文（当时状态，以页首为准）：
"""
            put(path,prefix+'\n'+old)
        elif old.startswith('# ') and '\n' in old:
            title,body=old.split('\n',1)
            put(path,title+'\n\n'+prefix+'\n'+body.lstrip('\n'))
        else:
            put(path,prefix+'\n'+old)

    state=read(state_path)
    changed_keys=['status','current_session_work','latest_analysis_report','latest_diagnosis','latest_experiment_plan',
                  'latest_experiment','latest_generalization_experiment','latest_user_perceptual_feedback','next_priority',
                  'next_experiment_plan','publication','experiment_execution_gate','active_experiment_plan','new_experiment_report',
                  'user_latest_decision']
    state['previous_fields_before_ghost_exit_round']={k:copy.deepcopy(state.get(k)) for k in changed_keys}
    state['previous_latest_experiment_before_ghost_exit']=copy.deepcopy(state.get('latest_experiment'))
    state['previous_generalization_before_ghost_exit']=copy.deepcopy(state.get('latest_generalization_experiment'))
    prior_feedback=copy.deepcopy(state['latest_user_perceptual_feedback'])
    state['previous_user_perceptual_feedback_before_localization']=prior_feedback
    state['status']='Q9_ghost_exit_round_complete'
    state['active_experiment_plan']=None
    state['current_session_work']='Completed same-shot ghost attribution, first-stage 10 candidates and adaptive C32 5 candidates, verification and original-speed previews. No candidate adopted.'
    state['new_experiment_report']=relative(report,REPO)
    state['latest_analysis_report']=relative(report,REPO)
    state['latest_experiment_plan']={'status':'completed','path':str(run/'execution_plan.json'),
                                     'adaptive_extension':str(run/'compact_extension/execution_plan.json'),'new_outputs':15}
    latest={k:v for k,v in result.items() if k!='changes'}
    latest.update(run=str(run),report=str(report),viewing=str(view),verification=str(run/'verification.json'),
                  changes_summary={c:{m:{k:v for k,v in d.items() if k!='user_event_crops'} for m,d in x.items()} for c,x in changes.items()},
                  visual_inspection=str(run/'visual_inspection.json'),evidence=str(evidence/'manifest.json'))
    state['latest_experiment']=latest
    state['latest_generalization_experiment']=copy.deepcopy(latest)
    state['latest_diagnosis']={
        'report':relative(report,REPO),'attribution':str(run/'diagnosis/attribution.json'),
        'same_shot_reef_frame61_old_anchor45_reuse_confirmed':True,
        'not_universal_cause':True,'video2_tail_not_resolved':True,
        'local_error_improvement_with_temporal_regression':True}
    state['latest_user_perceptual_feedback']={
        'date':'2026-11-12','type':'user_ghosting_location_clarification',
        'verbatim_user_feedback':read(run/'user_event_registration.json')['verbatim'],
        'preceding_feedback':prior_feedback['verbatim_user_feedback'],
        'scope':{'context_reference':'Q9_R25','reef_spotted_fish':{'time_seconds':4.07,'frame_zero_based':61},
                 'video2':{'region':'fish tail','time_seconds':None}},
        'prior_Q9_video1_video2_video3_positive_feedback_preserved':True,
        'Q9_subvariants_not_individually_accepted':True,
        'video2_goal':'Modest improvement; do not exaggerate severity',
        'long_video_priority_zh':prior_feedback['long_video_priority_zh'],
        'new_candidate_user_review':'pending',
        'localization_record':str(run/'user_event_registration.json'),
        'assistant_attribution_separate_from_user_feedback':str(run/'diagnosis/attribution.json'),
        'formal_adoption_changed':False}
    state['preferred_research_reference'].update(
        status='prior_positive_three_clips_with_localized_ghosting_feedback',
        latest_localized_feedback_path=str(run/'user_event_registration.json'),
        current_focus='Temporal evidence for invalid-history exit/recovery; static raw-conflict candidates have temporal regressions.')
    state['next_priority']=[
        'Inspect original-speed user-linked reef and video2 tail comparisons; keep positive prior Q9 assessment.',
        'Next round: temporal continuity and motion evidence for exit/recovery, retaining background stability checks.',
        'Treat long scene cuts and complexity as stress observations; defer strong whole-video score targets.']
    state['next_experiment_plan']={
        'status':'proposed_not_run','reference':'Q9_R25','focus':['temporal motion evidence','history exit and recovery'],
        'known_user_events':str(run/'user_event_registration.json'),'parameters_not_chosen':True,
        'whole_long_video_quality_priority':'deferred_per_user','deployment_unchanged':True}
    state['user_latest_decision']='那继续吧；补充定位：04礁石斑点鱼4.07s，然后video2也有残像问题，鱼尾巴那里'
    state['experiment_execution_gate']={'status':'completed_round','date':'2026-11-12','new_run_started':True,
                                        'currently_running':False,'automatic_next_run':False}
    state['publication']={'current_changes_committed':False,'pushed':False,
                          'scope':'Q9 localized ghost-exit round: 15 diagnostic outputs, evidence, previews and handoff.',
                          'receipt':str(run/'publication.json')}
    state['new_experiments_uncommitted']=True
    save(state_path,state)
    assert sha(config)==EXPECTED_CONFIG

    save(run/'publication.json', {
        'complete':True,'date':'2026-11-12','report':str(report),'viewing':str(view),'evidence':str(evidence),
        'state':str(state_path),'candidate_outputs':15,'videos':9,'contact_pngs':3,
        'adoption_changed':False,'committed':False,'pushed':False,'report_sha256':sha(report),
        'updated_entry_sha256':{str(p):sha(p) for p in entries+[state_path]},
        'publisher_sha256':sha(__file__),'previous_state_snapshot':str(before),
        'old_evidence_manifests_preserved':True})

    evidence.mkdir(parents=True)
    small_files=[p for p in run.rglob('*') if p.is_file() and p.suffix.lower() in ('.json','.csv','.py','.md','.txt')]
    for src in small_files:
        dst=evidence/'run'/relative(src,run)
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
    for src in [report, Path(__file__), STAGE/'finish_q9_ghost_exit.py']:
        dst=evidence/('report' if src==report else 'publication_source')/src.name
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
    for src in media_files:
        if src.suffix=='.png':
            dst=evidence/'visuals'/src.name
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dst)
    src=run/'diagnosis/reef_f61_old_fish_attribution.png'
    shutil.copy2(src,evidence/'visuals'/src.name)
    for src in [view/'README.md',view/'media_manifest.json']:
        dst=evidence/'viewing'/src.name
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
    for src in entries+[state_path]:
        dst=evidence/'entries_after'/relative(src,WORKSPACE)
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
    # Verify new Markdown links before sealing the small evidence manifest.
    checked_links=[]
    for path in [report,view/'README.md']+entries:
        content=path.read_text(encoding='utf-8-sig')
        if path in entries and path.suffix!='.txt':
            content=content.split('## 既往记录（以下保留当时状态，以页首本轮结果为准）',1)[0]
        elif path.suffix=='.txt':
            continue
        for href in re.findall(r'\]\(([^)]+)\)',content):
            target=(path.parent/href.split('#',1)[0]).resolve()
            assert target.exists() or target==evidence/'manifest.json',(str(path),href)
            checked_links.append(str(target))
    for path,h in historical_manifests.items():
        assert sha(path)==h
    for path,h in before_hashes.items():
        assert sha(before/relative(path,WORKSPACE))==h
    for src in media_files:
        assert sha(view/src.name)==sha(src)==media_records[src.name]['sha256']
    state=read(state_path)
    assert state['status']=='Q9_ghost_exit_round_complete' and state['active_experiment_plan'] is None
    assert state['latest_experiment']['new_candidate_outputs']==15 and state['latest_experiment']['selected'] is None
    json_files=[p for p in evidence.rglob('*.json')]
    for p in json_files:read(p)
    publication_qa={
        'complete':True,'current_state_complete_and_no_active_plan':True,'candidate_outputs':15,
        'videos':9,'pngs':3,'new_local_links_checked':len(checked_links),
        'old_evidence_manifests_preserved':len(historical_manifests),
        'snapshots_byte_exact':True,'all_media_hashes_match':True,'all_new_json_strict':True,
        'current_profile_sha256':sha(config),'publisher_sha256':sha(__file__),
        'original_speed_user_review':'pending','not_a_new_inference_run':True}
    save(run/'publication_verification.json',publication_qa)
    shutil.copy2(run/'publication_verification.json',evidence/'run/publication_verification.json')
    files={}
    for p in sorted(evidence.rglob('*')):
        if p.is_file():
            assert p.suffix.lower() not in ('.npy','.mp4')
            files[relative(p,evidence)]={'bytes':p.stat().st_size,'sha256':sha(p)}
    save(evidence/'manifest.json',{'complete':True,'date':'2026-11-12','files':files,
                                  'large_arrays_and_videos_excluded':True,'source_run':str(run),
                                  'note':'Manifest excludes itself. Existing historical evidence was not modified.'})
    for name,record in read(evidence/'manifest.json')['files'].items():
        assert sha(evidence/name)==record['sha256']
    print(json.dumps({'complete':True,'report':str(report),'viewing':str(view),
                      'evidence_files':len(files),'updated_entries':len(entries)+1,
                      'local_links_checked':len(checked_links),'current_config_unchanged':True},ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    publish(ap.parse_args().run)

