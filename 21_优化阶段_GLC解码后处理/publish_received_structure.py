"""Publish received-guide fusion and its separately registered blur control."""
import argparse
import copy
import csv
import math
import os
from pathlib import Path
import re
import shutil
from publish_q9_ghost_exit import sha,read,put,save,relative,link,table,delta,STAGE,REPO,WORKSPACE,EXPECTED_CONFIG

TITLE='Q9已传引导结构融合与纹理对照_20261113.md'
CASES=('video1','video2','video3')
METHODS=('GB25','GS25','GS50')
HISTORY='## 既往记录（以下保留当时状态，以页首本轮为准）'


def check_score_csv(folder, summary):
    def rows(name):
        with (folder/name).open(encoding='utf-8',newline='') as f:
            return list(csv.DictReader(f))
    tr,sr=rows('temporal.csv'),rows('spatial.csv')
    for scope,methods in summary['scores'].items():
        for name,v in methods.items():
            t=[r for r in tr if r['scope']==scope and r['method']==name]
            s=[r for r in sr if r['scope']==scope and r['method']==name]
            n=sum(int(r['pixels']) for r in t)
            sn=sum(int(r['pixels']) for r in s)
            assert len(s)==150 and [int(r['frame']) for r in s]==list(range(150))
            for band in ('coarse','fine','full'):
                assert math.isclose(sum(float(r[band+'_sum']) for r in t)/n,v[band+'_mae'],rel_tol=1e-12)
            assert math.isclose(sum(float(r['fine_ncc'])*int(r['pixels']) for r in t)/n,v['fine_ncc'],rel_tol=1e-12)
            assert math.isclose(sum(float(r['fine_sum']) for r in s)/sn,v['fine_amplitude'],rel_tol=1e-12)


def publish(run):
    control=run/'blur_control'
    plan,main=read(run/'execution_plan.json'),read(run/'summary.json')
    cp,extra=read(control/'execution_plan.json'),read(control/'summary.json')
    visual,auth=read(run/'visual_inspection.json'),read(run/'authorization.json')
    media,cm=read(run/'previews/verification.json'),read(control/'previews/verification.json')
    for root in (run,control):
        assert read(root/'verification.json')['complete']
        assert read(root/'summary.json')['complete']
    assert main['new_outputs']==9 and extra['new_outputs']==3
    assert media['video_count']==11 and cm['video_count']==3
    assert sha(STAGE/'当前采用配置.json')==EXPECTED_CONFIG
    assert sha(run/'execution_plan.json')==cp['first_phase_plan_sha256']
    assert sha(run/'summary.json')==cp['first_phase_summary_sha256']
    assert sha(run/'verification.json')==cp['first_phase_verification_sha256']
    for root,p in ((run,plan),(control,cp)):
        for n,h in p['code_sha256'].items():
            assert sha(STAGE/n)==h==sha(root/'source_snapshot'/n)
        for n,h in p['input_sha256'].items():
            assert sha(n)==h
    assert len(visual['images_actually_viewed'])==6
    assert all((run/n).exists() for n in visual['images_actually_viewed'])
    for case in CASES:
        for root,s in ((run,main),(control,extra)):
            check_score_csv(root/case/'evaluation',s['cases'][case])

    report=STAGE/TITLE
    view=WORKSPACE/'05_成果与观看/GLC_Q9引导结构融合_20261113'
    evidence=STAGE/'evidence_20261113/q9_received_structure'
    statepath=REPO/'05_资料与索引/会话收束_20261110/接续状态.json'
    entries=[REPO/n for n in ('README.md','交接文档.md','当前项目状态与接续.md','项目总结与未来方向.md',
                              '新会话接续.txt','05_资料与索引/会话收束_20261110/文件索引.md')]
    entries += [STAGE/n for n in ('README.md','当前方案总结与瓶颈路线_20261111.md',
                                  '下一阶段_海洋素材与长视频泛化计划_20261111.md')]
    entries += [WORKSPACE/'02_实验数据/README.md',WORKSPACE/'05_成果与观看/README.md']
    for p in (report,view,evidence,run/'publication.json'):
        assert not p.exists(), p
    before=run/'publication_before'
    before.mkdir(exist_ok=False)
    old_manifests={str(p):sha(p) for d in STAGE.glob('evidence_*') if d.is_dir() for p in d.rglob('manifest.json')}
    snapshots={}
    for p in entries+[statepath]:
        dst=before/relative(p,WORKSPACE)
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dst);snapshots[str(p)]=sha(p)
    save(before/'snapshot_manifest.json',dict(files=snapshots,old_manifests=old_manifests))

    changes,control_changes={},{}
    timing,spatial,texture,dynamic,regions,rate_rows,clipping,blur_rows,fine_rows,reef_rows=[],[],[],[],[],[],[],[],[],[]
    for case in CASES:
        s=main['cases'][case];b=s['scores']['full']['Q9_R25']
        changes[case]={m:dict(coarse_temporal_change_percent=delta(s['scores']['full'][m]['coarse_mae'],b['coarse_mae']),
            psnr_change_db=s['scores']['full'][m]['psnr_db']-b['psnr_db']) for m in METHODS}
        timing.append([case]+[f"{changes[case][m]['coarse_temporal_change_percent']:+.2f}%" for m in METHODS])
        spatial.append([case]+[f"{changes[case][m]['psnr_change_db']:+.4f}" for m in METHODS])
        texture.append([case]+[f"{s['scores']['full'][m]['fine_amplitude']:.4f}" for m in ('source','guide','Q9_R25','GB25','GS25','GS50')])
        dynamic.append([case]+[f"{s['dynamic'][m]['change_amplitude_ratio']:.4f}" for m in ('Q9_R25','GB25','GS25','GS50','guide')])
        ec=extra['cases'][case]
        blur=ec['scores']['full']['BLUR_MATCH'];g=s['scores']['full']['GS25']
        rec=read(control/case/'receive/receipt.json')
        c=dict(blur_coarse_change_vs_q9_percent=delta(blur['coarse_mae'],b['coarse_mae']),
               gs25_coarse_change_vs_blur_percent=delta(g['coarse_mae'],blur['coarse_mae']),
               gs25_psnr_change_vs_blur_db=g['psnr_db']-blur['psnr_db'],
               gs25_fine_temporal_change_vs_blur_percent=delta(g['fine_mae'],blur['fine_mae']),
               gs25_motion_amplitude_vs_blur_percent=delta(s['dynamic']['GS25']['change_amplitude_ratio'],ec['dynamic']['BLUR_MATCH']['change_amplitude_ratio']),
               gaussian_sigma=rec['sigma'],relative_amplitude_match_error=rec['relative_matching_error'])
        control_changes[case]=c
        blur_rows.append([case,f"{rec['sigma']:.6f}",f"{rec['relative_matching_error']*100:.5f}%",
            f"{c['blur_coarse_change_vs_q9_percent']:+.2f}%",f"{changes[case]['GS25']['coarse_temporal_change_percent']:+.2f}%",
            f"{c['gs25_coarse_change_vs_blur_percent']:+.2f}%",f"{c['gs25_psnr_change_vs_blur_db']:+.4f}"])
        fine_rows.append([case,f"{blur['fine_mae']:.6f}",f"{g['fine_mae']:.6f}",f"{c['gs25_fine_temporal_change_vs_blur_percent']:+.2f}%",
                          f"{ec['dynamic']['BLUR_MATCH']['change_amplitude_ratio']:.4f}",f"{s['dynamic']['GS25']['change_amplitude_ratio']:.4f}"])
        for label,v in s['current_regions'].items():
            q=v['Q9_R25']['mean_rgb_mae']
            regions.append([case+'/'+label,f"{q:.4f}"]+
                [f"{v[m]['mean_rgb_mae']:.4f} ({delta(v[m]['mean_rgb_mae'],q):+.2f}%)" for m in METHODS]+
                [f"{ec['current_regions'][label]['BLUR_MATCH']['mean_rgb_mae']:.4f}"])
        rate_rows.append([case,f"{s['total_kbps']:.4f}",f"{s['guide_kbps']:.4f}",0])
        rr=read(run/case/'receive/receipt.json')
        clipping.append([case]+[f"{100*rr['variants'][m]['clipping_channel_fraction']:.6f}%" for m in METHODS])
        q=s['scores']['reef']['Q9_R25']['coarse_mae']
        reef_rows.append([case]+[f"{delta(s['scores']['reef'][m]['coarse_mae'],q):+.2f}%" for m in METHODS])
    result=dict(date='2026-11-13',complete=True,cases=list(CASES),profiles=list(METHODS),
        new_outputs=12,new_candidate_outputs=9,postregistered_diagnostic_outputs=3,
        selected=None,viewing_candidate='GS25',automatic_adoption=False,
        current_profile_unchanged=True,current_profile_sha256=EXPECTED_CONFIG,reference='Q9_R25',
        changes=changes,matched_blur_comparison=control_changes,
        research_pixel_contract_changed=True,received_guide_rgb_enters_output=True,
        source_reads_blocked=True,extra_transmission_bytes=0,new_encoding=False,new_flow_model_inference=False,
        user_subjective_review='pending',static_inspection_completed=True,
        previous_Q9_positive_feedback_preserved=True,independent_holdout=False,
        reef_not_run=True,long_video_not_run=True,video4_awaiting_user=True,
        conclusion_zh='GS25三片低频时序与主体宽框误差改善，等细节高斯模糊不能解释全部收益；但GS25细纹理时序误差高于该控制，运动幅度更低，主体形状偏差仍在。仅作温和观看候选。',
        blur_limit_zh='只匹配全片平均细纹理幅度，未匹配逐像素纹理分布、完整频谱或运动幅度；不能由此宣称排除了所有平滑贡献或已恢复真实鱼形。')
    save(run/'round_result.json',result)
    rl=lambda label,p:link(label,p,STAGE)
    report_text=f"""# Q9 已传引导结构融合与纹理对照（2026-11-13）

本轮video1/2/3共完成9组统一参数候选，随后透明登记并完成3组等细节模糊诊断对照。**GS25保留为温和观看候选，尚未采用。** 三片的源对齐低频时序误差相对Q9_R25下降15%–17%；同等细节幅度的单帧高斯模糊没有取得同样收益。但GS25的细纹理时序误差高于模糊对照，运动幅度也更低，不能宣称鱼形、残像或泛化问题已解决。

{rl('原速观看入口',view/'README.md')} · {rl('结构化结果',run/'round_result.json')} · {rl('小证据',evidence/'manifest.json')}

## 1. 范围与用户优先级

最新原话：“{auth['verbatim']}”

本轮沿用video1/2/3各150帧、15 FPS、512×288、完整10秒。优先看整体可看、当前鱼形和运动。礁石旧影专题已降优先级，本轮未重跑礁石、海草或82.6秒长片。大面积珊瑚video4仍等用户提供文件，本轮未猜测时间段、未裁切、未提前运行。

用户认可Q9三段整体效果的反馈保持；Q9_R25是可追溯研究参照，不把认可外推至全部Q9子档。video2只需适度改善。历史数值筛选selected=null及video3珊瑚+5.373553108170714%是原比较口径，保持不变；本报告数值均明确采用Q9_R25或匹配模糊为基线。

## 2. 这轮改变了什么

已有的接收guide在部分粗轮廓上比原GLC更接近原片，但单独看明显模糊、带压缩块，并缺少很多细节。新候选让**已传输的当帧guide RGB直接参与输出**。历史Q9只用guide求光流与门控，输出像素来自原GLC；因此这是明确的研究像素来源变更，不是历史raw-only约束下的小调参。

设Q为Q9_R25，G为从原码包重新解码的guide，B1为sigma=1像素的GaussianBlur：

- GB25：clip/round(Q + 0.25 × (G − Q))，直接混合控制。
- GS25：clip/round(Q + 0.25 × B1(G − Q))，温和低频残差。
- GS50：clip/round(Q + 0.50 × B1(G − Q))，较强控制。

三个候选一并冻结，三片统一参数，无源图或源ROI参与输出。guide由同一基础码包绑定的原辅助包解码，与历史缓存逐位一致；未重新编码、未增加传输字节、未新增光流模型推理。clip/round只在输出末端执行一次。候选不保证处于raw/Q9通道区间，也不保证原alpha=0处不变，这是已登记的机制变化。

## 3. 第一阶段结果

低频时序误差变化（相对Q9_R25，负值为误差降低）：

{table(['素材','GB25','GS25','GS50'],timing)}

这是源光流对齐、共同可靠支持区上的低频时序误差，不是肉眼闪烁次数。每片完整149帧对；三片各评分范围均无低支持排除。本轮另保留全频、细纹理、四象限和珊瑚范围CSV，不能仅凭一个平均值验收。

全帧PSNR变化（dB，相对Q9_R25）：

{table(['素材','GB25','GS25','GS50'],spatial)}

源侧固定主体宽框RGB MAE：

{table(['区域','Q9_R25','GB25（相对Q9）','GS25（相对Q9）','GS50（相对Q9）','等细节模糊'],regions)}

宽框包含鱼与背景，不是语义mask，也不是鱼形恢复比例。源图只在独立评价和观看中读取，不进入接收输出或控制参数匹配。

原登记珊瑚范围低频时序变化（本表相对Q9_R25）：

{table(['素材','GB25','GS25','GS50'],reef_rows)}

此表不覆盖用户尚未提供的video4，也不替换历史+5.37%结论。

## 4. 纹理和运动的代价

全帧细纹理幅度（abs(gray − sigma1.2低通)均值；低不自动等于好）：

{table(['素材','原片','guide','Q9','GB25','GS25','GS50'],texture)}

GS25比GB25保留更多细纹理；GS50修正更强，细纹理进一步降低。guide单独PSNR较好但细节弱，说明不能据PSNR选方案。

源侧高变化像素上的变化幅度比（1表示与源变化幅度总量相等，不是运动速度）：

{table(['素材','Q9','GB25','GS25','GS50','guide'],dynamic)}

GS25/GS50仍压低动态幅度。动态RGB误差降低与幅度更小同时发生，不能据此宣称运动自然或闪烁已消失；全部逐帧动态记录在CSV。

## 5. 事后登记的等细节模糊控制

在看到第一阶段“误差与细节幅度同时下降”后，单独登记3组BLUR_MATCH诊断。未改原9组计划、结果、代码或核验。每片仅读取接收端Q9和GS25，匹配全10秒平均fine amplitude：Q9单帧高斯模糊，sigma范围0–1.5，12轮二分，选幅度绝对误差最小值；并固定clip/rint/uint8。匹配不读源图、评分summary、guide数组或诊断记录。

每片sigma不同只为诊断时匹配纹理幅度，**不是可部署的按片调参候选**，更不是另一个独立留出实验。

{table(['素材','模糊sigma','幅度匹配偏差','模糊低频误差vsQ9','GS25低频误差vsQ9','GS25低频误差vs模糊','GS25 PSNR−模糊'],blur_rows)}

在几乎相同的全片平均细节幅度下，GS25仍改善低频时序、PSNR与主体宽框误差，支持当帧guide提供了单纯空间模糊之外的信息。**该控制只排查了这一类等均值高斯模糊解释，没有匹配逐像素细节、完整频谱或运动幅度，不能说已经排除全部平滑贡献。**

反向指标也需保留：

{table(['素材','模糊细纹理时序MAE','GS25细纹理时序MAE','GS25相对模糊','模糊动态幅度比','GS25动态幅度比'],fine_rows)}

三片GS25的细纹理时序MAE都高于等细节模糊，动态幅度更低；细纹理NCC也较低，原值见summary。因此本轮不是所有尺度和运动指标都更优。

## 6. 实际静帧观察与观看顺序

已查看三片各30/61/90/120帧全图拼图，以及三片61帧的模糊控制四宫格。以下只属于静帧观察，没有冒充用户原速验收：

- video1：GS25保留珊瑚与鱼的细部，整体变化较温和；气泡/小鱼形状仍有偏差。
- video2：主鱼均可见，GS25仍保留较多斑点，但4.07秒尾部和斑点分布仍明显不同于原片；不能称鱼尾残像已修好。
- video3：前景鱼与珊瑚仍可见，GS25沿袭部分Q9身体/鳍线条偏差；原片中的形状没有完整恢复。
- guide单独明显模糊且有块状失真；强融合更可能带入这些问题。GS25仅作为下一次原速观看的温和候选。

先看左Q9/右GS25三段完整10秒；再看八宫格中的原片与guide，必要时看video2/video3主体放大。GS50和GB25保留作强度/机制对照，不自动替换Q9。

{rl('实际静帧观察记录',run/'visual_inspection.json')} · {rl('观看入口',view/'README.md')}

## 7. 码率、输入边界与核验

{table(['素材','实际总kbps','其中辅助kbps','新增字节'],rate_rows)}

码率按原始基础包与辅助包全部字节、10秒时长计算；观看MP4不是传输包。未重做端到端计时，不作实时或长片有界内存宣称。

候选输出通道裁剪比例：

{table(['素材','GB25','GS25','GS50'],clipping)}

9组候选逐帧独立公式重算、guide重新解码、原Q9数组一致性、禁读输入探测、CSV汇总及主体框误差核验通过。3组模糊控制的匹配过程重放、逐像素公式、输入禁读和汇总核验通过。禁读钩子用于复现边界检查，不是操作系统安全沙箱。

控制登记时修正过一个依赖文件路径（当时尚未生成计划/输出）；首个控制核验因float32/float64聚合精度不一致误报，差约1.94e-7。另写独立核验器修正聚合精度，**冻结实验源码和输出未改**。原失败与修复说明保留在{rl('实现记录',control/'implementation_notes.json')}。

14个观看视频均为10秒/15 FPS，已核对尺寸、150帧、时长及抽样面板，6张PNG已查看。源图用于评分；输出NPY用于度量；MP4仅用于观看。

{rl('主实验核验',run/'verification.json')} · {rl('模糊对照核验',control/'verification.json')} · {rl('主媒体核验',run/'previews/verification.json')} · {rl('对照媒体核验',control/'previews/verification.json')}

正式采用配置SHA256仍为{EXPECTED_CONFIG}。selected=null，未自动采用，未改部署默认，未提交/推送。

## 8. 接续基调

这一方向在已知三片上比上轮轻微回退原GLC更值得继续看，但目前证据只支持“接收guide低频残差带来额外改善”，不支持“真实鱼形已恢复”或“泛化问题已解决”。

下一步先核对GS25三片原速的鱼尾、前景鱼形和运动自然度；若继续研究，保留Q9细纹理，重点分离有效当帧结构与guide带来的动态幅度压低。video4收到后单独登记源文件、帧数、码率与固定基线，再评价珊瑚密集场景；当前未提前运行。长片整体可看性仍为后续目标。

本轮已完成，无活动实验。历史Q9正面反馈、旧selected=null及video3珊瑚+5.37%记录均保留。
"""
    put(report,report_text)

    view.mkdir()
    sources=sorted([p for root in (run/'previews',control/'previews') for p in root.iterdir() if p.suffix in ('.mp4','.png')])
    assert len(sources)==20 and len({p.name for p in sources})==20
    media_records={}
    for src in sources:
        dst=view/src.name;os.link(src,dst)
        h=sha(src)
        assert os.path.samefile(src,dst)
        if src.suffix=='.mp4':
            expected=(cm if src.parent.parent==control else media)['videos'][src.name]['sha256']
            assert h==expected
        media_records[src.name]=dict(source=str(src),sha256=h,bytes=src.stat().st_size,hardlink=True)
    save(view/'media_manifest.json',dict(complete=True,files=media_records,video_count=14,png_count=6))
    vl=lambda label,p:link(label,p,view)
    put(view/'README.md',f"""# Q9 引导结构融合：原速观看（2026-11-13）

本轮优先看video1/2/3的整体可看、当下鱼形与运动。**GS25仅为温和观看候选，未采用。** 它把已传输guide的低频残差加入Q9输出，无新增传输字节；与历史仅原GLC像素的Q9机制不同。

## 优先看三段双列

{chr(10).join('- '+vl(c+'：左Q9_R25 / 右GS25',view/(c+'_Q9_vs_GS25.mp4')) for c in CASES)}

三段均完整10秒、15 FPS。主要看鱼尾/前景鱼形是否更自然、纹理是否被削弱、运动是否变钝；不能仅凭平均误差下降判断成功。

## 原片与机制对照

{chr(10).join('- '+vl(c+'八宫格',view/(c+'_all_eight.mp4')) for c in CASES)}

每行左→右：原片/原GLC；guide/Q9_R25；CSM50/GB25；GS25/GS50。guide只作诊断参照，CSM50为前轮候选。GB25直接混合更软，GS50修正更强。

{chr(10).join('- '+vl(c+'：左Q9 / 右强档GS50',view/(c+'_Q9_vs_GS50.mp4')) for c in CASES)}

## 主体放大

- {vl('video2主鱼和鱼尾',view/'video2_current_fish_crop.mp4')}
- {vl('video3前景鱼',view/'video3_current_fish_crop.mp4')}

六宫格每行左→右：原片/guide；Q9/GS25；GS50/GB25。宽框包含背景。

## 等细节模糊诊断

{chr(10).join('- '+vl(c+'匹配模糊四宫格',view/(c+'_matched_blur_control.mp4')) for c in CASES)}

布局：原片/Q9；等细节模糊/GS25。BLUR_MATCH只为匹配全片平均细纹理幅度的诊断，不是部署候选。GS25低频误差较低，但细纹理时序误差较高、动态幅度更小。

## 静帧与报告

{chr(10).join('- '+vl(c+'全图拼图',view/(c+'_whole_scene_contact.png'))+' · '+vl('4.07秒模糊对照',view/(c+'_matched_blur_frame61.png')) for c in CASES)}

已看静图，用户原速反馈待记录；未证明鱼消失/残像已修复。礁石、长片本轮未跑，video4等待用户提供。

{vl('完整中文报告',report)} · {vl('结构化结果',run/'round_result.json')} · {vl('主核验',run/'verification.json')} · {vl('模糊核验',control/'verification.json')}

媒体为硬链接，编辑前请另存副本。观看MP4体积不代表通信码率。正式采用配置保持S8I + CURRENT_I12。
""")

    for p in entries:
        old=p.read_text(encoding='utf-8-sig')
        top=f"""**最新（2026-11-13）：Q9已传引导结构融合一轮完成，video1/2/3共9组候选，另加3组等细节模糊诊断。** GS25三片低频时序误差相对Q9下降17.24%/15.13%/16.77%，仅作温和观看候选，尚未采用。

新候选让已传guide RGB参与输出，明确改变历史raw-only像素来源；无新增传输字节、不读源图。相同细节幅度的高斯模糊不能解释全部改善，但GS25细纹理时序误差较高、动态幅度更低，鱼形偏差仍在，不能宣称已恢复。

{link('本轮报告',report,p.parent)} · {link('video1/2/3原速观看',view/'README.md',p.parent)} · {link('结果与控制',run/'round_result.json',p.parent)} · {link('小证据',evidence/'manifest.json',p.parent)}

本轮已收束，无活动实验。礁石与长片未重跑；video4等用户提供后单独登记，未提前裁片或处理。Q9三段正面反馈与正式采用配置保持，旧selected=null及video3珊瑚+5.37%记录不改。未提交/推送。

{HISTORY}
"""
        if p.suffix=='.txt':
            top=f"""GLC接续，2026-11-13：已传引导结构融合9组+匹配模糊诊断3组完成，无活动实验。
GS25仅温和观看候选，未采用；三片低频时序误差降低17.24%/15.13%/16.77%，鱼形偏差仍在。
新候选guide RGB参与输出，改变历史raw-only约定，无额外码率、源图不进入输出。
等细节高斯模糊不能解释全部收益，但GS25细纹理时序误差较高、运动幅度更小。
主攻video1/2/3整体可看与当前鱼形；礁石残像降优先级，本轮未跑礁石/长片。
video4用户正在准备，未收到路径、未猜测时间段、未裁切或提前处理。
Q9_R25为研究参照，正式S8I+CURRENT_I12及历史selected=null、video3珊瑚+5.37%不改。
报告：{report.as_posix()}
观看：{(view/'README.md').as_posix()}
运行：{run.as_posix()}
下一步看三段GS25原速，结合纹理和动态代价，再登记video4。未提交/推送。

此前记录（当时状态，以页首为准）：
"""
            put(p,top+'\n'+old)
        else:
            title,body=old.split('\n',1)
            put(p,title+'\n\n'+top+'\n'+body.lstrip('\n'))

    state=read(statepath)
    for n in ('latest_experiment','latest_generalization_experiment','latest_diagnosis',
              'next_experiment_plan','next_priority','user_latest_decision'):
        state['previous_'+n+'_before_received_structure']=copy.deepcopy(state.get(n))
    latest=dict(result,run=str(run),report=str(report),viewing=str(view),
                verification=str(run/'verification.json'),control_verification=str(control/'verification.json'),
                evidence=str(evidence/'manifest.json'))
    state.update(status='Q9_received_structure_round_complete',active_experiment_plan=None,
        latest_experiment=latest,latest_generalization_experiment=copy.deepcopy(latest),
        latest_analysis_report=relative(report,REPO),new_experiment_report=relative(report,REPO),
        latest_experiment_plan=dict(status='completed',path=str(run/'execution_plan.json'),
            diagnostic_control=str(control/'execution_plan.json'),new_candidate_outputs=9,diagnostic_outputs=3),
        latest_diagnosis=dict(report=str(report),fish_presence_priority=True,reef_ghost_optimization_deprioritized=True,
            guide_current_structure_promising=True,matched_gaussian_blur_not_full_explanation=True,
            fine_temporal_and_dynamic_tradeoff=True,fish_shape_not_resolved=True),
        latest_continue_authorization=auth,current_session_work='Completed 9 received-guide fusion outputs and 3 matched-fine-amplitude blur controls on video1/2/3; GS25 viewing only; video4 pending.',
        user_latest_decision=auth['verbatim'],new_experiments_uncommitted=True,
        next_priority=['Review GS25 vs Q9 on video1/2/3 at original speed for current fish shape, texture and motion.',
                       'Separate current-structure correction from guide-induced attenuation of dynamic changes.',
                       'Register video4 only after the user provides its file; do not infer a segment.',
                       'Keep whole-long-video watchability as future target; reef ghost footprint is secondary.'],
        next_experiment_plan=dict(status='proposed_not_run',reference='Q9_R25',viewing_candidate='GS25',
            focus=['current fish shape','whole-scene watchability','fine texture','motion amplitude'],
            video4=dict(status='awaiting_user_clip',source_path=None,time_interval=None),
            full_long_video_rerun='not_started',parameters_not_chosen=True),
        experiment_execution_gate=dict(status='completed_round',currently_running=False,new_run_started=True,
            automatic_next_run=False,date='2026-11-13'),
        publication=dict(current_changes_committed=False,pushed=False,receipt=str(run/'publication.json'),
            scope='Received-guide current structure fusion and matched blur diagnosis'))
    state['preferred_research_reference'].update(latest_experiment_report=str(report),
        current_focus='Video1/2/3 overall watchability, current fish and motion; GS25 viewing only; video4 pending.',
        viewing_candidate='GS25',viewing_candidate_not_adopted=True)
    save(statepath,state)
    save(run/'publication.json',dict(complete=True,date='2026-11-13',report=str(report),viewing=str(view),
        evidence=str(evidence),new_candidate_outputs=9,diagnostic_outputs=3,videos=14,pngs=6,
        report_sha256=sha(report),publisher_sha256=sha(__file__),
        updated_entry_sha256={str(p):sha(p) for p in entries+[statepath]},
        current_profile_unchanged=True,committed=False,pushed=False))

    evidence.mkdir(parents=True)
    for src in [p for p in run.rglob('*') if p.is_file() and p.suffix.lower() in ('.json','.csv','.py','.md','.txt')]:
        dst=evidence/'run'/relative(src,run);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for src in (report,Path(__file__),STAGE/'publish_q9_ghost_exit.py'):
        dst=evidence/('report' if src==report else 'publication_source')/src.name
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for src in [p for p in sources if p.suffix=='.png']:
        dst=evidence/'visuals'/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for src in (view/'README.md',view/'media_manifest.json'):
        dst=evidence/'viewing'/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for src in entries+[statepath]:
        dst=evidence/'entries_after'/relative(src,WORKSPACE)
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    checked=0
    for p in [report,view/'README.md']+entries:
        text=p.read_text(encoding='utf-8-sig')
        if p in entries:
            text=text.split(HISTORY)[0]
            if p.suffix=='.txt':text=text.split('此前记录（当时状态，以页首为准）：')[0]
        for href in re.findall(r'\]\(([^)]+)\)',text):
            target=(p.parent/href.split('#',1)[0]).resolve()
            assert target.exists() or target==evidence/'manifest.json',(p,href)
            checked+=1
    for p,h in old_manifests.items():assert sha(p)==h
    for p,h in snapshots.items():assert sha(before/relative(p,WORKSPACE))==h
    for src in sources:assert sha(src)==sha(view/src.name)==media_records[src.name]['sha256']
    for p in evidence.rglob('*.json'):read(p)
    final=read(statepath)
    assert final['active_experiment_plan'] is None and final['latest_experiment']['new_outputs']==12
    assert final['latest_user_perceptual_feedback']==read(before/relative(statepath,WORKSPACE))['latest_user_perceptual_feedback']
    assert final['next_experiment_plan']['video4']['source_path'] is None
    assert sha(STAGE/'当前采用配置.json')==EXPECTED_CONFIG
    qa=dict(complete=True,new_links_checked=checked,old_evidence_manifests_unchanged=True,
        media_sha_match=True,strict_json_checked=True,prior_user_feedback_preserved=True,
        state_complete_no_active_plan=True,full_fine_coarse_csv_reaggregation=True,
        current_config_sha256=sha(STAGE/'当前采用配置.json'))
    save(run/'publication_verification.json',qa)
    shutil.copy2(run/'publication_verification.json',evidence/'run/publication_verification.json')
    files={}
    for p in sorted(evidence.rglob('*')):
        if p.is_file():
            assert p.suffix not in ('.mp4','.npy','.npz')
            files[relative(p,evidence)]=dict(bytes=p.stat().st_size,sha256=sha(p))
    save(evidence/'manifest.json',dict(complete=True,date='2026-11-13',files=files,source_run=str(run),
         large_arrays_and_videos_excluded=True,self_excluded=True))
    for n,v in files.items():assert sha(evidence/n)==v['sha256']
    print('PUBLISHED',len(files),'evidence files;',checked,'links;',len(sources),'media files',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    publish(p.parse_args().run.resolve())

