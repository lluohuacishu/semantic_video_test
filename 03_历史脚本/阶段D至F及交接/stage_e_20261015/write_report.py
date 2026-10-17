"""Stage E evidence, bounded conclusions, and a runnable video example."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import shutil
import zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from run_stage_e import *

SCRIPT=Path(__file__).resolve().parent

def link(path,title):return f"[{title}]({Path(path).as_posix()})"

def rectangles(manifest,frame):
    return [p["box"] for p in manifest["patches"] if p["start"]<=frame<p["start"]+p["frames"]]

def coverage(manifest,semantics,labels,human):
    per=[];manual=[];area=0;covered_gt=0;total_gt=0;predicted=0
    lookup={x["frame_index"]-420:x for x in labels}
    for idx in range(50):
        roi=np.zeros((720,1280),bool)
        for x,y,w,h in rectangles(manifest,idx):
            assert not roi[y:y+h,x:x+w].any()
            roi[y:y+h,x:x+w]=True
        area+=int(roi.sum())
        for region in semantics["frames"][idx]["regions"]:
            x,y,w,h=region["box"];predicted+=1
            assert roi[y:y+h,x:x+w].all(),"A predicted box lost protection"
        if idx in lookup:
            gt=np.asarray(Image.open(lookup[idx]["fish_mask"]))>0
            n=int(gt.sum());covered=int((roi&gt).sum());total_gt+=n;covered_gt+=covered
            components,k=ndimage.label(gt,structure=np.ones((3,3),np.uint8));comp=[]
            for j in range(1,k+1):
                c=components==j;size=int(c.sum());comp.append({"area":size,"coverage":int((c&roi).sum())/size})
            per.append({"source_frame":420+idx,"gt_pixels":n,"covered_pixels":covered,"coverage":covered/n if n else None,"components":comp})
        boxes=np.zeros((720,1280),bool)
        for x,y,w,h in human["frames"][idx]["boxes"]:boxes[y:y+h,x:x+w]=True
        n=int(boxes.sum());inside=int((roi&boxes).sum())
        manual.append({"source_frame":420+idx,"reviewed_box_union_pixels":n,"covered_box_union_pixels":inside,"box_area_coverage":inside/n if n else None,"note":"rectangles are not pixel fish GT"})
    positive=[r for r in per if r["gt_pixels"]]
    human_positive=[r for r in manual if r["reviewed_box_union_pixels"]]
    return {"average_roi_area":area/(50*1280*720),"pooled_gt_fish_coverage":covered_gt/total_gt,"positive_gt_frame_hit":sum(r["covered_pixels"]>0 for r in positive)/len(positive),"positive_gt_frame_mean_coverage":float(np.mean([r["coverage"] for r in positive])),"predicted_boxes_fully_protected":predicted,"all_predicted_boxes_preserved":True,"per_gt_frame":per,"manual_positive_frames":len(human_positive),"manual_frame_mean_box_area_coverage":float(np.mean([r["box_area_coverage"] for r in human_positive])),"manual_frame_95_box_area_fraction":sum(r["box_area_coverage"]>=.95 for r in human_positive)/len(human_positive),"manual_frame_any_box_overlap_fraction":sum(r["covered_box_union_pixels"]>0 for r in human_positive)/len(human_positive),"manual_frames":manual}

def component_quality(groups,labels):
    rows=[]
    for label in labels:
        gt=np.asarray(Image.open(label["fish_mask"]))>0
        raw=np.asarray(Image.open(OLD/f"sender/private_frames/{label['frame_index']-420:05d}.png").convert("RGB"),dtype=np.float64)
        components,k=ndimage.label(gt,structure=np.ones((3,3),np.uint8))
        for j in range(1,k+1):
            mask=components==j
            row={"source_frame":label["frame_index"],"component":j,"area":int(mask.sum()),"psnr":{}}
            for group,directory in groups.items():
                rgb=np.asarray(Image.open(directory/f"{label['frame_index']-420:05d}.png").convert("RGB"),dtype=np.float64)
                mse=float(((raw-rgb)[mask]**2).mean())
                row["psnr"][group]=float(10*np.log10(255**2/mse))
            rows.append(row)
    return rows

def main():
    protocol=read(RUN/"protocol.json");selected=read(RUN/"selection.json")
    assert protocol["intent_sha256"]==sha256(INTENT)
    for name,expected in protocol["code_sha256"].items():
        path=SCRIPT/name if name=="run_stage_e.py" else PROJECT/"semantic_mvp"/name
        assert sha256(path)==expected,"Executable code changed after protocol freeze"
    legacy=read(OUT/"旧包与公开CLI验证.json");multi=read(OUT/"多区域恢复与协议检查.json")
    receipt=read(RUN/"network/end_to_end.json");decoder=read(RUN/"network/receiver/decode.json")
    assert receipt["success"] and receipt["packet_sha256_equal"] and receipt["separate_sender_receiver_processes"]
    assert receipt["frames"]==50 and selected["matched_within_2_percent"]
    assert sha256(selected["file"])==sha256(RUN/"network/receiver/message.scv")==decoder["file_sha256"]
    read_audit=read(RUN/"network/receiver/receiver_read_audit.json")
    assert all(x["allowed"] and Path(x["path"]).resolve().is_relative_to((RUN/"network/receiver").resolve()) for x in read_audit["python_reads"])
    with zipfile.ZipFile(selected["file"]) as z:
        assert z.testzip() is None
        manifest=json.loads(z.read("manifest.json"))
        _,semantics,semantic_bytes,_=inputs()
        assert z.read("semantics.json.gz")==semantic_bytes
        for name,entry in manifest["files"].items():
            payload=z.read(name)
            assert len(payload)==entry["bytes"] and hashlib.sha256(payload).hexdigest()==entry["sha256"]
    with zipfile.ZipFile(read(CONTROL/"A/selection.json")["file"]) as z:old_manifest=json.loads(z.read("manifest.json"))
    data=read(PROJECT/"data/phase_c_20261013/manifest.json")
    labels=[r for v in data["videos"] if v["video"]=="video131" for r in v["labels"] if 420<=r["frame_index"]<470]
    for label in labels:
        expected=next(x for x in read(CONTROL/"protocol.json")["gt_frames"] if x["frame"]==label["frame_index"])
        assert sha256(label["fish_mask"])==expected["sha256"]
    human=read(CONTROL/"manual_roi_snapshot.json")
    geometry={"A":coverage(old_manifest,semantics,labels,human),"E":coverage(manifest,semantics,labels,human)}
    write_json(OUT/"ROI与逐帧覆盖.json",geometry)
    directories={"A":CONTROL/"A/network/receiver/rgb","E":RUN/"network/receiver/rgb","H265":CONTROL/"H265/network/receiver/rgb"}
    comp=component_quality(directories,labels);write_json(OUT/"逐GT连通域画质.json",comp)
    old_metrics={r["group"]:r for r in read(WORKSPACE/"outputs/roi_control_20261015/实际码流与GT任务评价.json")}
    metrics={g:old_metrics[g] for g in ("A","H265")};metrics["E"]=read(OUT/"阶段E真实任务评价.json")
    assert all(m["teacher"]==metrics["E"]["teacher"] for m in metrics.values())
    selections={g:read(CONTROL/g/"selection.json") for g in ("A","H265")};selections["E"]=selected
    types=Counter()
    for stream in read(RUN/"stream_audit.json"):types.update(stream["frame_types"])
    write_json(OUT/"阶段E完整汇总.json",{"metrics":metrics,"selections":selections,"geometry":geometry,"component_quality":comp,"network":receipt,"stream_frame_types":dict(types)})

    figure=OUT/"自动ROI组织对比.png"
    raw=np.asarray(Image.open(OLD/"sender/private_frames/00000.png").convert("RGB"))
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
    from matplotlib.patches import Rectangle
    for ax,m,title in zip(axes,[old_manifest,manifest],["A: one rectangle / window 32","E: left and right crops / window 32"]):
        ax.imshow(raw);ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
        for x,y,w,h in rectangles(m,0):ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor="#ffdf60",linewidth=1.8))
    fig.suptitle("First-window geometry on source frame 420 (not detection GT)")
    fig.savefig(figure,dpi=140);plt.close(fig)

    launcher=PROJECT/"阶段E_双区域视频示例.cmd"
    launcher.write_text('@echo off\nsetlocal\ncd /d "%~dp0"\nset PYTHONUTF8=1\n".venv\\Scripts\\python.exe" "semantic_mvp\\system.py" demo --target fish --mode video --frames 50 --kbps 20 --roi-layout halves --roi-crf 31 --open\npause\n',encoding="ascii",newline="\r\n")
    rows=[]
    for g,label in (("A","阶段D原单区域"),("E","阶段E自动双区域"),("H265","普通H.265/B帧")):
        m=metrics[g]
        rows.append(f"| {label} | {selections[g]['file_bytes']:,} | {m['gt']['iou']*100:.3f} | {m['gt']['recall']*100:.3f} | {m['gt']['precision']*100:.3f} | {m['gt_fish_rgb_psnr_db']:.3f} | {m['gt_background_rgb_psnr_db']:.3f} | {m['global_rgb_psnr_db']:.3f} |")
    components=[f"| {r['source_frame']} | {r['area']} | {r['psnr']['A']:.3f} | {r['psnr']['E']:.3f} | {r['psnr']['H265']:.3f} |" for r in comp]
    e,a,h=metrics["E"],metrics["A"],metrics["H265"]
    a_fixed=selections["A"]["fixed_crf32_diagnostic"]["file_bytes"]
    fixed=selected["fixed_crf32_diagnostic"]["file_bytes"]
    roi=geometry["E"]
    counts=[f"| {g} | {metrics[g]['gt']['tp']} | {metrics[g]['gt']['fp']} | {metrics[g]['gt']['fn']} | {metrics[g]['positive_frame_mean_iou']*100:.3f}% | {metrics[g]['positive_frame_mean_recall']*100:.3f}% |" for g in ("A","E","H265")]
    report=OUT/"阶段E_自动双区域ROI视频与收发实测_2026-10-15.md"
    report.write_text(f"""# 阶段E：自动双区域ROI视频与收发实测

日期：2026-10-15。按用户要求，本轮推进至阶段E并收尾，只做一种自动ROI组织改动，没有增加或训练新模型。

## 1. 本轮完成了什么

主目标继续是**接收端恢复完整视频，语义信息保护关键目标**。发送端保留原分割器、阈值0.9、64像素筛选、外扩16像素和32帧时间窗口，改为在左右两个固定空间分区内分别求外接矩形，每窗口最多两路ROI，空分区不编码。跨中线的预测框分到两侧，保留其全部预测范围。

本轮实现了新发送选项`--roi-layout halves`和v2接收协议。新接收端支持同帧两块空间不重叠的ROI，也继续读取v1旧包。默认`single`模式保留；新包须使用包含`roi_layout.py`的新接收端，旧接收端会拒绝v2包。

这解决的是远距离预测框合并成巨型矩形的问题，不能自动补回检测器没有识别出的鱼，也没有实现神经网络动态比特分配。原人工ROI及像素GT仅用于编码选择之后的离线诊断。

## 2. 先锁定的执行边界

{link(INTENT,'阶段E执行约定')}在实现和新实验之前保存，SHA256为`{protocol['intent_sha256']}`。只比较一段开发视频、一档完整文件预算、一个新自动策略；不因结果好坏追加调参。

输入为video131原420–469连续50帧、1280×720、名义25fps，来自真实连续序列。它是已经用于训练/诊断的已见视频，不是独立测试集，拍摄时间戳未确认。复用完全相同的自动预测和背景码流，以隔离ROI组织因素；没有用人工框删改预测，也没有通过GT选择CRF。

完整文件目标100,000B十进制、容差2%，只按完整字节选CRF。背景仍为640×360/CRF42；ROI仍为x265 medium、P-only、GOP32、BT709 limited YUV420。H.265基线沿用此前原分辨率、允许B帧的完整MP4。全部背景、ROI、区域元数据、语义gzip、manifest和ZIP计入文件。

## 3. ROI面积与真实文件

新方案共3路ROI码流：第一窗口左右各一块，第二窗口只有右侧一块。旧方案2路。全部35个预测连通区域的目标框仍被ROI完整包含，没有删除预测；这不意味着真实鱼都被检测。

| 指标 | 旧单区域A | 新双区域E |
|---|---:|---:|
| 平均ROI画面面积 | {geometry['A']['average_roi_area']*100:.3f}% | {roi['average_roi_area']*100:.3f}% |
| 同ROI CRF32完整包 | {a_fixed:,}B | {fixed:,}B |
| 约100KB选定ROI CRF | {selections['A']['crf']} | {selected['crf']} |
| 稀疏GT鱼像素被ROI覆盖 | {geometry['A']['pooled_gt_fish_coverage']*100:.2f}% | {roi['pooled_gt_fish_coverage']*100:.2f}% |
| 三张含鱼GT中有覆盖的帧 | 3/3 | 1/3 |

相同CRF32下，新完整文件减少{100*(1-fixed/a_fixed):.2f}%。这只是该编码配置的字节诊断，不意味着两者重建失真相同。

![自动ROI组织示意](自动ROI组织对比.png)

更小的ROI也失去了大矩形顺带包含漏检鱼的能力：435、450帧的GT没有进入新ROI，465帧985个鱼像素进入ROI。69.12%的像素汇总覆盖仍受最大含鱼帧主导，不能宣传为大多数含鱼观测都得到保护。

在用户核对的38个含框帧上，逐帧人工框面积覆盖等权均值为{roi['manual_frame_mean_box_area_coverage']*100:.2f}%，覆盖≥95%框面积的帧占{roi['manual_frame_95_box_area_fraction']*100:.2f}%。这些是矩形几何诊断，不是新增鱼像素GT，完整逐帧记录保留。

## 4. 相近完整文件预算下的结果

IoU、召回、精确率为%；PSNR为dB。全局PSNR使用50帧的汇总RGB误差；GT鱼区、背景与分割准确率仅使用原4张发布者像素GT，其中3张含鱼、共1425个鱼像素。固定分割器对接收RGB重新推理，未使用传来的框计分。

| 方案 | 完整字节 | 鱼IoU | 鱼召回 | 鱼精确率 | GT鱼区PSNR | GT帧背景PSNR | 全局PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

E对A：IoU变化{100*(e['gt']['iou']-a['gt']['iou']):+.3f}个百分点，召回变化{100*(e['gt']['recall']-a['gt']['recall']):+.3f}个百分点；但汇总GT鱼区PSNR变化{e['gt_fish_rgb_psnr_db']-a['gt_fish_rgb_psnr_db']:+.3f}dB，全局PSNR变化{e['global_rgb_psnr_db']-a['global_rgb_psnr_db']:+.3f}dB。因此，不能把任务指标的积极结果解释为整体画质全面改善。

| 方案 | TP鱼像素 | FP像素 | FN鱼像素 | 含鱼帧IoU等权均值 | 含鱼帧召回等权均值 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(counts)}

新增TP集中在465帧801像素的大连通域，其召回为99.75%。435、450帧的TP仍为0；3个≤256像素连通域的召回也仍为0。原始未压缩输入的固定模型IoU为26.196%、召回33.053%，原图本身就漏掉小鱼；压缩输入上的预测响应还伴随大量FP。本轮不能宣称解决了小目标保护。

| GT源帧 | 连通域像素 | A区内PSNR | E区内PSNR | H.265区内PSNR |
|---|---:|---:|---:|---:|
{chr(10).join(components)}

不同鱼区质量和任务识别可能向不同方向变化。汇总值与逐帧/连通域值必须同时保留，不能只挑大鱼或只挑一项指标展示。

## 5. 独立收发与必要检查

实际通信包{selected['file_bytes']:,}B，距100,000B为{selected['relative_budget_difference']:+.3%}。组成：背景{selected['parts']['context']:,}B、ROI流{selected['parts']['roi']:,}B、冻结语义{selected['parts']['semantic']:,}B、manifest/ZIP{selected['parts']['container']:,}B。原有完整语义消息也计入，没有免费掩码或参考帧。

经本机真实TCP限速20kbps，应用层发送{receipt['application_transmit_bytes']:,}B，传输{receipt['transfer_seconds']:.3f}秒；独立接收进程恢复50帧，收发SHA256一致。接收恢复及展示生成{decoder['decode_and_preview_seconds']:.3f}秒。总视频流帧类型为I/P/B={types['I']}/{types['P']}/{types['B']}，包含背景与ROI层的重复空间层，不代表源视频多了帧。

传输时间不含缓存的发送端识别、背景编码或候选搜索。名义2秒视频仍需约40秒发送，属于批次系统，尚未实现20kbps实时视频。接收时间包含PNG、预览及叠加展示生成，不能当作纯解码速度。没有实际水声链路或跨机器试验。

- 跨中线、图像边缘、窄偶数尺寸和空ROI检查通过；全部预测范围保留。
- 32帧同时具有两块ROI，逐像素核对两块接收ROI均被贴回，ROI外像素与接收背景一致，50帧全量检查通过。
- 非法空间重叠、跨分区、未知版本及旧格式时间重叠均被拒绝；空ROI v2包可读取。
- 新接收器解码旧v1完整包，50帧RGB哈希与旧结果完全相同。
- 公开CLI用5帧新鲜模型预测完成编码与独立解码，验证新选项可以实际运行；这5帧冒烟检查不是新的准确率实验。
- 新包所有成员长度、CRC、SHA256、读取目录和固定代码指纹检查通过。读取约束是程序级审计，不是OS沙箱。

## 6. 怎么使用

{link(launcher,'双击阶段E视频示例')}可对既有鱼类连续示例执行新鲜识别、视频发送、独立接收并打开结果，默认20kbps。示例使用CRF31；任意其他视频的文件大小不会自动固定为100KB。同预算搜索是本轮实验脚本的步骤，不是新增的通用视频预算控制器。

自己的视频可用公开CLI，输出目录自动按时间创建：

```powershell
cd C:\\Users\\lenovo\\Desktop\\works\\work\\semantic-video-mlvc
& .\\.venv\\Scripts\\python.exe semantic_mvp/system.py demo --input 'C:\\视频\\水下.mp4' --target fish --mode video --frames 50 --kbps 20 --roi-layout halves --roi-crf 31 --open
```

本次已完成结果：[接收端视频结果](http://127.0.0.1:8776/results/semantic_mvp_stage_e_20261015/network/receiver/查看结果.html)。现有网页启动器继续使用旧单区域默认配置；新方式通过上述示例或命令显式选择，没有把有取舍的开发结果直接替换成全局默认。

## 7. 阶段E判断与今天的结束点

**阶段E的小范围工程迭代已完成。** 自动双区域减少了远距离框之间的额外编码，且该开发片段同预算下的部分任务表现改善；但漏保护小鱼、整体画质代价和分割器大量误检仍然存在。未证明跨场景优于普通H.265。

今天到此收尾：代码、包、日志、负结果和本报告全部保留，不继续增加网络、阈值搜索或阶段F。下一次优先处理自动ROI对漏检小目标的覆盖，并用独立视频检查已有取舍；阶段C的多帧修复不在今天同时展开。人工C/D仍只作为诊断参照，不作为新自动系统成绩。

## 8. 证据入口

- {link(OUT/'阶段E冻结协议.json','冻结协议与源/模型/代码指纹')}；{link(OUT/'阶段E完整汇总.json','字节、质量与任务汇总')}。
- {link(OUT/'ROI与逐帧覆盖.json','逐帧与人工框覆盖')}；{link(OUT/'阶段E真实任务评价.json','真实GT任务结果')}；{link(OUT/'逐GT连通域画质.json','小目标逐连通域质量')}。
- {link(OUT/'多区域恢复与协议检查.json','多区域与非法包检查')}；{link(OUT/'旧包与公开CLI验证.json','旧包与新鲜CLI验证')}。
- {link(OUT/'页面播放验证.json','本机页面视频播放验证')}；{link(OUT/'阶段E接收完成.jpg','收发完成页面凭证')}。结果页的29/50为发送端预测有目标的帧数，不是人工真实出现帧数。
- {link(RUN,'运行目录')}保存所有候选、完整命令、传输轨迹、接收RGB、读取审计和改动前代码；{link(SCRIPT/'run.log','本轮运行日志')}。
""",encoding="utf-8")
    project_report=report.read_text("utf-8").replace("(自动ROI组织对比.png)","("+figure.as_posix()+")")
    (PROJECT/"docs/phase_e_results_20261015.md").write_text(project_report,"utf-8")
    state={"stage":"E","status":"completed","today_work_finished":True,"scope":"one automatic ROI grouping change; one known clip; one budget","report":str(report),"new_training":False,"automatic_strategy_not_promoted_to_global_default":True,"material_limits":["small fish still missed","global and pooled fish PSNR costs","one seen clip; four sparse GT; no real acoustic link"],"intent_sha256":protocol["intent_sha256"],"packet_sha256":selected["sha256"]}
    write_json(OUT/"阶段E实验状态.json",state)
    archive=OUT/"阶段E_报告代码与实测证据.zip"
    entries=[(INTENT,INTENT.name),(report,report.name),(figure,figure.name),(launcher,launcher.name),(OUT/'阶段E接收完成.jpg','阶段E接收完成.jpg')]
    entries.extend((p,p.name) for p in OUT.glob("*.json") if p.name!="证据包指纹.json")
    entries.extend((p,"experiment_code/"+p.name) for p in SCRIPT.glob("*.py"))
    entries.extend((PROJECT/"semantic_mvp"/name,"runtime_code/"+name) for name in ("system.py","runtime.py","roi_layout.py","teacher.py","transport.py","README.md"))
    entries.append((SCRIPT/"run.log","logs/run.log"))
    entries.append((Path(selected["file"]),"actual_transmitted/E_message.scv"))
    entries.append((RUN/"network/receiver/reconstructed.mp4","post_receive_preview/E.mp4"))
    for g in ("A","H265"):
        source=Path(selections[g]["file"]);entries.append((source,"actual_transmitted/"+g+"_"+source.name))
    for name in ("selection.json","search.json","stream_audit.json"):
        entries.append((RUN/name,"evidence/"+name))
    for name in ("end_to_end.json","transfer.csv"):
        entries.append((RUN/"network"/name,"evidence/network/"+name))
    for name in ("decode.json","receiver_read_audit.json"):
        entries.append((RUN/"network/receiver"/name,"evidence/receiver/"+name))
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("阅读说明.txt","本包保留三组实际传输文件、新程序与实验脚本、结果和日志。模型与完整原始数据依赖本机既有工程；预览是接收后生成，不计入传输字节。本轮没有新模型或训练。\n")
        for path,name in entries:
            if path.is_file():z.write(path,name)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    write_json(OUT/"证据包指纹.json",{"file":str(archive),"bytes":archive.stat().st_size,"sha256":sha256(archive),"zip_crc_valid":True})
    print(json.dumps({"report":str(report),"archive":str(archive),"launcher":str(launcher)},ensure_ascii=False))

if __name__=="__main__":main()
