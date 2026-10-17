"""Small, data-backed report of both disjoint clips from the one new video."""
from __future__ import annotations
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from external_preflight import PROJECT, WORKSPACE, RUN, OUT, DATA, read, write

CASES = [
    ("初始片段0—49", RUN, OUT, DATA),
    ("含鱼补充片段50—99", RUN/"positive_case", OUT/"positive_case", DATA/"positive_case"),
]
FIG = OUT/"figures"
COLORS = {"original":"#667085", "E":"#168369", "H265":"#4979ae"}


def fmt(value, digits=2):
    return "不适用" if value is None else f"{value:.{digits}f}"


def pct(value):
    return "不适用" if value is None else f"{value*100:.2f}%"


def link(path, title):
    return f"[{title}]({Path(path).as_posix()})"


def row_for(result, name):
    return next(row for row in result["rows"] if row["condition"] == name)


def figures():
    fig, axes = plt.subplots(2,2,figsize=(12.5,7.6),layout="constrained")
    for column, (label,root,output,data_folder) in enumerate(CASES):
        for name in ("E","H265"):
            values = read(output/f"{name}_逐帧画质.json")
            axes[0,column].plot([v["source_frame"] for v in values],
                                [v["global_rgb_psnr_db"] for v in values],
                                label=name, color=COLORS[name])
        axes[0,column].set_title("Frames 0-49: empty sparse fish GT" if column == 0 else "Frames 50-99: positive-GT supplement",fontsize=11)
        axes[0,column].set_xlabel("Actual source frame")
        axes[0,column].set_ylabel("Global RGB PSNR (dB)")
        axes[0,column].grid(alpha=.2)
        axes[0,column].legend()
    observations = read(OUT/"逐标注帧分割评价.json")
    gt_frames = [0,15,30,45]
    for offset, name in enumerate(("original","E","H265")):
        values = [next(row["metrics_raw_09"]["fp"] for row in observations
                       if row["condition"] == name and row["source_frame"] == frame) for frame in gt_frames]
        axes[1,0].bar(np.arange(4)+(offset-1)*.24, values,width=.24,label=name,color=COLORS[name])
    axes[1,0].set_xticks(np.arange(4),gt_frames)
    axes[1,0].set_ylabel("False-positive fish pixels")
    axes[1,0].set_xlabel("Four publisher GT frames with no fish pixels")
    axes[1,0].set_title("Threshold 0.9, before 64-pixel filtering",fontsize=10)
    axes[1,0].legend()
    result = read(OUT/"positive_case/新视频实测评价.json")
    names = ["original","E","H265"]
    values = [row_for(result,name)["sparse_gt_metrics"]["iou"] for name in names]
    bars = axes[1,1].bar(names,[100*v if v is not None else 0 for v in values],
                       color=[COLORS[n] for n in names],width=.6)
    for bar,value in zip(bars,values):
        axes[1,1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,pct(value),
                       ha="center",fontsize=9)
    axes[1,1].set_ylim(0,max(15,max(v or 0 for v in values)*100+8))
    axes[1,1].set_ylabel("Fish IoU on three positive publisher masks (%)")
    axes[1,1].set_title("Same frozen model; one holdout video, not three datasets",fontsize=10)
    fig.suptitle("Frozen E vs native offline H.265: actual received RGB, about 100 KB per clip",fontsize=12)
    fig.savefig(FIG/"holdout_metrics.png",dpi=160); plt.close(fig)

    # Show every GT connected component, not a favorable subset.
    target = read(OUT/"positive_case/含鱼GT保护与逐连通域核对.json")
    components = target["gt_components"]
    width = max(240,max(row["box"][2]+64 for row in components))
    height = max(180,max(row["box"][3]+64 for row in components))
    count = len(components)
    crops = []
    positive_root = RUN/"positive_case"
    for page,start in enumerate(range(0,count,5),1):
        group = components[start:start+5]
        fig, axes = plt.subplots(len(group),3,figsize=(10.8,max(3.0,len(group)*2.7)),squeeze=False,
                                 layout="constrained")
        for i,row in enumerate(group):
            x,y,w,h = row["box"]
            left = int(np.clip((2*x+w-width)//2,0,1280-width))
            top = int(np.clip((2*y+h-height)//2,0,720-height))
            crops.append({"source_frame":row["source_frame"],"component":row["component"],
                          "source_crop_xywh":[left,top,width,height],"page":page})
            for column,name in enumerate(("original","E","H265")):
                path = (positive_root/f"initial_crf32/private_frames/{row['local_frame']:05d}.png"
                        if name == "original" else positive_root/f"{name}/network/receiver/rgb/{row['local_frame']:05d}.png")
                image = np.asarray(Image.open(path).convert("RGB"))[top:top+height,left:left+width]
                ax = axes[i,column]; ax.imshow(image,interpolation="nearest")
                ax.add_patch(Rectangle((x-left-.5,y-top-.5),w,h,fill=False,edgecolor="#ec4ab5",linewidth=1))
                suffix = "source reference" if name == "original" else f'{row["fish_rgb_psnr_db"][name]:.2f} dB on GT'
                ax.set_title(f'{row["source_frame"]} / c{row["component"]} / {name}\n{suffix}',fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Every positive GT component, page {page}: same {width} x {height} source crop\n"
                     "GT location is shown; prompted diagnosis, NOT a blinded readability rating",fontsize=10)
        fig.savefig(FIG/f"all_positive_target_crops_{page}.png",dpi=160); plt.close(fig)
    write(OUT/"图示裁剪坐标.json",{"all_GT_components_shown":True,"crops":crops,
                                  "source_RGB":True,"reconstructions":"actual independent receiver PNGs"})


def report():
    results = [read(output/"新视频实测评价.json") for _,_,output,_ in CASES]
    target = read(OUT/"positive_case/含鱼GT保护与逐连通域核对.json")
    data = read(DATA/"manifest.json")
    pos = results[1]
    pe,ph,po = (row_for(pos,k) for k in ("E","H265","original"))
    difference = pe["sparse_gt_metrics"]["iou"]-ph["sparse_gt_metrics"]["iou"]
    signed = f"{difference*100:+.2f}"
    text = f"""# 阶段F · 步骤3：冻结方案的新保留视频对照

日期：2026-10-15。目标仍是接收端恢复视频，由语义保护关键目标。本轮完成**一段新保留视频video6的两段不重叠连续片段**，不是两个独立视频；每段50帧，均实际发送、独立接收并恢复。

## 1. 先给判断

冻结的E在新视频上能够完成收发，但不能据此认定语义保护可靠。含鱼补充片段中，E与H.265的GT鱼IoU分别为{pct(pe["sparse_gt_metrics"]["iou"])}、{pct(ph["sparse_gt_metrics"]["iou"])}，差{signed}个百分点；原图为{pct(po["sparse_gt_metrics"]["iou"])}。鱼区PSNR分别为{fmt(pe["gt_fish_rgb_psnr_db"])}、{fmt(ph["gt_fish_rgb_psnr_db"])}dB。

E在含鱼片段上覆盖了{target["E_covered_GT_fish_pixels"]}/{target["pooled_GT_fish_pixels"]}个已标注鱼像素（{pct(target["E_pooled_GT_fish_coverage"])}），有任意鱼像素进入ROI的含鱼GT帧为{target["positive_GT_frame_any_coverage"]}/{target["positive_GT_frames"]}。这些是本次的真实保护覆盖，不再只看预测框是否全部保留。

**关键负结果：原图鱼IoU为25.80%，两种重建均为0；E鱼区PSNR高约1.79dB，但全局PSNR低约3.38dB。** 鱼像素保真改善没有带来当前固定模型的任务收益，不能宣布新视频上的语义保护成功。

在初始片段的4张无鱼GT上，E的鱼类误报为0，H.265为3,440像素；同时E全局PSNR低约4.24dB。**减少无鱼帧误报不能替代有鱼目标的发现与保留**。两段结果应一起保留。

## 2. 来源与样本边界

从[发布者Figshare记录](https://figshare.com/articles/dataset/30105151)中按需读取ZIP条目，并重新核对API元数据。文件名video7.zip有误导性：真实目录为video6和video2/video1，video6属于官方train划分。按阶段C记录，它没有参与本项目的训练、阈值或方法选择；此前训练视频为24/131/91，开发视频96、既有验证108。

因此本轮称为**本项目的新保留视频检验**，不称官方测试集。拍摄级独立性仍未确认；与此前5段实验视频的{data["previous_rgb_hashes_checked"]}个RGB指纹未发现精确重复，也不能据此证明不同拍摄来源或上游预训练数据无重叠。

先在看RGB和模型分数前固定前50帧0—49；其4张发布者掩码均无鱼，仍完整保留这个结果。随后另存补充规则：按真实帧号选择第一个≥50的含鱼GT锚点，首个为60，得到50—99连续帧；没有根据模型、编码质量或增益挑片。它是按标签是否含鱼选出的条件性诊断补充，**不是事先无条件抽取的第二个独立场景**。

RGB与同编号标注图逐像素一致，鱼色按发布者表取RGB(0,0,128)，条目CRC逐个检查；未下载整包，未声称通过完整归档MD5。两段共100张真实连续帧，7张稀疏GT：4张无鱼、3张含鱼，后者源帧60/75/90的鱼像素为199/310/1501。其余93帧不是已确认背景。

输入1280×720，沿用名义25fps；每段名义播放2秒，采集时间戳未确认。模型、0.9阈值、64像素过滤、16px外扩、32帧窗口、左右分区、0.5倍背景CRF42均保持冻结。E的ROI为P-only，普通H.265允许B4离线编码。只按完整文件字节选CRF，不读任务得分选候选；语义消息、所有流和封装均收费。

## 3. 完整文件、收发与画质

| 片段 | 方案 | 完整文件B | 20kbps发送秒 | 接收恢复帧 | 全局RGB PSNR |
|---|---|---:|---:|---:|---:|
"""
    for (label,root,output,_),result in zip(CASES,results):
        for method in ("E","H265"):
            selected = read(root/method/"selection.json")
            net = read(root/method/"network/end_to_end.json")
            row = row_for(result,method)
            text += (f'| {label} | {method} | {selected["file_bytes"]:,} | '
                     f'{net["transfer_seconds"]:.3f} | {net["frames"]} | {fmt(row["global_rgb_psnr_all50_db"])}dB |\n')
    text += """

上述时间只指发送方向的应用层传输，另有传输头字节；不含识别、候选搜索和接收恢复。每段都需要约40秒发送2秒名义播放内容，当前是离线批量原型，不能称实时视频或真实水声链路。

两组分别使用独立接收进程；完整文件SHA256一致，接收端仅依赖收到的包/MP4及FFmpeg，未读原图、GT或模型。评价读取接收端50张实际PNG，逐张核验RGB指纹，未使用展示视频再次压缩后的画面计分。读取限制是Python审计和显式FFmpeg路径检查，不是操作系统沙箱。

![两段实际接收帧的画质与任务指标](figures/holdout_metrics.png)

含鱼片段的三张GT，任务掩码统一采用固定模型原始0.9阈值，未做64像素过滤：

| 条件 | TP | FP | FN | 鱼IoU | 召回率 | 精确率 | GT鱼区PSNR | GT背景PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for method in ("original","E","H265"):
        row = row_for(pos,method); m = row["sparse_gt_metrics"]
        text += (f'| {method} | {m["tp"]} | {m["fp"]} | {m["fn"]} | {pct(m["iou"])} | '
                 f'{pct(m["recall"])} | {pct(m["precision"])} | {fmt(row["gt_fish_rgb_psnr_db"])} | '
                 f'{fmt(row["gt_background_rgb_psnr_fourframes_db"])} |\n')
    text += """

背景PSNR字段沿用评价脚本名称，含鱼片段实际仅汇总其3张GT的非鱼像素；不是50帧已知背景。原图是参考输入，PSNR应视为无穷，表中“不适用”不表示画质为零。

初始片段无正GT，召回率、鱼区PSNR均不适用；E连预测也为空时IoU分母为0，不能记为100%。原图有31个误报像素，E为0，H.265为3,440；初始GT四帧原图经过64像素筛选后均无候选，但窗口ROI仍会由其他帧产生。未标注帧的预测不能统一判为误报。

## 4. 真实目标覆盖与预算

| 真实源帧/GT连通域 | 鱼像素 | E保护覆盖 | 原图模型召回 | E模型召回 | H.265模型召回 | E鱼区PSNR | H.265鱼区PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for component in target["gt_components"]:
        text += (
            f'| {component["source_frame"]}/c{component["component"]} | {component["fish_pixels"]} | '
            f'{pct(component["E_roi_coverage"])} | {pct(component["model_gt_recall"]["original"])} | '
            f'{pct(component["model_gt_recall"]["E"])} | {pct(component["model_gt_recall"]["H265"])} | '
            f'{fmt(component["fish_rgb_psnr_db"]["E"])} | {fmt(component["fish_rgb_psnr_db"]["H265"])} |\n')
    text += """

连通域不是经人工验证的独立鱼实例。覆盖用发布者鱼像素GT计算，与步骤2的人工矩形面积口径不同；进入ROI也不等于一定被识别。

![全部含鱼GT连通域恢复裁剪，第1页](figures/all_positive_target_crops_1.png)

![全部含鱼GT连通域恢复裁剪，第2页](figures/all_positive_target_crops_2.png)

图中统一显示每个GT位置的原图、E和H.265接收帧，不挑选有利目标；提示了GT位置，不是人工盲测或人眼识别率结果。

| 片段 | 平均ROI面积 | 背景流B | ROI流B | 语义消息B | 容器等B |
|---|---:|---:|---:|---:|---:|
"""
    for (label,root,output,_),result in zip(CASES,results):
        selected = read(root/"E/selection.json"); parts = selected["parts"]
        text += (f'| {label} | {pct(result["roi_mean_frame_fraction"])} | {parts["context"]:,} | '
                 f'{parts["roi"]:,} | {parts["semantic"]} | {parts["container"]:,} |\n')
    text += """

初始片段ROI仅2.82%画面，但约100KB包中85,114B用于ROI流。固定背景质量后，增加预算主要靠提升少量候选的质量；小面积本身不能保证总预算分配合理。其46张未标注帧仍可能有鱼，不能将全部ROI字节都称为误报浪费。含鱼补充片段则用真实GT覆盖和任务结果检验保护是否落实。

## 5. 阶段F进度与下一步

步骤1的概率链路定位、步骤2的61框覆盖核对、本轮冻结方案的新视频首轮对照均已取得可复核结果。**这完成了阶段F的首轮诊断与小规模迁移检查，不等于可靠语义保护或跨场景优势已经成立。** 当前仍只有一个新增视频，且来源独立性未确认；7张稀疏GT和一次模型评价不支持普遍结论。

下一步先核对这批失败目标的人工可辨识性，区分“人仍能看清、固定模型失败”和“视频本身也难辨认”。不能把这里的0 IoU直接当成人看不见。若开展人工检查，应保留全部既定样例、固定显示尺度、隐藏编码方法；无提示发现目标与已知位置的轮廓辨认分开记录，本次尚未实施。

之后仅做一种预算分配小对照，优先检查固定低清背景、目标周边上下文及ROI占用的折中，先保持发送端模型与候选规则固定。当前9/10个GT连通域全部进入ROI却仍未识别，说明继续只缩小矩形或只提高候选召回不足以解释全部失败。局部PSNR、融合边界、上下文和0.9阈值稳定性都可能有关，尚未用受控实验区分。每轮同时报告覆盖、鱼区质量、任务、背景质量和无鱼误报，不增加新网络。

video6今后若参与选择新方法，就转为开发数据，新的修复方案还需另留视频检验。本轮没有更换网络、训练模型或修改生产协议；阶段C仍作为后续分支，不阻塞视频通信原型。人工可辨识性盲测及真实水声链路均未开展。

## 6. 视频与证据

"""
    for label,root,output,_ in CASES:
        text += f'- {label}：{link(root/"E/network/receiver/reconstructed.mp4","E接收端观看副本")} · {link(root/"H265/network/receiver/reconstructed.mp4","H.265接收端观看副本")} · {link(output/"新视频实测评价.json","完整评价JSON")} · {link(root/"E/network/receiver/message.scv","实际收到的E通信包")} · {link(root/"H265/network/receiver/video.mp4","实际收到的H.265文件")}\n'
    text += f"""
- {link(OUT/"positive_case/含鱼GT保护与逐连通域核对.json","含鱼GT逐区域数据")}、{link(OUT/"新视频完整文件选择.json","初始片段字节选择")}、{link(OUT/"positive_case/新视频完整文件选择.json","含鱼片段字节选择")}
- {link(OUT/"真实视频选择与划分说明.json","真实目录与划分纠正")}、{link(OUT/"步骤3a冻结实验协议.json","初始冻结协议")}、{link(OUT/"positive_case/步骤3b冻结实验协议.json","含鱼补充冻结协议")}
- {link(OUT/"真实收发核验.json","初始真实收发")}、{link(OUT/"positive_case/真实收发核验.json","含鱼补充真实收发")}
"""
    destination = OUT/"阶段F_步骤3_冻结方案新保留视频对照_2026-10-15.md"
    destination.write_text(text,"utf-8")
    project_copy = re.sub(r"!\[([^\]]*)\]\((figures/[^)]+)\)",
                          lambda m:f"![{m.group(1)}]({(OUT/m.group(2)).as_posix()})",text)
    (PROJECT/"docs/phase_f_step3_holdout_20261015.md").write_text(project_copy,"utf-8")
    return destination


def main():
    FIG.mkdir(exist_ok=True); figures(); path = report()
    print("REPORT_CREATED",str(path),flush=True)


if __name__ == "__main__":
    main()
