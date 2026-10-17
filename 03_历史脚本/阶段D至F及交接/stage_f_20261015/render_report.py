"""Visualize saved F1 probabilities and report a completed diagnostic step."""
from __future__ import annotations
import gzip
import json
import zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from scipy import ndimage
from diagnose import *

SCRIPT=Path(__file__).resolve().parent
COLORS={"gt":"#ff43bb","kept":"#25dbe8","discarded":"#ff9b39","roi":"#62ef86"}

def link(path,title):return f"[{title}]({Path(path).as_posix()})"

def outline(ax,mask):
    if mask.any() and not mask.all():ax.contour(mask.astype(float),levels=[.5],colors=[COLORS["gt"]],linewidths=1.1)

def box(ax,b,color,origin=(0,0),style="-",lw=1):
    x,y,w,h=b
    ax.add_patch(Rectangle((x-origin[0]-.5,y-origin[1]-.5),w,h,fill=False,edgecolor=color,linewidth=lw,linestyle=style))

def view(ax,im,title,gt=None):
    ax.imshow(im,interpolation="nearest");ax.set_title(title,fontsize=10)
    ax.set_xticks([]);ax.set_yticks([])
    if gt is not None:outline(ax,gt)

def render(records):
    figures=OUT/"figures";figures.mkdir(exist_ok=True)
    lookup={(r["frame"],r["condition"]):r for r in records}
    crop_log=[]
    for frame in FRAMES:
        original=lookup[(frame,"original")];received=lookup[(frame,"received_E")]
        gt=np.asarray(Image.open(original["gt_file"]))>0
        rgb=np.asarray(Image.open(original["image"]).convert("RGB"))
        rec=np.asarray(Image.open(received["image"]).convert("RGB"))
        yy,xx=np.where(gt)
        left=max(0,min(1280-192,int((xx.min()+xx.max())/2)-96))
        top=max(0,min(720-160,int((yy.min()+yy.max())/2)-80))
        sl=(slice(top,top+160),slice(left,left+192));local_gt=gt[sl]
        with np.load(original["arrays"]) as data:
            p=data["fish_probability"].copy();raw=data["threshold_mask"].copy();kept=data["filtered_mask"].copy()
        with np.load(received["arrays"]) as data:ep=data["fish_probability"].copy()
        fig,axes=plt.subplots(2,4,figsize=(14.5,7),layout="constrained")
        view(axes[0,0],rgb[sl],"1 Original RGB + GT",local_gt)
        ax=axes[0,1];heat=ax.imshow(p[sl],cmap="magma",vmin=0,vmax=1,interpolation="nearest");outline(ax,local_gt);ax.set_title("2 Original fish probability",fontsize=10)
        view(axes[0,2],np.repeat(raw[sl][...,None],3,2).astype(float),"3 Threshold >= 0.9",local_gt)
        view(axes[0,3],np.repeat(kept[sl][...,None],3,2).astype(float),"4 Components >= 64 pixels",local_gt)
        ax=axes[1,0];view(ax,rgb[sl],"5 Candidates / 16px padding",local_gt)
        nearby=0
        for c in original["raw_components"]:
            x,y,w,h=c["padded_box_16"]
            if x>=left+192 or x+w<=left or y>=top+160 or y+h<=top:continue
            nearby+=1;color=COLORS["kept"] if c["kept"] else COLORS["discarded"]
            box(ax,c["box"],color,(left,top),"-",1.3);box(ax,c["padded_box_16"],color,(left,top),"--",.8)
        if not nearby:ax.text(.03,.05,"No >=0.9 candidate in this view",color="white",fontsize=8,transform=ax.transAxes,bbox={"facecolor":"black","alpha":.65})
        roi=mask_for_boxes(original["actual_stage_e_window_boxes"])
        shade=rgb[sl].astype(float)/255;shade[~roi[sl]]*=.42
        view(axes[1,1],shade,"6 Actual E ROI (dark = outside)",local_gt)
        for b in original["actual_stage_e_window_boxes"]:box(axes[1,1],b,COLORS["roi"],(left,top),"-",1.5)
        view(axes[1,2],rec[sl],"7 Received E RGB + same GT",local_gt)
        ax=axes[1,3];ax.imshow(ep[sl],cmap="magma",vmin=0,vmax=1,interpolation="nearest");outline(ax,local_gt);ax.set_title("8 Received E fish probability",fontsize=10)
        for ax in axes.ravel():
            ax.set_xlim(-.5,191.5);ax.set_ylim(159.5,-.5);ax.set_xticks([]);ax.set_yticks([])
        cb=fig.colorbar(heat,ax=[axes[0,1],axes[1,3]],shrink=.8,ticks=[0,.25,.5,.75,.9,1]);cb.set_label("Fish probability (fixed 0..1)")
        fig.suptitle(f"Source frame {frame} | fixed 192x160 view | magenta = publisher GT\nFrozen model; threshold 0.9; min 64 pixels; padding 16; no retuning",fontsize=12)
        fig.savefig(figures/f"frame_{frame}_pipeline.png",dpi=150)
        plt.close(fig)
        crop_log.append({"frame":frame,"view_box":[left,top,192,160],"same_rgb_scale":True,"probability_color_scale":[0,1],"gt_is_hint_for_diagnosis_not_blind_review":True})
    write_json(figures/"crop_coordinates.json",crop_log)

    fig,axes=plt.subplots(3,4,figsize=(15,8.8),layout="constrained")
    titles=["Original RGB + GT boxes","Original probability (0..1)","Raw candidates / padded boxes","Actual E window ROI"]
    for row,frame in enumerate(FRAMES):
        r=lookup[(frame,"original")];rgb=np.asarray(Image.open(r["image"]).convert("RGB"));gt=np.asarray(Image.open(r["gt_file"]))>0
        with np.load(r["arrays"]) as data:p=data["fish_probability"].copy()
        axes[row,0].imshow(rgb);axes[row,1].imshow(p,cmap="magma",vmin=0,vmax=1)
        axes[row,2].imshow(rgb);axes[row,3].imshow(rgb)
        for ax in axes[row]:
            for target in r["targets"]:box(ax,target["box"],COLORS["gt"],lw=1.4)
            ax.set_xlim(-.5,1279.5);ax.set_ylim(719.5,-.5);ax.set_xticks([]);ax.set_yticks([])
        for c in r["raw_components"]:
            color=COLORS["kept"] if c["kept"] else COLORS["discarded"]
            box(axes[row,2],c["box"],color,lw=1.2);box(axes[row,2],c["padded_box_16"],color,style="--",lw=.7)
        kept_sizes=[c["pixels"] for c in r["raw_components"] if c["kept"]]
        dropped_sizes=[c["pixels"] for c in r["raw_components"] if not c["kept"]]
        axes[row,2].text(.02,.97,f"Kept: {kept_sizes or 'none'} px\nDiscarded: {dropped_sizes or 'none'} px",ha="left",va="top",transform=axes[row,2].transAxes,color="white",fontsize=8,bbox={"facecolor":"black","alpha":.6,"pad":3})
        for b in r["actual_stage_e_window_boxes"]:box(axes[row,3],b,COLORS["roi"],lw=1.5)
        axes[row,0].set_ylabel(str(frame),fontsize=12)
    for i,title in enumerate(titles):axes[0,i].set_title(title,fontsize=10)
    fig.suptitle("Full-frame source diagnosis | magenta GT; cyan kept; orange discarded; green final ROI",fontsize=12)
    fig.savefig(figures/"full_frame_overview.png",dpi=145);plt.close(fig)

def main():
    records=read(OUT/"概率与逐步骤诊断.json");protocol=read(OUT/"步骤1冻结协议.json")
    assert len(records)==6
    for path,expected in protocol["code_sha256"].items():assert sha256(path)==expected
    assert sha256(protocol["model"]["checkpoint"])==protocol["model"]["checkpoint_sha256"]
    for r in records:assert sha256(r["arrays"])==r["arrays_sha256"]
    render(records)
    lookup={(r["frame"],r["condition"]):r for r in records}
    table=[]
    for frame in FRAMES:
        raw=lookup[(frame,"original")];rec=lookup[(frame,"received_E")]
        for target,er in zip(raw["targets"],rec["targets"]):
            p,q=target["probability"],er["probability"]
            table.append(f"| {frame} | {target['pixels']} | {p['mean']:.4f} | {p['max']:.4f} | {q['mean']:.4f} | {q['max']:.4f} | {p['pixels_at_threshold']}→{q['pixels_at_threshold']} | {target['actual_stage_e_window_roi_coverage']*100:.0f}% |")
    c435=lookup[(435,"original")];c450=lookup[(450,"original")]
    assert all(t["window_if_this_frame_rejected_components_restored"]==0 for r in (c435,c450) for t in r["targets"])
    with zipfile.ZipFile(OLD/"receiver/message.scv") as z:semantics=json.loads(gzip.decompress(z.read("semantics.json.gz")))
    small_gt=np.asarray(Image.open(lookup[(465,"original")]["gt_file"]))>0
    cc,_=ndimage.label(small_gt,structure=np.ones((3,3),np.uint8));small_gt=cc==1
    contributors=[]
    for r in semantics["frames"][32:50]:
        for region in r["regions"]:
            padded=roi_box([{"regions":[region]}],1280,720,16)
            hits=int((mask_for_boxes([padded])&small_gt).sum())
            if hits:contributors.append({"source_frame":r["frame"]+420,"region_box":region["box"],"padded_box":padded,"overlap_with_fixed_frame465_small_gt":hits,"note":"geometric evidence across frames, not a verified tracking identity"})
    write_json(OUT/"465顶部目标的窗口几何来源.json",contributors)
    report=OUT/"阶段F_步骤1_漏保护与概率诊断_2026-10-15.md"
    report.write_text(f"""# 阶段F · 步骤1：漏保护与概率诊断

日期：2026-10-15。按用户“可以分步推进”的要求，本轮只完成第一步。阶段F尚未整体完成；未进行61个人工框的完整逐框分析、新视频测试、盲评或算法调整。

## 1. 主要发现

**435、450帧在0.9阈值阶段就没有产生GT鱼像素候选；本帧64像素过滤并未删掉能保护这两条GT鱼的候选。**

- 435帧GT鱼像素最高鱼概率0.6871，平均0.2453，均未产生≥0.9的GT命中像素。
- 450帧GT鱼像素最高概率0.5714，平均0.2248，同样没有GT命中像素。
- 两帧被删除的小连通域离GT目标较远；将本帧所有被删候选按原16像素外扩补回原32帧窗口，GT覆盖仍为0。它们不是“刚检测到鱼，随后被64像素规则直接删掉”的情况。

这不表示模型完全没有响应，也不表示0.9阈值在所有场景都错误。**定位到哪一步失败，不等于已经解释了概率为何偏低。** 跨域、类别混淆、边缘/截断、模型能力与置信度决策的影响尚未区分。

**465帧184像素小目标的情况也已更清楚：** 它在原图和E恢复图上都没有超过0.9，但平均概率由0.3625升至0.7123，最高由0.7939升至0.8566。分割召回仍为0，掩盖了连续概率响应的改善；不能将其描述为“保护后毫无响应”或据此认定压缩损坏了识别特征。

## 2. 本轮锁定条件与核验

只推理三个原始帧及三个阶段E实际恢复帧，共6次。模型、512分块/384步长、阈值0.9、64像素筛选、16像素外扩及E的32帧双区域规则全部保持不变。发送端及接收端程序、既有实际码流和标注均未改动，没有训练和阈值搜索。

模型SHA256：`{protocol['model']['checkpoint_sha256']}`。原始RGB、E接收RGB和GT文件指纹核验通过；六次阈值掩码的TP/FP/FN均精确复现既有报告，三个原图的筛选后目标框也与既有发送记录一致。完整float32概率图与掩码保存在NPZ中，可复核数字和图像。

## 3. 四个GT连通域的连续概率

下表只在对应GT像素内统计概率，避免将附近大鱼或背景高概率误当作小鱼响应。≥0.9列是像素数量，不是物体数量。

| 原始帧 | GT像素 | 原图均值 | 原图最大 | E恢复均值 | E恢复最大 | ≥0.9像素：原图→E | 实际E ROI覆盖 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

465帧801像素连通域从471个GT命中像素增加到799个，召回58.80%→99.75%。184像素小连通域仍为0个命中像素。这是同一固定模型在原图/恢复图上的响应，不等同于人眼可辨识性结论。

435帧恢复后的GT均值约0.00710、最大0.03088；450帧均值约0.000408、最大0.001522。未被ROI保护的这两个目标在E输入上的模型响应进一步下降。这里没有隔离局部像素与全局上下文各自的作用，不能声称已经证明唯一原因。

## 4. 64像素规则究竟删了什么

| 帧 | 原图≥0.9像素 | 原图连通域 | 64像素筛选结果 | 与本次GT鱼的位置关系 |
|---|---:|---|---|---|
| 435 | 17 | 6像素、11像素两块 | 两块均删除 | 位于右上，最短GT像素距离约220像素；16像素外扩均不覆盖鱼 |
| 450 | 190 | 142像素一块、48像素一块 | 保留142，删除48 | 保留块在上方；删除块在左下，与右中GT鱼相距约1253像素；均不覆盖该GT鱼 |
| 465 | 481 | 一块481像素区域 | 保留 | 对应较大鱼；本帧外扩可覆盖801像素GT，但不能覆盖顶部184像素GT |

450帧上方的142像素候选不在当前185像素GT内，但该处也涉及用户新增人工框。此处只分析它与当前GT的位置关系，不据此认定它一定是误检，也不改动原像素GT。

几何反事实只做一次固定检查：保留本帧全部0.9候选、包括小于64像素者，替换该帧候选列表，其他49帧保持原记录，重新生成同一个32帧双区域窗口。435和450的GT覆盖仍为0；**没有重新编码或测量此反事实的字节、质量及任务收益。**

结论范围仅限这两帧的本帧候选与该反事实；本轮未重新提取窗口内其余所有帧的小连通域，不能排除其他帧被删候选的潜在影响。

## 5. 为什么465顶部小鱼进入了ROI

465本帧的保留框外扩不覆盖顶部184像素GT，而实际32帧窗口ROI覆盖100%。窗口还合并了其他帧的候选，包括466等之后帧向画面顶部延伸的区域。

这是跨帧区域合并产生的几何保护，不能说465帧自身已检出这条小鱼。{link(OUT/'465顶部目标的窗口几何来源.json','窗口来源记录')}保留了与465固定GT位置相交的邻帧候选；它不是经过身份核验的目标跟踪结果。当前系统按批次处理，窗口可使用后续帧，不能当作因果实时保护。

## 6. 诊断图

所有图使用冻结输出，没有美化重建。概率色标固定为0–1；洋红为发布者GT，青色为保留候选，橙色为删除候选，绿色为实际窗口ROI。局部视野固定192×160，按原像素最近邻显示，位置来源和坐标另存。图中GT提示了目标位置，**这些图用于诊断，不是盲评结果。**

全图总览可检查候选实际在哪里，局部图依次显示：原图、原始概率、0.9掩码、64像素筛选、候选及外扩、实际E保护范围、接收RGB、接收后鱼概率。图中后两项属于接收后评价，没有反过来参与发送端ROI生成。

![全帧概率、候选与实际ROI](figures/full_frame_overview.png)

### 435帧

![435帧逐步骤诊断](figures/frame_435_pipeline.png)

### 450帧

![450帧逐步骤诊断](figures/frame_450_pipeline.png)

### 465帧

![465帧逐步骤诊断](figures/frame_465_pipeline.png)

## 7. 本步判断与后续安排

第一步已经把“原图没生成合格候选”“64像素过滤”“窗口顺带覆盖”“接收后概率仍低于决策阈值”分开了。现有证据不支持单独取消64像素过滤来修好435/450，也不支持凭三帧把阈值改成某个更低数值。

**本轮到此结束，不在诊断后立即调参。** 下一小步应复用现有61个人工框，把这类漏保护在50帧内按目标大小和边缘位置量化，并保留12个人工确认无鱼帧用于检查误检和区域成本，再决定是否设计更谨慎的候选保留规则。不能仅凭三张正样本选一个新阈值。冻结E的新独立视频检查仍保留，不能等调好video131后再拿它冒充新测试。

人工可辨识性检查、阈值校准、分割器适配及阶段C训练都尚未开展。阶段F状态为进行中，只有步骤1诊断完成。

## 8. 数据与复现入口

- {link(OUT/'步骤1冻结协议.json','本步冻结协议')}；{link(OUT/'步骤1数据核验.json','数据与复现核验')}。
- {link(OUT/'概率与逐步骤诊断.json','每帧、每连通域及候选的完整数值')}；{link(RUN/'probabilities','六份原始概率与掩码NPZ')}。
- {link(SCRIPT/'diagnose.py','诊断脚本')}；{link(SCRIPT/'render_report.py','作图与报告脚本')}；{link(SCRIPT/'diagnose.log','推理记录')}。
- {link(WORKSPACE/'阶段E后_联合意见与下一步共识_2026-10-15.md','本轮依据的联合共识')}。
""",encoding="utf-8")
    project_text=report.read_text("utf-8").replace("(figures/","("+(OUT/"figures").as_posix()+"/")
    (PROJECT/"docs/phase_f_step1_diagnosis_20261015.md").write_text(project_text,"utf-8")
    delivery=read(OUT/"步骤1数据核验.json")
    delivery.update({"step1":"completed","report_generated":True,"diagnostic_figures_generated":4})
    write_json(OUT/"步骤1数据核验.json",delivery)
    state={"stage":"F","status":"in_progress","completed_steps":["step1 frozen probability and ROI diagnosis"],"remaining":["61-box object-level coverage","frozen-E independent-video validation"],"human_blind_review":"not started","model_training":False,"algorithm_or_threshold_changed":False,"report":str(report)}
    write_json(OUT.parent/"阶段F进度.json",state)
    archive=OUT/"阶段F_步骤1_报告图表与数值.zip"
    entries=[(report,report.name)]
    entries.extend((p,p.name) for p in OUT.glob("*.json") if p.name!="证据包指纹.json")
    entries.extend((p,"figures/"+p.name) for p in (OUT/"figures").iterdir() if p.is_file())
    entries.extend((p,"code/"+p.name) for p in SCRIPT.glob("*.py"))
    entries.append((SCRIPT/"diagnose.log","logs/diagnose.log"))
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("阅读说明.txt","这是阶段F步骤1的冻结模型诊断，不是算法修改或新视频测试。完整概率NPZ留在报告指向的本机运行目录，未重复打包模型和原始视频。所有GT提示图均为非盲诊断。\n")
        for path,name in entries:z.write(path,name)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    write_json(OUT/"证据包指纹.json",{"file":str(archive),"bytes":archive.stat().st_size,"sha256":sha256(archive),"crc_valid":True})
    print(json.dumps({"report":str(report),"archive":str(archive),"stage_f":"in_progress","step1":"completed"},ensure_ascii=False))

if __name__=="__main__":main()
