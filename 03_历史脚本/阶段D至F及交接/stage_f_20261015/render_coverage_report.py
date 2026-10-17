"""Render the bounded F2 audit, its full 61-row appendix, and verified figures."""
from __future__ import annotations
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, Patch
import numpy as np
from PIL import Image
from coverage_audit import WORKSPACE, PROJECT, OUT, OLD, MANUAL, read, write, mask

FIG = OUT / "figures"
COLORS = {"full95": "#26856a", "partial": "#e1a335", "missed": "#c94d4d", "empty": "#e6e9ed"}
SUMMARY = read(OUT/"覆盖统计汇总.json")
BOXES = read(OUT/"逐框覆盖明细.json")
FRAMES = read(OUT/"逐帧覆盖与无鱼开销.json")
STREAMS = read(OUT/"码流与人工框空间对应.json")


def pct(v):
    return f"{v*100:.2f}%"


def link(path, title):
    return f"[{title}]({path.as_posix()})"


def plot_overview():
    matrix = np.full((2, 50), np.nan)
    for row in BOXES:
        matrix[row["box_ordinal"]-1, row["local_frame"]] = (
            0 if row["E"]["status"] == "missed" else 1 if row["E"]["status"] == "partial" else 2)
    fig = plt.figure(figsize=(13.2, 7.0), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[.65, 1])
    ax = fig.add_subplot(grid[0, :])
    cmap = ListedColormap([COLORS["missed"], COLORS["partial"], COLORS["full95"]])
    cmap.set_bad(COLORS["empty"])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect="auto", interpolation="nearest",
              extent=[419.5, 469.5, 2.5, .5])
    ax.set_yticks([1, 2], ["Box #1", "Box #2"])
    ax.set_xticks([420, 425, 430, 435, 440, 445, 450, 452, 455, 460, 465, 469])
    ax.axvline(451.5, color="#202b38", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Real source frame; box order is NOT a track identity")
    ax.set_title("Frozen E: 39 full / 2 partial / 20 missed out of 61 frame-box observations", loc="left")
    ax.legend(handles=[Patch(color=COLORS[k], label=t) for k, t in [
        ("full95", "Full: >=95% box area"), ("partial", "Partial: >0, <95%"),
        ("missed", "Missed: 0%"), ("empty", "No annotated box")]],
        loc="center left", bbox_to_anchor=(.005, .5), ncol=1, fontsize=8)
    ax = fig.add_subplot(grid[1, 0])
    groups = SUMMARY["methods"]["E"]["size_groups"]
    keys = ["le512", "513to2048", "gt2048"]
    labels = ["<=512 px\n(n=5)", "513-2048 px\n(n=32)", ">2048 px\n(n=24)"]
    bottom = np.zeros(3)
    for status, title in [("full95", "Full"), ("partial", "Partial"), ("missed", "Missed")]:
        values = np.array([groups[k][status]/groups[k]["boxes"]*100 for k in keys])
        ax.bar(labels, values, bottom=bottom, color=COLORS[status], label=title, width=.65)
        for i, value in enumerate(values):
            if value:
                ax.text(i, bottom[i]+value/2, str(groups[keys[i]][status]),
                        ha="center", va="center", color="white", fontweight="bold")
        bottom += values
    ax.set_ylim(0, 106)
    ax.set_ylabel("Fraction of box observations (%)")
    ax.set_title("Small rectangles are poorly covered\nSize means BBOX area, not fish mask pixels", loc="left", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.19), ncol=3, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(grid[1, 1])
    e_streams = [s for s in STREAMS if s["method"] == "E"]
    ax.barh(np.arange(3), [s["bytes"]/1000 for s in e_streams],
            color=["#c94d4d", "#527caa", "#26856a"])
    ax.set_yticks(np.arange(3), ["420-451 left\nno reviewed target overlap",
                               "420-451 right", "452-469 right"])
    ax.invert_yaxis()
    ax.set_xlim(0, 49)
    for i, stream in enumerate(e_streams):
        ax.text(stream["bytes"]/1000+.5, i, f'{stream["bytes"]:,} B', va="center", fontsize=9)
    ax.set_xlabel("Existing MP4 stream size (decimal KB)")
    ax.set_title("Where E's ROI bytes went\nLeft strip: 52.47% of ROI stream bytes", loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Stage F / step 2: frozen video131 geometry audit (no new encode)", fontsize=13)
    fig.savefig(FIG/"coverage_overview.png", dpi=155)
    plt.close(fig)


def outlined(ax, b, color, origin=(0, 0), label=None, linestyle="-", lw=1.5):
    x, y, w, h = b
    x -= origin[0]; y -= origin[1]
    ax.add_patch(Rectangle((x-.5, y-.5), w, h, fill=False,
                           edgecolor=color, linewidth=lw, linestyle=linestyle))
    if label is not None:
        ax.text(x, max(1, y-3), label, color="white", fontsize=7,
                va="bottom", bbox={"facecolor": "#182130", "alpha": .85, "pad": 1})


def overlay(ax, image, frame, method, crop=None):
    origin = (crop[0], crop[1]) if crop else (0, 0)
    roi = mask(frame[method]["boxes"])
    if crop:
        x, y, w, h = crop
        visible, rm = image[y:y+h, x:x+w], roi[y:y+h, x:x+w]
    else:
        visible, rm = image, roi
    ax.imshow(visible, interpolation="nearest")
    tint = np.zeros((*rm.shape, 4), dtype=np.float32)
    tint[rm] = [.12, .82, .58, .18]
    ax.imshow(tint, interpolation="nearest")
    for b in frame[method]["boxes"]:
        outlined(ax, b, "#45e7ab", origin, linestyle="--", lw=1)
    for row in [r for r in BOXES if r["source_frame"] == frame["source_frame"]]:
        outlined(ax, row["box"], COLORS[row[method]["status"]], origin,
                 f'#{row["box_ordinal"]} {row[method]["coverage"]*100:.1f}%')
    ax.set_xlim(-.5, visible.shape[1]-.5)
    ax.set_ylim(visible.shape[0]-.5, -.5)
    ax.set_xticks([]); ax.set_yticks([])


def plot_geometry():
    examples = [435, 450, 452]
    crop = (1040, 0, 240, 340)
    fig = plt.figure(figsize=(13.0, 10.0), layout="constrained")
    grid = fig.add_gridspec(3, 3, width_ratios=[2.5, 1, 1])
    rgb_audit = read(OLD/"sender/source_audit.json")
    import hashlib
    for index, number in enumerate(examples):
        frame = FRAMES[number-420]
        image = np.asarray(Image.open(OLD/f"sender/private_frames/{number-420:05d}.png").convert("RGB"))
        assert hashlib.sha256(image.tobytes()).hexdigest() == rgb_audit["rgb_sha256"][number-420]
        ax = fig.add_subplot(grid[index, 0])
        overlay(ax, image, frame, "E")
        ax.set_title(f"Frame {number}: source image + E rectangles", fontsize=10)
        outlined(ax, list(crop), "#f5e5b4", lw=1)
        ax = fig.add_subplot(grid[index, 1])
        overlay(ax, image, frame, "A", crop)
        ax.set_title(f"{number}: A crop overlay", fontsize=10)
        ax = fig.add_subplot(grid[index, 2])
        overlay(ax, image, frame, "E", crop)
        ax.set_title(f"{number}: E crop overlay", fontsize=10)
    fig.suptitle("Original RGB with ROI overlays: geometry only, NOT a reconstructed-quality comparison\n"
                 "Green tint = ROI; box colors = coverage; same 240 x 340 source crop in A/E columns",
                 fontsize=11)
    fig.savefig(FIG/"spatial_coverage_examples.png", dpi=160)
    plt.close(fig)
    write(OUT/"图示坐标与来源.json", {
        "source_frames": examples, "source_crop_xywh": list(crop),
        "images": "original RGB, not A/E decoded RGB",
        "purpose": "annotated geometric diagnosis with known locations; not blinded readability test",
        "examples_selected_before_render": "435,450 previous failures;452 immediate next window boundary",
    })


def appendix():
    rows = ["# 阶段F步骤2：61个逐帧人工框覆盖表", "",
        "日期：2026-10-15。框序号只在当前帧内有效，不是跟踪ID；矩形内像素不等于鱼像素。"
        "达标：覆盖框面积≥95%；完全漏保护：覆盖为0；其余为部分覆盖。A/E均沿用已实际接收的通信包。",
        "", "| 框ID | 来源状态 | xywh | 框面积 | 距边缘px | A覆盖 | E覆盖 | E状态 | 同帧候选扩框覆盖* |",
        "|---|---|---|---:|---:|---:|---:|---|---:|"]
    words = {"full95": "达标", "partial": "部分", "missed": "完全漏保护"}
    for r in BOXES:
        rows.append(
            f'| {r["id"]} | {r["annotation_status"]} | {",".join(map(str,r["box"]))} | '
            f'{r["bbox_area"]} | {r["minimum_edge_gap"]} | {pct(r["A"]["coverage"])} | '
            f'{pct(r["E"]["coverage"])} | {words[r["E"]["status"]]} | '
            f'{pct(r["same_frame_E_geometry"]["coverage"])} |')
    rows += ["", "* 同帧候选扩框仅将该帧现有保留预测送入同样的左右分区及16px外扩，"
             "不利用窗口其他帧；未重新编码，不能视为单帧通信实验或检测召回率。", ""]
    (OUT/"逐框覆盖表.md").write_text("\n".join(rows), "utf-8")
    misses = ["# 阶段F步骤2：漏保护与部分覆盖清单", "",
        "这是已核对矩形的几何清单，不是新增像素GT或独立鱼数量。", "",
        "| 源帧 | 框序号 | xywh | 框面积 | E覆盖 |", "|---:|---:|---|---:|---:|"]
    for r in BOXES:
        if r["E"]["status"] != "full95":
            misses.append(
                f'| {r["source_frame"]} | {r["box_ordinal"]} | {",".join(map(str,r["box"]))} | '
                f'{r["bbox_area"]} | {pct(r["E"]["coverage"])} |')
    misses += ["", "完全漏保护共20个，均为432—451帧中的第1框；部分覆盖为450、451帧的第2框。",
        "上述框ID不代表跟踪身份，不能据此断言是20条不同的鱼。", ""]
    (OUT/"漏保护清单.md").write_text("\n".join(misses), "utf-8")


def write_report():
    a, e = (SUMMARY["methods"][k] for k in ("A", "E"))
    e_left = next(s for s in STREAMS if s["method"] == "E" and s["partition"] == "left")
    partial = [r for r in BOXES if r["E"]["status"] == "partial"]
    assert [r["id"] for r in partial] == ["00450-2", "00451-2"]
    assert e["all_boxes_full95_frames"] == 18
    negative = [f for f in FRAMES if f["no_fish"]]
    negative_candidates = [f["source_frame"] for f in negative if f["predicted_region_count"]]
    negative_without = [f["source_frame"] for f in negative if not f["predicted_region_count"]]
    report = f"""# 阶段F · 步骤2：61个人工框覆盖与ROI开销核对

日期：2026-10-15。项目目标仍是**接收端恢复可观看视频，语义负责保护关键目标**。本次完成阶段F第二步；独立新视频检验尚未开展，阶段F仍在进行中。

## 1. 本次结论

阶段E的空间效率改善成立，但目标保护覆盖仍不足：**61个逐帧框中，39个达标、2个部分覆盖、20个完全漏保护；38张含目标帧中，只有18张的所有框都达标。** 最小一组5个框有4个完全漏保护。

另一个直接影响预算的发现是：E第一窗口的左侧ROI条带占**40,102B，即ROI视频字节的52.47%、完整通信包的39.79%**，在其全部32帧中均没有与已核对的人工框相交。以当前人工核对为依据，误检位置仍在消耗大量目标保护预算。

这不能证明删掉条带后会等质量节省39.79%，也不能把框进入ROI直接当成目标已清晰恢复。下一步仍先让冻结的E接受一次新视频检验，再选择一个因素做修复。

## 2. 数据与口径

沿用video131真实连续420—469帧、1280×720输入，以及既有A/E通信包中**已经由独立接收端解码的ROI计划**。A为32帧单大框，E为相同预测、相同32帧窗口的左右双区域。本次无新推理、编码、收发、训练或阈值调整。

人工核对快照：50帧，38帧有框、12帧标为无鱼，共61个逐帧框；47帧状态为human_reviewed、3帧保留publisher_verified。用户此前声明已经完成核对，系统据此保存坐标和确认状态；本次未重新开展独立人工复核。框不是鱼像素GT，也没有跟踪身份，不能把61个框当成61条独立鱼。450帧上方新增的人工框保留参与几何统计，未写入发布者像素GT，不能因为该处未获像素标注就将候选直接判为误检。

- 覆盖率＝ROI与当前框交集面积÷框面积。≥95%为达标，0为完全漏保护，其余为部分覆盖。含目标帧只有在**每个框都达标**时才记成功。框内含背景，部分框面积被裁掉不等于相同比例的鱼像素被裁掉。
- 预先固定框面积分组为≤512、513—2048、>2048px；这是**矩形面积**，与此前184/255等鱼像素数不是一个口径。实际框面积242—5325px。
- 边缘框定义为距任一画面边界≤16px；另列真正接触边界的框。大小、位置和时间相互关联，分组仅作描述，不推断因果或显著性。
- 采用精确整数判定95%，并将每个框的解析交集与逐像素矩形掩码交叉核验。输入快照、预测载荷、代码及模型指纹不变；A/E计划复现且包指纹与既有接收端记录一致。

冻结协议见{link(OUT/"步骤2冻结协议.json", "步骤2冻结协议")}；人工快照SHA256为{SUMMARY["annotation_sha256"]}。

## 3. 逐框结果：面积平均值掩盖了部分遗漏

| 指标 | A：预测＋32帧单大框 | E：预测＋32帧双区域 |
|---|---:|---:|
| 框覆盖≥95% | {a["full95"]}/61（{pct(a["full95_fraction"])}） | {e["full95"]}/61（{pct(e["full95_fraction"])}） |
| 部分覆盖 | 0 | 2 |
| 完全漏保护 | 0 | 20 |
| 所有框均达标的含目标帧 | 38/38 | 18/38（47.37%） |
| 至少有一个框与ROI相交的帧 | 38/38 | 34/38 |
| 逐框覆盖率等权均值 | 100.00% | {pct(e["mean_box_coverage"])} |
| 此前逐帧框并集覆盖率均值 | 100.00% | 78.69% |
| 平均ROI画面覆盖面积 | {pct(a["average_roi_area"])} | {pct(e["average_roi_area"])} |
| 既有约100KB完整包 | 99,497B | 100,788B |

此前78.69%先按每帧框并集面积计算、再对帧取平均，较大的框会在帧内占更多权重；本次每个框等权后仅66.87%。这两个数的权重不同，**本身不是新旧系统性能变化**。本片段“框并集≥95%”和“所有框各自≥95%”恰好都为18/38，一般情况下两种判定并不等价。

完全漏保护集中于**432—451帧的第1框**。这些框位于画面右侧中部，而E第一窗口的右ROI只有[1166,0,114,88]，没有覆盖其纵向位置。450、451帧第2框分别只覆盖{pct(partial[0]["E"]["coverage"])}和{pct(partial[1]["E"]["coverage"])}，还存在上方目标横向被裁掉一部分的现象。

从452帧开始切到下一窗口，右ROI变为[1058,0,222,330]，452—469帧的所有已核对框都达标。这是固定窗口与不同时间目标位置共同形成的几何现象，不能当成识别器在452帧突然变好。

只使用同帧现有预测和相同扩框规则时，61个框中仅12个达标；实际32帧窗口有39个达标，其中27个依赖窗口聚合才跨过95%线。**窗口覆盖不能等同于检测成功**，也未证明这些跨帧候选属于同一条鱼。窗口计划可使用后续帧；当前是整段离线处理与收发，不能将此解释为实时跟踪效果。该补充仅为几何运算，没有实施新的短窗口编码。

![61个框覆盖、大小分组及真实ROI子码流字节](figures/coverage_overview.png)

| 框面积组 | 框数 | E达标 | E部分 | E完全漏保护 |
|---|---:|---:|---:|---:|
"""
    for key, label in [("le512", "≤512px"), ("513to2048", "513—2048px"), ("gt2048", ">2048px")]:
        g = e["size_groups"][key]
        report += f'| {label} | {g["boxes"]} | {g["full95"]}（{pct(g["full95_fraction"])}） | {g["partial"]} | {g["missed"]} |\n'
    report += f"""
≤16px边缘组有51个框，29个达标、2个部分覆盖、20个漏保护；内部组10/10达标。真正接触边界的13个框中8个达标、5个漏保护。**全部最小框都在边缘组，不能分离“小”和“边缘”各自的作用**；同一连续片段的重复观测也不支持显著性结论。

![435、450及452帧原图上的几何核对](figures/spatial_coverage_examples.png)

图中使用原始RGB、提示目标位置并叠加ROI，仅用于检查几何；不是接收端画质对比，也不是人工盲测。435/450显示漏保护，452显示窗口切换后的覆盖。完整记录见{link(OUT/"逐框覆盖表.md", "61框逐项表")}和{link(OUT/"漏保护清单.md", "22个未达标框清单")}。

## 4. 无鱼帧与子码流：预算确实存在偏配

人工/发布者确认无鱼的12帧为420—431；其中{len(negative_candidates)}/12帧仍有保留预测，共13个连通域、1,747个预测掩码像素。具体有预测帧：{", ".join(map(str,negative_candidates))}。{", ".join(map(str,negative_without))}帧自身无保留候选，最终仍被时间窗口ROI覆盖。这区分了**当前帧误报**与**窗口持续保护**。

| 无鱼帧上的几何量 | A | E |
|---|---:|---:|
| 仍有最终ROI的帧 | 12/12 | 12/12 |
| 这些帧平均ROI面积占画面 | {pct(a["negative_mean_roi_fraction"])} | {pct(e["negative_mean_roi_fraction"])} |
| 占全部ROI“像素×帧”的比例 | {pct(a["negative_share_of_all_roi_pixel_frames"])} | {pct(e["negative_share_of_all_roi_pixel_frames"])} |

这里的面积时间占比**不是字节占比**。窗口内帧共用压缩参考，不能按面积给单帧分摊码量。可精确读取的是既有完整子码流文件：

| E的ROI码流 | 源帧 | 区域xywh | 实际MP4字节 | ROI字节占比 | 与人工框关系 |
|---|---|---|---:|---:|---|
"""
    for stream in [s for s in STREAMS if s["method"] == "E"]:
        relation = "全部活动帧均不相交" if stream["zero_overlap_with_reviewed_boxes_all_active_frames"] else "包含部分已核对目标框区域"
        report += (
            f'| {stream["file"]} | {stream["source_frames"][0]}—{stream["source_frames"][1]} | '
            f'{stream["box"]} | {stream["bytes"]:,} | {pct(stream["share_of_roi_stream_bytes"])} | {relation} |\n')
    report += f"""
E的左条带由15个预测区域撑出，活动32帧中没有人工目标框与之相交，却占{pct(e_left["share_of_roi_stream_bytes"])}的ROI视频字节；这比单看ROI平均面积更直接地显示预算偏配。背景流21,056B、语义消息929B、容器等2,370B也已计入100,788B完整包。

本结果以已核对框完整性为前提；该条带仍可能提供背景上下文。**没有做删流重编码或同预算重分配，尚不能承诺删除后画质不变或恢复质量必然上升。** 尤其不能推广成“删除左边/边缘框”的规则，真实目标也可能出现在这些位置。

完整5条A/E子流空间对应记录见{link(OUT/"码流与人工框空间对应.json", "子流空间与字节核对")}；该补充对全部子流统一计算，未只挑出左条带。

## 5. 合并步骤1后的判断及下一步

步骤1已发现435/450原图GT位置最大鱼概率分别约0.687/0.571，未达到0.9；恢复这两帧自身被64像素过滤的小连通域也不能覆盖它们。步骤2进一步表明：E没有丢掉已有35个预测区域，但**保留全部预测不代表保护了真实目标**。旧A覆盖全部人工框主要受益于大面积几何兜底，不能因此认定其检测更准。

465帧184像素目标虽然被保护，原图和恢复图的0.9阈值识别仍都失败。目标发现、ROI几何覆盖、接收重建质量、下游识别须继续分开评价；这次增加的61框统计只回答中间的几何覆盖。

下一步按原共识推进一小步：**保持模型、0.9阈值、64像素过滤和双区域实现冻结，选一段未参与训练/方法选择的新连续视频，与同完整文件预算H.265做实际收发比较。** 首先核对来源和原图识别能力，所有画质与任务评价使用独立接收端实际RGB；没有像素GT时不报真实分割IoU。若现有小样本都曾用于训练或方法选择，就先补一个小的外部片段，不把已见样本换名当新测试。

完成这次冻结检验后，再在开发数据上单独比较一种改动：提高真目标候选召回，或抑制持续误检的预算占用，并同时记录小框覆盖和无鱼帧开销。当前证据不支持直接降低全局阈值，也不支持跳回大矩形来宣称问题已解决。阶段C多帧微调仍保留为后续研究分支，不阻塞视频原型。

本次没有新视频效果结论，没有人工可辨识性盲测，也没有跨场景优势结论。阶段F已完成步骤1和2，步骤3待进行。

## 6. 可复核材料

- {link(OUT/"逐框覆盖明细.json", "61框机器可读明细")}、{link(OUT/"逐帧覆盖与无鱼开销.json", "50帧逐项数据")}、{link(OUT/"覆盖统计汇总.json", "统计汇总")}
- {link(OUT/"步骤2数据核验.json", "输入与计算核验")}、{link(OUT/"码流空间对应补充约定.json", "子流补充约定")}、{link(MANUAL, "冻结人工框快照")}
- {link(WORKSPACE/"outputs/semantic_roi_phase_f/step1_probability_diagnosis/阶段F_步骤1_漏保护与概率诊断_2026-10-15.md", "步骤1概率与漏保护诊断")}
- {link(WORKSPACE/"阶段E后_联合意见与下一步共识_2026-10-15.md", "已确定的联合共识")}
"""
    report_path = OUT/"阶段F_步骤2_人工框覆盖与ROI开销核对_2026-10-15.md"
    report_path.write_text(report, "utf-8")
    project_report = PROJECT/"docs/phase_f_step2_coverage_20261015.md"
    project_copy = re.sub(
        r"!\[([^\]]*)\]\((figures/[^)]+)\)",
        lambda m: f"![{m.group(1)}]({(OUT/m.group(2)).as_posix()})",
        report,
    )
    project_report.write_text(project_copy, "utf-8")
    return report_path, project_report


def main():
    FIG.mkdir(exist_ok=True)
    plot_overview()
    plot_geometry()
    appendix()
    report_path, project_report = write_report()
    print({"report": str(report_path), "project_copy": str(project_report),
           "figures": [str(p) for p in sorted(FIG.glob("*.png"))]}, flush=True)


if __name__ == "__main__":
    main()
