"""Summarize frozen probability evidence; never assign human readability ratings."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from probability_diagnosis import OUT, RUN, WORKSPACE, PROJECT


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def link(path):
    return str(Path(path)).replace("\\", "/")


def main():
    summary = read(OUT/"概率诊断汇总.json")
    rows = read(OUT/"逐帧逐区域概率.json")
    curves = read(OUT/"七标注帧PR曲线.json")
    frozen = read(OUT/"概率诊断冻结协议.json")
    for path, expected in frozen["file_sha256"].items():
        assert sha(path) == expected, path
    for row in rows:
        for path_key, hash_key in (("image", "image_file_sha256"),
                                   ("fish_mask", "fish_mask_sha256"),
                                   ("array", "array_sha256")):
            assert sha(row[path_key]) == row[hash_key]
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 10, "figure.facecolor": "white"})
    names = {"original": "原始 RGB", "E": "E 接收恢复", "H265": "H.265 接收恢复"}
    colors = {"original": "#3a6aa0", "E": "#168979", "H265": "#c77c33"}
    conditions = ("original", "E", "H265")
    vals = {}
    for condition in conditions:
        chunks = []
        for row in rows:
            if row["condition"] != condition or row["GT_fish_probability"] is None:
                continue
            with np.load(row["array"]) as archive:
                probability = archive[archive.files[0]]
            mask = np.asarray(Image.open(row["fish_mask"])) > 0
            chunks.append(probability[mask])
        vals[condition] = np.concatenate(chunks)
        assert len(vals[condition]) == 2010
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for condition in conditions:
        v = np.sort(vals[condition])
        axes[0, 0].step(v, np.arange(1, len(v)+1)/len(v), where="post",
                        color=colors[condition], label=names[condition])
    axes[0, 0].axvline(.9, ls="--", lw=1, color="#666", label="固定阈值 0.9")
    axes[0, 0].set(xlim=(0, 1), ylim=(0, 1.02), xlabel="GT 鱼像素的鱼类概率",
                   ylabel="累计像素比例", title="① GT 鱼像素概率分布（2,010 像素）")
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 0].legend(fontsize=8, loc="lower right")
    bins = np.array(summary["original"]["GT_fish_probability"]["histogram_bins"])
    for index, condition in enumerate(conditions):
        h = np.array(summary[condition]["GT_fish_probability"]["histogram"])/2010
        axes[0, 1].bar(np.arange(8)+(index-1)*.24, h, width=.24,
                       color=colors[condition], label=names[condition])
    axes[0, 1].set_xticks(np.arange(8), [f"{a:g}–{b:g}" for a,b in zip(bins[:-1], bins[1:])], rotation=35)
    axes[0, 1].set(ylabel="GT 鱼像素比例", title="② 固定分箱；两种恢复均无 [0.85, 0.9) 像素")
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1))
    for condition in conditions:
        curve = curves[condition]
        ap = curve["AP"]
        axes[1, 0].plot(curve["curve"]["recall"], curve["curve"]["precision"],
                        color=colors[condition], lw=1.3,
                        label=f"{names[condition]}，像素 AP {100*ap:.4f}%")
        point = curve["fixed_09"]
        axes[1, 0].scatter(point["recall"], point["precision"], c=colors[condition], s=40, marker="x")
        if condition != "original":
            axes[1, 1].plot(curve["curve"]["recall"], curve["curve"]["precision"],
                            color=colors[condition], lw=1.3, label=names[condition])
    prevalence = curves["E"]["positive_prevalence"]
    for ax in axes[1]:
        ax.axhline(prevalence, ls=":", color="#999", lw=1, label="GT 鱼像素占比 0.0312%")
        ax.set(xlim=(0, 1), xlabel="像素召回率", ylabel="像素精确率")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(fontsize=8, loc="upper right")
    axes[1, 0].set(ylim=(-.015, 1.02), title="③ 全部 7 张 GT 的 PR；× 为固定 0.9 工作点")
    axes[1, 1].set(ylim=(-.0001, .006), title="④ 恢复结果 PR 纵轴放大；不用于选择新阈值")
    for ax in axes.flat:
        ax.grid(alpha=.16)
    fig.suptitle("video6 固定模型概率诊断：一段视频的既有 7 张稀疏 GT", fontsize=14)
    fig.savefig(OUT/"概率分布与PR诊断.png", dpi=170)
    plt.close(fig)

    component_keys = [(r["source_frame"], c["component"])
                      for r in rows if r["condition"] == "original" for c in r["components"]]
    matrix = []
    labels = []
    component_table = []
    for frame, component in component_keys:
        stats = []
        for condition in conditions:
            row = next(r for r in rows if r["condition"] == condition and r["source_frame"] == frame)
            region = next(c for c in row["components"] if c["component"] == component)
            stats.append(region["fish_probability"])
        labels.append(f"{frame}/c{component}  ({region['fish_pixels']} px)")
        matrix.append([s[k] for k in ("mean", "p50", "max") for s in stats])
        component_table.append(f"| {frame}/c{component} | {region['fish_pixels']} | " +
                               " | ".join(f"{s['mean']:.4f} / {s['max']:.4f}" for s in stats) + " |")
    fig, ax = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
    values = np.array(matrix)
    plot = ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_yticks(np.arange(10), labels)
    ax.set_xticks(np.arange(9), [f"{stat}\n{names[c]}" for stat in ("均值", "中位数", "最大值") for c in conditions])
    for (y,x), value in np.ndenumerate(values):
        ax.text(x, y, f"{value:.3f}", va="center", ha="center", fontsize=8,
                color="white" if value > .6 else "#152535")
    ax.axvline(2.5, color="white", lw=2)
    ax.axvline(5.5, color="white", lw=2)
    fig.colorbar(plot, ax=ax, label="鱼类预测概率（统一 0–1）", shrink=.8)
    ax.set_title("全部 10 个 GT 连通域；不能当作 10 条独立鱼", pad=12)
    fig.savefig(OUT/"逐连通域概率核对.png", dpi=170)
    plt.close(fig)

    human_files = sorted((RUN/"human_review/answers").glob("*.json"))
    human = [read(path) for path in human_files]
    assert all(not h["qa_mode"] for h in human)
    completed = sum(len(h["answers"]) == 37 for h in human)
    confirmed = sum(len(h["answers"]) for h in human)
    status = {"date":"2026-10-16", "stage_F_remains":"first_cycle_complete",
              "probability_diagnosis":"complete", "report":"complete",
              "human_review":"pending" if not confirmed else "in_progress",
              "human_observers_started":len(human), "human_observers_completed":completed,
              "confirmed_human_items":confirmed, "qa_responses_excluded":True,
              "human_review_url":"http://127.0.0.1:8780/", "budget_experiment":"not_started",
              "new_training":0, "new_video_encodes":0, "new_TCP_transfers":0,
              "reason_for_next_step":"collect human observations before choosing a budget allocation change"}
    (OUT/"本次推进状态.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n", "utf-8")
    summary_table = []
    metric_names = [("GT 鱼概率均值", "mean"),("中位数", "p50"),("第 90 百分位", "p90"),
                    ("最大值", "max")]
    for label, key in metric_names:
        summary_table.append(f"| {label} | " + " | ".join(f"{summary[c]['GT_fish_probability'][key]:.6f}" for c in conditions)+" |")
    report = f"""# 阶段 F 后续：video6 概率诊断与人工核对准备

日期：2026-10-16。本次完成概率诊断，准备并验证人工核对工具。阶段 F 的首轮完成状态保持不变；本次没有新命名阶段，没有训练模型、重新编码视频或新增通信测试。

**目前的关键结论：video6 两种恢复结果的零召回，不能解释为鱼类概率只是略微低于 0.9。GT 鱼像素上的模型响应大幅减弱；鱼对人是否仍可辨认，尚待真实人工核对。**

## 1. 范围与冻结条件

沿用阶段 F 的 video6：0—49 和 50—99 两段来自同一个视频，不能算两个独立视频。使用已经固定的 7 张发布者 GT：0、15、30、45、60、75、90；其中前 4 张无鱼，后 3 张合计 2,010 个鱼像素、10 个像素连通域。剩余 93 帧未标注，不进入本轮像素统计，也不当背景。

原图使用既存 RGB；E 和 H.265 使用阶段 F **实际接收端解码的 RGB**，未使用再次导出的展示视频。模型仍为冻结的 8 类 SegFormer，512 瓦片、384 步长；主任务阈值保持 0.9。未对 video6 选择新模型、阈值或编码配置。

源 RGB、接收 RGB、GT、模型与生产代码的哈希均已核对；21 张概率图对应的固定 0.9 下 TP/FP/FN 全部复现阶段 F 记录。完整配置见[冻结协议]({link(OUT/'概率诊断冻结协议.json')})，详细逐帧结果见[逐帧逐区域概率]({link(OUT/'逐帧逐区域概率.json')})。

## 2. 鱼类概率发生了什么

以下仅在已标注的 GT 鱼像素上统计，概率数值范围为 0—1；它是模型响应，不是经过校准的人类可见性或真实存在概率。

| 统计量 | 原图 | E 接收恢复 | H.265 接收恢复 |
|---|---:|---:|---:|
{chr(10).join(summary_table)}
| 达到 0.9 的 GT 鱼像素 | 615/2,010 | 0/2,010 | 0/2,010 |
| 位于预先约定 [0.85, 0.9) 的 GT 鱼像素 | 225/2,010 | 0/2,010 | 0/2,010 |
| 低于 0.75 的 GT 鱼像素比例 | 42.19% | 99.55% | 100% |

E 的 GT 鱼像素最大概率只有 0.7803，H.265 只有 0.1737，两种恢复图在 [0.85, 0.9) 区间都为零。原先提出的“0.89 与 0.84 都被 0.9 判成非鱼”是假设性例子，本片段不呈现这种临界现象。

E 的响应仍高于 H.265，但远低于原图。等权平均 10 个连通域的平均概率分别为 0.7077、0.0322、0.0037，下降方向没有因消除区域面积权重而改变。逐区域均值和最大值如下；全部分位数保留在 JSON。

| 源帧/连通域 | GT 像素 | 原图均值 / 最大值 | E 均值 / 最大值 | H.265 均值 / 最大值 |
|---|---:|---:|---:|---:|
{chr(10).join(component_table)}

![逐连通域概率]({link(OUT/'逐连通域概率核对.png')})

这可以排除“本片段零召回仅仅来自略低于 0.9”的解释，**不能证明鱼的视觉信息彻底消失，也不能确定局部失真、周围上下文、融合边界或模型鲁棒性中哪一个是唯一原因**。没有据此直接降低阈值。

## 3. PR 诊断与固定工作点

PR 使用全部 7 张已标注帧的鱼和背景，共 6,451,200 像素；GT 鱼占比仅 0.03116%。采用 float32 相同分数分组后的精确像素 AP，即每次召回增加量乘相应精确率后求和；不是梯形面积，也不是目标检测 mAP。曲线扫分数仅为诊断，不据此选阈值。

| 全部 7 张 GT | 原图 | E 接收恢复 | H.265 接收恢复 |
|---|---:|---:|---:|
| 像素 AP | 37.9427% | 0.1703% | 0.03548% |
| 固定 0.9：TP | 615 | 0 | 0 |
| 固定 0.9：FP | 405 | 3,409 | 6,593 |
| 固定 0.9：FN | 1,395 | 2,010 | 2,010 |
| 固定 0.9：鱼 IoU | 25.47% | 0% | 0% |
| 固定 0.9：鱼召回率 | 30.60% | 0% | 0% |

阶段 F 含鱼片段原图 IoU 为 25.80%，本表为 25.47%，是因为这里额外纳入 4 张无鱼 GT 的 31 个原图 FP。H.265 的 FP 同理从含鱼片段 3,153 增至 6,593；既有阶段 F 数字没有被改写。

![概率与 PR]({link(OUT/'概率分布与PR诊断.png')})

E 的像素 AP 高于 H.265，但二者绝对值均很低，而且只来自同一视频的稀疏 GT。曲线上的差距不能当成已经实现目标可辨识性或跨场景任务优势。像素数量也不能当成独立实验样本数。

## 4. 人工核对：材料与工具完成，结果待填写

正式入口：[本机人工核对](http://127.0.0.1:8780/)。服务只监听本机。请先完成页面核对，再查看带目标位置提示的对照材料；若已看过图像、框或报告，开始时如实记录即可。

每名观察者共 **37 项**，分两部分顺序进行：

1. **7 项整幅找鱼**：不提示位置，标记发现的位置，也可选择未发现或不确定。每人每个源帧只看一种匿名版本，避免先看原图后记住恢复图中的位置。三名观察者的轮换分组才能使每帧覆盖原图、E、H.265 三种条件；单人比较是探索性的非配对发现结果。
2. **30 项已知位置辨认**：全部 10 个 GT 连通域 × 3 种输入，随机排列。相同源坐标裁剪，统一 240×180 源像素、480×360 显示；只画位置外接框，不显示 GT 轮廓。分别选择可辨认是鱼及基本轮廓、不确定、无法辨认。这个结果不称无提示发现率。

整幅画面统一显示 960×540；请保持浏览器缩放一致，较窄窗口横向滚动。匿名页不展示方法、真实帧号、概率、PSNR 或正确答案。每次操作保存草稿，每次“确认并下一张”保存正式答案；刷新会恢复，全部完成无需再点总保存。同一人继续时使用同一代号；换人使用“切换观察者”，独立作答并如实记录既往接触情况。

发现任务的后续几何核对预先约定为点击位置与外扩 8 源像素的 GT 连通域框作一对一匹配，只是定位代理，不能称像素召回率或经确认的鱼实例召回率。已看过材料的发现答卷单列，不当作无提示的初次发现证据。单名观察者及已知位置辨认均有探索性限制。

协议、随机映射和显示规则已在人工作答前保存。[人工核对预先约定]({link(OUT/'人工核对预先约定.json')})及[接口核验]({link(OUT/'人工核对接口核验.json')})可复核。界面及接口测试使用合成图、单独测试答卷目录，绝不混入人工结果。**本报告生成时，正式已确认答案为 {confirmed} 项，已完成观察者 {completed} 名；尚无可报告的人工可辨识性结论。**

## 5. 本次判断与下一步

本次解决了一个诊断问题：在 video6 既定 GT 上，恢复后模型响应是明显减弱，而非集中在 0.9 附近。阶段 F 的“鱼区 PSNR 改善而识别仍为零”仍成立，现阶段不能仅靠调整判定阈值解释或宣称修复。

接下来先收集上述人工观察，分开看原图本身的难度、无提示发现和已知位置辨认。若人仍能辨认而模型失败，优先研究评价模型对压缩的鲁棒性；若原图能辨认、恢复图难辨认，再依据证据选择预算或上下文改动。两者也可能同时发生。

人工核对后只开展一组小规模的背景/ROI 预算分配对照：固定预测、筛选、双区域组织、分辨率与时序配置，在相近完整文件预算下评价接收端实际恢复结果。所有流、消息和封装纳入预算，保留负结果。video6 已进入诊断开发，后续方案选择后需要另留新视频验证。MLVC 多帧研究保持次要分支。

**当前完成的是概率诊断与人工核对准备，不是可靠目标保护的验证完成。** 状态保存在[本次推进状态]({link(OUT/'本次推进状态.json')})；昨日阶段 F 总结及历史实测证据包保持原样。
"""
    path = OUT/"2026-10-16_概率诊断与人工核对准备.md"
    path.write_text(report, "utf-8")
    (PROJECT/"docs/probability_review_20261016.md").write_text(report, "utf-8")
    (WORKSPACE/"2026-10-16_继续推进记录.md").write_text(
        "# 2026-10-16 继续推进记录\n\n已完成 video6 概率诊断；正式人工核对材料与工具已准备，结果待真人填写。"
        "阶段 F 保持首轮完成状态，本次未重新训练、编码或进行新通信测试。\n\n"
        f"- [本次详细报告]({link(path)})\n- [人工核对入口](http://127.0.0.1:8780/)\n"
        f"- [本次推进状态]({link(OUT/'本次推进状态.json')})\n", "utf-8")
    check = read(OUT/"概率诊断核验.json")
    check.update({"status":"probability_and_report_complete_human_review_pending",
                  "source_hashes_rechecked_after_report":True,
                  "figures_generated_from_frozen_data":2})
    (OUT/"概率诊断核验.json").write_text(json.dumps(check,ensure_ascii=False,indent=2)+"\n", "utf-8")
    print(json.dumps({"report":str(path),"human_confirmed":confirmed,"model_unchanged":True},ensure_ascii=False))


if __name__ == "__main__":
    main()
