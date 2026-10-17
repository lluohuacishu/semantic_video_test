"""Analyze actual human answers without changing them or the frozen review protocol."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE = Path(r"C:\Users\lenovo\Documents\ChatGPT\语义通信")
PROJECT = Path(r"C:\Users\lenovo\Desktop\works\work\semantic-video-mlvc")
RUN = PROJECT/"runs/semantic_mvp_probability_review_20261016"
HUMAN = RUN/"human_review"
OUT = WORKSPACE/"outputs/semantic_review_20261016"
WORK = Path(__file__).parent
CONDITIONS = ("original", "E", "H265")
CHOICES = ("recognizable", "uncertain", "not_recognizable")
LABELS = {"recognizable":"可辨认", "uncertain":"不确定", "not_recognizable":"无法辨认",
          "found":"发现并标记", "none":"未发现"}
NAMES = {"original":"原图", "E":"E 接收恢复", "H265":"H.265 接收恢复"}


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", "utf-8")


def link(path):
    return str(path).replace("\\", "/")


def match_points(points, regions):
    """Maximum one-to-one matching using the prospectively specified 8px bbox margin."""
    if not points or not regions:
        return []
    valid = np.zeros((len(points), len(regions)), dtype=np.int8)
    for i, (px, py) in enumerate(points):
        for j, region in enumerate(regions):
            x,y,w,h = region["box"]
            valid[i,j] = max(0,x-8) <= px < min(1280,x+w+8) and max(0,y-8) <= py < min(720,y+h+8)
    point_indices, region_indices = linear_sum_assignment(valid, maximize=True)
    return [{"point_index":int(i), "component":regions[j]["component"]}
            for i,j in zip(point_indices, region_indices) if valid[i,j]]


def matching_checks():
    # First point overlaps both boxes, second only the first: greedy matching would fail.
    regions = [{"component":1, "box":[10,10,10,10]}, {"component":2, "box":[25,10,10,10]}]
    assert len(match_points([[22,15],[10,15]], regions)) == 2
    assert len(match_points([[10,15],[10,15]], regions[:1])) == 1
    assert not match_points([], regions)
    assert not match_points([[100,100]], regions)


def main():
    matching_checks()
    key = read(HUMAN/"private_key.json")
    protocol = read(HUMAN/"protocol.json")
    assert sha(HUMAN/"protocol.json") == key["protocol_sha256"]
    for field, path in (("asset_builder_sha256",WORK/"prepare_human_review.py"),
                        ("server_sha256",WORK/"review_server.py"), ("page_sha256",WORK/"review.html")):
        assert sha(path) == protocol[field], path
    probability_protocol = OUT/"概率诊断冻结协议.json"
    assert sha(probability_protocol) == protocol["source_probability_protocol_sha256"]
    frozen = read(probability_protocol)
    for path, expected in frozen["file_sha256"].items():
        assert sha(path) == expected, path
    source_files = sorted((HUMAN/"answers").glob("*.json"))
    assert len(source_files) == 1, "This report is explicitly for the one completed actual observer"
    source = source_files[0]
    before_hash = sha(source)
    session = read(source)
    assert session["qa_mode"] is False
    assert session["protocol_sha256"] == key["protocol_sha256"]
    assert len(session["tasks"]) == len(session["answers"]) == 37
    assert session.get("draft") is None
    assert len({a["task_id"] for a in session["answers"]}) == 37
    assert [a["task_id"] for a in session["answers"]] == [t["id"] for t in session["tasks"]]
    assert [a["phase"] for a in session["answers"]] == ["discovery"]*7+["recognition"]*30
    archive = RUN/"human_analysis/source_answers"
    archive.mkdir(parents=True, exist_ok=True)
    snapshot = archive/f"{before_hash}.json"
    if not snapshot.exists():
        shutil.copyfile(source, snapshot)
    assert sha(snapshot) == before_hash

    probability_rows = read(OUT/"逐帧逐区域概率.json")
    reference = read(WORKSPACE/"outputs/semantic_roi_phase_f/step3_frozen_external/positive_case/含鱼GT保护与逐连通域核对.json")
    regions = reference["gt_components"]
    region_lookup = {(r["source_frame"],r["component"]):r for r in regions}
    record_lookup = {(r["source_frame"],r["condition"]):r for r in probability_rows}
    discovery = []
    recognition = []
    verified_assets = set()
    for task, answer in zip(session["tasks"],session["answers"]):
        asset = key["assets"][task["asset"]]
        assert asset["phase"] == answer["phase"]
        if task["asset"] not in verified_assets:
            assert sha(asset["path"]) == asset["file_sha256"]
            verified_assets.add(task["asset"])
        display = answer["display"]
        assert abs(display["image_css_width"] - asset["display_width"]) < .01
        assert abs(display["image_css_height"] - asset["display_height"]) < .01
        row = {"observer":"observer_1", "phase":answer["phase"], "source_frame":asset["source_frame"],
               "condition":asset["condition"], "choice":answer["choice"], "display":display,
               "elapsed_ms":answer["elapsed_ms"], "saved_utc":answer["saved_utc"]}
        if answer["phase"] == "discovery":
            assert answer["choice"] in {"found","none","uncertain"}
            gt = [r for r in regions if r["source_frame"] == asset["source_frame"]]
            row.update({"GT_components":len(gt), "points":answer["points"],
                        "viewport_narrower_than_canvas":display["viewport_width"] < display["image_css_width"]})
            if answer["choice"] == "uncertain":
                row.update({"confirmed_geometry_matches":None, "unmatched_clicks":None,
                            "unmatched_GT_components":None, "abstention_not_scored_as_negative":True})
            else:
                matches = match_points(answer["points"],gt)
                row.update({"confirmed_geometry_matches":matches,
                            "unmatched_clicks":len(answer["points"])-len(matches),
                            "unmatched_GT_components":len(gt)-len(matches)})
            discovery.append(row)
        else:
            assert answer["choice"] in CHOICES
            region = region_lookup[(asset["source_frame"],asset["component"])]
            probability_region = next(r for r in record_lookup[(asset["source_frame"],asset["condition"])]["components"]
                                      if r["component"] == asset["component"])
            row.update({"component":asset["component"], "GT_pixels":region["fish_pixels"],
                        "E_roi_coverage":region["E_roi_coverage"],
                        "model_GT_probability":probability_region["fish_probability"],
                        "model_GT_recall_09":region["model_gt_recall"][asset["condition"]],
                        "fish_RGB_PSNR_db":None if asset["condition"] == "original" else region["fish_rgb_psnr_db"][asset["condition"]]})
            recognition.append(row)
    assert sorted(r["source_frame"] for r in discovery) == [0,15,30,45,60,75,90]
    lookup = {(r["source_frame"],r["component"],r["condition"]):r for r in recognition}
    assert len(lookup) == 30
    assert set(lookup) == {(f,c,m) for f,c in region_lookup for m in CONDITIONS}
    counts = {c:dict(Counter(r["choice"] for r in recognition if r["condition"] == c)) for c in CONDITIONS}
    for c in CONDITIONS:
        counts[c] = {choice:counts[c].get(choice,0) for choice in CHOICES}
        assert sum(counts[c].values()) == 10
    paired = {a:{b:0 for b in CHOICES} for a in CHOICES}
    original_visible = []
    for frame, component in sorted(region_lookup):
        e = lookup[(frame,component,"E")]["choice"]
        h = lookup[(frame,component,"H265")]["choice"]
        paired[e][h] += 1
        if lookup[(frame,component,"original")]["choice"] == "recognizable":
            original_visible.append((frame,component))
    retained = {c:sum(lookup[(f,k,c)]["choice"] == "recognizable" for f,k in original_visible) for c in CONDITIONS}
    contradictions = {c:sum(r["choice"] == "recognizable" and r["model_GT_recall_09"] == 0
                            for r in recognition if r["condition"] == c) for c in ("E","H265")}
    summary = {"observers":1,"confirmed":37,"prior_exposure_self_report":session["prior_exposure"],
               "recognition_counts":counts, "paired_E_rows_H265_columns":paired,
               "posthoc_original_recognizable_components":len(original_visible),
               "posthoc_retained_original_recognizable":retained,
               "human_recognizable_but_zero_model_GT_recall":contradictions,
               "discovery_items":7,"discovery_same_frame_paired_comparison":False,
               "discovery_viewport_narrower_than_canvas_count":sum(r["viewport_narrower_than_canvas"] for r in discovery),
               "whole_frame_horizontal_scroll_coverage_recorded":False,
               "recognition_uncertain_not_counted_as_recognizable":True,
               "scope":"one observer, one video, ten GT connected components; descriptive diagnostics, no significance/generalization claim",
               "no_revised_answers_or_protocol":True}
    save(OUT/"单人核对汇总.json", summary)
    save(OUT/"单人核对逐项结果.json", {"discovery":discovery,"recognition":recognition})

    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":11})
    fig, ax = plt.subplots(figsize=(8.4,4.2), constrained_layout=True)
    left = np.zeros(3)
    for choice, color in zip(CHOICES,("#168979","#e5b655","#aab6c3")):
        values = np.array([counts[c][choice] for c in CONDITIONS])
        ax.barh(np.arange(3), values, left=left, color=color, label=LABELS[choice], height=.58)
        for y, (value,start) in enumerate(zip(values,left)):
            if value:
                ax.text(start+value/2,y,str(value),va="center",ha="center",color="#15313b")
        left += values
    ax.set_yticks(np.arange(3), [NAMES[c] for c in CONDITIONS])
    ax.invert_yaxis()
    ax.set(xlim=(0,10),xticks=range(11),xlabel="GT 连通域局部辨认项数（每种输入 10 项）",
           title="1 名观察者：已知位置后的局部辨认，非整幅发现率")
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.17),ncol=3,frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x",alpha=.15)
    fig.savefig(OUT/"单人局部辨认统计.png",dpi=170)
    plt.close(fig)

    component_table = []
    for f,c in sorted(region_lookup):
        region = region_lookup[(f,c)]
        outcomes = [LABELS[lookup[(f,c,m)]["choice"]] for m in CONDITIONS]
        probabilities = [lookup[(f,c,m)]["model_GT_probability"]["mean"] for m in ("E","H265")]
        component_table.append(f"| {f}/c{c} | {region['fish_pixels']} | {100*region['E_roi_coverage']:.0f}% | " +
                               " | ".join(outcomes)+f" | {probabilities[0]:.4f} / {probabilities[1]:.4f} |")
    discovery_table = []
    for r in sorted(discovery,key=lambda r:r["source_frame"]):
        result = "不评分（不确定）" if r["confirmed_geometry_matches"] is None else f"{len(r['confirmed_geometry_matches'])} / {r['GT_components']}"
        unmatched = "—" if r["unmatched_clicks"] is None else str(r["unmatched_clicks"])
        discovery_table.append(f"| {r['source_frame']} | {NAMES[r['condition']]} | {r['GT_components']} | {LABELS[r['choice']]} | {result} | {unmatched} |")
    exposure = {"no":"未看过本批图像或目标位置","yes":"已看过本批图像、目标框或对照报告","unsure":"不确定是否已接触"}[session["prior_exposure"]]
    report = f"""# 阶段 F 后续：单人可辨识性核对与下一步判断

日期：2026-10-16。已收到 **1 名真实观察者完整的 37/37 项答卷**：7 项整幅找鱼、30 项已知位置局部辨认，无未确认草稿。本报告只统计正式答卷，不纳入任何合成图测试记录；不要求补充其他观察者才能推进。

**关键发现：这名观察者在位置提示后，原图、E、H.265 分别有 8/10、6/10、4/10 项判为可辨认。模型在两种恢复结果上的 GT 鱼召回均为零，不能由此推断人也完全无法辨认鱼。** 这是同一视频、同一观察者的探索性结果，尚不能证明 E 的稳定优势。

## 1. 保存与完整性核对

答卷保存于 2026-10-16，观察者自报“{exposure}”。顺序为先完成 7 张整幅找鱼，再进入局部辨认；所有确认项与随机任务逐一匹配，无重复或缺失，且所有 10 个 GT 连通域均具有原图、E、H.265 三份局部答复。方法与帧号在作答界面隐藏，局部任务随机排列。

已核对答卷、素材、冻结协议和页面哈希，归档原始答卷的字节副本；未更改答案、模型、阈值、ROI 或编码配置。以下“10 项”对应 10 个 GT 像素连通域，不是经确认的 10 条独立鱼。不同连通域的裁剪还可能包含相邻目标，共享同一帧上下文，不能当作完全独立样本。

## 2. 已知位置后的辨认结果

| 输入 | 可辨认是鱼及基本轮廓 | 不确定 | 无法辨认 | 合计 |
|---|---:|---:|---:|---:|
| 原图 | {counts['original']['recognizable']} | {counts['original']['uncertain']} | {counts['original']['not_recognizable']} | 10 |
| E 接收恢复 | {counts['E']['recognizable']} | {counts['E']['uncertain']} | {counts['E']['not_recognizable']} | 10 |
| H.265 接收恢复 | {counts['H265']['recognizable']} | {counts['H265']['uncertain']} | {counts['H265']['not_recognizable']} | 10 |

“不确定”原样保留，不合并为可辨认或无法辨认。这里判断的是固定位置提示、固定尺度下的局部辨认，不能称为无提示发现率。两种接收结果沿用阶段 F 相近完整文件预算：E 99,269 B、H.265 98,551 B，未重新编码。

![单人局部辨认统计]({link(OUT/'单人局部辨认统计.png')})

逐项配对看：E 与 H.265 均判为可辨认的有 3 项；E 可辨认、H.265 不确定的有 3 项；H.265 可辨认、E 不确定的有 1 项。不存在 E 可辨认而 H.265 被判为明确无法辨认的项。两法各有 1 项无法辨认，但不是同一个连通域。因此差异主要体现在可辨认与不确定之间，不能表述为“E 把 H.265 完全看不见的鱼恢复出来了”。

事后按原图已被判为可辨认的 8 项作描述性核对，E 有 6 项仍可辨认，H.265 有 4 项。这个子集核对不是预先登记的独立验证，也不作显著性检验。原图自身还有 1 项不确定、1 项无法辨认，不能把这些困难全部归于压缩。

| 帧/连通域 | GT 像素 | E 的 GT 像素覆盖 | 原图判读 | E 判读 | H.265 判读 | E / H.265 的 GT 平均概率 |
|---|---:|---:|---|---|---|---:|
{chr(10).join(component_table)}

## 3. 与模型概率诊断合起来怎么解释

E 中 6 项、H.265 中 4 项被这名观察者判为可辨认，但对应区域的模型 GT 召回都为零。这提供了“人能辨认而当前模型失败”的具体个案证据。比如 60/c1：原图与 E 均可辨认，H.265 不确定，E 在该 GT 区域的平均鱼概率仅约 0.007。

同时，原图 GT 鱼概率均值为 0.718，E 为 0.051、H.265 为 0.005；恢复图的 GT 鱼像素没有落在 [0.85, 0.9) 区间。因此两点可以同时成立：模型响应确实大幅减弱，但部分目标仍能被人辨认。不能把降低 0.9 阈值当作已经解决问题，也不能把零 IoU 当作全部视觉信息丢失。

现有模型对低码率重建的响应和人工辨认出现了不一致，评价模型的鲁棒性与代表性需要提升优先级；这还不足以把全部原因归为分割器。原图 75/c2 可辨认，E 不确定、H.265 无法辨认；编码仍可能损失了有用细节或上下文。接收结果中也有 H.265 判读优于 E 的个案，不能只保留有利样例。

局部框是由 GT 提供的位置提示，所以本结果不能证明发送端能够发现这些鱼；也不能证明整体视频已经具备相同的观看质量。阶段 F 中 E 的全局 PSNR 仍比 H.265 低约 3.38 dB，这个代价没有因局部辨认统计而消失。

## 4. 整幅找鱼：保留原始观察，不给方法排名

每个源帧只随机展示一种版本，只有一名观察者，三种方法面对的鱼区域不同。这是非配对观察，不能用各方法的找到数量作公平比较。不确定保持弃答，不计作确定的负结果。

| 源帧 | 实际展示版本 | GT 连通域数 | 观察者选择 | 外扩 8 像素框几何匹配数 / GT 数 | 未匹配点击 |
|---|---|---:|---|---|---:|
{chr(10).join(discovery_table)}

上述点击使用事前约定的最大一对一匹配。它只是点击与 GT 外接框的几何定位代理，不是像素召回率或经确认的鱼实例召回率；未匹配点击也不能直接等同于已证实的人工错误。

**界面还有一项限制：整幅画布为 960×540，记录的视口宽度为 641—927，7 项均需横向滚动才能看完整幅图；程序未记录是否滚动看遍所有位置，且作答中窗口宽度有变化。** 因此这部分尤其不适合得出发现能力结论。局部图固定显示 480×360，记录的视口足以容纳其宽度，30 项显示尺寸均一致。

本次不要求重测。后续若正式评价整幅发现，先把所有方法统一为完整可见的显示条件并记录覆盖情况，再另设材料与协议；不能在已知道目标位置后重测并当成首次发现。

## 5. 接下来怎么推进

当前第一优先的“人工辨认 + 固定模型概率诊断”已按单人探索性范围完成，无需等待更多人。项目结论更新为：**目标 ROI 的像素质量优势，可能对应部分人工局部辨认的积极迹象；现有分割 IoU 不足以单独代表目标可辨识性，但可靠的系统任务收益仍未证实。**

下一次仍只做一组背景/ROI 预算小对照，不增加新模型：

1. 固定预测、0.9 阈值、筛选、双区域布局、分辨率与时序设置，只改背景和 ROI 的质量分配；预先限制少量候选，不反复按 video6 的 GT 选“最好看”的结果。
2. 在同一完整文件预算下，使用实际接收端 RGB，联合比较全局/背景质量、GT 鱼区质量、固定模型概率与任务指标；目标覆盖和全部流、消息、封装开销继续单列。检查增加背景质量及上下文是否有帮助，不预设一定有效。
3. 本轮单人结果保留为诊断证据。模型适配与阈值校准如果另行开展，必须用开发数据并把旧/新评价模型分列，不能为了让 E 获胜替换评价口径。方案选择后仍需新的保留视频验证。

本次只完成答卷核验、统计与方向更新，尚未启动预算实验，没有新增训练、编码或通信测试。阶段 F 既有报告和证据保持不变。

## 6. 可复核资料

- [逐项人工答案与几何匹配]({link(OUT/'单人核对逐项结果.json')})
- [单人统计汇总]({link(OUT/'单人核对汇总.json')})
- [答卷及素材核验]({link(OUT/'单人核对核验.json')})
- [作答前固定协议]({link(OUT/'人工核对预先约定.json')})
- [前置概率诊断报告]({link(OUT/'2026-10-16_概率诊断与人工核对准备.md')})
"""
    report_path = OUT/"2026-10-16_单人可辨识性核对与方向更新.md"
    report_path.write_text(report,"utf-8")
    (PROJECT/"docs/human_review_20261016.md").write_text(report,"utf-8")
    assert sha(source) == before_hash
    checks = {"observer_count":1,"actual_confirmed":37,"discovery_confirmed":7,"recognition_confirmed":30,
              "draft":None,"QA_directories_excluded":True,"original_answer_sha256":before_hash,
              "answer_source":str(source),"answer_byte_snapshot":str(snapshot),
              "answer_and_protocol_unmodified":True,"all37_assets_hash_verified":True,
              "all10_components_have3_conditions":True,"all_display_sizes_match_protocol":True,
              "maximum_matching_overlap_duplicate_and_empty_checks_passed":True,
              "no_method_retraining_or_reencoding":True,"analysis_code_sha256":sha(__file__),
              "generated_utc":datetime.now(timezone.utc).isoformat()}
    save(OUT/"单人核对核验.json",checks)
    state = read(OUT/"本次推进状态.json")
    state.update({"human_review":"complete_single_observer_exploratory","human_observers_started":1,
                  "human_observers_completed":1,"confirmed_human_items":37,"human_report":str(report_path),
                  "next_action":"one controlled budget allocation comparison; no new model or threshold change",
                  "budget_experiment":"not_started","reason_for_next_step":"human-model discrepancy confirmed as case evidence; next inspect allocation/context under fixed budget"})
    save(OUT/"本次推进状态.json",state)
    old_report = OUT/"2026-10-16_概率诊断与人工核对准备.md"
    note = f"> 同日后续：1 名观察者已完成 37/37 项。[查看单人结果与方向更新]({link(report_path)})。本文第 4 节的“待填写”及零答案数字保留为准备完成时的历史状态。\n\n"
    old_text = old_report.read_text("utf-8")
    if note not in old_text:
        title, body = old_text.split("\n",1)
        old_report.write_text(title+"\n\n"+note+body.lstrip("\n"),"utf-8")
        (PROJECT/"docs/probability_review_20261016.md").write_text(old_report.read_text("utf-8"),"utf-8")
    (WORKSPACE/"2026-10-16_继续推进记录.md").write_text(
        "# 2026-10-16 继续推进记录\n\n已完成 video6 概率诊断及 1 名真实观察者的 37/37 项核对。"
        "位置提示后原图、E、H.265 分别有 8/10、6/10、4/10 项可辨认；这是单人探索性结果，不能当作稳定优势或无提示发现率。"
        "人工与模型响应存在差异，下一次做固定文件预算下的背景/ROI 质量分配小对照。阶段 F 保持首轮完成，本次未新增训练、编码或通信测试。\n\n"
        f"- [单人核对结果与方向更新]({link(report_path)})\n"
        f"- [前置概率诊断]({link(old_report)})\n"
        f"- [本次推进状态]({link(OUT/'本次推进状态.json')})\n","utf-8")
    print(json.dumps({"report":str(report_path),"recognition_counts":counts,"real_observers":1,
                      "all37_saved":True,"raw_answers_unmodified":True},ensure_ascii=False))


if __name__ == "__main__":
    main()
