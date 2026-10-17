"""Publish measured results, controls, negative outcomes, and reproducible artifacts."""
import csv
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from experiment_common import *


def csv_write(path,rows,fields):
    with Path(path).open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)


def pct(x):return "—" if x is None else f"{100*x:.2f}%"
def number(x):return "—" if x is None else f"{x:.3f}"


def main():
    measured=json.loads((RUN/"task_quality_results.json").read_text("utf-8"));rows=measured["rows"]
    if len(rows)!=24:raise AssertionError("Incomplete final comparison")
    m=manifest();teacher=measured["teacher"]
    if sha256(teacher["checkpoint"])!=teacher["checkpoint_sha256"]:raise AssertionError("Fixed teacher changed")
    cold=json.loads((RUN/"cold_decoder_summary.json").read_text("utf-8"))
    if not cold["all_selected_files_verified"]:raise AssertionError("Cold decoding incomplete")
    protocol=json.loads((RUN/"protocol.json").read_text("utf-8"));schedule=json.loads((RUN/"schedule.json").read_text("utf-8"))
    gradient=[];controls=[];channels=[]
    sample_keys=["step","video","target_index","reference_index","left","top","crop","horizontal_flip","q_index","gt_available","crop_kind","mask"]
    for branch in ("M_RD","M_ROI4","M_ROI16"):
        logs=[json.loads(x) for x in (RUN/"training"/branch/"steps.jsonl").read_text("utf-8").splitlines()]
        if len(logs)!=512:raise AssertionError("Update count differs")
        if any(any(log[k]!=sample[k] for k in sample_keys) for log,sample in zip(logs,schedule,strict=True)):raise AssertionError("Paired data/crop/Q schedule differs")
        result=json.loads((RUN/"training"/branch/"result.json").read_text("utf-8"))
        if result["schedule_sha256"]!=sha256(RUN/"schedule.json"):raise AssertionError("Schedule hash mismatch")
        controls.append({"branch":branch,"updates":len(logs),"changed_parameter_tensors":result["changed_parameter_tensors"],"peak_allocated_MiB":result["peak_allocated_bytes"]/2**20,"peak_reserved_MiB":result["peak_reserved_bytes"]/2**20,"seconds":result["seconds"],"paired_samples_verified":True})
        for log in logs:
            y=log.get("weighted_channel_mse",{}).get("Y",{})
            gradient.append({k:log.get(k) for k in ("branch","step","video","target_index","reference_index","gt_available","crop_kind","roi_fraction","q_index","active_roi_ratio","loss","bpp_estimate","distortion","encoder_gradient_norm","gradient_norm_before_clip","distortion_gradient_roi_abs_sum","distortion_gradient_background_abs_sum","encoder_probe_Y_roi_gradient_norm","encoder_probe_Y_background_gradient_norm")}|y)
        labelled=[x for x in logs if x["gt_available"]]
        for channel in ("Y","U420","V420"):
            a=sum(x["weighted_channel_mse"][channel]["roi_weighted_mse_contribution"] for x in labelled)
            b=sum(x["weighted_channel_mse"][channel]["background_weighted_mse_contribution"] for x in labelled)
            channels.append({"branch":branch,"channel":channel,"labelled_updates":len(labelled),"ROI_weighted_MSE_sum":a,"background_weighted_MSE_sum":b,"ROI_share_actual_MSE":a/(a+b)})
    compare=[]
    for r in rows:
        if r["branch"] not in ("M_ROI4","M_ROI16"):continue
        baseline=next(x for x in rows if x["branch"]=="M_RD" and x["video"]==r["video"] and x["anchor"]==r["anchor"])
        compare.append({"video":r["video"],"role":r["experiment_role"],"budget":{8:"低",24:"中",40:"高"}[r["anchor"]],"branch":r["branch"],"M_RD_file_bytes":baseline["file_bytes"],"ROI_file_bytes":r["file_bytes"],"file_difference_percent":100*r["relative_file_difference"],"matched_within_2_percent":r["matched_within_2_percent"],
                        "M_RD_fish_IoU_percent":100*baseline["fish_iou"],"ROI_fish_IoU_percent":100*r["fish_iou"],"delta_fish_IoU_points":100*(r["fish_iou"]-baseline["fish_iou"]),"M_RD_fish_recall_percent":100*baseline["fish_recall"],"ROI_fish_recall_percent":100*r["fish_recall"],
                        "M_RD_fish_PSNR_dB":baseline["gt_fish_rgb_psnr_db"],"ROI_fish_PSNR_dB":r["gt_fish_rgb_psnr_db"],"delta_fish_PSNR_dB":r["gt_fish_rgb_psnr_db"]-baseline["gt_fish_rgb_psnr_db"],"global_PSNR_cost_dB":baseline["global_rgb_psnr_db"]-r["global_rgb_psnr_db"],"background_PSNR_cost_dB":baseline["gt_background_rgb_psnr_db"]-r["gt_background_rgb_psnr_db"]})
    OUT.mkdir(parents=True,exist_ok=True)
    csv_write(OUT/"实测对照.csv",compare,list(compare[0]))
    csv_write(OUT/"完整质量点.csv",rows,["video","experiment_role","official_split","branch","anchor","q_index","file_bytes","payload_bytes","container_bytes","i_bytes","p_bytes","relative_file_difference","matched_within_2_percent","global_rgb_psnr_db","gt_fish_rgb_psnr_db","gt_background_rgb_psnr_db","fish_iou","fish_recall","fish_precision","fish_f1","gt_frames","gt_positive_frames","teacher_sha256","independent_decode_verified"])
    csv_write(OUT/"训练梯度与损失.csv",gradient,list(gradient[0]))
    csv_write(OUT/"实际ROI误差贡献.csv",channels,list(channels[0]))
    save_json(OUT/"实验审计.json",{"paired_schedule_verified":True,"control_results":controls,"schedule_sha256":sha256(RUN/"schedule.json"),"protocol_sha256":sha256(RUN/"protocol.json"),"data_manifest_sha256":sha256(DATA/"manifest.json"),"fixed_teacher":teacher,"cold_decode":cold,"test_used":False})
    font=FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams.update({"font.family":font.get_name(),"axes.unicode_minus":False,"font.size":10})
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    colors={"M_RD":"#245c96","M_ROI4":"#d66b2e","M_ROI16":"#2f8f68","M_PRE":"#777777"}
    for col,video in enumerate(("video96","video108")):
        for branch in ("M_PRE","M_RD","M_ROI4","M_ROI16"):
            group=sorted([r for r in rows if r["video"]==video and r["branch"]==branch],key=lambda x:x["file_bytes"])
            axes[0,col].plot([r["file_bytes"]/1024 for r in group],[r["fish_iou"]*100 for r in group],"o--" if branch=="M_PRE" else "o-",label=branch,color=colors[branch])
            axes[1,col].plot([r["file_bytes"]/1024 for r in group],[r["gt_fish_rgb_psnr_db"] for r in group],"o--" if branch=="M_PRE" else "o-",label=branch,color=colors[branch])
        original=next(x for x in measured["original"] if x["video"]==video)
        axes[0,col].axhline(100*original["fish_iou"],color="black",linestyle=":",label="参考RGB识别成绩")
        axes[0,col].set_title(video+("（开发验证）" if video=="video96" else "（官方验证）"))
        axes[0,col].set_ylabel("鱼类 IoU（%）");axes[1,col].set_ylabel("GT 鱼区 RGB PSNR（dB）")
        for row in (0,1):axes[row,col].set_xlabel("完整文件大小（KiB）");axes[row,col].grid(alpha=.25);axes[row,col].legend(fontsize=8)
    fig.savefig(OUT/"完整文件预算与任务质量.png",dpi=160);plt.close(fig)
    lines=["# 阶段 C：小规模 GT-ROI 微调实测", "", "日期：2026-10-13。本报告记录实际训练、真实熵编码、独立解码和 GT 评价。当前方法是**语义感知损失驱动的视频编码微调**；编码器输入没有 ROI 掩码，未实现显式动态语义比特分配。", "", "## 数据和可信边界", "", "| 视频 | 官方划分 / 本轮用途 | 连续 RGB 帧 | GT 帧 | 含鱼 GT 帧 |", "|---|---|---:|---:|---:|"]
    for v in m["videos"]:lines.append(f'| {v["video"]} | {v["official_split"]} / {v["experiment_role"]} | {v["frames"]} | {v["annotated_frames"]} | {v["annotated_frames_with_fish"]} |')
    lines += ["", "来源：[UVMulti 作者仓库](https://github.com/pingchuan/UVMulti)、[官方 Figshare](https://figshare.com/articles/dataset/30105151)。用户下载的 video131 已移到 `C:/Users/lenovo/Desktop/works/work/semantic-video-mlvc/data/uvmulti_downloads/video131`，2,303 个原始文件保留，其中 834 个实际使用的 RGB/GT 条目与官方 ZIP 长度和 CRC32 一致。整包发布 MD5 没有完整核验。", "", "发现并修正归档命名：`video25.zip` 实际含 video24；其嵌套 `video109.zip` 实际含官方 val 的 video108。采用实际目录编号查询发布者划分，未按压缩包名称误分训练/验证。全部标注 RGB 与同编号连续 RGB 逐像素一致；序列编号无缺口，目标 P 帧引用前一个序列帧。训练与验证无完全相同的 RGB 帧。", "", "video96 从官方 train 按完整视频留出，用于分割器配置和 ROI 方案开发选择；video108 属于官方 val，未用于分割器适配或阈值选择。官方 test 及已分析的 SFISHTRACK 001/027/054 本轮均未用于优化。公开样例缺少多数片段的原始拍摄文件信息，尚不能证明全部片段在拍摄组层面独立。", "", "video91 的 time.txt 对应 20 秒 300 帧，约 15 fps，与论文统一声明 25 fps 不同。两段评价视频缺少相同时间记录，视频缓存采用论文声明的 25 fps，不能据此宣称真实采集码率；本轮核心比较采用完整文件字节数。GT 稀疏且 video108 仅有 3 张含鱼标注，结论范围有限。", "", "## 受控训练与梯度", "", "三组均从相同官方 MLVC-S 权重开始，固定相同 512 次更新、Adam 1e-5、种子、q={8,24,40}、256 裁剪和冻结 DMCI 前序重建。50% 更新来自有 GT 目标帧，50% 为无 GT 目标帧；25% 为鱼目标裁剪，其他为确认背景或随机裁剪。无 GT 更新使用普通 RD 损失，没有把未知标签填成背景。", "", "普通分支与 ROI 分支使用同一原生几何 YUV420 失真（Y 权重8）和非零码率项。ROI 权重1在空、全和部分掩码上与普通分支的损失、重建梯度差均为0；本轮只使用已验证的 batch1。固定上游的 batch2 ROI 权重广播存在问题，未修改上游，也未宣称支持 batch2。", "", "| 方法 | 实际更新 | 发生更新的参数张量 | 分配峰值 MiB |", "|---|---:|---:|---:|"]
    for c in controls:lines.append(f'| {c["branch"]} | {c["updates"]} | {c["changed_parameter_tensors"]} | {c["peak_allocated_MiB"]:.2f} |')
    lines += ["", "三组逐步样本、裁剪、翻转、q 和 GT 状态逐项核对一致。记录了真实 ROI 面积、Y/U/V 内 ROI 与背景加权误差、重建梯度的区域分布及编码器梯度；见 CSV。几何总失真是非线性的，未把各区域的加权 MSE 错写成可直接相加的几何总损失。", "", "| 方法 | 有 GT 更新内实际 Y-MSE 的 ROI 贡献占比 |", "|---|---:|"]
    for c in channels:
        if c["channel"]=="Y":lines.append(f'| {c["branch"]} | {pct(c["ROI_share_actual_MSE"])} |')
    lines += ["", "## 固定分割评价器", "", "最初的二分类头适配在开发原图上的最佳 IoU 仅 1.36%，该负结果和权重已保留。分块尺度检查也未使其可靠。随后保留 SUIM 原八类结构，冻结编码器和 BatchNorm 统计，仅适配解码头；fish 对其余七类概率总和进行二值监督。", "", f'最终模型在开发原图上选择 step={teacher["selected_step"]}，512 分块 / stride384、重叠 logits 平均，fish 阈值={teacher["threshold"]}。开发原图 IoU={pct(teacher["development_original_fish_iou"])}，仍标记为可靠性有限。评价权重 SHA256：`{teacher["checkpoint_sha256"]}`。压缩任务计分前固定，计分期间未修改。', "", "| 视频 | 参考 RGB 鱼 IoU | 参考 RGB 鱼召回 | 参考 RGB 鱼精确率 |", "|---|---:|---:|---:|"]
    for r in measured["original"]:lines.append(f'| {r["video"]} | {pct(r["fish_iou"])} | {pct(r["fish_recall"])} | {pct(r["fish_precision"])} |')
    lines += ["", "参考成绩用于观察分割器本身的局限，不是压缩后成绩的理论上限。", "", "## 完整文件预算下的实测", "", "包括完整 I/P 帧和容器开销，GOP32。以 M_RD 三个质量点的完整字节数为目标，ROI 的 q 仅按已测候选的字节差选择，不使用 GT、PSNR 或 IoU。离散质量点无法保证完全相等；所有实际误差保留，超过预先采用的2%容差的点不按严格匹配结果解释。M_PRE 为预训练参考曲线，其相同 q 并非匹配预算。", "", "全局 / 背景 PSNR 代价定义为 M_RD−M_ROI，正数表示退化，负数表示改善。鱼区和背景 PSNR 只在带 GT 的帧累计 RGB 误差；全局 PSNR 评价整段视频。IoU、召回率按每段视频全部 GT 帧累计，未仅汇总全部鱼像素给出一个项目总分。"]
    lines += ["", "最终原图与重建计分均采用 FP32、关闭 TF32 和确定性设置。首次计分漏调用统一配置，初次结果另存为 first_eval_default_precision；最终24点已全部重算，评价权重和阈值未修改。"]
    for video in ("video96","video108"):
        lines += ["",f"### {video}","", "| 档位 / 方法 | RD / ROI 完整字节 | 大小差 | RD / ROI 鱼 IoU | IoU变化 / 点 | 全局PSNR代价 / dB | 背景PSNR代价 / dB |", "|---|---:|---:|---:|---:|---:|---:|"]
        for c in compare:
            if c["video"]==video:
                mark="" if c["matched_within_2_percent"] else " †"
                lines.append(f'| {c["budget"]} / {c["branch"]}{mark} | {c["M_RD_file_bytes"]:,} / {c["ROI_file_bytes"]:,} | {c["file_difference_percent"]:+.3f}% | {c["M_RD_fish_IoU_percent"]:.2f}% / {c["ROI_fish_IoU_percent"]:.2f}% | {c["delta_fish_IoU_points"]:+.3f} | {c["global_PSNR_cost_dB"]:+.3f} | {c["background_PSNR_cost_dB"]:+.3f} |')
    lines += ["", "† 超过2%预算容差。鱼区 PSNR、召回率及全部质量点在配套 CSV 中完整列出。", "", f'![完整文件预算与任务质量]({(OUT/"完整文件预算与任务质量.png").as_posix()})', "", "## 实验判断与下一步", ""]
    matched=[c for c in compare if c["matched_within_2_percent"]]
    task_positive=sum(c["delta_fish_IoU_points"]>0 for c in matched);pixel_positive=sum(c["delta_fish_PSNR_dB"]>0 for c in matched)
    lines += [f'满足2%文件容差的 ROI 对照共 {len(matched)} 个，其中鱼区 PSNR 改善 {pixel_positive} 个、鱼类 IoU 改善 {task_positive} 个。正负结果均保留；这些是小样本观测，没有统计显著性或跨场景普遍性的证明。']
    for branch in ("M_ROI4","M_ROI16"):
        group=[c for c in matched if c["branch"]==branch]
        if group:lines.append(f'{branch}：匹配点平均鱼区 PSNR 变化 {np.mean([c["delta_fish_PSNR_dB"] for c in group]):+.3f} dB，平均 IoU 变化 {np.mean([c["delta_fish_IoU_points"] for c in group]):+.3f} 个百分点。均值仅辅助阅读，主要证据是各视频、各预算结果。')
    official_raw=next(r for r in measured["original"] if r["video"]=="video108")
    if official_raw["fish_iou"]<.2:
        lines.append("官方验证原图的鱼识别表现仍偏低，当前评价器的跨场景可靠性是重要限制。因此本轮不能仅凭局部 PSNR 或少数 IoU 提升宣布已证明实际语义任务收益。")
    elif task_positive<len(matched):lines.append("任务变化存在正负混合，尚未满足多个独立场景、多个预算下稳定提高任务效果的验收。")
    else:lines.append("这两段视频的匹配点观察到一致的任务改善，但场景数量、GT 覆盖及一个随机种子仍限制结论；下一轮必须在更多拍摄组复验。")
    temporal=json.loads((RUN/"temporal_degradation_diagnostic.json").read_text("utf-8"))
    lines += ["", "**更应优先处理的负结果是普通微调自身的退化。**", "", "| 视频 / 档位 | 预训练 / 普通微调字节 | 大小差 | 预训练 / 普通微调全局 PSNR |", "|---|---:|---:|---:|"]
    for r in temporal["matched_pretrained_comparisons"]:lines.append(f'| {r["video"]} / q{r["anchor"]} | {r["M_PRE_bytes"]:,} / {r["M_RD_bytes"]:,} | {r["file_difference_percent"]:+.3f}% | {r["M_PRE_PSNR"]:.3f} / {r["M_RD_PSNR"]:.3f} dB |')
    pos=temporal["video96_q8_by_GOP_position"]
    lines += ["", f'video96 q8 的 I 帧一致；普通微调相对预训练的分组 PSNR 差从首个 P 帧的 {pos[1]["delta_group_PSNR_dB"]:.3f} dB，扩大到 GOP 第31位置的 {pos[31]["delta_group_PSNR_dB"]:.3f} dB。这与时序误差累积相符，但不能证明它是唯一原因。当前训练只覆盖一个可微 P 帧、冻结 I 帧重建参考和 fa_idx=1，完整编码还使用 fa_idx=0/2 与连续更新的 P 参考特征。', "", f'![GOP内质量退化]({(OUT/"GOP内重建质量退化.png").as_posix()})', "", "因此，ROI 相对一个已经退化的 M_RD 的局部提升不能直接作为方法有效的证明。下一轮首先修复普通微调的完整视频 RD 性能。"]
    probe=json.loads((RUN/"multiframe_probe.json").read_text("utf-8"))
    for p in probe["rows"]:lines.append(f'{p["frames"]} 帧（{p["gradient_P_frames"]} 个可微 P 帧）反向传播检查通过，分配峰值 {p["peak_allocated_bytes"]/2**30:.3f} GiB，DPB 的帧和特征均保留梯度；没有优化器更新。这是显存/梯度可行性，不是多帧训练收益。')
    lines += ["", "下一轮先从相同初始化开展4帧普通 RD 训练，覆盖真实 fa 索引及更新后的 P 参考特征，检查完整 GOP 的退化，并扩充质量参数采样。普通微调恢复到接近预训练性能后，再做同条件4帧 ROI 对照。并行补更多含鱼连续训练片段、拍摄组信息和官方验证目标覆盖，提高分割器对原图及压缩图的稳定性；暂缓可微任务损失。", "", "本轮没有更新 I 帧模型；继续使用阶段 A 的同一 DMCI 镜像权重。完整文件共享模型权重假定预部署，若实际发送微调权重，应另算模型更新成本。没有做 H.265 新对照、跨设备解码或水声信道试验，不据此宣称胜过强传统基线或完成通信链路。", "", "## 结果位置", "", f'- [原始实验目录]({RUN.as_posix()})', f'- [实测对照 CSV]({(OUT/"实测对照.csv").as_posix()})', f'- [全部质量点 CSV]({(OUT/"完整质量点.csv").as_posix()})', f'- [逐步训练梯度与损失]({(OUT/"训练梯度与损失.csv").as_posix()})', f'- [审计记录]({(OUT/"实验审计.json").as_posix()})', ""]
    report=OUT/"阶段C_小规模ROI对照实测_2026-10-13.md";report.write_text("\n".join(lines),"utf-8")
    package=OUT/"配置日志代码与权重.zip"
    with zipfile.ZipFile(package,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(Path(__file__).parent.glob("*.py")):z.write(p,"scripts/"+p.name)
        for p in sorted(Path(__file__).parent.glob("*.log")):z.write(p,"logs/"+p.name)
        for p in RUN.rglob("*"):
            if p.is_file() and (p.suffix in (".json",".jsonl") or (p.suffix in (".ckpt",".pt") and ("training" in p.parts or "segformer_adaptation" in p.parts or "segformer_adaptation_preserve8" in p.parts))):
                if "training_cache" not in p.parts:z.write(p,"experiment/"+p.relative_to(RUN).as_posix())
        for p in (DATA/"protocol.json",DATA/"video_list.txt",OUT/"数据核验统计.json",OUT/"实验审计.json"):z.write(p,"provenance/"+p.name)
        for row in rows:z.write(Path(row["run"])/"video.mlvcv",f'bitstreams/{row["video"]}/{row["branch"]}/anchor{row["anchor"]}_q{row["q_index"]}.mlvcv')
    with zipfile.ZipFile(package) as z:
        if z.testzip() is not None:raise AssertionError("Artifact ZIP failed integrity check")
    print("REPORT_OK",report,"ZIP_BYTES",package.stat().st_size,"MATCHED",len(matched),"IOU_POSITIVE",task_positive,"ROI_PSNR_POSITIVE",pixel_positive,flush=True)


if __name__=="__main__":main()
