"""Report the completed controls without running new training or inference."""
from __future__ import annotations

from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

from control_common import OUT, RUN, PROJECT, DIRECTION, OLD, read, sha256, write_json, frozen

SCRIPT = Path(__file__).resolve().parent
FIGURES = OUT / "figures"


def link(path: Path, title: str) -> str:
    return f"[{title}]({path.as_posix()})"


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def collect():
    protocol = read(RUN / "protocol.json")
    assert sha256(DIRECTION) == protocol["direction_sha256"]
    metrics = {x["group"]: x for x in read(OUT / "实际码流与GT任务评价.json")}
    geometry = {x["window"]: x for x in read(OUT / "逐帧与小目标覆盖复核.json")}
    controls = {}
    for group in ("A", "B", "H265"):
        selected = read(RUN / group / "selection.json")
        receipt = read(RUN / group / "network/end_to_end.json")
        decoder = read(RUN / group / "network/receiver/decode.json")
        audit = read(RUN / group / "network/receiver/receiver_read_audit.json")
        streams = read(RUN / group / "stream_audit.json")
        packet = Path(selected["file"])
        received = RUN / group / "network/receiver" / ("received.mp4" if group == "H265" else "message.scv")
        assert sha256(packet) == sha256(received) == selected["sha256"] == decoder["file_sha256"]
        assert packet.stat().st_size == sum(selected["parts"].values()) == receipt["file_bytes"]
        assert receipt["success"] and receipt["separate_sender_receiver_processes"] and receipt["packet_sha256_equal"]
        assert decoder["frames"] == 50 and len(list((received.parent / "rgb").glob("*.png"))) == 50
        assert audit["guard_installed"] and all(row["allowed"] for row in audit["python_reads"])
        assert all(Path(row["path"]).resolve().is_relative_to(received.parent.resolve()) for row in audit["python_reads"])
        if group != "H265":
            with zipfile.ZipFile(packet) as archive:
                assert archive.testzip() is None
                manifest = json.loads(archive.read("manifest.json"))
                for name, record in manifest["files"].items():
                    assert len(archive.read(name)) == record["bytes"]
                assert len(manifest["patches"]) == geometry[32 if group == "A" else 4]["segments"]
        counts = Counter()
        for stream in streams["streams"]:
            counts.update(stream["frames_by_type"])
        controls[group] = {"selection": selected, "receipt": receipt, "decoder": decoder,
                           "stream_count": len(streams["streams"]), "coded_frames_by_type": dict(counts)}
    assert metrics["original"]["gt"]["tp"] == 471
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "Pillow", "scipy", "matplotlib")}}
    write_json(OUT / "交付核验.json", {"direction_sha256_unchanged": True, "groups": list(controls),
               "each_received_file_sha256_matches": True, "reconstructed_frames_each": 50,
               "zip_member_sizes_and_crc_validated": True, "receiver_python_media_reads_stay_in_output": True,
               "guard_is_not_os_sandbox": True, "environment": environment})
    return protocol, metrics, geometry, controls


def figures(geometry):
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    windows = [4, 8, 16, 32]
    x = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    ax = axes[0]
    ax.bar(x - .25, [100 * geometry[w]["mean_rectangle_fraction"] for w in windows], .25, label="ROI image area")
    ax.bar(x, [100 * geometry[w]["pooled_gt_pixel_coverage"] for w in windows], .25, label="Pooled fish-pixel coverage")
    ax.bar(x + .25, [100 * geometry[w]["positive_frame_hit_fraction"] for w in windows], .25, label="Positive GT frames hit")
    ax.set_xticks(x, [str(w) for w in windows]); ax.set_xlabel("Temporal window (frames)")
    ax.set_ylabel("Percent"); ax.set_ylim(0, 110); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Geometry: 4 GT frames, 3 contain fish")
    ax = axes[1]
    sizes = [read(RUN / g / "selection.json")["fixed_crf32_diagnostic"]["file_bytes"] for g in ("A", "B")]
    bars = ax.bar(["A: window 32", "B: window 4"], np.array(sizes) / 1000, color=["#386cb0", "#f28e2b"])
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f"{size:,} B", ha="center")
    ax.set_ylim(0, 380); ax.set_ylabel("Complete packet (decimal KB)")
    ax.set_title("Actual complete files at ROI CRF 32")
    fig.suptitle("Stage D diagnosis: smaller ROI reduces bytes, but misses small targets")
    fig.savefig(FIGURES / "coverage_and_complete_bytes.png", dpi=160)
    fig.savefig(FIGURES / "coverage_and_complete_bytes.svg")
    plt.close(fig)

    # Use exactly the same GT-centered view and unaltered RGB for every column.
    # The yellow contour is publisher GT, never a received detection prediction.
    _, _, _, _, labels = frozen()
    crops = []
    for label in labels:
        gt = np.asarray(Image.open(label["fish_mask"])) > 0
        components, count = ndimage.label(gt, structure=np.ones((3, 3), dtype=np.uint8))
        for index in range(1, count + 1):
            yy, xx = np.where(components == index)
            left = max(0, min(1280 - 128, int((xx.min() + xx.max()) / 2) - 64))
            top = max(0, min(720 - 96, int((yy.min() + yy.max()) / 2) - 48))
            crops.append((label["frame_index"], index, int(len(xx)), left, top, gt))
    fig, axes = plt.subplots(len(crops), 4, figsize=(12, 8.5), layout="constrained")
    titles = ["Original", "A: 32-frame ROI", "B: 4-frame ROI", "H.265: B frames"]
    directories = [OLD / "sender/private_frames", *[RUN / g / "network/receiver/rgb" for g in ("A", "B", "H265")]]
    crop_audit = []
    for row, (frame, component, area, left, top, gt) in enumerate(crops):
        for column, directory in enumerate(directories):
            rgb = np.asarray(Image.open(directory / f"{frame - 420:05d}.png").convert("RGB"))
            ax = axes[row, column]
            ax.imshow(rgb[top:top + 96, left:left + 128], interpolation="nearest")
            ax.contour(gt[top:top + 96, left:left + 128], levels=[.5], colors=["#ffe066"], linewidths=.7)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0: ax.set_title(titles[column])
            if column == 0: ax.set_ylabel(f"Frame {frame}\nGT component {component}: {area} px", fontsize=9)
        crop_audit.append({"source_frame": frame, "gt_component": component, "pixels": area, "view_box": [left, top, 128, 96]})
    fig.suptitle("Same decoded-RGB crop; yellow contour = publisher GT\nKnown training clip; sparse GT; visual diagnostic only", fontsize=12)
    fig.savefig(FIGURES / "small_target_same_view.png", dpi=150)
    plt.close(fig)
    write_json(FIGURES / "crop_coordinates.json", crop_audit)


def report(protocol, metrics, geometry, controls):
    annotation = read(OUT / "连续人工ROI核对/annotations.json")
    known = sum(x["status"] in {"publisher_verified", "human_reviewed"} for x in annotation["frames"])
    rows = []
    labels = {"A": "A：预测ROI，32帧", "B": "B：相同预测ROI，4帧", "H265": "普通H.265，允许B帧"}
    for group in ("A", "B", "H265"):
        c, m = controls[group], metrics[group]
        s = c["selection"]
        rows.append(f"| {labels[group]} | {s['file_bytes']:,} | {s['relative_budget_difference']:+.3%} | {m['gt']['iou'] * 100:.2f} | {m['gt']['recall'] * 100:.2f} | {m['gt']['precision'] * 100:.2f} | {m['global_rgb_psnr_db']:.3f} | {m['gt_fish_rgb_psnr_db']:.3f} | {m['gt_background_rgb_psnr_db']:.3f} |")
    geo_rows = []
    for w in (32, 16, 8, 4):
        g = geometry[w]
        geo_rows.append(f"| {w} | {pct(g['mean_rectangle_fraction'])} | {pct(g['pooled_gt_pixel_coverage'])} | {round(g['positive_frame_hit_fraction'] * 3)}/3 | {pct(g['positive_frame_mean_coverage'])} | {pct(g['tiny_component_summary']['mean_coverage'])} |")
    parts_rows, network_rows = [], []
    for group in ("A", "B", "H265"):
        c = controls[group]; s = c["selection"]; p = s["parts"]; ft = c["coded_frames_by_type"]
        parts_rows.append(f"| {group} | {p.get('context', '—')} | {p.get('roi', '—')} | {p.get('semantic', '—')} | {p.get('container', '—')} | {c['stream_count']} | {ft.get('I', 0)}/{ft.get('P', 0)}/{ft.get('B', 0)} |")
        network_rows.append(f"| {group} | {c['receipt']['application_transmit_bytes']:,} | {c['receipt']['transfer_seconds']:.3f} | {c['decoder']['decode_and_preview_seconds']:.3f} | 50 | 一致 |")
    counts_rows = [f"| {name} | {metrics[name]['gt']['tp']} | {metrics[name]['gt']['fp']} | {metrics[name]['gt']['fn']} | {metrics[name]['positive_frame_mean_iou'] * 100:.3f}% | {metrics[name]['positive_frame_mean_recall'] * 100:.3f}% |" for name in ("original", "A", "B", "H265")]
    afixed, bfixed = [controls[g]["selection"]["fixed_crf32_diagnostic"]["file_bytes"] for g in ("A", "B")]
    text = f"""# 阶段D：ROI窗口受控实测与下一步判断

日期：2026-10-15。本轮先保存《现状总结与后续方向》，冻结协议和指纹，再执行实验。未新增阶段，未训练或更换分割器、神经编解码器。

## 1. 结论与现状

项目主目标继续固定为：**接收端恢复可观看的视频，语义用于保护关键目标。** 信息模式保留作辅助能力。

本轮得到的是明确取舍与负结果：4帧窗口使相同ROI CRF32下的完整文件由{afixed:,}B降至{bfixed:,}B，减少{100 * (1 - bfixed / afixed):.2f}%；但在约100KB相同完整文件预算下，4帧方案的鱼类IoU、召回率、GT鱼区PSNR和全局PSNR均低于32帧方案。不能将“ROI变小、字节减少”当成“目标保护成功”。

32帧方案对普通H.265的IoU有微弱优势，但真实鱼像素命中数更少，召回率和鱼区PSNR也略低，整体画质下降。因此，本轮**没有证明语义方案在约100KB预算下更好地保留鱼类**。不能凭单项IoU宣称优于传统编码。

阶段D的完整收发与恢复仍可运行；可靠的关键目标保护尚待证明。阶段C仍保留为研究分支，普通RD长GOP退化未修复，本轮没有启动新训练。

## 2. 对外部评价的复核

认可其主线与优先级。69.12%确实不是三次有鱼观测中多数被保护，而是435、450帧全部漏保护，仅465帧的985像素被覆盖。GT一共1425像素，最大含鱼帧主导了像素汇总。

| 窗口/帧 | 平均ROI画面面积 | 汇总GT鱼像素覆盖 | 含鱼GT帧命中 | 含鱼帧覆盖的等权均值 | ≤256像素连通域平均覆盖 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(geo_rows)}

420帧无鱼；435、450、465帧鱼像素分别为255、185、985。465帧含184、801像素两个连通域。≤256像素组包含255、185、184三个连通域；连通域不是经过确认的鱼实例。4帧窗口在该组仅覆盖184像素连通域，另外两个未覆盖。≥95%覆盖阈值及最小帧覆盖也保存在逐帧JSON中。

这里的“覆盖/命中”只表示GT与高质量ROI层的几何重叠，不代表目标实际恢复或识别成功。原始未压缩视频的固定分割器也只在3张含鱼GT中的1张命中鱼；语义提取缺口在编码之前已经存在。

![ROI几何与完整文件](figures/coverage_and_complete_bytes.png)

## 3. 锁定的实验条件

- 输入：video131原始420–469连续50帧，1280×720，名义25fps；来源时间戳不可得。视频曾用于阶段C适配，是开发诊断片段，不是独立测试集。
- 固定鱼类SegFormer权重、512分块/384步长、阈值0.9、64像素筛选、外扩16像素。A/B直接复用同一份已冻结预测，未用GT选择阈值。
- A为32帧单矩形，B为4帧单矩形。背景都为640×360、CRF42；ROI为x265 medium、P-only、GOP32、zerolatency。首次只改变窗口，不同时增加分块、跟踪或检测器改动。
- H.265基线为原分辨率普通完整MP4，medium、允许最多4个B帧、GOP32。它是离线视频比较，编码延迟约束与分层方案不同；各方案颜色配置均为BT709 limited YUV420。
- 每组预算100,000B十进制，允许偏差≤2%，按完整文件字节选CRF，不看任务分数。实际三组均距目标小于1%；最大组间差约1.28%。不做无效字节填充。
- 分层方案计背景、所有ROI流、语义gzip、manifest和ZIP；普通H.265计完整MP4。传输协议另计实际应用层字节。文件预算不含TCP/IP底层报文，未抓包计算物理链路成本。
- 接收端只靠收到的文件独立解码；任务评价另行对接收端RGB重新运行固定模型，不用随包发送的框计分。

原32帧CRF32流及公共背景流复用既有码流，并验证输入和配置；加入本轮实验manifest后A完整文件为333,726B，比阶段D旧包多138B。候选编码耗时排除了缓存识别、公共背景等步骤，**不能解释为新的端到端耗时或实时性能**。

## 4. 约100KB完整文件结果

IoU、召回率、精确率单位为%；PSNR单位dB。全局PSNR在全部50帧汇总RGB误差后计算；鱼区/背景PSNR及任务指标只在4张发布者GT上计算。其余46帧未当作已知背景。

| 方案 | 完整字节 | 距100,000B | 鱼IoU | 鱼召回 | 鱼精确率 | 全局PSNR | GT鱼区PSNR | GT帧背景PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

原始未压缩输入：鱼IoU26.196%、召回33.053%、精确率55.806%。三种约100KB方案均大幅降低任务表现，当前强压缩点没有达到可靠关键目标保护。

| 输入/方案 | TP鱼像素 | FP像素 | FN鱼像素 | 含鱼帧IoU等权均值 | 含鱼帧召回等权均值 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(counts_rows)}

A相对H.265的汇总IoU增加约{100 * (metrics['A']['gt']['iou'] - metrics['H265']['gt']['iou']):.3f}个百分点，却少命中5个真实鱼像素；主要变化是FP从14,989降至7,634。因此这项IoU差异不能单独解释为目标细节保留更好。B相对A的鱼IoU下降约{100 * (metrics['A']['gt']['iou'] - metrics['B']['gt']['iou']):.3f}个百分点，全局PSNR下降约{metrics['A']['global_rgb_psnr_db'] - metrics['B']['global_rgb_psnr_db']:.3f}dB。

所有输入/方案在435、450帧的TP均为0；全部3个≤256像素GT连通域的分割召回均为0。465帧801像素连通域的召回依次为原图58.80%、A17.98%、B6.74%、H.26518.60%。这是当前小目标和检测可靠性缺口的直接证据，样本量不足以推断普遍性能。

![固定GT位置的同视野局部比较](figures/small_target_same_view.png)

图中使用同一个GT中心裁剪，保持原始RGB值，以最近邻显示；黄色轮廓为发布者GT，不是分割器预测。ROI框标注也不会生成新像素GT。

## 5. 真实字节开销和收发

| 组 | 背景B | ROI流合计B | 语义gzip B | manifest/ZIP B | 视频流数 | 总I/P/B帧 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(parts_rows)}

H.265的上述分层开销列不适用，其完整MP4为99,520B。A/B的ROI MP4内已经包含各自MP4容器、编码头等开销；manifest/ZIP列仅为外层封装。区域位置和时间索引已经计入manifest，不能忽略。

短窗使ROI段数由2增至11、总I帧由4增至13，外层封装约从2KB增至4.45KB。即使面积大幅缩小，也仍有明显的初始化和封装成本。不能把ROI层与背景层的总编码帧数当作新增源帧；各组恢复源视频均为50帧。

| 组 | 实际应用层发送B | 20kbps传输秒 | 恢复与预览生成秒 | 恢复帧数 | 收发SHA256 |
|---|---:|---:|---:|---:|---|
{chr(10).join(network_rows)}

三组均由独立接收进程通过真实本机TCP收到文件，哈希一致后完整恢复。读取审计中Python媒体读取位于各自接收目录，FFmpeg输入命令留存；这是程序级审计，不是操作系统沙箱或跨机器物理链路验收。

接收耗时包含PNG输出和预览生成；A/B还生成叠加展示，因此不同方案的该耗时不能作为纯解码速度排名。约2秒名义视频需要约40秒发送，本轮是批次传输，未实现20kbps实时视频。发送端识别时间不包含在本轮传输秒数里。

## 6. 人工ROI对照与下一步

四组实验目前只完成A/B。C（人工ROI×32帧）与D（人工ROI×4帧）暂未运行；本报告生成时连续标注核对状态为{known}/50帧，余{50 - known}帧未核对。未用预测、插值或空框伪造连续GT。

已准备 {link(OUT / '连续人工ROI核对/标注须知.md', '标注说明')} 与 [本机逐帧标注页面](http://127.0.0.1:8777/)。页面支持放大、画框、明确确认无鱼、保留不确定、保存及版本记录。4张已有发布者标注保留来源。人工框只用于ROI控制，IoU仍只用已发布像素掩码。保存后不会自动启动耗时实验。

下一步顺序不改变：

1. 补齐这50帧连续的真实人工/发布者ROI核对，再沿用冻结协议执行C/D和普通H.265对照，判断识别缺口和区域组织的作用。人工ROI是诊断条件，不是部署成绩或严格性能上界。
2. 根据四组结果，优先调整有证据支持的瓶颈；若人工ROI下短窗仍不佳，检查小流封装、I帧、背景预算和区域编码；若人工ROI改善而预测组失败，优先解决语义提取遗漏。当前不能预先认定唯一瓶颈。
3. 只有出现可重复收益，再扩大预算档位和真正独立连续视频，固定模型后评价，保留负结果。已反复使用的131、001、027、054都不冒充未见最终测试。
4. 阶段C按零更新一致性→少量多帧普通RD更新→完整GOP质量复核推进；普通RD稳定之前不新增ROI/任务损失训练。该研究分支不阻塞阶段D工程交付。

这轮不追加新网络，也不因结果不好修改阈值、换预算或挑帧重新包装为成功。结论范围仅限一个已见训练片段、一档预算、4张稀疏GT。

## 7. 证据与复现入口

- {link(DIRECTION, '先行保存的现状与方向文档')}；SHA256 `{protocol['direction_sha256']}`。
- {link(OUT / '实验协议.json', '冻结协议与输入指纹')}；{link(OUT / '逐帧与小目标覆盖复核.json', '逐帧/连通域覆盖')}。
- {link(OUT / '实际码流与GT任务评价.json', '完整GT任务结果')}；{link(OUT / '完整文件预算选择.json', '完整文件候选及选择')}；{link(OUT / '交付核验.json', '收发和环境核验')}。
- 码流、所有候选、完整命令与收发日志目录：{link(RUN, '本轮原始运行目录')}。每组`selection.json`指明最终文件，每组`network/receiver/rgb`为评价用恢复RGB。
- 复现脚本：{link(SCRIPT / 'run_controls.py', 'run_controls.py')}；共享准备：{link(SCRIPT / 'control_common.py', 'control_common.py')}。使用项目既有venv；依赖已冻结的原始帧、模型和公共背景，不是脱离工程即可训练的独立数据包。

接收端预览是恢复RGB后再次编码生成的展示文件，未参与传输字节或质量评价。真实传输包和显示预览在证据包中分别命名。
"""
    target = OUT / "阶段D_ROI窗口对照实测_2026-10-15.md"
    target.write_text(text, encoding="utf-8")
    # The project-docs copy lives outside this artifact folder.
    project_copy = text.replace("(figures/", "(" + FIGURES.as_posix() + "/")
    (PROJECT / "docs/phase_d_roi_control_20261015.md").write_text(project_copy, encoding="utf-8")
    return target


def bundle(report_file, controls):
    archive_path = OUT / "阶段D_ROI对照_报告与证据.zip"
    entries = [(DIRECTION, "现状总结与后续方向_2026-10-15.md"), (report_file, report_file.name)]
    # The hash of the final ZIP stays outside that ZIP (avoid a stale self hash).
    entries += [(p, p.name) for p in OUT.glob("*.json") if p.name != "证据包指纹.json"]
    entries += [(p, "figures/" + p.name) for p in FIGURES.iterdir() if p.is_file()]
    entries += [(p, "code/" + p.name) for p in SCRIPT.iterdir() if p.suffix in {".py", ".html"}]
    for name in ("controls.log", "annotation_server.log"):
        entries.append((SCRIPT / name, "logs/" + name))
    if (OUT / "人工ROI核对界面.jpg").exists():
        entries.append((OUT / "人工ROI核对界面.jpg", "人工ROI核对界面.jpg"))
    for group, c in controls.items():
        packet = Path(c["selection"]["file"])
        entries.append((packet, f"actual_transmitted/{group}_{packet.name}"))
        group_root = RUN / group
        for name in ("selection.json", "search.json", "stream_audit.json"):
            entries.append((group_root / name, f"evidence/{group}/{name}"))
        for name in ("end_to_end.json", "transfer.csv"):
            entries.append((group_root / "network" / name, f"evidence/{group}/network/{name}"))
        for name in ("decode.json", "receiver_read_audit.json"):
            entries.append((group_root / "network/receiver" / name, f"evidence/{group}/receiver/{name}"))
        entries.append((group_root / "network/receiver/reconstructed.mp4", f"post_receive_preview/{group}.mp4"))
    notes = "本包含实际传输文件、结果JSON、日志、脚本和恢复后展示视频。展示视频不计入传输文件。人工标注未完成，C/D没有结果；没有新训练。完整原始数据和模型仍在既有工程，不包含在此压缩包。MD使用的绝对路径供本机定位。\n"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("阅读说明.txt", notes)
        for path, name in entries:
            if path.is_file(): archive.write(path, name)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        assert len([n for n in names if n.startswith("actual_transmitted/")]) == 3
    write_json(OUT / "证据包指纹.json", {"file": str(archive_path), "bytes": archive_path.stat().st_size,
               "sha256": sha256(archive_path), "zip_crc_valid": True})
    return archive_path


if __name__ == "__main__":
    p, m, g, c = collect()
    figures(g)
    document = report(p, m, g, c)
    archive = bundle(document, c)
    print(json.dumps({"report": str(document), "archive": str(archive), "bytes": archive.stat().st_size}, ensure_ascii=False))
