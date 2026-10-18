"""Archive completed B2 results and render an evidence-based development report."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import shutil
import sys
from b2_common import git, now, read, require, sha, write


def record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--network", default="network")
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    network, sender, evaluation = run / args.network, run / "sender", run / "evaluation"
    summary = read(network / "summary.json")
    require(summary["status"] == "B2_three_rate_engineering_passed" and summary["all_rates_identical_enhanced_RGB"], "Network not complete")
    require([r["rate_kbps"] for r in summary["results"]] == [20, 30, 40], "Missing rate")
    evaluated = read(evaluation / "completion.json")
    require(evaluated["status"] == "B2_diagnostics_complete" and evaluated["enhanced_rgb_sequence_sha256"] == summary["results"][0]["enhanced_rgb_sequence_sha256"], "Evaluation uses different RGB")
    require(not args.report.exists(), "Report already exists")
    output.mkdir(parents=True, exist_ok=False)

    def result_directory(rate):
        directory = Path(summary.get("result_directories", {}).get(str(rate), network / f"rate_{rate}kbps")).resolve()
        require(directory.is_relative_to(run), "Accepted result outside experiment root")
        return directory

    def copy(source, destination):
        destination = output / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    for name in ("encode.json", "config.json", "code_hashes.json", "source_audit.json", "manifest.json", "roi_plan.json", "fish_prediction_summary.json", "diver_prediction_summary.json", "commands.json"):
        copy(sender / name, Path("sender") / name)
    for name in ("summary.json", "environment.json", "commands.json", "progress.json"):
        copy(network / name, Path("network") / name)
    for name in ("metrics.json", "prediction_coverage.json", "per_frame_metrics.json", "per_frame_metrics.csv", "viewing.json", "completion.json", "commands.json"):
        copy(evaluation / name, Path("evaluation") / name)
    copy(result_directory(40) / "receiver/target_information.json", Path("sender_predictions_received.json"))
    write(output / "accepted_result_directories.json", {str(rate): str(result_directory(rate)) for rate in (20, 30, 40)})
    artifacts = [record(sender / "message.scv")]
    for result in summary["results"]:
        tag = f"rate_{result['rate_kbps']}kbps"
        source = result_directory(result["rate_kbps"])
        require(read(source / "result.json") == result and result["decoded_frames"] == 250 and result["outside_roi_unchanged"]
                and result["composed_roi_matches_decoded_patch"] and result["playback_verified"] and result["source_or_sender_cache_reads"] == 0, "Incomplete rate evidence")
        for name in ("result.json", "transfer.json", "transfer.csv", "send_schedule.csv", "pacing_verification.json", "frame_verification.json", "playback_probe.json", "receiver_process.log"):
            copy(source / name, Path("network") / tag / name)
        for name in ("decode.json", "completion.json", "receive.json", "deployment.json", "commands.json", "decode_started.json"):
            copy(source / "receiver" / name, Path("network") / tag / name)
        for name in ("decode_allowed.json", "decode_input.json"):
            if (source / "receiver" / name).exists():
                copy(source / "receiver" / name, Path("network") / tag / name)
        if (source / "receipt_provenance.json").exists():
            copy(source / "receipt_provenance.json", Path("network") / tag / "receipt_provenance.json")
        copy(source / "receiver/base_decoded/decode.json", Path("network") / tag / "base_decode.json")
        audit_path = source / "receiver/receiver_read_audit.json"
        audit = read(audit_path)
        counts = Counter()
        for row in audit["python_reads"]:
            counts[row["category"]] += row["count"]
        write(output / "network" / tag / "receiver_audit_summary.json", {"scope": audit["scope"], "python_open_attempts_by_category": dict(counts),
              "shared_model_allowlist": audit["shared_model_allowlist"], "denied_reads": audit["denied_reads"],
              "external_source_or_sender_cache_reads": audit["external_source_or_sender_cache_reads"], "subprocesses": audit["subprocesses"],
              "full_local_audit": record(audit_path), "receiver_pid": result["receiver_pid"], "evaluator_pid": result["evaluator_pid"]})
        for name in ("message.scv", "enhanced_preview.mp4", "base_decoded/reconstruction_preview.mp4", "receiver_read_audit.json"):
            artifacts.append(record(source / "receiver" / name))
    viewing = read(evaluation / "viewing.json")
    artifacts += [record(video["path"]) for video in viewing["videos"]]
    artifacts += [record(evaluation / name) for name in ("固定关键帧对比.jpg", "固定鱼纹局部对比.jpg")]
    for name in ("sender_stdout.log", "sender_stderr.log", "network_stdout.log", "network_stderr.log", "evaluation_stdout.log", "evaluation_stderr.log"):
        copy(run / name, Path("process_logs") / name)
    if "recovery" in summary:
        for name in ("recovery_stdout.log", "recovery_stderr.log"):
            copy(run / name, Path("process_logs") / name)
        original = Path(summary["recovery"]["original_network"])
        for name in ("failure.json", "environment.json", "commands.json"):
            copy(original / name, Path("failed_network_attempt") / name)
        for name in ("failure.json", "commands.json"):
            copy(original / "rate_40kbps/receiver" / name, Path("failed_network_attempt/receiver_40kbps") / name)
        failed_audit_path = original / "rate_40kbps/receiver/receiver_read_audit.json"
        failed_audit = read(failed_audit_path)
        write(output / "failed_network_attempt/receiver_40kbps/audit_summary.json", {
              "scope": failed_audit["scope"], "denied_reads": failed_audit["denied_reads"],
              "external_source_or_sender_cache_reads": failed_audit["external_source_or_sender_cache_reads"],
              "subprocesses": failed_audit["subprocesses"], "full_local_audit": record(failed_audit_path)})
        copy(original / "rate_40kbps/receiver_process.log", Path("failed_network_attempt/receiver_40kbps/process.log"))
        previous_recovery = summary["recovery"].get("previous_recovery")
        if previous_recovery:
            previous = Path(previous_recovery)
            for name in ("failure.json", "environment.json", "commands.json", "progress.json"):
                copy(previous / name, Path("previous_recovery") / name)
            for path in sorted(previous.glob("rate_*kbps/receiver_process.log")):
                if not (path.parent / "result.json").exists():
                    for evidence_path in sorted(path.parent.rglob("*")):
                        if evidence_path.is_file() and evidence_path.suffix in (".json", ".log", ".txt"):
                            copy(evidence_path, Path("previous_recovery") / evidence_path.relative_to(previous))
            for name in ("recovery2_stdout.log", "recovery2_stderr.log"):
                copy(run / name, Path("process_logs") / name)
    write(output / "private_mask_checksums.json", [{"relative_path": str(p.relative_to(run)), "bytes": p.stat().st_size, "sha256": sha(p)}
          for p in sorted((sender / "private_masks").rglob("*.png"))])
    write(output / "local_artifacts.json", artifacts)
    write(output / "archive_provenance.json", {"created_at": now(), "argv": [sys.executable, *sys.argv],
          "collector_script_sha256": sha(__file__), "repository_commit_at_collection": git("rev-parse", "HEAD"),
          "working_tree_clean_at_collection": not bool(git("status", "--porcelain")),
          "accepted_network_directory": str(network), "network_summary_sha256": sha(network / "summary.json"),
          "evaluation_completion_sha256": sha(evaluation / "completion.json")})
    write(output / "evidence_checksums.json", [{"relative_path": str(p.relative_to(output)), "bytes": p.stat().st_size, "sha256": sha(p)}
          for p in sorted(output.rglob("*")) if p.is_file()])
    metrics = read(evaluation / "metrics.json")
    encoded = read(sender / "encode.json")
    methods = {r["method"]: r for r in metrics["methods"]}
    before, after = methods["GLC_base"], methods["GLC_auto_ROI"]
    numeric = lambda value: "无预测像素／不适用" if value is None else f"{value:.4f}"
    metric_keys = {"全画面": "full", "源端原始鱼类预测掩码": "fish_raw_source_prediction", "过滤后的鱼类预测掩码": "fish_kept_source_prediction",
                   "A既有固定鱼纹裁剪（帧210–249）": "fixed_stripe_crop", "传输ROI矩形内": "inside_transmitted_ROI", "传输ROI矩形外": "outside_transmitted_ROI"}
    quality_rows = "\n".join(f"| {label} | {numeric(before['rgb_psnr_db'][key])} | {numeric(after['rgb_psnr_db'][key])} |"
                             for label, key in metric_keys.items())
    transfer_rows = "\n".join(f"| {r['rate_kbps']} | {r['transfer']['application_bytes']:,} | {r['transfer']['minimum_application_seconds']:.4f} | {r['transfer']['elapsed_seconds']:.4f} | {r['transfer']['average_application_kbps']:.6f} | {r['decode_and_preview_seconds']:.2f} |"
                              for r in summary["results"])
    byte_labels = {"glc_file_bytes": "GLC完整基础码流", "roi_h265_bytes": "H.265 ROI补充视频", "semantic_message_gzip_bytes": "压缩语义消息",
                   "roi_position_time_value_bytes": "ROI条目数组（文件名／位置／时段）", "manifest_other_bytes": "清单其余字段及语法", "zip_container_bytes": "ZIP容器"}
    byte_rows = "\n".join(f"| {label} | {encoded['byte_ledger'][key]:,} |" for key, label in byte_labels.items())
    coverage = metrics["source_prediction_coverage"]
    geometry = read(sender / "manifest.json")
    fish_mask_fraction = coverage["fish"]["kept"]["pixels"] / (geometry["width"] * geometry["height"] * geometry["frames"])
    percentage = lambda value: "不适用" if value is None else f"{value * 100:.4f}%"
    coverage_rows = "\n".join(f"| {target}／{kind} | {row['present_frames']} | {row['pixels']:,} | {percentage(row['coverage'])} |"
                              for target, values in coverage.items() for kind, row in values.items())
    temporal_rows = "\n".join(f"| {label} | {numeric(before['error_change_mae_0_255'][key])} | {numeric(after['error_change_mae_0_255'][key])} |"
                             for key, label in (("all", "全片相邻误差变化"), ("window_boundaries", "32帧窗口切换处"), ("non_boundaries", "非窗口切换处"), ("fixed_stripe_crop", "固定鱼纹裁剪内")))
    first_source_commit = read(sender / "commands.json")[0]["code_commit"]
    decode_commits = ", ".join(sorted(set(summary.get("independent_decode_commits", {}).values()))) or summary["code_commit"]
    additional_recovery = ""
    if summary.get("recovery", {}).get("previous_recovery"):
        additional_recovery = "\n\n第一次恢复中40 kbps已通过；随后20 kbps接收进程以3221225477（0xC0000005）退出，尚无逐帧输出，且未能落盘Python异常或最终读取审计。该失败不算通过，原目录和外层退出记录原样保留；根因未确证。后续恢复保留40 kbps已通过结果，在另一新目录只补做20／30 kbps两次独立解码，增加即时输出与原生故障栈记录选项，没有改变解码算法、权重或包内容。三档接受结果的实际目录与执行提交分别登记；没有把已完成40 kbps重计为新的解码。"
    path_label = output.name
    report = f"""# B2首片实验报告：GLC基础层与自动语义ROI

日期：2026-10-18。**B2首个开发片段的三档工程闭环通过，自动ROI补充层已实现；B3等预算／独立片段对照和真实观看评价未完成，语义选择收益尚未证实。**

## 实现与冻结

沿用video91源帧0–249，1280×720、15 fps及A/B0的GLC q0／GOP64／reset32基础码流。发送端逐帧核对A标准化输入的PNG和RGB哈希；没有覆盖A或B0/B1产物。

鱼类模型沿用class6、0.9阈值、512分块／384步长；潜水员模型沿用SUIM class1、长边256与0.7置信度。两个模型独立执行、分开记录；未训练、未调阈值、未读取GT。采用既有32帧双区域布局、外扩16、最小连通区域64像素；H.265 CRF32来自原工程默认值，在B2结果出现前固定。

新profile为`semantic-video-glc-roi-v1`，保留旧封包解析入口并显式分派GLC。包成员、版本、哈希、尺寸、帧数、帧率、ROI时间窗与空间分区均需核验；接收端从实际收到的GLC基础层和ROI视频重建，再按固定矩形直接贴回。基础层未更换，补丁外RGB逐像素保持不变。

发送执行提交：`{first_source_commit}`；接受的独立解码执行提交：`{decode_commits}`。各档与提交的对应关系见汇总JSON；评价脚本SHA-256及命令另行记录。24项测试通过，覆盖旧v1/v2解析、新GLC分派、错版／坏包／重复成员、越界路径、窗口与分区、语义帧序、颜色标签缺失／冲突和外部读取限制。

## 文件与代价

| 内容 | 字节 |
|---|---:|
{byte_rows}
| 完整发送文件 | {encoded['packet_bytes']:,} |
| 每档应用层传输头 | {summary['results'][0]['transfer']['application_bytes'] - encoded['packet_bytes']} |

完整文件约为GLC基础码流的{encoded['packet_bytes'] / encoded['byte_ledger']['glc_file_bytes']:.3f}倍。共{encoded['patches']}段ROI视频，矩形面积按帧加权平均占画面{encoded['average_patch_rectangle_fraction'] * 100:.2f}%。ROI矩形包含较多目标周边像素，因此不能把全局改善全归因于目标语义。

作为布局诊断，过滤后的鱼类预测掩码按帧加权仅占画面{fish_mask_fraction * 100:.4f}%；它与{encoded['average_patch_rectangle_fraction'] * 100:.2f}%的传输矩形面积相差很大。这是源模型预测与固定窗口矩形布局的统计，不是真实鱼类面积或目标召回证明；后续需要检查稀疏预测、跨帧并集和大片周边背景带来的字节成本。

GLC内部的码本、熵流和容器开销继续包含在168,338 B基础流内；两个GLC共享权重共1,169,616,826 B，作为部署前置成本另列，本轮未传权重。

## 三档真实发送与独立接收

| kbps | 应用层 B | 理论最短 s | 实测发送 s | 平均 kbps | 接收解码与预览 s |
|---:|---:|---:|---:|---:|---:|
{transfer_rows}

实际发送顺序40→20→30，三次发送独立计时。每档均收到同一完整包，SHA-256为`{encoded['packet_sha256']}`。首轮接收在读取未写入的可选颜色标签时发生异常，原始三档收发记录及失败日志均保留。修复后将每档实际收到的文件复制到新的接收目录，核对其原接收PID、文件哈希和传输证据，再各自重新独立解码；没有重传，没有复用任何RGB重建缓存。表内发送时间来自原实际传输，解码时间来自修复后的独立解码，两阶段明确区分。
{additional_recovery}

每档250帧，共750帧通过最终验收；GLC基础RGB与B1一致，三档增强RGB序列一致，补丁像素与实际H.265解码相同，补丁外像素保持原GLC值。视频帧数、尺寸、15 fps及完整播放解码检查通过。

颜色核验继续强制要求有限范围与BT709矩阵。现有FFmpeg/x265输出未携带`color_transfer`和`color_primaries`，按缺失记录为null，不能写成已核验这两个标签；如显式存在但与BT709冲突则拒绝。未更改实际码流、色彩转换或解码像素来绕过异常。

接收Python审计未记录源媒体或发送端缓存访问，预共享模型按路径与哈希列出，FFmpeg输入限定接收目录。发送调度按每次调用前的累计字节截止时间核验；实际单调时钟分辨率为{summary['results'][0]['pacing']['clock_resolution_seconds']:.6f}秒，逐档原始误差值见证据。此结果是同机TCP回环的软件应用层限速，不含TCP/IP、链路层或反向ACK，不代表物理水声、跨设备或实时性测试。Python审计不是操作系统沙箱。

## 预测覆盖：不是GT召回

| 源端预测／处理 | 出现帧数 / 250 | 预测像素合计 | 传输ROI覆盖率 |
|---|---:|---:|---:|
{coverage_rows}

raw为固定阈值后的掩码，kept为剔除小于64像素连通区域后的掩码。过滤前后的帧数分别保留；模型从未检出的真实目标无法通过此覆盖指标发现。连通区域实例不等于真实鱼数。潜水员未检出，相关像素指标为空，本片不能证明潜水员保护效果。

## 接收RGB失真诊断

| 区域 | GLC基础 PSNR dB | GLC+自动ROI PSNR dB |
|---|---:|---:|
{quality_rows}

基础层与增强层文件大小不同。鱼区掩码来自同一个源端选择模型，属于开发诊断，不能视为独立任务准确率。固定鱼纹裁剪沿用A阶段预先选定的坐标与帧范围，不参与本次传输ROI选取。上述PSNR变化不能替代鱼纹、轮廓和运动的真实观看核对。

本片全画面PSNR增加{after['rgb_psnr_db']['full'] - before['rgb_psnr_db']['full']:.4f} dB，固定鱼纹裁剪增加{after['rgb_psnr_db']['fixed_stripe_crop'] - before['rgb_psnr_db']['fixed_stripe_crop']:.4f} dB；这说明当前增加补充字节后的接收RGB误差下降，不等于已证明语义选区优于相同预算的非语义方案。

## 窗口与边界诊断

| 误差变化MAE（0–255像素值） | GLC基础 | GLC+自动ROI |
|---|---:|---:|
{temporal_rows}
| ROI边缘相邻重建误差差分 | {numeric(before['ROI_edge_error_difference_mae_0_255'])} | {numeric(after['ROI_edge_error_difference_mae_0_255'])} |

**负结果：ROI边缘相邻重建误差差分增加{(after['ROI_edge_error_difference_mae_0_255'] / before['ROI_edge_error_difference_mae_0_255'] - 1) * 100:.2f}%。** 当前直接贴回的空间边界是明确待检查项；不能只报告平均PSNR和相邻误差变化改善而忽略它。先保留本次冻结配置与正负结果，再在B3对照中区分额外字节、区域选择和合成边界的影响。

相邻误差变化定义为`mean(abs((重建_t-源图_t)-(重建_t-1-源图_t-1)))`；边缘诊断使用相同ROI边界两侧相邻像素的重建误差差分。它们是误差变化代理，不是已经验证的感知闪烁评分；较小数值不能自动排除贴回边界或鱼纹错误。固定关键帧包含223／224窗口切换，鱼纹半速视频也跨越该边界。

## 交付与未完成项

- [小型证据]({path_label})：发送／接收配置、字节账本、预测、三档轨迹、逐帧校验、审计摘要与完整命令记录。
- [本地媒体与审计索引]({path_label}/local_artifacts.json)：路径、字节数与SHA-256；实际视频、掩码、码流和模型留本地。
- [复现说明](B2复现说明.md)：运行命令、版本化接口、串行解码和指标定义。
- [观看记录](B2观看记录_20261018.md)：观看顺序、固定问题与待填写的真实用户反馈。
- 本地完整结果：`{run}`；正常三路对比、固定鱼纹半速、预测与ROI叠加视频位于`evaluation`。
- 用户观看答卷：待评，没有自动生成用户评分。B3至少三段独立连续素材仍待登记。

下一步应先观看本片的鱼纹与窗口边界，保留正负例，再完成等完整文件预算的GLC对照及同补充字节预算的固定几何区域消融。只有这些对照与独立片段支持结果后，才能评价语义选择的净收益；本报告不将增加文件大小后的失真变化写成语义方案胜出。
"""
    args.report.write_text(report, encoding="utf-8")
    print(f"B2 report archived: {args.report}", flush=True)


if __name__ == "__main__":
    main()
