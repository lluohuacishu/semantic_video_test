"""Archive small B0/B1 evidence and render the report only after all trials pass."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def record(path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--failed-attempt", type=Path)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    summary = read(run / "summary.json")
    if summary["status"] != "B0_B1_passed" or [r["rate_kbps"] for r in summary["results"]] != [20, 30, 40]:
        raise ValueError("All three accepted trials are required")
    if not summary["all_rates_identical_received_RGB"] or len({r["rgb_sequence_sha256"] for r in summary["results"]}) != 1:
        raise ValueError("All received RGB sequences must agree")
    for result in summary["results"]:
        if result != read(run / f"rate_{result['rate_kbps']}kbps/result.json"):
            raise ValueError("Summary differs from per-rate evidence")
        if (result["decoded_frames"] != 250 or not result["all_rgb_and_latent_checksums_match"]
                or not result["playback_verified"] or result["external_source_or_sender_cache_reads"] != 0):
            raise ValueError("A required acceptance check is missing")
    if args.report.exists():
        raise FileExistsError(args.report)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("summary.json", "config.json", "progress.json", "b0_frozen.json", "environment.json", "semantic_models.json", "input_frame_mapping.json", "commands.json", "glc_packages.txt", "gpu_driver.txt", "ffmpeg_version.txt"):
        shutil.copyfile(run / name, output / name)
    recovery_note = ""
    if "recovery" in summary:
        recovery = read(run / "recovery.json")
        if recovery["status"] != "completed":
            raise ValueError("Recovery did not complete")
        for name in ("recovery.json", "recovery_commands.json", "recovery_stdout.log", "recovery_stderr.log"):
            shutil.copyfile(run / name, output / name)
        for attempt in recovery["preserved_incomplete_attempts"]:
            source = Path(attempt["preserved_path"])
            destination = output / source.name
            destination.mkdir()
            for name in ("transfer.json", "transfer.csv", "send_schedule.csv", "receiver_process.log", "receiver/ready.json", "receiver/receive.json", "receiver/deployment.json"):
                shutil.copyfile(source / name, destination / Path(name).name)
        recovery_note = """
第二轮在20、30 kbps已通过、40 kbps已收完文件但未完成解码时被中断。恢复检查确认原发送／接收进程均已退出；完整保留未完成的40 kbps目录，并只重新执行该档。保留的20、30 kbps各250帧RGB和潜变量哈希、帧序、收发文件、审计与限速轨迹已再次只读核验。三档接受结果分别来自前两档原运行及40 kbps补跑，不能称为一次不中断运行。

原主进程在写出`commands.json`前退出，**原命令总日志、退出码及部分命令耗时未保留下来**；没有事后补造。逐档结果、独立接收完成记录、传输和解码证据仍完整。`commands.json`明确标注该缺口，恢复命令与代码哈希记录于`recovery_commands.json`和`recovery.json`。因此工程闭环检查通过，但原运行命令总日志完整性存在缺口；复现命令另列于复现说明。
"""
    artifacts = []
    for result in summary["results"]:
        tag = f"rate_{result['rate_kbps']}kbps"
        source, destination = run / tag, output / tag
        destination.mkdir()
        for name in ("transfer.json", "transfer.csv", "send_schedule.csv", "pacing_verification.json", "frame_verification.json", "playback_probe.json", "playback_decode.log", "receiver_process.log", "result.json"):
            shutil.copyfile(source / name, destination / name)
        audit = read(source / "receiver/receiver_read_audit.json")
        counts = Counter()
        for row in audit["python_reads"]:
            counts[row["category"]] += row["count"]
        write(destination / "receiver_audit_summary.json", {
            "scope": audit["scope"], "python_open_attempts_by_category": dict(counts),
            "shared_model_allowlist": audit["shared_model_allowlist"],
            "denied_reads": audit["denied_reads"], "external_source_or_sender_cache_reads": audit["external_source_or_sender_cache_reads"],
            "subprocesses": audit["subprocesses"], "full_local_audit": record(source / "receiver/receiver_read_audit.json"),
            "receiver_pid": result["receiver_pid"], "evaluator_pid": result["evaluator_pid"],
        })
        for name in ("receiver/receive.json", "receiver/decoded/decode.json", "receiver/completion.json"):
            shutil.copyfile(source / name, destination / Path(name).name)
        for name in ("receiver/received.glcv", "receiver/decoded/reconstruction_preview.mp4", "receiver_process.log", "playback_decode.log", "receiver/receiver_read_audit.json"):
            artifacts.append(record(source / name))
    if args.failed_attempt:
        failed = output / "failed_attempt_1"
        failed.mkdir()
        for path, name in (("failure.json", "runner_failure.json"), ("rate_20kbps/receiver/failure.json", "receiver_failure.json"),
                           ("rate_20kbps/transfer.json", "transfer.json"), ("commands.json", "commands.json")):
            shutil.copyfile(args.failed_attempt / path, failed / name)
        failed_audit_path = args.failed_attempt / "rate_20kbps/receiver/receiver_read_audit.json"
        failed_audit = read(failed_audit_path)
        write(failed / "audit_failure_summary.json", {"full_local_audit": record(failed_audit_path),
              "denied_read_attempts": failed_audit["denied_reads"], "scope": failed_audit["scope"]})
    write(output / "local_artifacts.json", artifacts)
    write(output / "evidence_checksums.json", [
        {"relative_path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": record(path)["sha256"]}
        for path in sorted(output.rglob("*")) if path.is_file()
    ])
    results = summary["results"]
    rows = "\n".join(f"| {r['rate_kbps']} | {r['transfer']['application_bytes']:,} | {r['transfer']['minimum_application_seconds']:.4f} | {r['transfer']['elapsed_seconds']:.4f} | {r['transfer']['average_application_kbps']:.6f} | {r['decoded_frames']} |" for r in results)
    timing = "\n".join(f"| {r['rate_kbps']} | {r['pacing']['calls_including_header']} | {r['pacing']['maximum_early_seconds']:.3e} | {r['pacing']['clock_resolution_seconds']:.6f} | {r['decode_seconds_before_preview']:.2f} |" for r in results)
    evidence_link = output.name
    report = f"""# B0+B1 实验报告：GLC 文件通信闭环

日期：2026-10-18。**B0+B1通过：三档真实收发、三次独立解码，共750帧。B2/B3尚未执行，语义保护收益未评价。**

## 输入与冻结

沿用video91源帧0–249，1280×720、15 fps、16.67秒；GLC q0／GOP64／reset32，完整码流168,338 B。它是开发片段和可复现起点。250帧的原始文件、标准化PNG及实际编码输入RGB均已核对；A产物未覆盖。

基线执行提交：`{summary['code_commit']}`，原运行开始时工作区干净。恢复运行重新核验原执行代码与模型哈希，新增恢复脚本另记SHA-256。GLC适配器与A运行版本逐文件一致。两个GLC共享权重合计1,169,616,826 B，本轮未传权重；沿用RTX4060 Laptop、PyTorch2.5.1+cu124及FP32确定性环境。

语义配置沿用现有Teacher：鱼类适配模型class6、概率阈值0.9、512分块与384步长；潜水员SUIM底模class1、长边256与居中补边、argmax及0.7置信度。严格加载、权重哈希与冻结状态通过；两套模型分开记录，不混合评分。ROI固定为halves、32帧窗口、外扩16、最小连通区域64像素。本次未执行ROI推断、训练或阈值搜索。

## 三档实际收发

| 限速 kbps | 应用层发送 B | 理论最短 s | 实测 s | 平均应用层 kbps | 独立重建帧 |
|---:|---:|---:|---:|---:|---:|
{rows}

三档发送／接收SHA-256均为`{results[0]['received_sha256']}`。接收器和发送／评价主进程PID不同；每档均从实际收到的文件重建全部250帧，RGB与潜变量逐帧校验通过。三档接收RGB序列哈希完全一致，尺寸和源帧序正确。

全局PSNR从实际接收RGB计算，各档均为{results[0]['full_psnr_db_from_received_RGB']:.6f} dB，仅作为基线复现记录，不代表目标识别或语义改善。观看副本均经FFmpeg完整解码和FFprobe帧数／尺寸／帧率检查；H.264 CRF16、YUV420、正常15 fps，属于接收后的额外编码，未计入通信码流，也未生成用户观看评分。

## 限速调用与时间精度

| 档位 kbps | 含传输头的发送调用 | 最大提前量 s | 时钟分辨率 s | 解码至预览前 s |
|---:|---:|---:|---:|---:|
{timing}

每次`sendall`调用前后分别计时。验收使用调用前时间与累计字节对应的期限比较；没有末尾补等待来掩盖早发。表内微小提前量如非零，仍小于时钟分辨率，按原值公开，不强行写为精确零。解码时间包括模型加载与PNG写出，发送时间与它分开报告。

环境实际报告的单调时钟分辨率为0.015625秒；此前临时核验中手工传入的1e-7秒并非本机时钟分辨率，以本报告及实际保存的环境／限速核验JSON为准。

## 完整字节账本与读取审计

| 项目 | 字节 |
|---|---:|
| GLC码本索引 | 8,568 |
| GLC熵码流 | 137,534 |
| GLC容器及校验 | 22,236 |
| 完整GLC文件 | 168,338 |
| 额外ROI、位置／时段、语义消息、外层封装 | 0 |
| 应用层传输头 | 96 |
| 每档实际发送合计 | 168,434 |

接收进程只收到部署依赖清单与TCP文件。共享权重按精确路径和哈希放行；源码、推理环境和允许的子进程参数另行记录。三档审计均无禁止的外部源媒体／发送缓存读取，模型依赖明确列出。源帧仅在独立的输入校验与结果评价进程读取。

这是Python文件打开与显式子进程参数约束，**不是操作系统沙箱**，不声称捕获原生DLL的全部底层读取。实验为同机TCP回环的软件应用层限速，排除TCP/IP、链路层及反向ACK字节；没有跨设备、真实水声物理链路或实时性验收结论。

## 调试记录与交付

首轮20 kbps文件传输成功，但Windows的`subprocess.Popen`审计事件使用已拼接命令字符串及`executable=None`，新增审计代码尚未兼容，解码启动失败；Python还尝试读取旧外部字节码缓存，被策略拒绝。修正为精确允许命令比较，并将字节码查找重定向到接收目录后，在新目录重新执行全部三档。首轮单独保留，不纳入通过结果。9项检查通过，覆盖提前发送、遗漏传输头、真实小文件收发、未声明模型、源文件读写模式及Windows子进程格式。
{recovery_note}

- [小型证据目录]({evidence_link})：汇总、冻结清单、250帧映射、三档发送轨迹、逐帧核验、读取审计摘要及失败首轮。
- [完整本地文件索引]({evidence_link}/local_artifacts.json)：码流、视频与完整审计的路径、大小和SHA-256；媒体不上传Git。
- [复现说明](B0_B1复现说明.md)：命令、输入、读取边界及所有产物说明。
- 本地完整结果：`{run}`。

## 下一步

进入B2：在固定GLC基础层外加入自动预测的目标ROI补充流，明确新封包版本与接收分派，保存完整字节账本和同步观看材料。重点检查小鱼漏保护、条纹改变、ROI边界及跨窗口跳变。B3的至少三段独立连续视频仍未登记；当前工程闭环通过，不能预填语义收益有效。
"""
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(args.report), "evidence": str(output), "local_artifacts": len(artifacts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
