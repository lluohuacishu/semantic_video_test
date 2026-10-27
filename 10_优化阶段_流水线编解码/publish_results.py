"""Export small reviewable evidence and documentation; keep media local."""
from pathlib import Path
import hashlib
import json
import shutil
import statistics

import stream_protocol as p
from glc_container import read_stream
from pipeline_probe import HERE, REPO, WORK, dump, sha

RUN = WORK / '02_实验数据/优化_流水线编解码/pipeline_pairs_20261027_01'
FIRST = WORK / '02_实验数据/优化_流水线编解码/pipeline_probe_20261027_01'
EVIDENCE = HERE / 'evidence_20261027'


def read(path):
    return json.loads(path.read_text('utf-8'))


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + '\n', 'utf-8')


def main():
    result = read(RUN / 'summary.json')
    assert result['complete'] and len(result['runs']) == 8
    EVIDENCE.mkdir(exist_ok=True)
    for filename in ('plan.json', 'summary.json'):
        shutil.copy2(RUN / filename, EVIDENCE / filename)
    shutil.copy2(FIRST / 'summary.json', EVIDENCE / 'first_prototype_summary.json')
    shutil.copy2(FIRST / 'frame_timeline.json', EVIDENCE / 'first_prototype_timeline.json')
    for row in result['runs']:
        name = row['name']
        folder = EVIDENCE / name
        folder.mkdir(exist_ok=True)
        shutil.copy2(RUN / name / 'manifest.json', folder / 'manifest.json')
        for role in ('sender', 'receiver'):
            for file in ('result.json', 'events.jsonl'):
                shutil.copy2(RUN / name / role / file, folder / f'{role}_{file}')
    header, packets = read_stream(FIRST / 'sender/encoded.glcv')
    old_frames = read(FIRST / 'sender/result.json')['frames']
    sizes = {}
    for version in (1, 2):
        transcript = hashlib.sha256()
        total = len(p.encode_record(p.HEADER, 0, p.json_bytes(header), version))
        for i, packet in enumerate(packets):
            packet = {**packet, **{key: old_frames[i][key] for key in ('rgb_sha256', 'latent_sha256')}}
            body = p.encode_frame(header, i, packet, version)
            transcript.update(body)
            total += len(p.encode_record(p.FRAME, i, body, version))
        trailer = dict(frames=len(packets), frame_records_sha256=transcript.hexdigest(),
                       archive_sha256=sha(FIRST / 'sender/encoded.glcv'))
        total += len(p.encode_record(p.END, len(packets), p.json_bytes(trailer), version))
        sizes[str(version)] = total
    assert sizes['1'] == read(FIRST / 'summary.json')['forward_application_bytes']
    size16 = {'frames': 16, 'method': 'same saved compressed payload, deterministic protocol repack; no new neural timing',
        'archive_bytes': 3776, 'v1_forward_bytes': sizes['1'], 'v2_forward_bytes': sizes['2']}
    dump(EVIDENCE / 'same_payload_protocol_bytes.json', size16)
    base = result['runs'][0]
    wire, archive = base['forward_application_bytes'], base['archival_bytes']
    aggregate_table = []
    pair_table = []
    for a in result['aggregates']:
        kbps = a['rate_kbps']
        serial = [r for r in result['runs'] if r['rate_kbps'] == kbps and r['mode'] == 'serial']
        pipeline = [r for r in result['runs'] if r['rate_kbps'] == kbps and r['mode'] == 'pipeline']
        sf = statistics.mean(r['seconds_from_first_encode_start']['first_frame_decoded'] for r in serial)
        pf = statistics.mean(r['seconds_from_first_encode_start']['first_frame_decoded'] for r in pipeline)
        aggregate_table.append(f"| {kbps} | {a['mean_serial_seconds']:.3f} | {a['mean_pipeline_seconds']:.3f} | {a['median_paired_reduction_percent']:.2f}% | {sf:.3f} → {pf:.3f} |")
    for a in result['pairs']:
        pair_table.append(f"| {a['rate_kbps']} | {a['pair']} | {a['serial_seconds']:.3f} | {a['pipeline_seconds']:.3f} | {a['reduction_percent']:.2f}% |")
    report = f'''# 紧凑流式协议与串行／流水线配对结果

日期：2026-10-27。分支：`codex/glc-streaming-optimization`。

新协议去掉线上逐帧重建哈希和重复的帧布局字段，直接传输紧凑 z/y 记录。逐帧做 CRC32 检错；开始/结束/确认消息保留 SHA-256，结束时校验整段传输记录摘要及完整 GLCV 文件摘要。编码网络、量化、熵编码与重建结果不变。

## 传输开销

| 对象 | GLCV归档 | 原诊断协议正向 | 紧凑协议正向 | 紧凑协议相对归档增加 |
|---|---:|---:|---:|---:|
| 同一16帧载荷重新封装 | 3,776 B | {sizes['1']:,} B | {sizes['2']:,} B | {(sizes['2']/3776-1)*100:.2f}% |
| 66帧真实收发 | {archive:,} B | 未重跑 | {wire:,} B | {(wire/archive-1)*100:.2f}% |

16 帧行是对同一份已保存载荷的格式核算，不是新的神经网络耗时实验。66 帧的八次运行发送字节数及归档哈希全部一致。反向应用消息每次 {base['reverse_application_bytes']} B，另外报告；不把 TCP/IP、系统 ACK 或真实链路开销混入应用层字节。这不是 FEC，尚无纠错冗余；不能据此承诺所有片段满足历史 10% 预算，16 帧结果即超过该门槛。

## 配对条件

同一源视频，640×360、12.5 fps、66 帧（5.28秒）、q0、GOP64、reset32，FP32、TF32关闭、原生 C rANS，无语义预处理，无预览视频输出。两端先各暖机8帧，然后清空参考状态；初始化和暖机不计入下表。发送端和接收端是独立进程，但共用一块 RTX4060 Laptop GPU。本机 TCP 按正向应用字节限速，不代表真实水声链路。

串行与流水线使用相同新协议、同样校验和完全相同发送字节。串行模式先编码完，再完整接收，之后解码；流水线模式逐帧交付。每个速率两对，第二对交换运行顺序；共8次编码与8次独立解码，各66帧。测量终点为接收端完整文件验证完成，起点为第一帧开始编码。

接收端暖机读取预置的已知正确压缩码流，不读取原视频；暖机完成即重置 `ref_latent/ref_y`。正式接收解码只使用当前网络收到的数据，双方逐帧日志再独立比对。

| 应用限速 kbps | 串行均值 s | 流水线均值 s | 配对降幅中位数 | 首帧等待均值 s（串行→流水线） |
|---|---:|---:|---:|---:|
{chr(10).join(aggregate_table)}

| kbps | 配对 | 串行 s | 流水线 s | 总耗时降幅 |
|---|---:|---:|---:|---:|
{chr(10).join(pair_table)}

40 kbps 下两种方式总耗时接近，不据此宣称稳健提速；20 kbps 的结果按上表报告。首帧等待明显缩短，但首帧提前不等于整段按相同比例加速。仅两对、单素材、桌面仍运行，结论限于本次条件。两个进程共享同一 GPU，包含并发争用，不能推算“Orin编码＋另一设备解码”的速度。计时覆盖网络调用和等待，不能把重叠的编/解码区间相加作为 GPU 内核工作量。

## 正确性和失败处理

- 八次 66 帧的发送/接收归档 SHA-256 全部一致，等于冻结的 native_05 基线。逐帧 z/y、RGB和参考状态与原基线一致，覆盖第65帧I帧和参考重置位置。
- 十项协议检查在禁用第三方 site-packages 的 Python 下通过，无需模型或本地素材。覆盖两个版本、拆包、示例帧每一个单比特翻转、所有截断位置、重复/乱序、长度越界、缺结束消息和错误最终摘要。
- 逐帧 CRC 是随机错误检测，不是纠错、认证或无碰撞保证。CRC 通过后产生的图像仍为暂定输出，只有最终 SHA-256 校验通过才标记整段成功。此次没有测 FEC、实际丢包或水下恢复能力。
- 校验失败或截断不会发布最终 `received.glcv`；未通过的内容保留在本次实验的临时文件中。网络线程退出时关闭连接，避免继续处理错误任务。
- 当前仅本机原型，最大66帧；网页接入、未知总帧数、跨设备数值兼容及完整取消生命周期仍待完成。原 C5 网页和生产编解码器未替换。

## 复查与复现

仓库小证据在 [evidence_20261027](evidence_20261027/summary.json)，含预登记顺序、八次汇总、两端逐帧结果和事件时间。完整本地目录为工作区 `02_实验数据/优化_流水线编解码/pipeline_pairs_20261027_01`；视频、码流和模型不上传 Git。

CPU 检查：在本目录执行 `python -S -m unittest -v test_stream_protocol`。

真实任务使用现有 GLC 运行环境与 `c4_config.py` 配置，单次：

```text
python pipeline_probe.py run --input <视频路径> --frames 66 --mode pipeline --kbps 40 --warmup-frames 8 --warmup-stream <已知正确码流> --output <新目录>
```

`--mode serial` 为同协议串行对照。可选 `--baseline-root` 指向含 `stream.glcv` 和 `encoder/encode.json` 的旧结果，以检查逐帧基线。完整有限配对由 `benchmark_pairs.py --output <新目录>` 执行，默认使用本工作区固定素材；其他机器先配置素材与运行环境，不直接照搬本机路径。
'''
    write(HERE / '紧凑协议与配对结果_20261027.md', report)
    write(RUN / '实验报告.md', report)
    write(HERE / 'README.md', f'''# GLC 流水线优化

更新：2026-10-27。分支 `codex/glc-streaming-optimization`。目标为 Orin 上文件任务的编码、传输、解码总等待，不要求实时播放；旧轮次已收束。

紧凑协议和8次66帧配对已完成。每次归档 {archive:,} B，正向实际发送 {wire:,} B，增加 {(wire/archive-1)*100:.2f}%；同一16帧的协议开销从58.5%降至{(sizes['2']/3776-1)*100:.2f}%。模型、压缩载荷和重建结果不变。

- [最新配对报告](紧凑协议与配对结果_20261027.md)：首帧、总耗时、全部配对及限制。
- [小型原始证据](evidence_20261027/summary.json)：可在 GitHub 查看；完整媒体和运行环境保留本地。
- [研究方案](流水线研究方案_20261027.md)、[首个16帧原型](流水线原型结果_20261027.md)：历史机制证明，旧协议/冷态数据不与新配对混用。
- [执行程序](pipeline_probe.py)、[协议](stream_protocol.py)、[配对程序](benchmark_pairs.py)、[CPU检查](test_stream_protocol.py)。

检查命令：`python -S -m unittest -v test_stream_protocol`，不依赖模型和数据集。实际编解码需要已配置的 GLC Python、权重和原生 rANS。

当前支持明确帧数、两端有界队列、逐帧完整性与最终文件验证。网页尚未接入；无FEC、真实水下或Orin实验。下一步完善取消/超时和未知帧数接口，再接现有网页；跨设备必须先验证数值兼容。40 kbps 下本机共享GPU的总耗时收益不显著，不能将首帧等待下降说成同等整段加速。
''')
    state = read(HERE / 'project_status.json')
    state.update(branch='codex/glc-streaming-optimization', prototype='compact_v2_8x66_paired_verified',
        protocol_version=2, protocol_tests_passed=10, experiment_summary='evidence_20261027/summary.json',
        next='Lifecycle and unknown-frame-count support, then web integration; verify cross-device arithmetic',
        paired_measurement=result['aggregates'], forward_bytes=wire, archive_bytes=archive)
    dump(HERE / 'project_status.json', state)
    prior = WORK / '02_实验数据/C_编解码加速与网页展示'
    closed = REPO / '05_资料与索引/收束证据_20261027'
    for folder, names in [
        ('GLC_operator_profile_20261025_recovered_01', ['validation.json', 'GLC算子级诊断报告.md']),
        ('GLC_precision_20261026_native_05', ['summary.json', '文件任务与Orin优化说明.md']),
        ('GLC_BER_20261026_01', ['summary.json', 'validation.json', '误码实验报告.md'])]:
        target = closed / folder
        target.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(prior / folder / name, target / name)
    write(closed / 'README.md', '''# 已收束实验的小型证据

这些报告和JSON是此前已经完成实验的原文副本，不代表新一轮运行。原文中的相对媒体链接需要在完整本地结果目录打开；模型、视频、码流及详细日志保留在工作区 `02_实验数据/C_编解码加速与网页展示`。

- [算子诊断](GLC_operator_profile_20261025_recovered_01/GLC算子级诊断报告.md)
- [原生后端精度补测](GLC_precision_20261026_native_05/文件任务与Orin优化说明.md)
- [裸误码试验](GLC_BER_20261026_01/误码实验报告.md)

旧Python熵后端试探、首次中断和失败记录仍在本地原目录；不将它们混入有效原生后端基准。
''')
    write(REPO / '05_资料与索引/文件索引_20261027.md', '''# 项目文件索引

更新：2026-10-27。旧轮次已收束，当前分支 `codex/glc-streaming-optimization`。

## GitHub可直接阅读

- [项目总结](../项目总结与未来方向.md)、[当前接续](../交接文档.md)。
- [流水线优化入口](../10_优化阶段_流水线编解码/README.md)、[最新配对结果](../10_优化阶段_流水线编解码/紧凑协议与配对结果_20261027.md)。
- [收束证据](收束证据_20261027/README.md)：算子、原生精度补测与误码报告及汇总。
- [语义 H.265 展示代码](../08_新阶段C_编解码加速与网页展示/h265_demo)、[语义分割展示代码](../08_新阶段C_编解码加速与网页展示/segmentation_demo)。

## 本地资产（未上传）

下列路径相对本机工作区 `C:/Users/lenovo/Documents/ChatGPT/语义通信`。

| 路径 | 内容 |
|---|---|
| `02_实验数据/优化_流水线编解码` | 新流水线完整实验 |
| `03_历史归档/阶段收束_20261027` | 37份旧入口、代码和证据快照 |
| `02_实验数据/C_编解码加速与网页展示` | 历史性能、精度、误码与网页任务 |
| `02_实验数据/D_语义目标增强` | 历史语义实验及原始视频数据 |
| `汇报展示材料` | 上传视频、码流、重建和对比网页 |
| `01_项目/运行环境` | Python、FFmpeg、模型及旧工程路径 |
| `01_项目/代码/FaunaCodec` | 单独的 FaunaCodec 工程 |

桌面 `C:/Users/lenovo/Desktop/临时/材料` 的启动入口保留：`打开网页.cmd`（GLC 8773/8774）、`H265.cmd`（8777）、`语义分割.cmd`（8778）、`FaunaCodec.cmd`（8775）。端口是启动配置，不表示服务正在运行。
''')
    handoff = (REPO / '交接文档.md').read_text('utf-8')
    handoff = handoff.split('\n## 紧凑协议与配对续更')[0]
    handoff = handoff.replace('分支仍为 `codex/glc-d`；当前整理不切换分支、提交、推送或覆盖已有修改。',
        '2026-10-27用户授权继续优化并上传GitHub，当前分支 `codex/glc-streaming-optimization`；保留全部已有修改，按代码、报告和小证据范围提交推送。')
    handoff += '\n## 紧凑协议与配对续更\n\n已完成协议v2和8次66帧配对，见[最新报告](10_优化阶段_流水线编解码/紧凑协议与配对结果_20261027.md)。旧16帧记录是历史机制证明。当前下一步为生命周期与未知帧数，再接网页；不重复已完成矩阵。CPU协议检查10项通过，数据集/模型/视频/码流保留本地。\n'
    write(REPO / '交接文档.md', handoff)
    write(REPO / '上传说明.md', '''# GitHub上传范围

更新：2026-10-27。仓库 `lluohuacishu/semantic_video_test`，优化分支 `codex/glc-streaming-optimization`，从既有 `codex/glc-d` 的工作状态延续。

本次提交项目收束文档、此前未提交的本地展示和诊断源码、逐帧传输原型、紧凑协议、有限配对程序、检查、报告和小型JSON/时间轴证据。既有代码修改一起保存，不覆盖或丢弃。分支未合并到main。

模型权重、数据集、原视频、重建视频、实际码流、环境、原生二进制和完整归档继续保留本地，使用现有 .gitignore 排除；不使用Git LFS、不打包Release。FaunaCodec为工作区另一个独立工程，本仓库仅记录其本地展示入口。

CPU协议检查不依赖本地素材，可在克隆后使用Python标准库运行。神经编解码仍需按运行环境配置准备GLC权重、PyTorch与原生rANS。实验报告中的本机路径用于溯源，仓库内入口使用相对链接；不宣称仓库包含完整实验媒体或可直接部署到Orin。
''')
    evidence_manifest = {file.relative_to(EVIDENCE).as_posix(): {'bytes': file.stat().st_size, 'sha256': sha(file)}
                         for file in EVIDENCE.rglob('*') if file.is_file() and file.name != 'evidence_manifest.json'}
    dump(EVIDENCE / 'evidence_manifest.json', evidence_manifest)
    print(json.dumps({'evidence_files': len(evidence_manifest), 'v1_v2_same16': size16,
                      'aggregates': result['aggregates']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
