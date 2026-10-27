from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WORK = REPO.parents[2]
RUN = WORK / '02_实验数据/优化_流水线编解码/pipeline_probe_20261027_01'
ARCHIVE = WORK / '03_历史归档/阶段收束_20261027'

def read(path):
    return json.loads(path.read_text('utf-8'))

def write(path, text):
    path.write_text(text.rstrip() + '\n', 'utf-8')

def dump(path, obj):
    write(path, json.dumps(obj, ensure_ascii=False, indent=2))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if read(HERE / 'project_status.json').get('protocol_version', 1) != 1:
    raise SystemExit('Historical v1 finalizer is closed; use publish_results.py for the current results.')
result = read(RUN / 'summary.json')
assert result['verified'] and result['first_decoded_before_encode_finished']
assert result['first_decoded_before_last_frame_received'] and result['identical_to_baseline_prefix']
manifest = read(ARCHIVE / '归档清单.json')
for item in manifest['files']:
    assert sha(Path(item['snapshot'])) == item['sha256']
    if item['category'] in ('基线代码', '关键证据'):
        assert sha(Path(item['source'])) == item['sha256'], item['source']
for name, digest in read(RUN / 'manifest.json')['code_sha256'].items():
    assert sha(HERE / name) == digest
for filename in ('FaunaCodec.cmd', 'H265.cmd', '语义分割.cmd', '打开网页.cmd'):
    text = (Path('C:/Users/lenovo/Desktop/临时/材料') / filename).read_text('utf-8-sig')
    for value in re.findall(r'"([A-Za-z]:\\[^"\r\n]+)"', text):
        assert Path(value).exists(), value

send = read(RUN / 'sender/result.json')
recv = read(RUN / 'receiver/result.json')
origin = send['frames'][0]['start']
send_events = [json.loads(line) for line in (RUN / 'sender/events.jsonl').read_text('utf-8').splitlines()]
recv_events = [json.loads(line) for line in (RUN / 'receiver/events.jsonl').read_text('utf-8').splitlines()]
complete = next(r['seconds'] for r in recv_events if r['event'] == 'complete_verified')
timeline = []
for a, b in zip(send['frames'], recv['frames']):
    e = next(row for row in send_events if row['event'] == 'send_end' and row['kind'] == 2 and row['frame'] == a['frame'])
    timeline.append({'frame': a['frame'], 'encode_start': a['start'] - origin,
        'encode_end': a['end'] - origin, 'sent': e['seconds'] - origin,
        'received_verified': b['received_verified'] - origin,
        'decode_start': b['start'] - origin, 'decode_end': b['end'] - origin})
dump(RUN / 'frame_timeline.json', timeline)
selected = [timeline[i] for i in (0, 1, 2, 15)]
table = '\n'.join(f"| {r['frame']} | {r['encode_end']:.3f} | {r['sent']:.3f} | {r['decode_start']:.3f} | {r['decode_end']:.3f} |" for r in selected)
t = result['seconds_from_first_encode_start']
extra = result['added_forward_bytes_over_glcv'] / result['archival_bytes'] * 100
report = f'''# GLC 流水线原型结果

日期：2026-10-27。**16 帧真实编码、逐帧 TCP 发送、独立接收进程解码已通过。首帧解码完成时，发送端仍在编码后续帧。** 这证明现有 GLC 可以支持流水线；原网页尚未替换。

## 条件

纯 GLC、640×360、12.5 fps、16 帧（1.28 秒）、q0、GOP64、reset32；FP32、TF32关闭、确定性算法、原生 C rANS。沿用旧基线相同输入和采样，未改变模型/权重。两个独立进程在同一台 RTX 4060 Laptop 上共享一块 GPU，经 localhost TCP 收发。

应用层发送上限为 40 kbps，按不超过 256 B 的块进行发送前计时；包含正向原型协议、诊断信息和完整性校验，不含 TCP/IP、系统 ACK、物理链路和重传开销。未添加人为编解码延时，未模拟真实水声、误码、丢包或 FEC。

## 实测时间

以下时间统一从发送端开始编码第 1 帧计算，**不含之前模型和进程初始化**；未进行暖机，首帧含首次推理成本。

| 事件 | 时间 |
|---|---:|
| 第 1 帧完整解码 | {t['first_frame_decoded']:.3f} s |
| 第 16 帧编码完成 | {t['encoding_finished']:.3f} s |
| 第 16 帧完整接收并验证 | {t['last_frame_received']:.3f} s |
| 第 16 帧解码完成 | {t['last_frame_decoded']:.3f} s |
| 接收任务最终验证完成 | {complete - origin:.3f} s |

若从父进程启动计时，接收端完整验证完成在 {complete:.3f} s；进入第 1 帧编码之前约 {origin:.3f} s，包括依次启动进程、导入、加载模型和建立连接。不能把这个冷启动总时间与历史模型常驻的任务时间直接比较。

| 帧号 | 编码完成 | 发送完成 | 开始解码 | 解码完成 |
|---|---:|---:|---:|---:|
{table}

第 1 帧在整段编码结束前 {t['encoding_finished'] - t['first_frame_decoded']:.3f} 秒就已解码完成。所有 16 帧完整时间见 `frame_timeline.json`；原始事件在两端 `events.jsonl`。没有做同条件串行/流水线配对，因此不报告加速倍数，不推算 Orin 速度。

## 正确性与资源

- 16 帧编码与独立接收解码的 RGB 和参考状态哈希全部一致。
- 每帧 z/y 压缩载荷、RGB 和参考状态与 native_05 既有 66 帧基线的前 16 帧一致。
- 发送端 `encoded.glcv` 与接收端 `received.glcv` 的 SHA-256 相同，现有 `read_stream` 验证通过。
- 接收端没有读取原视频；只有发送端进程使用源视频。
- 两端应用队列容量均为 2；观察峰值发送 {result['queue_peaks']['sender']}、接收 {result['queue_peaks']['receiver']}。另有正在处理的帧及 TCP 缓冲，不能把总内存说成仅两帧。
- PyTorch 峰值 allocated：发送 {result['peak_allocated_mib']['sender']:.1f} MiB，接收 {result['peak_allocated_mib']['receiver']:.1f} MiB；不含整机桌面占用，非总设备显存峰值。
- CPU 协议检查 5 项通过：真实 I/P 帧拆成小片接收、正文翻位、头部长度翻位、截断、帧索引错误。异常数据在进入网络解码前被拒绝，测试没有声称能够纠错。

## 协议开销需要继续优化

| 内容 | 字节数 |
|---|---:|
| 原 GLCV v2 归档文件 | {result['archival_bytes']:,} B |
| 原型正向应用传输 | {result['forward_application_bytes']:,} B |
| 原型反向就绪与完成消息 | {result['reverse_application_bytes']:,} B |
| 正向相对归档文件增加 | {result['added_forward_bytes_over_glcv']:,} B（{extra:.1f}%） |

为了逐帧发现数值分歧，原型传输了逐帧 RGB/参考状态哈希及较完整的记录头。它们在低码率小帧中占比明显，所以这不是最终部署格式；还没有任何纠错冗余。下一步应优先减少诊断元数据、明确业务校验结构，再做同条件配对。不能只用 `.glcv` 大小代替实际网络字节数，也不能将原型认定为满足历史逐片 10% 增幅约束。

## 实验文件

结果根目录：`{RUN.as_posix()}`。

`manifest.json` 保存条件和代码快照哈希，`summary.json` 保存通过条件；两端有独立日志、时间轴、逐帧结果和归档码流。源代码为优化目录中的 `pipeline_probe.py`、`stream_protocol.py`、`test_stream_protocol.py`；生产 C5 服务和默认精度没有修改。这个原型只接受本机连接，未增加网页、对外服务、自动重试或 FEC。
'''
write(RUN / '流水线原型结果.md', report)
write(HERE / '流水线原型结果_20261027.md', report)
write(HERE / 'README.md', '''# GLC 流水线优化

更新：2026-10-27。旧轮次已收束，当前优化文件编码、传输和解码的总等待，最终面向 Orin NX Super 16GB，不要求实时播放。

**首个真实原型已通过：16 帧编码、逐帧 TCP 发送、独立进程逐帧解码；第 1 帧在 2.77 秒完成解码，整段编码在 8.65 秒结束。** 从首帧开始编码计时，不含此前启动与模型加载；本机两个进程共享一块 GPU，不是速度对照或 Orin 测量。

- [研究方案与改造位置](流水线研究方案_20261027.md)
- [首个原型实测结果](流水线原型结果_20261027.md)
- [原型程序](pipeline_probe.py)、[传输协议](stream_protocol.py)、[协议检查](test_stream_protocol.py)
- 实验目录：工作区 `02_实验数据/优化_流水线编解码/pipeline_probe_20261027_01`。

码流、RGB 和参考状态均与旧基线相同。接收端没有读取原视频。原型正向发送 5,985 B，原归档 3,776 B，增加 58.5%；下一步先精简诊断元数据，再做串行/流水线同条件配对。现有网页、语义处理、纠错、真实水下链路和 Orin 部署尚未接入。

## 复现

使用现有 GLC Python，在本目录运行 `python -m unittest -v test_stream_protocol`。真实试验用 `python pipeline_probe.py run --frames 16 --kbps 40 --output <新的绝对目录>`；输出目录必须不存在。此程序是受限机制原型，接收器只监听本机，每次启动两个独立模型进程，不需要重新安装环境。

不重启已完成目录，不自动恢复旧 D3、训练或裸误码矩阵。FP32、原生 rANS、画质和采样先保持不变。每次实验都披露实际传输字节和证据边界。
''')
state = read(HERE / 'project_status.json')
state.update(prototype='verified_16_real_frames', prototype_result=str(RUN / 'summary.json'),
    next='Reduce protocol diagnostic overhead, then matched serial/pipeline tests',
    acceptance={'first_frame_before_encode_end': True, 'all_frame_hashes_match': True,
                'baseline_payload_unchanged': True, 'source_reads_on_receiver': []},
    production_web_changed=False, FEC_implemented=False, Orin_tested=False)
dump(HERE / 'project_status.json', state)
dump(ARCHIVE / '整理完成.json', {'date': '2026-10-27', 'complete': True,
    'snapshots_verified': len(manifest['files']), 'production_code_unchanged': True,
    'desktop_launcher_targets_exist': True, 'large_assets_moved': False,
    'old_rounds_closed': True, 'new_optimization': str(HERE)})
validation = {'date': '2026-10-27', 'snapshot_files_verified': len(manifest['files']),
    'production_code_unchanged_since_snapshot': True, 'existing_launcher_targets_valid': True,
    'protocol_tests_passed': 5, 'real_neural_frames_verified': 16,
    'source_and_result_locations_preserved': True}
dump(HERE / '整理与原型核验.json', validation)
dump(RUN / 'final_validation.json', validation)
print(json.dumps(validation, ensure_ascii=False, indent=2))
