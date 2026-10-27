from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
WORK = REPO.parents[2]
ARCHIVE = WORK / '03_历史归档/阶段收束_20261027'
OPT = REPO / '10_优化阶段_流水线编解码'
DATA = WORK / '02_实验数据/优化_流水线编解码'
C = REPO / '08_新阶段C_编解码加速与网页展示'
RESULTS = WORK / '02_实验数据/C_编解码加速与网页展示'

def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + '\n', encoding='utf-8')

def dump(path, obj):
    write(path, json.dumps(obj, ensure_ascii=False, indent=2))

def link(path, title):
    return f'[{title}](<{path.as_posix()}>)'

if (ARCHIVE / '整理完成.json').exists():
    raise SystemExit('Closure is complete; do not rerun the one-time organizer.')
if ARCHIVE.exists() and '--resume' not in sys.argv:
    raise SystemExit('Archive already exists; refusing to overwrite the closure snapshot.')
ARCHIVE.mkdir(parents=True, exist_ok=True)
OPT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
before = [WORK / '开始这里.md', WORK / '01_项目/README.md', WORK / '02_实验数据/README.md',
          WORK / '03_历史归档/README.md', REPO / 'README.md', REPO / '交接文档.md',
          REPO / '当前项目状态与接续.md', REPO / '项目总结与未来方向.md',
          REPO / '05_资料与索引/文件与会话接续索引_20261022.md',
          REPO / '09_新阶段D_语义目标增强/README.md',
          REPO / '09_新阶段D_语义目标增强/d_status_20261021.json', RESULTS / 'README.md']
code = [C / name for name in ('c1_runtime.py', 'c1_io.py', 'c2_native.py', 'c4_config.py',
        'c5_server.py', 'c5_transport.py', 'c5_resident.py', 'c5_compat.py',
        'bench_glc_precision.py', 'bench_glc_errors.py', 'profile_glc_operators.py')]
code += [REPO / '04_外部依赖快照/semantic-video-training/scripts' / name
         for name in ('glc_video_codec.py', 'glc_container.py', 'glc_entropy.py')]
code += [C / 'c5_console/index.html']
evidence = [RESULTS / p for p in (
    'GLC_operator_profile_20261025_recovered_01/validation.json',
    'GLC_operator_profile_20261025_recovered_01/GLC算子级诊断报告.md',
    'GLC_precision_20261026_native_05/summary.json',
    'GLC_precision_20261026_native_05/文件任务与Orin优化说明.md',
    'GLC_BER_20261026_01/manifest.json', 'GLC_BER_20261026_01/summary.json',
    'GLC_BER_20261026_01/validation.json', 'GLC_BER_20261026_01/误码实验报告.md')]
manifest = []
for category, paths in [('入口更新前', before), ('基线代码', code), ('关键证据', evidence)]:
    for source in paths:
        target = ARCHIVE / category / source.relative_to(WORK)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
            assert target.read_bytes() == source.read_bytes()
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest.append({'category': category, 'source': str(source), 'snapshot': str(target),
                         'bytes': target.stat().st_size, 'sha256': digest})
git = subprocess.check_output(['git', 'status', '--short'], cwd=REPO, encoding='utf-8')
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, encoding='utf-8').strip()
if not (ARCHIVE / '整理前git状态.txt').exists():
    write(ARCHIVE / '整理前git状态.txt', git)
dump(ARCHIVE / '归档清单.json', {'date': '2026-10-27', 'git_head': head, 'files': manifest,
     'copied_bytes': sum(r['bytes'] for r in manifest), 'moved_or_deleted': [],
     'reason': '用户要求旧轮次到此为止，整理归档后开始流水线优化研究。'})

summary = '''# 项目收束总结：转入流水线与部署优化

更新：2026-10-27。按用户要求，之前所有轮次到此收束，旧待办不再自动继续。未完成的验证仍记为未完成，不因归档而视为通过。新的工作目标是：在保持可用画质和码流正确性的前提下，缩短水下视频文件从编码到接收重建的总等待时间，最终面向 Jetson Orin NX Super 16GB。用户不要求实时播放帧率。

## 已有成果与结论

| 工作 | 已有成果 | 结论和限制 |
|---|---|---|
| 旧 A–F / H.265、MLVC | 语义选区、ROI、实际码流与收发原型 | 历史路线保留；H.265 展示现在使用旧语义改善流程 |
| 新 A / GLC 选型 | 官方模型本地运行、图像与视频权重、真实码流 | GLC 保留为当前优化基线，未换模型 |
| 新 B / GLC 闭环 | 独立接收解码、四档压缩与观看副本 | MP4 是预览，真正发送数据为 .glcv |
| 新 C / 加速和网页 | 原生 C rANS、常驻任务、独立收发网页 | 历史同期组合约 1.991×，不能等同新实验或 Orin 速度 |
| 新 D / 语义增强 | 低通、预测目标保护、GT 诊断与网页集成 | 语义选择净收益未证明；011 预测回退，021 可运行；D3 只有协议与静态预算，结束待续状态 |
| FaunaCodec / H.265 / 分割展示 | 独立网页与桌面启动入口 | 保持可用，作为对照与展示资产，不开启新模型研究 |
| 算子诊断 | 66 帧真实码流及独立解码验证 | GPU 神经网络为主要热点；该分辨率无显存容量不足证据，未取得硬件吞吐/占用计数 |
| 精度试验 | TF32、FP16 试探及原生后端补测 | 旧 01–04 误用 Python 熵后端，不作当前原生基准；native_05 混合精度收益有限，不代表优化上限 |
| 误码试验 | 6000 个整文件注错、252 个载荷试验 | 实际损坏的 161 个载荷样本均未完整解码，包含 48 次单比特试验；保护与纠错尚未实现 |

## 冻结的性能口径

本机 RTX 4060 Laptop 8GB，Python 3.12、PyTorch 2.5.1+cu124。正式补测为纯 GLC、640×360、12.5 fps、66 帧、q0、GOP64、reset32，原生 rANS，进程暖机后运行。

| native_05（两轮平均） | 编码整任务 | 解码整任务 | 编码核心 fps | 解码核心 fps |
|---|---:|---:|---:|---:|
| FP32 | 18.382 s | 11.643 s | 3.784 | 5.822 |
| RGB 部分 FP16、其余 FP32 | 18.060 s | 11.037 s | 3.854 | 6.147 |

这些任务包含预览与校验，不含首次加载和网络传输；不同日期的单次诊断不能直接拼成加速比。两模式压缩载荷均为 13,717 B，载荷哈希一致；FP32 文件为 15,049 B。混合方案没有改成生产默认。全 FP16 曾出现非有限值，不能直接上线。

Orin 尚未实测，也没有安装新的 WSL 环境或替换当前模型。157 TOPS 不能直接换算成这个 FP32 工作负载的速度。当前没有证据证明 GPU 已被用满。

## 误码与流式的结论

正常入口整文件 SHA-256 检错，不能自行纠错。裸载荷即使只改一个比特也可能打断熵解码和参考链；一个样本在第 33 帧仅产生极小视觉变化，第 34 帧即失败。6 次手动跳到干净 I 帧的诊断成功，但正式接收器没有自动恢复功能。

流水线仍可实现：当前帧完整、校验通过且前序参考正确，就可以先解码，后续帧继续接收。真实水下链路需要确认调制解调器能力，再选择纠错、交织和必要的补传。不能把软件翻位/TCP限速称为实测水声信道。先研究流水线机制，随后把纠错加入同一条链路；本阶段不自动恢复旧语义矩阵。

## 文件管理决定

源码、大视频、权重、环境、完整结果和兼容链接保持原路径，避免破坏网页与证据。旧入口已在历史归档集中保存，关键源码/报告/JSON建立小快照和哈希清单。新代码统一放在 `10_优化阶段_流水线编解码`，新实验统一放在工作区 `02_实验数据/优化_流水线编解码`。既有失败、中断和后端误用记录全部保留。

旧 D 预算协议保留用于解释历史结论；优化原型按相同编码负载报告新增传输开销，不把原型自动认定为满足 40 kbps / 10% 应用验收。模型精度、画质设置、音频范围和传输速率分别记录。
'''
write(REPO / '项目总结与未来方向.md', summary)

index = '# 项目文件索引（2026-10-27）\n\n旧轮次已收束；新优化只在新目录开展。\n\n'
entries = [
    (REPO / '项目总结与未来方向.md', '项目总结与结论'),
    (OPT / 'README.md', '当前流水线优化入口'),
    (DATA, '新优化实验结果'),
    (ARCHIVE / 'README.md', '本次归档说明与快照'),
    (RESULTS / 'GLC_operator_profile_20261025_recovered_01/GLC算子级诊断报告.md', '有效算子诊断'),
    (RESULTS / 'GLC_precision_20261026_native_05/文件任务与Orin优化说明.md', '有效原生后端精度补测'),
    (RESULTS / 'GLC_BER_20261026_01/误码实验报告.md', '误码报告与样本'),
    (WORK / '02_实验数据/D_语义目标增强', '历史语义实验'),
    (WORK / '汇报展示材料', '原视频、码流、重建和对比网页'),
    (WORK / '01_项目/运行环境', 'Python、FFmpeg、模型和旧工程路径'),
    (WORK / '02_实验数据/D_语义目标增强/原始数据/SFISHTRACK.zip', '原始视频数据集 SFISHTRACK'),
    (REPO / '01_主工程/data', '历史数据与来源索引'),
    (WORK / '01_项目/代码/FaunaCodec', 'FaunaCodec 展示代码'),
    (C / 'h265_demo', '语义 H.265 展示代码'),
    (C / 'segmentation_demo', '语义分割展示代码')]
for path, name in entries:
    assert path.exists() or path in (OPT / 'README.md', ARCHIVE / 'README.md'), str(path)
    index += '- ' + link(path, name) + '\n'
index += '\n## 桌面启动入口\n\n'
for name, description in [('打开网页.cmd', 'GLC：8773 / 8774'), ('H265.cmd', '语义 H.265：8777'),
                          ('语义分割.cmd', '语义分割：8778'), ('FaunaCodec.cmd', 'FaunaCodec：8775')]:
    path = Path('C:/Users/lenovo/Desktop/临时/材料') / name
    assert path.is_file()
    index += '- ' + link(path, description) + '\n'
index += '\n端口是启动配置，目录整理不代表服务正在运行。运行环境和旧依赖链接不迁移。\n'
write(REPO / '05_资料与索引/文件索引_20261027.md', index)

handoff = '''# 当前接续：GLC 流水线与部署优化

更新：2026-10-27。用户明确要求“之前那些轮到此为止”，先整理项目，再研究边编码、边传输、边解码。旧 A–F、新 A–D、性能诊断和裸误码试验均收束为历史基线；没有把未通过的项目标成通过。D3 未开展，不自动恢复。

阅读顺序：[项目总结](项目总结与未来方向.md) → [文件索引](05_资料与索引/文件索引_20261027.md) → [优化入口](10_优化阶段_流水线编解码/README.md)。旧入口和关键证据快照在工作区 `03_历史归档/阶段收束_20261027`。

当前目标是文件任务总等待时间，最终目标设备为 Jetson Orin NX Super 16GB；不要求实时播放。先保持普通 GLC / FP32 / 原生 rANS、相同采样与质量参数，研究调度和传输重叠。纯 GLC 是本轮基线，不恢复语义训练、D3 双流或旧矩阵，不重跑已完成诊断。

已确认模型逐帧依赖过去参考，没有等待未来帧的网络要求。现有阻塞位于：C1 收齐全部压缩帧才写出文件；C5 等完整编码文件才能预约和传输；接收端全文件校验后才派发解码；GLCV读取器先读完整文件。具体方案和原型证据写入优化目录。

码流正确性、传输保护和数值兼容分别验证。纠错不能用取消校验代替；水下设备、纠错后误码/丢包率和往返时间尚未提供。本机 TCP 原型只验证流水线，不代表真实水声表现。本机单 GPU 上发送与接收争抢计算，不能据此承诺两台设备或 Orin 的提速。

唯一 Git 工程为本目录，分支仍为 `codex/glc-d`；当前整理不切换分支、提交、推送或覆盖已有修改。工作区根部 `.git` 不用于提交。模型、原视频和运行环境保持原位置；桌面入口保持原路径。

新代码只放在 `10_优化阶段_流水线编解码`，实验放在工作区 `02_实验数据/优化_流水线编解码`，每次用新的目录并保存参数、逐帧时间、校验与失败原因。网页当前仍是整文件收发；不要把独立原型描述成已经完成网页替换。
'''
write(REPO / '交接文档.md', handoff)
write(REPO / '当前项目状态与接续.md', '''# 当前项目状态

更新：2026-10-27。

| 项目 | 状态 |
|---|---|
| 旧轮次 | 全部收束；保留成果、失败证据和未验证限制 |
| D3 | 仅历史规划，未开展，旧待办结束 |
| 当前任务 | 研究 GLC 边编码、边传输、边解码 |
| 最终设备 | Jetson Orin NX Super 16GB，尚未实测 |
| 优化目标 | 缩短文件端到端等待，不以实时 fps 为验收条件 |
| 基线 | 普通 GLC，FP32，原生 C rANS，编码质量参数固定 |
| 误码 | 裸码流敏感，纠错与交织待研究，不以关闭校验处理 |
| 现有网页 | 保留；目前仍是整文件收发流程 |
| 新代码/数据 | 10_优化阶段_流水线编解码 / 工作区02_实验数据/优化_流水线编解码 |

[完整接续](交接文档.md)、[项目总结](项目总结与未来方向.md)、[优化进展](10_优化阶段_流水线编解码/README.md)、[文件索引](05_资料与索引/文件索引_20261027.md)。
''')
write(REPO / 'README.md', '''# 水下视频语义通信

更新：2026-10-27。之前各轮研究已收束，当前转入 GLC 流水线与 Orin 部署优化。目标是减少文件编码、传输、接收重建的总等待；用户不要求实时播放。

- [当前接续](交接文档.md)
- [项目总结与已知限制](项目总结与未来方向.md)
- [文件索引与网页启动](05_资料与索引/文件索引_20261027.md)
- [流水线优化研究](10_优化阶段_流水线编解码/README.md)

GLC、语义 H.265、语义分割与 FaunaCodec 展示均保留。旧语义增强净收益未证明，D3 只做过规划；原生后端精度补测和裸误码试验已归档。旧报告的“下一步”不再自动执行。

唯一工程是本目录，保留当前分支和所有未提交修改；大文件、权重和环境保持原路径。当前网页仍采用整文件流程，新的流水线原型独立验证后再接入。
''')
write(WORK / '开始这里.md', '''# 水下视频通信项目：从这里开始

更新：2026-10-27。**之前所有轮次到此收束，当前开始研究 GLC 边编码、边传输、边解码，最终面向 Orin 文件任务。**

1. [项目总结](01_项目/代码/semantic_video_test/项目总结与未来方向.md)：已有成果、失败结果和边界。
2. [文件索引](01_项目/代码/semantic_video_test/05_资料与索引/文件索引_20261027.md)：报告、数据、环境和四个网页入口。
3. [当前优化](01_项目/代码/semantic_video_test/10_优化阶段_流水线编解码/README.md)：机制研究、原型和下一步。
4. [本次归档](03_历史归档/阶段收束_20261027/README.md)：整理前入口、基线源码与关键证据快照。

新会话先读仓库[交接文档](01_项目/代码/semantic_video_test/交接文档.md)。新实验结果统一放在[优化数据](02_实验数据/优化_流水线编解码)。原数据、权重、运行环境和桌面启动器保持原位置。

桌面 `C:\\Users\\lenovo\\Desktop\\临时\\材料` 中保留：`打开网页.cmd`（GLC）、`H265.cmd`、`语义分割.cmd`、`FaunaCodec.cmd`。现有网页仍是整文件收发；新流水线研究进展以优化入口为准。
''')
write(WORK / '01_项目/README.md', '''# 当前项目

更新：2026-10-27。旧轮次收束，新工作为 GLC 流水线与 Orin 文件任务优化。

- [唯一 Git 工程](代码/semantic_video_test)
- [当前接续](代码/semantic_video_test/交接文档.md)
- [项目总结](代码/semantic_video_test/项目总结与未来方向.md)
- [新优化代码与研究](代码/semantic_video_test/10_优化阶段_流水线编解码)
- [运行环境](运行环境)

现有工程、交付包、运行环境和兼容链接保留原位。工作区根部隐藏 .git 不作为工程仓库。
''')
banner = '> 2026-10-27 收束说明：本文属于历史记录，旧“下一步”不再自动执行。当前转入 GLC 流水线优化，详见工作区“开始这里.md”和仓库“交接文档.md”。\n\n'
for path in (WORK / '02_实验数据/README.md', WORK / '03_历史归档/README.md', RESULTS / 'README.md',
             REPO / '05_资料与索引/文件与会话接续索引_20261022.md', REPO / '09_新阶段D_语义目标增强/README.md'):
    original = path.read_text('utf-8-sig')
    if not original.startswith(banner):
        write(path, banner + original)
for path in (WORK / '02_实验数据/D_语义目标增强/README.md', C / 'README.md'):
    target = ARCHIVE / '入口更新前' / path.relative_to(WORK)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(path, target)
    manifest.append({'category': '入口更新前', 'source': str(path), 'snapshot': str(target),
        'bytes': target.stat().st_size, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    original = path.read_text('utf-8-sig')
    if not original.startswith(banner):
        write(path, banner + original)
status_path = REPO / '09_新阶段D_语义目标增强/d_status_20261021.json'
status = json.loads(status_path.read_text('utf-8'))
status.setdefault('status_before_closure', status['status'])
status.update(date='2026-10-27', status='closed_at_user_request_D3_not_executed',
              automatic_continuation=False, current_handoff='../交接文档.md',
              closure_reason='用户要求旧轮次到此为止，转入独立流水线优化。历史阶段结果和限制保留。')
dump(status_path, status)
write(DATA / 'README.md', '''# 流水线优化实验

开始日期：2026-10-27。每次实验使用独立新目录；旧 C/D 实验不迁入、不覆盖。

当前研究：编码出一帧即发送，接收端逐帧校验后按参考顺序解码。必须记录首帧可用时间、编码结束、传输结束、最后一帧完成、实际应用字节、队列等待和独立解码校验。区分本机共享 GPU、双机以及实际 Orin 证据。

代码和最新结果见工作区 `01_项目/代码/semantic_video_test/10_优化阶段_流水线编解码`。
''')
write(OPT / 'README.md', '''# GLC 流水线优化

开始日期：2026-10-27。旧轮次已经收束。本目录研究编码、传输、解码重叠，目标是减少文件任务总等待，最终面向 Orin。当前先做独立小原型，未接入原网页。

第一步：检查现有逐帧接口、封装和完整文件前置条件，设计有界队列与逐帧校验；用有限帧数证明接收端在发送端编码结束之前已经开始并完成首帧解码，检查双方 RGB/参考状态一致，以及接收器不读取源视频。

保持普通 GLC / FP32 / 原生 rANS，不改变模型、权重、GOP、精度或语义路径。原型中新增的传输元数据和哈希必须计入实际字节。TCP 原型不代表水下纠错能力；FEC/交织以后接在交付完整帧之前。

研究结果、代码与验证完成后在此补充链接。
''')
write(ARCHIVE / 'README.md', '''# 旧轮次收束归档（2026-10-27）

用户要求之前各轮到此为止，整理后转入 GLC 流水线优化。本目录集中保存更新前的入口文档、当前关键代码和已完成实验的小型报告/JSON副本。

- `入口更新前`：旧 README、交接、状态与总结；其中旧待办不再执行。
- `基线代码`：编码、封装、传输、原生熵后端及诊断脚本快照。
- `关键证据`：算子、原生后端精度、误码试验的报告与汇总。
- `归档清单.json`：原路径、快照路径、字节数和 SHA-256。
- `整理前git状态.txt`：整理之前已有修改和未跟踪文件。

快照保持原文，内部相对链接以原位置为准；需要完整视频/码流/日志时按清单回到原结果目录。未复制数据集、权重、环境或大视频，未移动或删除旧结果。没有提交或推送。

当前权威入口是工作区“开始这里.md”及仓库“交接文档.md”。收束不意味着所有研究目标已达成：D3未开展、语义净收益未证明、Orin未实测、纠错未实现。
''')
dump(ARCHIVE / '归档清单.json', {'date': '2026-10-27', 'git_head': head, 'files': manifest,
     'copied_bytes': sum(r['bytes'] for r in manifest), 'moved_or_deleted': [],
     'reason': '用户要求旧轮次到此为止，整理归档后开始流水线优化研究。'})
dump(OPT / 'project_status.json', {'date': '2026-10-27', 'phase': 'pipeline_optimization',
     'old_rounds': 'closed', 'D3_executed': False, 'web_pipeline_integrated': False,
     'target': 'Orin NX Super 16GB file completion latency; no realtime requirement',
     'prototype': 'in_progress', 'data_root': str(DATA), 'closure_archive': str(ARCHIVE)})
dump(ARCHIVE / '整理完成.json', {'date': '2026-10-27', 'complete': True})
print(json.dumps({'archive': str(ARCHIVE), 'snapshot_files': len(manifest),
                  'snapshot_bytes': sum(r['bytes'] for r in manifest), 'new_code': str(OPT),
                  'new_data': str(DATA)}, ensure_ascii=False))
