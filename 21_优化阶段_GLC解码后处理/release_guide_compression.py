"""Archive the completed video1 guide compression pilot without running inference."""

import argparse
from collections import Counter
import importlib
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

from receiver_structure import file_sha, load_json, save_json


PROFILES = ('M10', 'S10', 'M8', 'S8', 'S8I', 'S8C4')


def copy_verified(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    digest = file_sha(source)
    assert file_sha(destination) == digest, destination
    return dict(source=str(source), bytes=destination.stat().st_size, sha256=digest)


def encoder_audit(run):
    records = []
    for name in PROFILES:
        folder = run/'video1'/name
        send = load_json(folder/'send.json')
        for path in sorted(folder.glob('encode_*/encode.json')):
            trial = load_json(path)
            logfile = path.parent/'guide_pass.log'
            lines = logfile.read_text(encoding='utf-8').splitlines()
            options = dict(re.findall(r'(\S+?)=([^ ]+)', lines[0]))
            packet = path.parent/'guide.glcguide'
            assert file_sha(packet) == trial['sha256']
            assert packet.stat().st_size == trial['bytes']
            records.append(dict(profile=name, trial=path.parent.name,
                selected=trial['sha256'] == send['sha256'],
                requested_HEVC_kbps=trial['HEVC_target_kbps'],
                logged_integer_HEVC_kbps=int(options['bitrate']),
                complete_bytes=trial['bytes'], payload_bytes=trial['payload_bytes'],
                wrapper_bytes=trial['bytes']-trial['payload_bytes'],
                logged_options={k: options[k] for k in ('bframes', 'b-adapt', 'rc-lookahead',
                    'ref', 'me', 'subme', 'rd', 'rdoq-level', 'cbqpoffs', 'crqpoffs')},
                first_pass_frame_types=dict(Counter(re.findall(r'\btype:(\w)', '\n'.join(lines[1:])))),
                first_pass_log_sha256=file_sha(logfile), first_pass_header=lines[0]))
    save_json(run/'encoder_audit.json', dict(complete=True, attempts=records,
        interpretation='First-pass logs establish integer bitrate and actual preset options; caps are shared, measured rates are not matched.'))


def snapshot_code(stage, run, workspace):
    for name in ('evaluate_guide_compression', 'refine_guide_byte_search', 'receive_structure_guide'):
        importlib.import_module(name)
    repository = stage.parent
    files = {Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        value = getattr(module, '__file__', None)
        if value:
            path = Path(value).resolve()
            if path.suffix == '.py' and path.is_relative_to(repository):
                files.add(path)
    snapshot = run/'source_final'
    snapshot.mkdir(exist_ok=False)
    for source in sorted(files):
        copy_verified(source, snapshot/'repository'/source.relative_to(repository))
    upstream = run.parent/'deflicker_reef80_20261108_01/upstream'
    # The receiver dynamically imports this RAFT implementation during inference.
    for source in sorted((upstream/'src/models/stage_1/core').rglob('*.py')):
        copy_verified(source, snapshot/'upstream'/source.relative_to(upstream))
    for relative in ('src/__init__.py', 'src/models/__init__.py', 'src/models/stage_1/__init__.py'):
        if (upstream/relative).exists():
            copy_verified(upstream/relative, snapshot/'upstream'/relative)
    import cv2
    import numpy
    import torch
    ffmpeg = workspace/'01_项目/运行环境/FFmpeg/ffmpeg.exe'
    version = subprocess.run([str(ffmpeg), '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    save_json(snapshot/'environment.json', dict(python=sys.version, executable=sys.executable,
        platform=platform.platform(), numpy=numpy.__version__, opencv=cv2.__version__, torch=torch.__version__,
        ffmpeg=version, ffmpeg_sha256=file_sha(ffmpeg),
        note='Repository layout is retained; external runtime, base assets and shared weights remain required.'))
    manifest = {p.relative_to(snapshot).as_posix(): file_sha(p) for p in snapshot.rglob('*') if p.is_file()}
    save_json(snapshot/'manifest.json', manifest)
    return len(files), len(manifest)


def write_report(stage, run, view, summary, baseline):
    candidates = summary['candidates']
    rows = [f"| A12，已认可比较基线 | 16,187 | 12.9496 | — | {baseline['scores']['coral']['A12']['coarse_mae']:.6f} | {baseline['scores']['coral']['A12']['fine_ncc']:.6f} | — | 历史观感认可 |"]
    for name in PROFILES:
        r = candidates[name]
        label = 'S8（目标未贴近）' if name == 'S8' else name
        rows.append(f"| {label} | {r['bytes']:,} | {r['auxiliary_kbps']:.4f} | {r['auxiliary_saved_percent']:.2f}% | {r['coral_MAE']:.6f} | {r['coral_NCC']:.6f} | {r['full_MAE_change_percent']:+.2f}% | {'通过' if r['numeric_screen_pass'] else '未通过'} |")
    regional = []
    for name in ('S8I', 'S10', 'S8C4'):
        r = candidates[name]
        regional.append('| '+name+' | '+' | '.join(f"{r['quadrant_MAE_change_percent'][q]:+.2f}%" for q in ('top_left', 'top_right', 'bottom_left', 'bottom_right'))+' |')
    timings = '\n'.join(f"| {n} | {candidates[n]['encoder_selected_seconds']:.3f} | {candidates[n]['flow_seconds']:.3f} | {candidates[n]['render_seconds']:.3f} |" for n in ('M10', 'M8', 'S10', 'S8I', 'S8C4'))
    report = '''# 结构引导继续压码率：固定 T3 接收端的 video1 试验

2026-11-11。用户已对上一轮展示的 video1 G256_b12_T3 反馈“这回的还可以”，并要求先压缩辅助码率、视觉优化后做；讨论后的“那继续吧”授权本轮验证。将该版本简称 A12，作为本轮已认可的比较基线。认可不扩展到其他素材、其他引导档位或全部动态主体；K/U 保留为历史回归参照，零辅助 R0 已被用户拒绝。

本轮重点候选 **S8I：辅助 16,187 → 8,865 B，12.9496 → 7.0920 kbps，减少 45.23%**。原 GLC 基础流仍是 31,963 B /25.5704 kbps，两路合计由 38.5200 降到 32.6624 kbps。S10 为较高码率的备选：12,134 B /9.7072 kbps，辅助少 25.04%，珊瑚时序指标更好。两者的原速观看结果尚未由用户确认，暂不替代 A12。

## 设置与计费

仅 video1，同一段 150 帧、15 FPS、10 秒、512×288。引导保持 256×144、15 FPS、YUV420、同样源 RGB 下采样。改变的是发送端 libx265 两遍编码的码率与完整 preset（medium/slower）。实际 GLCGDE01 文件包含 HEVC 和 251–252 B 封装（含基础流绑定与 CRC），全部计费；观看 MP4 不作为传输字节。封装长度随载荷长度字段的位数变化，S8I 为 8,614 B HEVC 加 251 B 封装。

接收端规则完全保持已认可 T3：收到引导后放大、FP32 RAFT 24 次迭代估计双向运动，原可靠性检查再与 sigma3 局部亮度标准差至少 3 的纹理检查取交集；锚点 0/15/45/75/105/135、六帧旧参考过渡、2px 羽化与两次取整均保持。输出纹理来自历史原始 GLC RGB。引导只帮助计算运动和可靠区；这些计算结果是接收端本地缓存，不传输。基础包、原解码 RGB 和潜变量参考链保持原样。

## 所有候选结果

MAE 指在相同外部源对应支持上，运动对齐后扣除原片真实变化的低频时序误差，越低越好；NCC 为细频相邻相关，越高越好。“整帧 MAE 变化”也指该时序指标，不是静态 RGB 误差。百分比均相对 A12，正数表示误差增大。评分使用无损 RGB。

| 方案 | 辅助 B | 实际辅助 kbps | 辅助节省 | 珊瑚 MAE ↓ | 珊瑚 NCC ↑ | 整帧 MAE 变化 | 数值筛选 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
'''+ '\n'.join(rows)+'''

预先冻结的候选筛选为：珊瑚/整帧 MAE 不高于 A12 的 1.05 倍，珊瑚 NCC 下降不超过 0.02，整帧 PSNR 下降不超过 0.15 dB，源高变化像素帧差误差不超过 1.03 倍。它只用于候选排序，不是用户新增的硬标准，也不能代替视觉验收。

普通 medium 的两档降码率均未通过，slower 的 S10/S8I 通过。观察支持先用更高编码计算量争取压缩收益；不能把本轮结果只归因于 B 帧、运动搜索或某一个内部选项，也不能概括成所有视频都能得到相同收益。

## 极低码率控制与色度结果

最初 M10/S10 共用 12,500 B 上限，M8/S8 共用 10,000 B 上限，每档最多四次两遍编码，只根据完整文件字节选择。实际码率未严格配平，属于相同上限下的方案对照，不是严格同码率 preset 因果消融。

x265 第一遍日志记录整数 bitrate。小数 kbps 请求会映射到相同整数档，十秒短片极低码率 ABR 还有明显跳变：初始 S8 尝试内部 7/5/9/7 kbps，跳过了 6，最终只能选 6,114 B /4.8912 kbps。该结果已保留并计分，不能称为“8 kbps 实测”。补充冻结 S8I，按 6/7/5 整数档只看字节选择，得到 8,865 B；没有根据接收画面搜索编码参数。实际尝试、完整文件和第一遍配置头记录在 encoder_audit.json。

另冻结 S8C4：相同整数列表、slower、Cb/Cr QP offset +4。得到 9,153 B /7.3224 kbps，比 S8I 多 288 B；珊瑚 MAE 稍差，整帧时序 MAE 稍好。它未实现继续节省字节，因此当前压码率优先展示 S8I，保留色度结果供追溯。未扫描更多色度参数。

## 局部退步与观看重点

S8I 珊瑚 MAE 比 A12 低 15.17%，整帧时序 MAE 低 0.63%，但整帧 RGB PSNR 从 22.59024 到 22.53732 dB（低 0.05292 dB）。源独立选择的高变化像素帧差误差从 30.01215 到 29.94509，仅低 0.22%。该诊断不是语义分割，也不能独立证明鱼、气泡或边界运动正常。

| 方案 | 左上时序 MAE 变化 | 右上 | 左下 | 右下 |
| --- | ---: | ---: | ---: | ---: |
'''+ '\n'.join(regional)+'''

因此并非所有区域改善。S10 珊瑚 MAE 低 23.01%，整帧时序 MAE 低 5.59%，但左下仍退步 5.18%，静态 PSNR 也略降。优先检查 S8I 原速全画面的鱼体、气泡、下方结构与遮挡退出，再看珊瑚放大；S10 用于比较压缩幅度与稳定指标的取舍。静态接触表已检查，原速主观反馈待用户提供。

## 核验与计算成本

六份实际候选的两阶段接收均核验：第一阶段只有五项获准数据输入，T3 阶段九项（包括本地缓存），各自七个未传输源输入负例均被阻断。核对实际包、基础包、接收回执与输出哈希；实际引导解码逐位相同，旧可靠性函数和固定 T3 的 CPU 重放 RGB/alpha/共享数组逐位相同，零 alpha 像素等于原 GLC，T3 alpha 只能减小。Python 读取审计是实验边界约束，不是操作系统安全沙箱。

A12、K、raw 的历史全部区域分数与高变化诊断精确重现；13 份冻结输入及五份冻结代码哈希保持一致。六段 MP4 均为 150 帧、15 FPS、10 秒，首/中/末帧面板身份已检查；实际传输副本、源码快照和小证据另有哈希清单。沿用既有包格式，未重新宣称执行本轮未运行的格式负例测试。

| 方案 | 选中档两遍编码秒 | RAFT 秒 | T3 门控融合秒 |
| --- | ---: | ---: | ---: |
'''+timings+'''

时间对应本机十秒片段的单次测量；表内编码不包含未选中的搜索尝试，RAFT 与门控也不代表完整端到端延迟。slower 编码比 medium 花更多计算时间。当前包含双遍编码、B 帧和整段推理，未验证实时分包、丢包恢复或实时性能。

## 当前决定与复现入口

保持“先降辅助码率，再优化视觉”的顺序。S8I 是待看画面的压缩重点，S10 为较高码率备选；A12 保留为已认可的 video1 比较基线。此次没有扩展 video2/3、留出或长片，也没有重新运行两路合计码率的 H.265 主输出比较。

- 运行：工作区 `02_实验数据/优化_GLC后处理/guide_compression_20261111_01`。
- 观看：工作区 `05_成果与观看/GLC_结构引导继续压码率_20261111`。01–03 为 S8I；04–06 为 S10。左右均为左 A12、右候选；四宫格上排原片/原 GLC，下排 A12/候选。
- 实际文件：观看目录 `实际传输文件/video1/{A12,S8I,S10}/base.glcsnd + guide.glcguide`。每套只需要一份基础包与一份引导包，不是三套合并计费。
- 小证据：[evidence_20261111/guide_compression](evidence_20261111/guide_compression/README.md)，含冻结计划、编码日志摘要、全部评分、接收核验、decision/visual_review、源码及清单。大数组与原始日志留在运行目录。
- 编码与独立接收：`compress_structure_guide.py initial`；补整数档：`refine_guide_byte_search.py`（色度档加 `--chroma 4`）；评分与观看：`evaluate_guide_compression.py score/previews`。均接受 `--workspace` 和 `--run`，评分另需 `--profiles` 与唯一 `--tag`。复现请使用新输出目录，脚本不会覆盖已有实验。

新代码、证据和文档仍未提交或推送；当前分支 codex/glc-temporal-stability，记录时 HEAD 7f717e5。
'''
    (stage/'结构引导继续压码率_20261111.md').write_text(report, encoding='utf-8')
    (view/'README.md').write_text('''# 结构引导继续压码率：video1 原速对照

重点 S8I：辅助从 A12 的 12.9496 降到 7.0920 kbps，少 45.23%。备选 S10 为 9.7072 kbps，珊瑚时序指标更好。新候选观感待确认，A12 是此前用户认可的比较基线。

01/02：左 A12，右 S8I，全画面/珊瑚放大；03：上排原片/原 GLC，下排 A12/S8I。
04/05：左 A12，右 S10，全画面/珊瑚放大；06：上排原片/原 GLC，下排 A12/S10。
均为 150 帧、15 FPS、10 秒原速。评分用无损 RGB，MP4 只供观看。

S8I 下方两个区域的时序误差仍有约 7% 的退步，数值筛选不能代替观看鱼、气泡和边界。此次仅 video1。
实际传输文件每套为 base.glcsnd + guide.glcguide；共享软件/模型预装，接收端本地光流与掩码不传输。
''', encoding='utf-8')


def release(workspace, run):
    stage = Path(__file__).resolve().parent
    view = workspace/'05_成果与观看/GLC_结构引导继续压码率_20261111'
    evidence = stage/'evidence_20261111/guide_compression'
    for destination in (view, evidence, run/'source_final'):
        if destination.exists():
            raise FileExistsError(destination)
    plan = load_json(run/'execution_plan.json')
    assert len(plan['inputs_sha256']) == 13
    assert all(file_sha(Path(p)) == sha for p, sha in plan['inputs_sha256'].items())
    assert all(file_sha(stage/n) == sha for n, sha in plan['code_sha256'].items())
    summary = load_json(run/'evaluation_summary.json')
    assert summary['complete'] and set(summary['candidates']) == set(PROFILES)
    for name in PROFILES:
        folder = run/'video1'/name
        assert all(load_json(folder/'verification.json').values())
        send = load_json(folder/'send.json')
        assert file_sha(folder/'guide.glcguide') == send['sha256']
        assert send['bytes'] == (folder/'guide.glcguide').stat().st_size <= send['cap_bytes']
        for sub, code in (('flow_receive', 'receive_structure_guide.py'), ('receive', 'receive_guide_texture.py')):
            receipt = load_json(folder/sub/'receipt.json')
            assert receipt['complete'] and receipt['source_reads_blocked']
            assert receipt['code_sha256'] == file_sha(stage/code)
            assert receipt['guide_codec_sha256'] == file_sha(stage/'structure_guide_stream.py')
            for filename, digest in receipt['outputs_sha256'].items():
                assert file_sha(folder/sub/filename) == digest
    baseline = load_json(run/'evaluation/initial/summary.json')
    for tag in ('initial', 'chroma'):
        record = load_json(run/'evaluation'/tag/'summary.json')
        assert record['archived_scores_exact'] and record['archived_dynamic_diagnostic_exact']
        assert record['code_sha256'] == file_sha(stage/'evaluate_guide_compression.py')
    prior_review = load_json(run.parent/'structure_guidance_20261111_01/visual_review.json')
    assert prior_review['status'] == 'user_accepted_video1_direction'
    save_json(run/'visual_review.json', dict(date='2026-11-11',
        accepted_reference='A12 = G256_b12_T3, video1 only', user_quote=prior_review['user_quote'],
        priority='Compress auxiliary first; visual optimization later.',
        candidates={n: 'pending_user_original_speed_review' for n in PROFILES},
        static_contact_sheets_checked=['S8I', 'S10'], candidate_promoted=False))
    save_json(run/'decision.json', dict(complete=True, focus_candidate='S8I', alternative='S10',
        accepted_reference='A12', new_candidates_visually_accepted=False,
        numeric_screen_pass=[n for n in PROFILES if summary['candidates'][n]['numeric_screen_pass']],
        chroma_result='S8C4 adds 288 bytes versus S8I; no further bitrate saving.',
        scope='Only video1 150 frames, fixed T3 receiver; no transfer or visual-rule tuning.',
        next='Review S8I and S10 original-speed comparisons against accepted A12 before promoting a new operating point.'))
    encoder_audit(run)
    view.mkdir(); copies = {}; index = 1
    preview_check = load_json(run/'previews/verification.json')
    assert preview_check['complete'] and len(preview_check['previews']) == 6
    for name in ('S8I', 'S10'):
        for source_name, label in ((f'A12_vs_{name}_full.mp4', f'左A12_右{name}_全画面'),
                                   (f'A12_vs_{name}_coral.mp4', f'左A12_右{name}_珊瑚放大'),
                                   (f'source_raw_A12_{name}.mp4', f'四宫格_原片_原GLC_A12_{name}')):
            source = run/'previews'/source_name
            assert file_sha(source) == preview_check['previews'][source_name]['sha256']
            destination = view/f'{index:02d}_video1_{label}.mp4'
            copies[destination.name] = copy_verified(source, destination); index += 1
    base = run.parent/'source_dense_flow_three_clips_20261109_01/video1/base.glcsnd'
    packets = dict(A12=run.parent/'structure_guidance_20261111_01/video1/G256_b12_T3/guide.glcguide',
        **{n: run/'video1'/n/'guide.glcguide' for n in ('S8I', 'S10')})
    for name, packet in packets.items():
        for source in (base, packet):
            destination = view/'实际传输文件/video1'/name/source.name
            copies[destination.relative_to(view).as_posix()] = copy_verified(source, destination)
    save_json(view/'来源核验.json', dict(complete=True, files=copies))
    save_json(run/'viewing_copies.json', dict(complete=True, directory=str(view), files=copies))
    source_count, manifest_count = snapshot_code(stage, run, workspace)
    save_json(run/'release_verification.json', dict(complete=True, profiles=list(PROFILES),
        frozen_input_count=13, frozen_inputs_unchanged=True, frozen_code_count=len(plan['code_sha256']),
        frozen_code_unchanged=True, receiver_outputs_still_match_receipts=True,
        candidate_replay_verified_at_scoring=True, archived_scores_exact=True,
        preview_count=6, original_speed_and_panel_identity_verified=True, copied_files_exact=True,
        repository_source_files=source_count, source_manifest_files=manifest_count,
        visual_acceptance='pending', new_inference_during_release=False))
    write_report(stage, run, view, summary, baseline)
    evidence.mkdir(parents=True)
    for source in run.rglob('*'):
        if source.is_file() and source.suffix in ('.json', '.csv', '.py'):
            copy_verified(source, evidence/source.relative_to(run))
    (evidence/'README.md').write_text('''# 固定 T3 的结构引导压码率证据

execution_plan.json 是初始冻结计划；video1/S8I 与 S8C4 的 plan.json 是补整数档和色度档的预先计划。
evaluation_summary.json 汇总全部六档；evaluation 内保存逐帧/逐区域评分，原速视频身份核验在 previews/verification.json。
encoder_audit.json 保留全部编码尝试的整数码率、完整字节、选中状态及第一遍日志头；原始日志、实际码流与大数组在本机运行目录。
每档 send/flow_receive/receive/verification 记录发送、独立接收和不变 T3 重放。
decision.json 与 visual_review.json 区分已认可 A12 和新候选待观看。source_final 保留仓库相对结构、RAFT 源码、环境及哈希。
release_verification.json / viewing_copies.json 核验冻结输入和展示副本；manifest.json 覆盖本目录文件（不包含自身）。
''', encoding='utf-8')
    save_json(evidence/'manifest.json', {p.relative_to(evidence).as_posix(): file_sha(p)
        for p in evidence.rglob('*') if p.is_file()})
    print('GUIDE COMPRESSION RELEASE COMPLETE', view, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    arguments = parser.parse_args()
    release(arguments.workspace.resolve(), arguments.run.resolve())
