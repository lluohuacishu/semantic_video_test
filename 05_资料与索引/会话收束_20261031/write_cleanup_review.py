"""Render the existing proposal for human review. No files are deleted or moved."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    proposal = HERE / '待确认清理清单.json'
    data = json.loads(proposal.read_text(encoding='utf-8'))
    digest = hashlib.sha256(proposal.read_bytes()).hexdigest()
    execution_file = HERE / '清理执行记录.json'
    execution = json.loads(execution_file.read_text(encoding='utf-8-sig')) if execution_file.exists() else None
    completed = bool(execution and execution.get('passed'))
    gib = lambda value: f'{value / 1024 ** 3:.3f}'
    lines = [
        '# 清理清单与执行结果（2026-10-31）' if completed else '# 待确认清理清单（2026-10-31）', '',
        '**状态：用户已明确批准仅A组；48个文件已删除，B组11个文件全部保留，删除后核验通过。**' if completed else '**状态：仅盘点和校验，尚未删除任何文件；本清单本身不是删除授权。**', '',
        '按用户要求，先展示清单，确认范围后再次逐文件核对，最后才删除。此次A组已完成，不重复执行；B组未获删除授权，继续保留。' if completed else '按用户要求，先展示清单，确认范围后再进行最后一次逐文件核对，最后才删除。新会话不得因看到本清单而自动执行清理。', '',
        '工作区：`' + data['workspace'] + '`。本次只列出近期实验中能够明确判断用途的派生文件，不移动大目录，不改网页或模型。', '',
        '只读盘点共121,264个文件、约106.95 GiB逻辑大小；未遍历`.git`或目录联接。25.60 GiB的`SFISHTRACK.zip`是原始数据集，保留。', '',
        '| 组别 | 内容 | 文件数 | 逻辑大小 GiB | 建议 |',
        '|---|---|---:|---:|---|',
    ]
    for group in data['groups']:
        state = ('已清理' if group['id'].startswith('A') else '本次保留') if completed else ('建议清理' if group['id'].startswith('A') else '可选，影响离线重评')
        lines.append(f"| {group['id']} | {group['title']} | {group['file_count']} | {gib(group['logical_bytes'])} | {state} |")
    lines += [
        '', f"A组共48个文件，{gib(data['recommended_bytes'])} GiB；B组共11个文件，额外{gib(data['optional_bytes'])} GiB。",
        f"原始提案共59个文件、{gib(data['recommended_bytes'] + data['optional_bytes'])} GiB；本次只处理A组{gib(data['recommended_bytes'])} GiB。C盘空闲空间实测增加约6.210 GiB，该差值可能包含其他程序的磁盘活动。" if completed else f"全部合计59个文件，{gib(data['recommended_bytes'] + data['optional_bytes'])} GiB。实际磁盘释放量受NTFS分配和硬链接影响，尚未发生释放。", '',
        '## 删除的影响', '',
    ]
    for group in data['groups']:
        lines.append(f"- **{group['id']}：** {group['impact']}")
    lines += [
        '', 'A组ONNX只用于重新导出、建引擎或手动图诊断；现有有效`.engine`及其元数据均保留。需要重跑图诊断时应先重新导出ONNX。', '',
        '执行依据：[用户授权](清理授权.json) · [逐文件执行记录](清理执行记录.json) · [清理后核验](清理后核验.json)。初始提案JSON保持原样用于核对哈希，其中`approved=false`和`deletion_performed=false`描述提案生成时刻，当前结果以这些回执为准。' if completed else '本清单仍等待用户明确选择，不能从核验回执推断批准。', '',
        'B组删除后，现有MP4、码流和已计算指标仍可查看；但原实验脚本会复用已完成任务，不能指望它自动补回缺失RGB。重新进行逐像素评价须在新结果目录解码。', '',
        '## 明确保留', '',
        '- 所有原始数据集、原上传视频、检测框标注、模型/蒸馏权重、训练与推理环境。',
        '- 第16、17、18阶段的有效TensorRT引擎和构建元数据；`engine_packages_1050`运行依赖及角色精简权重包。',
        '- 整个`C5_loopback`目录，包含上传、现有码流、任务记录和播放视频；不打断正在运行的8773网页服务。',
        '- 第16阶段网页重启会读取的5份源数组与12份解码数组；不能因为已有MP4就删除这些数组。',
        '- 第17阶段三种后端各一份首轮短片RGB；第18阶段全部latent、两轮RGB与引擎。',
        '- 所有报告、JSON、CSV、日志、代码快照、分支、`.git`和目录联接。', '',
        '## 执行前最后核对', '',
        '1. 记录用户明确同意的组别以及本清单JSON的SHA256；没有确认则不执行。',
        '2. 对每个批准文件重新检查绝对路径、所有上级目录的联接属性、大小、修改时间和SHA256；任一项改变就停止。',
        '3. 确认无新实验在使用这些文件，网页源数组、有效引擎、依赖与保留副本仍然完整。',
        '4. 只删除JSON中明确列出的批准文件，不递归删除父目录、不扩展通配符范围。保留逐文件删除记录并复核网页入口及保留资产。', '',
        '删除前已执行`verify_cleanup_plan.py`并再次核对确切路径/哈希；删除后使用`verify_after_cleanup.py`核验。不要重跑删除前脚本或清理执行脚本：A组已不存在，执行记录也会阻止重复清理。' if completed else '只读校验脚本：`verify_cleanup_plan.py`；该脚本没有删除功能，不能代替用户批准。', '',
        '## 逐文件范围', '',
        '以下路径均相对于工作区。A组路径为已删记录，B组路径仍存在。' if completed else '以下路径均相对于工作区。机器可读清单含字节数、修改时间、SHA256与重复文件的保留对应项；另附CSV便于查阅。', '',
        f'清单JSON SHA256：`{digest}`。', '',
        '[完整JSON](待确认清理清单.json) · [逐文件CSV](待确认清理清单.csv) · [校验回执](清理前只读校验.json)', '',
    ]
    for group in data['groups']:
        lines += [f"### {group['id']} {group['title']}", '', '```text']
        for item in sorted(group['files'], key=lambda item: item['relative_path']):
            lines.append(item['relative_path'].replace('\\', '/'))
        lines += ['```', '']
    (HERE / '待确认清理清单.md').write_text('\n'.join(lines), encoding='utf-8')
    with (HERE / '待确认清理清单.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['group', 'relative_path', 'bytes', 'MiB', 'mtime_ns', 'sha256', 'hardlinks', 'retained_identical_file'])
        for group in data['groups']:
            for item in sorted(group['files'], key=lambda item: item['relative_path']):
                writer.writerow([group['id'], item['relative_path'], item['bytes'], f"{item['bytes'] / 1024 ** 2:.3f}",
                                 item['mtime_ns'], item['sha256'], item['hardlinks'], item.get('retained_identical_file', '')])
    print('Rendered review and CSV. No files deleted. Proposal SHA256:', digest)


if __name__ == '__main__':
    main()
