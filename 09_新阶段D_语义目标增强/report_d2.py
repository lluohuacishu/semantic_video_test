"""Render the measured D2 appendix without making a visual acceptance claim."""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from prepare_d2 import read_json, save_json


def number(value, digits=3, signed=False):
    if value is None:
        return 'NA'
    return format(value, f'{"+" if signed else ""}.{digits}f')


def table(headers, rows):
    return '\n'.join([
        '| ' + ' | '.join(headers) + ' |',
        '|' + '|'.join('---' for _ in headers) + '|',
        *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows],
    ])


def run(args):
    root = args.run
    result = read_json(root / 'evaluation.json')
    points = {point['id']: point for point in result['points']}
    lines = [
        '# D2 两片机制诊断：完整数值附表', '',
        '实验登记日期：2026-10-21。本表由 `report_d2.py` 从 `evaluation.json` 生成；'
        '所有画质指标来自独立接收端解码的无损 PNG，与未处理原始 RGB 比较。'
        '鱼区覆盖原 GT 中的全部鱼，包括预测漏检；其余画面不等于纯背景。', '',
        '本轮仅有同一来源组的两个开发片段，各 64 帧、1920×1080、30000/1001 fps，'
        '每片时长 4004/1875 秒。没有训练、留出推理或潜水员视频验收。'
        '本表不作完整 D 轮门槛、主观观看或泛化通过声明。', '',
        'A=原图 GLC，B=整图低通，C=冻结预测保护后低通，O=鱼 GT 保护后低通。'
        'q 只取 0/32/63；按实际字节选点，未用画质选 q。', '',
        '## 1. 预先冻结的预算', '',
    ]
    lines.append(table(['片段', 'A@q32 / S0（B）', 'B10 上限（B）', '基准文件码率（kbps）'], [
        [f'{int(video):03d}', f'{anchor["S0"]:,}', f'{anchor["B10"]:,}',
         number(points[anchor['task']]['file_kbps'])]
        for video, anchor in result['budget_anchors'].items()
    ]))
    lines += ['', 'B0=S0，B10=floor(11×S0/10)。硬上限逐片检查，'
              '不能相互抵消。完整 glcv 已包含封装，发送端掩码不传输，新增每片侧信息为 0 B。'
              '30–40 kbps 为文件码率参考，不是本轮已达到的结论。', '',
              '## 2. 全部 24 个实际文件', '']
    lines.append(table(['点', '文件（B）', '相对 S0', '≤B10', '文件 kbps', '鱼 PSNR', '其余 PSNR', '全图 PSNR'], [
        [p['id'], f'{p["file_bytes"]:,}', number(100*p['growth_vs_S0'], 2, True)+'%',
         '是' if p['within_B10'] else '否：仅诊断', number(p['file_kbps']),
         *[number(p['regions'][region]['psnr_db']) for region in ('fish', 'rest', 'all')]]
        for p in result['points']
    ]))
    lines += ['', 'PSNR 单位 dB。先跨帧汇总三通道 SSE/样本数，再转换为 PSNR；'
              '不是逐帧 PSNR 的平均。未压缩尺寸和容器压缩倍数不能替代这些实际文件码率。', '',
              '## 3. 同一上限下按字节选择的点', '']
    rows = []
    for video, budgets in result['budget_selections'].items():
        for budget, selected in budgets.items():
            for method, point_id in selected['selected'].items():
                if point_id is None:
                    rows.append([f'{int(video):03d}', budget, method, '无合格点', 'NA', 'NA', 'NA'])
                    continue
                p = points[point_id]
                rows.append([f'{int(video):03d}', budget, method, point_id,
                             f'{p["file_bytes"]:,}', f'{selected["unused_bytes"][method]:,}',
                             number(p['regions']['fish']['psnr_db'])])
    lines.append(table(['片段', '预算', '方法', '所选点', '实际 B', '剩余预算 B', '鱼 PSNR'], rows))
    lines += ['', '选择各方法在上限内最大的实际文件；字节相同选较低 q。'
              '同上限不等于同体积。若 B0/B10 选中同一个点，加出的预算没有被当前三档 q 利用。', '',
              '## 4. C/O 与同预算 A/B 的差值', '']
    comparisons = result['budget_comparisons']
    lines.append(table(['片段/预算', '候选 → 对照', '文件增幅', '严格近等', '鱼 ΔdB', '其余 ΔdB'], [
        [f'{c["video_id"]:03d}/{c["budget"]}', f'{c["candidate"]} → {c["comparator"]}',
         number(100*c['relative_byte_increase_vs_comparator'], 2, True)+'%',
         '是' if c['near_equal_no_larger_2pct'] else '否',
         number(c['target_psnr_delta_db'], 3, True), number(c['rest_psnr_delta_db'], 3, True)]
        for c in comparisons
    ]))
    lines += ['', '严格近等条件：候选不大于对照，且 `(对照字节−候选字节)/对照字节≤2%`。'
              '正画质差代表候选更好；正文件增幅代表候选更大。'
              '没有通过该条件的结果仍可报告实际质量/大小权衡，但不能标为近等体积收益。', '',
              '## 5. 两片等权平均', '']
    averages = []
    for budget in ('B0', 'B10'):
        for method in ('C', 'O'):
            for comparator in ('A', 'B'):
                pairs = [c for c in comparisons if c['budget'] == budget
                         and points[c['candidate']]['method'] == method
                         and points[c['comparator']]['method'] == comparator]
                if not pairs:
                    continue
                metrics = {key: mean(c[key] for c in pairs)
                           if all(c[key] is not None for c in pairs) else None
                           for key in ('target_psnr_delta_db', 'rest_psnr_delta_db')}
                averages.append(dict(budget=budget, method=method, comparator=comparator,
                                     clips=len(pairs), all_pairs_near_equal=all(c['near_equal_no_larger_2pct'] for c in pairs),
                                     **metrics))
    lines.append(table(['预算', '比较', '片数', '全部严格近等', '平均鱼 ΔdB', '平均其余 ΔdB'], [
        [a['budget'], a['method']+' vs '+a['comparator'], a['clips'],
         '是' if a['all_pairs_near_equal'] else '否',
         number(a['target_psnr_delta_db'], 3, True), number(a['rest_psnr_delta_db'], 3, True)]
        for a in averages
    ]))
    lines += ['', '平均不允许掩盖单片文件超限或画质下降；完整 D 轮还需要第三个开发片段、'
              '连续观看和预定其他证据，本次没有执行完整门槛验收。', '',
              '## 6. 固定 q 的诊断', '']
    lines.append(table(['片段/q', '候选 vs A', '文件增幅', '鱼 ΔdB', '其余 ΔdB'], [
        [f'{c["video_id"]:03d}/q{c["q"]}', points[c['candidate']]['method'],
         number(100*c['relative_byte_increase_vs_comparator'], 2, True)+'%',
         number(c['target_psnr_delta_db'], 3, True), number(c['rest_psnr_delta_db'], 3, True)]
        for c in result['same_q_vs_original']
    ]))
    lines += ['', '固定 q 只帮助区分预处理效应；不能直接解释成等字节资源重新分配。'
              'O 为 GT 诊断参考，不能保证其表现是 C 的上界。', '',
              '## 7. 处理耗时和理论传输', '']
    rows = []
    for video, budgets in result['budget_selections'].items():
        for method, point_id in budgets['B10']['selected'].items():
            if point_id:
                p = points[point_id]
                rows.append([point_id, number(p['encoder_seconds'], 2), number(p['decoder_seconds'], 2),
                             number(8*p['file_bytes']/20000, 2), number(8*p['file_bytes']/16000, 2)])
    lines.append(table(['B10 所选点', '编码 s', '独立解码 s', '20 kbps 理论发送 s', '16 kbps 理论发送 s'], rows))
    lines += ['', '编码/解码为本次各点墙钟耗时，模型加载与 CPU 输入准备另列。'
              '解码耗时含研究输出和预览生成；发送时间只是文件字节除以假定有效吞吐，'
              '16 kbps 为 20 kbps×80% 场景，均非真实网络或水声实测。', '']
    phases = (read_json(root/'execution_summary.json')['phases'] if (root/'execution_summary.json').exists() else
              [dict(phase=phase, **read_json(root/(phase+'_summary.json')))
               for phase in ('encode_anchors', 'decode_anchors', 'encode_remaining', 'decode_remaining')])
    lines.append(table(['阶段', '状态', '完成点数', '阶段耗时 s', '其中模型加载 s'], [
        [p['phase'], p['state'], len(p['tasks']),
         ('≥' if p.get('seconds_is_lower_bound') else '')+number(p['seconds'], 2),
         number(p['model_load_seconds'], 2)] for p in phases
    ]))
    lines += ['', '若含中断阶段，其耗时为最后一帧进度记录的下界，完成点数不含未完成半成品；'
              '继续运行另列实际阶段，不虚构一次完整执行。跨日暂停时间不计入编解码耗时。']
    lines += ['', f'CPU 输入准备 {read_json(root/"preparation.json")["seconds"]:.2f} s；'
              f'CPU 画质评价 {result["quality_evaluation_seconds"]:.2f} s。'
              '本轮复用 D1 预测缓存，没有重复语义推理，不能据此宣称完整发送端耗时已测。', '',
              '## 8. 证据与观看边界', '',
              '- `evaluation.json`：全部点、预算选择与成对差值；`frame_metrics.json`：逐帧误差。',
              '- `budget_anchors.json`：语义结果前冻结的普通 GLC 字节与新码流指纹。',
              '- `points/*/encoder/encode.json`、`decoder/decode.json`：逐点编码、独立接收和逐帧验证。',
              '- `viewing/index.html`：四份隐藏方法名的同步观看视频；`viewing_key.json`：独立映射。',
              '- 同步裁剪固定为轨迹 38 的原标注全时段包围框加 48 像素；数值仍统计全部鱼。',
              '- 观看 MP4 是有损副本，不用于上述数值；生成页面或看静帧均不表示连续观看验收通过。',
              '- 大文件、原图、掩码、模型与重建 PNG 只保留本地；仓库保存脚本、报告和小型证据。', '',
    ]
    args.output.write_text('\n'.join(lines), encoding='utf-8')
    save_json(root/'report_summary.json', {'clip_equal_weight_averages': averages,
                                         'numeric_appendix': args.output.name,
                                         'continuous_visual_validation_completed': False})
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
