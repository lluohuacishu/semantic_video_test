"""Generate numerical report from completed evidence, with distinct comparison scopes."""
from __future__ import annotations
import argparse
import statistics
from experiment_common import *


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output
    fill=load_json(out/'fill_summary.json');enc=load_json(out/'encoder_summary.json');q=load_json(out/'quality.json');full=load_json(out/'full_summary.json')
    if not all(x['complete'] for x in [fill,enc,q,full]):raise ValueError('Incomplete evidence')
    lines=['# 输入编码器推理加速实测：无需训练的GLC优化','',
      '2026-10-30，RTX4060 Laptop 8GB、Windows、PyTorch2.5.1+cu124、TensorRT10.5.0/cu12。原始GLC权重不变，发送端免RGB，原生C rANS，接收端保留原完整生成器。所有优化仅在诊断子进程使用，生产默认和网页未切换。','',
      '本阶段完成五组素材的填充扩大对照、24个I/P输入编码器单模块任务，以及四组视频上的三种候选完整编解码。它验证了现有执行方式仍有优化空间；现阶段不需要先启动编码器蒸馏。','',
      '## 填充开关：五组素材、五档q、40次完整运行','',
      '547个输入帧，每组每端两对新进程，顺序on/off、off/on，8帧暖机。FP32、确定性算法True及全部符号/CDF/参考校验保留。原三组匹配历史码流和RGB，新增两组独立解码通过。','',
      '| 素材 | 帧数/q | 编码on→off中位秒 | 编码逐对加速中位 | 解码on→off中位秒 | 解码逐对加速中位 |','|---|---:|---:|---:|---:|---:|']
    for a in [r for r in fill['reports'] if r['role']=='sender']:
        b=next(r for r in fill['reports'] if r['clip']==a['clip'] and r['role']=='receiver')
        lines.append(f'| {a["clip"]} | {a["frames"]}/{a["q"]} | {a["on_median_seconds"]:.3f}→{a["off_median_seconds"]:.3f} | {a["paired_speedup_median"]:.3f}× | {b["on_median_seconds"]:.3f}→{b["off_median_seconds"]:.3f} | {b["paired_speedup_median"]:.3f}× |')
    lines+=['','逐对比值先计算再取中位数，不等同于两列中位数相除；省时比例为`1-1/加速比`。这些结果扩大了单片q0证据，仍不能证明所有输入都不读取未初始化内存。UV096的255帧覆盖四个GOP，不是长期部署稳定性测试。','',
      '## 单模块：布局、图、精度和引擎分开比较','',
      '真实输入640×384（有效640×360），帧0/2/33，I/P编码器分别测。每模式每样本12次、三轮交替顺序，独立原布局控制；所有观测保留。下表为包含输入转换及输出复制的同步墙钟中位数，GPU Event另存。均使用fill-off。控制输入实测连续NCHW；channels-last候选使用单独的等值权重副本。','',
      '| 实现 | P输入编码器ms | I输入编码器ms | 采样潜变量逐位一致（P/I） |','|---|---:|---:|---|']
    for a in [r for r in enc['reports'] if r['model']=='video']:
        b=next(r for r in enc['reports'] if r['model']=='image' and r['variant']==a['variant'])
        lines.append(f'| {a["variant"]} | {a["candidate_wall_median_ms"]:.3f} | {b["candidate_wall_median_ms"]:.3f} | {all(d["bit_exact"] for d in a["changes"])}/{all(d["bit_exact"] for d in b["changes"])} |')
    lines+=['','channels-last、TF32和FP16+channels-last没有获得普遍收益；TorchScript、CUDA Graph、显式连续输入在本机收益很小。TensorRT默认流警告触发了独立流补测；其收益按实测记录，不把警告当作加速保证。','',
      '## 完整编码与原接收端独立解码','',
      '候选依据单模块速度、部署复杂度选择，先冻结selection.json，再评价完整画质。每组两次发送，第一轮候选正序、第二轮逆序；随后独立接收器禁止读取源视频、源数组和GT。压缩符号、CDF和参考状态逐帧核对。所有方向都采用相同fill-off条件，不能与第15阶段默认fill-on或其他时段基线直接相除。','',
      '| 素材 | 实现 | 编码核心中位秒 | 逐对加速中位 | P帧PSNR变化dB | 标注区域P帧PSNR变化dB | 压缩载荷变化 |','|---|---|---:|---:|---:|---:|---:|']
    for r in q['reports']:
        p=r['quality']['groups']['P'];fish='—' if p['fish_psnr_db'] is None else f'{p["fish_psnr_db"]-p["fish_baseline_psnr_db"]:+.4f}'
        speed=statistics.median(r['paired_speedup_vs_eager'])
        lines.append(f'| {r["clip"]} | {r["variant"]} | {r["sender_core_median_seconds"]:.3f} | {speed:.3f}× | {p["psnr_delta_db"]:+.4f} | {fish} | {(r["payload_ratio_vs_eager"]-1)*100:+.4f}% |')
    exact=[r for r in q['reports'] if r['variant']=='trt_fp32_stream']
    fp16=[r for r in q['reports'] if r['variant']=='trt_fp16_stream']
    lines+=['',f'TensorRT FP32：{sum(r["payload_and_reference_exact_vs_eager"] for r in exact)}/{len(exact)}组压缩载荷/符号/参考与原执行方式一致，{sum(r["rgb_exact_vs_eager"] for r in exact)}/{len(exact)}组RGB逐位一致。这是本次素材的实测结果，不是跨框架FP32普遍逐位一致的保证。',
      f'TensorRT FP16：完整编码加速范围{min(statistics.median(r["paired_speedup_vs_eager"]) for r in fp16):.3f}–{max(statistics.median(r["paired_speedup_vs_eager"]) for r in fp16):.3f}×；其码流和重建可以变化，仍全部由原接收端正确解码。具体画质代价见上表，不能称为无损替换。','',
      '码流头明确记录实验输入编码器实现；原arithmetic字段对应未改的参考/熵/解码路径。因此全文件SHA不同可能仅因头部元数据，表中单独比较压缩载荷和逐帧状态；文件bpp及容器开销另存quality.json。完整输出仍是实验GLCSND01协议，非官方GLC标准码流。','',
      '![推理对照](evidence_20261030/inference_overview.png)','',
      '![025帧序34对比](evidence_20261030/sf025_q32_512_frame034.png)','',
      '## 时间、画质和部署边界','',
      '- 核心计时前完成输入上卡，结束同步，含原熵编码及诊断校验；实际任务时间、初始化和引擎构建另存。单模块采用同步墙钟/CUDA Event，不将其称为新一轮CUPTI内核归因。',
      '- 首次TensorRT建立包含ONNX导出、算法搜索、序列化，可能明显长于编码一段短视频；这些成本没有计入上表。后续复用同一固定形状引擎。640×384和512×320分别构建，I/P权重分别构建，不声称动态形状通用。',
      '- TensorRT FP32显式禁用TF32；FP16标志允许混合精度，不代表每层均为FP16。局部PyTorch精度开关离开输入编码器后恢复。参考/熵/接收端仍为原FP32。',
      '- PSNR先聚合RGB MSE，再转dB，整段/I/P分组分别记录。标注区域仅用已有SFishTrack标签并记录掩码SHA；UV未标注区域不作背景。帧序误差变化只是诊断代理，未证明解决感知闪动。',
      '- 无独立拍摄来源保证，UV/SFishTrack均是项目已有素材。本次是推理实现对照，未用这些指标训练权重。',
      '- PyTorch显存计数不包含TensorRT自行分配的内存；实验封装还保留原模块用于对照，不能当作最终部署包峰值。桌面其他程序保留运行，没有能耗或SM硬件计数器结论。',
      '- Windows/RTX4060引擎不是Orin部署产物。仍需依据盒子的JetPack/TensorRT重新构建，测试目标功耗模式、长时运行及岸上跨设备解码。生产默认未切换。','',
      '## 证据与接续','',
      '[运行方法](README.md)、[填充结果](evidence_20261030/fill_results.json)、[单模块数值](evidence_20261030/encoder_results.json)、[完整链结果](evidence_20261030/full_results.json)、[画质与码率](evidence_20261030/quality.json)、[逐帧校验表](evidence_20261030/frame_checks.csv)、[执行清单](evidence_20261030/execution_manifest.json)。9项CPU校验通过，覆盖数值/形状拒绝、符号/CDF/参考/RGB变更检测及ROI/PSNR聚合。完整运行均检查参考与压缩符号。',
      '原始码流、RGB数组、引擎及视频留在工作区`02_实验数据/优化_输入编码器推理/inference_20261030_01`。实际接收CLI另生成观看副本，媒体/帧序对比摘要在执行清单；H264观看副本不用于画质计分。',
      '下一步优先将经过完整链验证的引擎方案在真实盒子上复现；FP32路线与FP16路线分别保留。暂不启动编码器蒸馏。','',
      '原理参考：[PyTorch数值准确性](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/numerical_accuracy.rst)、[CUDA Graph](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/cuda.rst)、[TensorRT跨框架FP32差异](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/troubleshooting-reporting.html#reporting-an-accuracy-issue)、[TensorRT10.5支持矩阵](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1050/pdf/TensorRT-Support-Matrix-Guide.pdf)。','']
    compact=load_json(out/'compact_transport.json')
    extra=['## 实际传输文件：精简研究元数据','',
      '详细实验头曾额外携带代码SHA和长解释，短码流的整文件增幅明显大于压缩载荷增幅。因此另外生成精简实现标识的副本，溯源信息放旁证；不改变任何z/y载荷、符号或有序参考摘要，也不计作神经编码加速。16份副本均通过逐载荷/参考摘要验证，原执行方式的重打包还要求全文件逐位一致。两种TensorRT的66帧精简副本另由原接收CLI实际输出视频。','',
      '| 素材 | 实现 | 详细实验文件大小变化 | 精简标识文件大小变化 | 压缩载荷变化 |','|---|---|---:|---:|---:|']
    for r in compact['reports']:
        if r['variant'].startswith('trt'):
            extra.append(f'| {r["clip"]} | {r["variant"]} | {(r["research_file_ratio_vs_baseline"]-1)*100:+.4f}% | {(r["file_ratio_vs_baseline"]-1)*100:+.4f}% | {(r["payload_ratio_vs_baseline"]-1)*100:+.4f}% |')
    extra+=['','比较对象均为本段原执行方式的相同协议文件。精简后的文件仍需保留实现标识，并非自动完全等长；详细记录见[compact_transport.json](evidence_20261030/compact_transport.json)。','']
    position=lines.index('## 时间、画质和部署边界')
    lines[position:position]=extra
    (HERE/'输入编码器推理加速实测_20261030.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')


if __name__=='__main__':main()
