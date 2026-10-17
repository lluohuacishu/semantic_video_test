"""Evidence-backed semantic evaluation report, plots and deterministic examples."""
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ROOT, configure, metrics, save_json

OUT=Path(r"C:\Users\lenovo\Documents\Codex\2026-10-12\segformer-c-users-lenovo-desktop-works\outputs")
OUT.mkdir(parents=True,exist_ok=True)
load=lambda path:json.loads(Path(path).read_text("utf-8"))
corpus=load(ROOT / "data/phase_b_annotation_corpus/manifest.json")
semantic=load(ROOT / "runs/phase_b_gt_codec_semantics/summary.json")
quality=load(ROOT / "runs/phase_b_quality/comparison.json")
pilot=load(ROOT / "runs/phase_b_pilot_semantics/summary.json")
validation=load(ROOT / "runs/phase_b_teacher_validation/result.json")
resolution=load(ROOT / "runs/phase_b_resolution_validation/summary.json")
raw512=load(ROOT / "runs/phase_b_gt_raw_size512/summary.json")
roi_interface=load(ROOT / "runs/phase_b_roi_interface/result.json")
roi_backprop=load(ROOT / "runs/phase_b_roi_backprop/result.json")
groups=load(ROOT / "data/recording_groups_review.json")
uv=load(ROOT / "data/uvmulti_preflight/mask_availability.json")
matched=load(ROOT / "runs/phase_b_rate_match/selected.json")
matched_semantic=load(ROOT / "runs/phase_b_rate_match/semantic_results/summary.json")["sequences"][0]
configure()
matched_hevc=matched_semantic["streams"][2]
matched_quality_file=ROOT / "runs/phase_b_rate_match/quality.json"
if matched_quality_file.exists():
    matched_quality=load(matched_quality_file)
else:
    matched_quality=metrics(corpus["sequences"][1]["reference_dir"],ROOT / "runs/phase_b_rate_match/hevc_rgb")
    save_json(matched_quality_file,matched_quality)

plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(1,3,figsize=(13.8,4.5),layout="constrained")
labels={"mlvc":"MLVC-S + DMCI (mirror)","hevc":"HEVC medium, P-only","glc":"GLC q0"}
colors={"mlvc":"#d77720","hevc":"#2669aa","glc":"#268348"}
for ax,sequence in zip(axes,semantic["sequences"],strict=True):
    for prefix in ["mlvc","hevc","glc"]:
        rows=sorted([r for r in sequence["streams"] if r["name"].startswith(prefix)],key=lambda r:r["complete_file_bytes"])
        ax.plot([r["complete_file_bytes"]/1000 for r in rows],[r["fish_gt_iou"]*100 for r in rows],"o-" if prefix!="glc" else "s",color=colors[prefix],label=labels[prefix])
    ref=sequence["streams"][0]
    ax.axhline(ref["fish_gt_iou"]*100,linestyle="--",color="#657480",label="Uncompressed reference prediction")
    ax.set_title(sequence["name"])
    ax.set_xlabel("Complete file size (decimal KB)")
    ax.set_ylabel("GT fish foreground IoU (%)")
    ax.grid(alpha=.2)
    ax.set_ylim(0,max(r['fish_gt_iou'] for r in sequence['streams'])*115)
axes[0].legend(fontsize=7,loc="best")
fig.suptitle("Frozen SegFormer input 256: real fish GT, fixed codec weights\nAnnotated-frame sequences, nominal 5fps; panel y-axis scales differ; rate points not identical",fontsize=10)
curve_name="阶段B_真实鱼类语义曲线_2026-10-13"
fig.savefig(OUT / f"{curve_name}.png",dpi=180)
fig.savefig(OUT / f"{curve_name}.svg")
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11.4,4.3),layout="constrained")
x=np.arange(3)
for size,color,offset in [(256,"#286baa",-.17),(512,"#d77822",.17)]:
    vals=[s["streams"][0]["fish_gt_iou"]*100 for s in (semantic["sequences"] if size==256 else raw512["sequences"])]
    bars=axes[0].bar(x+offset,vals,.32,label=f"input {size}",color=color)
    axes[0].bar_label(bars,fmt="%.1f",fontsize=8)
    result=next(r for r in resolution["results"] if r["input_size"]==size)
    vals=[result["mean_iou_present_union"]*100,result["per_class"][6]["iou"]*100,result["per_class"][1]["iou"]*100]
    bars=axes[1].bar(x+offset,vals,.32,label=f"input {size}",color=color)
    axes[1].bar_label(bars,fmt="%.1f",fontsize=8)
axes[0].set_xticks(x,["001","027","054"])
axes[0].set_title("Reference-frame fish IoU: SFISHTRACK")
axes[1].set_xticks(x,["mIoU (8 classes)","FV IoU","HD IoU"])
axes[1].set_title("SUIM validation: same original GT pixel grid")
for ax in axes:
    ax.set_ylim(0,100);ax.set_ylabel("Percent");ax.grid(axis="y",alpha=.2);ax.legend(fontsize=8)
fig.suptitle("Resolution diagnostic only — same frozen weights, no training",fontsize=11)
resolution_name="阶段B_分割器分辨率诊断_2026-10-13"
fig.savefig(OUT / f"{resolution_name}.png",dpi=180)
plt.close(fig)

font=ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf",15)
sheet=Image.new("RGB",(1980,750),"#f4f6f8")
draw=ImageDraw.Draw(sheet)
for row,(sequence,record) in enumerate(zip(corpus["sequences"],semantic["sequences"],strict=True)):
    idx=sequence["frame_count"]//2
    gt=np.asarray(Image.open(Path(sequence["ground_truth_dir"]) / f"{idx:05d}.png")).astype(bool)
    ref=np.asarray(Image.open(Path(sequence["reference_dir"]) / f"{idx:05d}.png").convert("RGB"))
    panels=[("Reference RGB",None,None),("Publisher fish GT",gt,(40,235,80))]
    for name in ["reference","mlvc_q21","hevc_crf50","glc_q0"]:
        pred=np.asarray(Image.open(ROOT / "runs/phase_b_gt_codec_semantics" / sequence["name"] / name / "prediction_index" / f"{idx:05d}.png"))==6
        path=next(s["input_frames_dir"] for s in record["streams"] if s["name"]==name)
        image=np.asarray(Image.open(sorted(Path(path).glob("*.png"))[idx]).convert("RGB"))
        panels.append(("SegFormer ref" if name=="reference" else name,pred,(255,110,35),image))
    for col,panel in enumerate(panels):
        label,mask,color=panel[:3]
        image=panel[3] if len(panel)==4 else ref
        pixels=image.copy()
        if mask is not None: pixels[mask]=(image[mask]*.42+np.asarray(color)*.58).round().astype(np.uint8)
        tile=Image.fromarray(pixels).resize((320,180),Image.Resampling.LANCZOS)
        px,py=col*330+5,row*240+36
        sheet.paste(tile,(px,py))
        draw.text((px,row*240+12),f"{sequence['name'][-3:]} f{idx} / {label}",font=font,fill="#192a38")
draw.text((8,721),"Fixed middle frame per sequence. Green = published fish GT; orange = predicted FV. File sizes differ; full-sequence metrics are in the report.",font=font,fill="#394b5c")
examples_name="阶段B_鱼类标注与预测对照_2026-10-13.png"
sheet.save(OUT / examples_name)

def link(path,label=None):
    p=Path(path).resolve()
    return f"[{label or p.name}]({p.as_posix()})"

raw_table="| 场景 | 帧数 | GT 鱼像素占比 | 参考帧鱼 IoU | 参考帧鱼 F1 | 鱼召回 | 鱼精确率 |\n|---|---:|---:|---:|---:|---:|---:|\n"
for sequence,result in zip(corpus["sequences"],semantic["sequences"],strict=True):
    raw=result["streams"][0]; fish=raw["fish_ground_truth"]["per_class"][1]
    fraction=fish["target_pixels"]/(sequence["frame_count"]*1280*720)
    raw_table+=f"| {sequence['name']} | {sequence['frame_count']} | {fraction*100:.4f}% | {raw['fish_gt_iou']*100:.2f}% | {raw['fish_gt_f1']*100:.2f}% | {fish['recall']*100:.2f}% | {fish['precision']*100:.2f}% |\n"

table="| 场景 / 方法 | 完整字节 | 名义 kbps | RGB PSNR / dB | GT 鱼区 PSNR / dB | GT 鱼 IoU | GT 鱼 F1 | 20 kbps 理论秒数 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
for r in quality["rows"]:
    table+=f"| {r['sequence'][-3:]} / {r['method']} | {r['file_bytes']:,} | {r['video_kbps']:.3f} | {r['rgb_psnr_global_db']:.3f} | {r['gt_fish_rgb_psnr_db']:.3f} | {r['fish_iou']*100:.2f}% | {r['fish_f1']*100:.2f}% | {r['transfer_seconds_at_20kbps']:.4f} |\n"

pilot_table="| 既有潜水员场景 / 方法 | 完整字节 | HD 预测区域一致性 IoU | 教师参考 ROI 区 RGB PSNR / dB |\n|---|---:|---:|---:|\n"
for r in pilot["sequences"][0]["streams"]:
    if r["name"]=="reference": continue
    if r["name"] not in {"glc_q00","glc_q63","mlvc_q0","mlvc_q21","mlvc_q42","mlvc_q63","hevc_crf44","hevc_crf50","hevc_crf51"}:continue
    pilot_table+=f"| {r['name']} | {r['complete_file_bytes']:,} | {r['source_prediction_agreement_exclusions']['per_class'][1]['iou']*100:.2f}% | {r['teacher_roi_rgb_psnr_db']:.3f} |\n"

val256,val512=resolution["results"]
matched_mlvc=next(r for r in quality["rows"] if r["sequence"]=="sfishtrack_027" and r["method"]=="mlvc_q21")
data_text=f'''# 水下视频语义通信：阶段 B 语义评价与 ROI 准备

日期：2026-10-13。以下数值来自实际文件、冻结模型推理与发布者标注；本轮没有对编解码器或 SegFormer 更新参数。

## 1. 当前结论与技术选择

继续保留 MLVC-S 作为可微调的候选，下一步先用可信 GT 掩码验证 ROI 训练，并建立同预算的普通微调对照。SegFormer 的任务损失需要在场景适配和可靠性核对后再接入。

理由已经有实验证据：未压缩参考帧的鱼类 IoU 在三个场景中差异很大；普通 MLVC-S 在一个严格匹配文件大小的样例上，整体画质优于 H.265，但鱼区画质和鱼类分割反而更差。现在需要检验语义加权是否能改变这种折中，不能仅凭模型较新或整体 PSNR 较高确定最终方案。

H.265 继续作为强基线，GLC 保留作为基线和备选。本轮没有训练出 ROI 优化模型，没有证明 ROI 增益，也没有宣称完成联合源信道通信。

## 2. 实際完成清单

| 工作 | 状态与证据 |
|---|---|
| 现有分割器迁移 | SegFormer 严格加载，权重哈希核对，冻结全部参数 |
| 原 SUIM 测试复现 | 110 张，mIoU 69.1347%，与历史记录几乎一致 |
| 新增 GT 数据 | 官方 SFISHTRACK 3 组标注帧，50+20+50=120 帧 |
| 图像与 GT 对应 | 使用官方标注帧，检查尺寸、RLE 面积、类别与映射 |
| 无损参考视频 | 三组 FFV1 缓存均通过所有帧 RGB 像素哈希往返核对 |
| 新的完整文件编解码 | MLVC q0/q21、GLC q0、H.265 CRF44/50，共 15 组文件 |
| 独立解码验证 | MLVC 共 240 帧状态/像素核对；GLC 共 120 帧核对；H.265 共 240 帧独立解码、GOP/数量检查 |
| GT 语义评价 | 120 参考帧 + 600 重建帧，真实鱼类 IoU/F1、鱼区 RGB 误差 |
| 原潜水员视频诊断 | 参考与 13 个压缩质量点，共 700 帧预测；只报告预测一致性 |
| 分辨率诊断 | 原 SUIM 验证集与新参考帧比较输入 256/512，权重保持冻结 |
| 文件预算匹配 | 027 样例 H.265 与 MLVC 完整文件差 {matched['relative_file_size_difference']*100:.4f}% |
| 官方 ROI 接口 | 复用原损失类，预计算掩码无需辅助分割模型初始化 |
| ROI 梯度/实际模型反向 | ROI 像素梯度比 4；真实 MLVC 编码器获得有效梯度，参数未更新 |
| 后续独立数据来源 | UVMulti 官方 Figshare 可获取，标签表与 ZIP 索引及 6 张 mask 已核对 |

完整工程：{link(ROOT / 'README.md', 'README')}。上一轮阶段 A 结果：{link(ROOT / 'docs/phase_a_results_20261013.md')}。

## 3. 数据来源与许可记录

新增数据来自 [SFISHTRACK 作者仓库](https://github.com/JosepSanchezCano/SFISHTRACK) 与 [官方 Zenodo 记录](https://zenodo.org/records/20617668)。发布者说明采用半自动分割/传播与人工核验。本轮采用发布者 GT，没有重新人工逐帧重标。

官方 ZIP 为 27,487,705,333 字节。按 HTTP Range 读取目录及所需条目，没有下载整个归档。下载条目经过 ZIP CRC32 检查并记录 SHA256；整个 ZIP 的发布 MD5 没有被完整验证，不能把条目校验写成整包校验。

元数据声明数据许可为 CC-BY-4.0；原始记录、来源、发布者及许可保存在 {link(ROOT / 'data/sfishtrack_official/zenodo_metadata.json')}。来源链接与记录应在后续成果中保留。

三个评价样例按编号均匀选取 001、027、054，选择发生在分割/编码质量计分之前。其元数据中的拍摄日期分别是 03/06/2025、26/07/2022、09/05/2023。

## 4. 标注帧与 MP4 的核对过程

直接把 `frame_id` 当作 MP4 物理帧编号会出问题。027 的 COCO `frame_id` 从 0 开始，但图像名称从 `000002.png` 开始；使用发布者随附图像核对后，编号到视频帧的偏移也并非完全固定。

此外，部分标注名称是通用 `object`。本轮依据作者单一 fish 类定义，把 `category_id=1` 映射为鱼，保留原始类别名称与转换说明；没有据此生成八类视频 GT。

本轮选择官方提供的、与 COCO 图像条目对应的 RGB 帧作为实际输入，采用原始图片名/ID关联 RLE 掩码。RLE 由 `pycocotools` 解码，检查面积和尺寸，合并鱼实例为二值语义掩码。

RGB 等比缩放到 1280×720；GT 用最近邻缩放。生成 FFV1 无损视频缓存后，逐帧解码回 RGB，120/120 帧哈希完全一致。

早期 MP4 对齐尝试和失败日志保留，见 {link(ROOT / 'runs/sfishtrack_alignment_v2/result.json')}、`audit/prepare_sfishtrack.log`、`audit/sfishtrack_clean_alignment.log`。初次 rawvideo 对齐诊断发现 FFmpeg 默认补帧，已改为 `fps_mode=passthrough` 并校验输出帧数；错误的首次搜索结果没有用于 GT 评分。

## 5. 帧序列与时间基准的边界

本轮输入是按发布者标注序列顺序抽取的帧，统一以名义 5 fps 播放。001、054 分别为 50 帧、名义 10 秒；027 为 20 帧、名义 4 秒。

部分原始编号存在缺口：054 抽取帧的最大图片编号跳跃为 {corpus['sequences'][2]['max_selected_source_filename_gap']}。因此这轮可评价共同输入上的压缩和鱼类语义保留，不能把所有序列当作已经核实的、等间隔连续采集视频。

实际原始 MP4 的 PTS 没有被用来对这些 GT 计分。理论链路时间取决于完整文件字节数；表中的视频 kbps 按名义播放时间换算。真实连续视频的时序优势还需要用完整源视频及可靠的帧关联再验证。

完整关联与图像/标注哈希：{link(ROOT / 'data/phase_b_annotation_corpus/manifest.json')}。

## 6. 分割器复现与冻结协议

使用原 SegFormer-B0/SUIM 最佳权重，epoch 39，参数 3,716,200。SHA256 为 `ff00cb439a4af290dc3efde8ae76d4acfad797784483b669ab6db79aaf5b1476`。

类别顺序固定为 BW、HD、PF、WR、RO、RI、FV、SR；鱼预测为原八类 argmax 的 FV=6。新 GT 是 fish / non-fish 二值，不把其他六类自动当成 SUIM 背景标签。

主协议使用 checkpoint 训练尺寸 256，RGB [0,1]，原 mean/std，长边等比缩放、居中黑色补边。输出 logits 去除补边，双线性上采样到 1280×720，再 argmax；GT 在共同原图网格计分。

110 张原 SUIM 测试复现 mIoU 为 {validation['mean_iou_present_union']*100:.6f}%，历史值 {validation['historical_miou']*100:.6f}%，差 {validation['delta_miou']*100:.8f} 个百分点。这个复现用于检查实现一致性，没有重新选择权重或训练轮次。

新数据只评价鱼类，网格/类别/场景与 SUIM 八类测试不同，不能直接把这些鱼 IoU 与 69.13% mIoU 当成同一指标比较。

兼容性验证：{link(ROOT / 'runs/phase_b_teacher_validation/result.json')}。

## 7. 压缩前的真实鱼类识别

{raw_table}

001 和 054 的 GT 鱼面积占比都很小，原参考图上已经存在明显漏检。027 的鱼类识别较可靠。因此任务评价需要同时保留“参考帧性能”和“重建帧性能”，把分割器本身的局限与压缩额外造成的变化分开观察。

ROI 训练阶段不能把现有 SegFormer 对所有场景的预测直接当作 GT；尤其弱场景中的高置信度或空 ROI 仍需要校验。

## 8. 真实文件与 GT 鱼类评价

{table}

主表每一行都采用完整文件大小。MLVC、GLC 均包含首帧、后续帧和必要协议/容器信息；H.265 为完整 MP4。三个编解码器采用相同 RGB 输入、分辨率、名义 fps 与 GOP32，但内部色彩域不同：GLC 为 RGB、MLVC 为浮点 YUV420、H.265 为 8-bit YUV420。

GT 鱼区 PSNR 使用真实二值鱼掩码累计 RGB 平方误差；鱼 IoU/F1 使用所有帧累计 TP、FP、FN。没有用全背景占优的 pixel accuracy 代替鱼类效果。MS-SSIM 与逐帧详细记录另存 JSON。

这些预设 q/CRF 没有保证相等文件大小，因此主表用于显示可复现质量点，不据此宣称全面优越性。

![真实鱼类语义曲线]({(OUT / (curve_name+'.png')).as_posix()})

图中灰色虚线是参考帧的分割表现，不是任务性能上限；压缩的平滑/变化也可能影响分类器。各子图 y 轴范围不同，054 的全部成绩仍处于很低水平。

完整数据：{link(ROOT / 'runs/phase_b_quality/comparison.json')}。语义逐帧统计：{link(ROOT / 'runs/phase_b_gt_codec_semantics/summary.json')}。

## 9. 文件大小匹配的关键单例

027 场景固定 MLVC q21 的文件为 {matched['target_bytes']:,} 字节。H.265 使用相同 medium、P-only、GOP32、BT.709 路径，只按文件字节数搜索 CRF，不用 GT、IoU 或 PSNR 选候选。

得到 H.265 文件 {matched['hevc_bytes']:,} 字节，CRF {matched['hevc_crf']:.6f}，实际大小差 {matched['relative_file_size_difference']*100:.4f}%。

| 方法 | 完整字节 | RGB 全局 PSNR | GT 鱼区 RGB PSNR | GT 鱼 IoU | GT 鱼 F1 |
|---|---:|---:|---:|---:|---:|
| MLVC-S q21 | {matched['target_bytes']:,} | {matched_mlvc['rgb_psnr_global_db']:.3f} dB | {matched_mlvc['gt_fish_rgb_psnr_db']:.3f} dB | {matched_mlvc['fish_iou']*100:.2f}% | {matched_mlvc['fish_f1']*100:.2f}% |
| H.265 匹配文件大小 | {matched['hevc_bytes']:,} | {matched_quality['rgb_psnr_global_db']:.3f} dB | {matched_hevc['ground_truth_fish_rgb_psnr_db']:.3f} dB | {matched_hevc['fish_gt_iou']*100:.2f}% | {matched_hevc['fish_gt_f1']*100:.2f}% |

这一单例说明：整体 RGB 画质更好，并不保证关键鱼区域重建或任务结果更好。当前普通 MLVC-S 的语义优势尚未建立。下一步 ROI 实验的目的，就是检验同预算下能否改善目标区域的这种损失。

该结果只有一个匹配样例，不能作为完整 RD 曲线、跨场景优势或统计显著性的结论。搜索配置和全部候选保留在 {link(ROOT / 'runs/phase_b_rate_match/search.json')}；最终选择与语义评价分别记录。

## 10. 既有潜水员视频的预测一致性

既有视频仍没有人工视频 GT。本轮冻结教师，比较输入视频和重建视频的 HD 区域预测一致性；主诊断统一排除水印矩形 `[0,0,320,144]`，也保存完整画面的结果。

{pilot_table}

“HD 一致性 IoU”只表示同一模型前后预测区域重合程度，不是真实潜水员 IoU。数据中 FV 预测区域很少，也可能包括误预测，不把其一致性写成真实鱼类准确率。

该诊断仍有工程价值：MLVC q0 的 HD 一致性明显低于较高质量点，可以据此检查低码率区域变化，并结合后续人工标注判断实际意义。

原始记录：{link(ROOT / 'runs/phase_b_pilot_semantics/summary.json')}。

## 11. 256 / 512 输入诊断

权重没有改变。SUIM 验证集 297 张使用共同的原始 GT 网格计分；该网格与历史的补边 256 网格不同，数值不能混用。

| 输入尺寸 | SUIM 验证原图网格 mIoU | FV IoU | HD IoU |
|---|---:|---:|---:|
| 256 | {val256['mean_iou_present_union']*100:.2f}% | {val256['per_class'][6]['iou']*100:.2f}% | {val256['per_class'][1]['iou']*100:.2f}% |
| 512 | {val512['mean_iou_present_union']*100:.2f}% | {val512['per_class'][6]['iou']*100:.2f}% | {val512['per_class'][1]['iou']*100:.2f}% |

新参考帧鱼类 IoU：256 为 14.68%、76.35%、3.14%；512 为 {raw512['sequences'][0]['streams'][0]['fish_gt_iou']*100:.2f}%、{raw512['sequences'][1]['streams'][0]['fish_gt_iou']*100:.2f}%、{raw512['sequences'][2]['streams'][0]['fish_gt_iou']*100:.2f}%。没有稳定改善所有鱼类场景。

主基线继续保持 256。后续分割器适配需要基于训练/验证数据设计尺度、裁剪与场景增强，再使用保留的评价集测试；不会只因一个测试样例改善就更换所有协议。

![分割器分辨率诊断]({(OUT / (resolution_name+'.png')).as_posix()})

## 12. 图像、GT 与预测核验

![中间帧对照]({(OUT / examples_name).as_posix()})

每组固定取中间帧，未按预测质量挑选。绿色为发布者鱼 GT，橙色为 SegFormer FV 预测。该图用于检查对应关系和典型漏检，主结论来自完整帧集的累计统计。

## 13. 官方 ROI 损失的接口适配

复用固定上游的 `loss_total_rdc_psnr_roi_per_pixel`，没有重写其加权逻辑，也没有修改 `video/src/losses/codec.py`。

原通用损失工厂会索取辅助分割模型。本轮通过直接构造原类，提供一个缺少预计算 mask 时会报错的空接口，避免初始化与下载人脸分割辅助权重。训练调用必须显式给出对齐的 mask。

当前适配锁定 YUV420、`mse_rgb_weight=0`、原生几何 YUV 失真、码率权重 1。mask 接受 B×H×W 或单通道 B×1×H×W，并检查尺寸、有限性与 0～1 值域。

`p_roi=None` 使用每帧归一化；`roi_bg_ratio=4` 在此模式下表示目标像素相对于背景像素的权重为 4，平均像素权重保持 1。不是把整个 ROI 总误差强行设成背景总误差的 4 倍。

实样本 mask 的重建 Y 通道梯度权重比为 {roi_interface['observed_y_gradient_ratio']:.6f}，空/全 mask 的损失均有限，缺少 mask 会拒绝运行。

接口代码：{link(ROOT / 'adapters/precomputed_roi_loss.py')}。检查结果：{link(ROOT / 'runs/phase_b_roi_interface/result.json')}。

## 14. 真实 MLVC 前向/反向 smoke test

预训练 MLVC-S、冻结 I 帧重建参考、batch=1、两帧、256×256 裁剪、q32、GT 鱼 ROI。调用官方 ROI 损失并反向传播，{roi_backprop['parameter_tensors_with_nonzero_gradient']} 个参数张量产生非零梯度，编码器梯度有效且有限。

显存分配峰值 {roi_backprop['peak_allocated_bytes']/2**20:.2f} MiB，保留峰值 {roi_backprop['peak_reserved_bytes']/2**20:.2f} MiB。此测试没有优化器，没有更新任何编解码器参数，也没有加载 SegFormer。

保留的 GT 仅用于形状/梯度通路诊断和固定接口检查，没有用于训练、选择 ROI 权重或构造声称有增益的模型。ROI 训练后的真实文件编码和增益验证尚待下一阶段。

实际记录：{link(ROOT / 'runs/phase_b_roi_backprop/result.json')}。

## 15. 拍摄组与正式划分风险

54 个视频的元数据已全部检查。去除日期字段中的空格后，有 4 种有效日期文本及 1 个非法日期记录；`26 /27/2022` 的月份不合法，不能直接用它构造独立训练组。

候选日期 `05/09/2023` 与已保留的 `09/05/2023` 存在日/月交换可能，部分拍摄时段、相机与地点也接近。当前未确认这批片段是独立拍摄来源，因此不把它们自动加入训练或验证。

三组本轮 GT 数据继续保留为评价/工程诊断数据，ROI/分割器正式优化需来自独立确认的训练来源。原始拍摄的相邻剪辑也不能仅凭不同 `video_id` 被拆进不同集合。

日期核对与隔离结果：{link(ROOT / 'data/recording_groups_review.json')}。本轮没有生成一个假装已经确认独立的训练/验证划分。

## 16. 下一阶段数据来源已经核对到哪里

[UVMulti 原论文](https://doi.org/10.1038/s41597-026-07402-0) 和 [作者仓库](https://github.com/pingchuan/UVMulti) 给出了独立的水下视频与语义标注资源。[官方 Figshare 记录](https://figshare.com/articles/dataset/30105151) 可直接访问部分视频 ZIP 与标签表；其记录的数据许可为 CC0，与本文使用的 SFISHTRACK 来源分别记录。

本轮已校验 `seg_label.txt` 的发布 MD5，读取 `video91.zip`、`video96.zip` 的 ZIP 索引，按需取得首/中/末三张 mask，全部通过条目 CRC32 检查并保存 SHA256。

鱼类颜色编码明确为 RGB `(0,0,128)`。6 张采样 mask 都含鱼像素，尺寸为 1280×720。没有把 UVMulti 类别表直接当作 SUIM 八类编号使用。

目前只是标签/资源预检，尚未完整下载 RGB 配对、核对原拍摄组或制定正式训练/验证划分。其原始连续 RGB 帧和稀疏标注可作为下一阶段候选；必须检查同名图像与 mask、帧间隔和独立性后再训练。

预检记录：{link(ROOT / 'data/uvmulti_preflight/provenance.json')}，{link(ROOT / 'data/uvmulti_preflight/mask_availability.json')}。

## 17. 下一步实施顺序与验收

1. 从独立、可确认的连续视频来源准备训练与验证。优先核对 UVMulti 的原 RGB/GT 配对、序列边界、鱼颜色映射和每个训练帧的参考关系；保留本轮 SFISHTRACK 评价样例。
2. 建立普通预训练微调与 GT-ROI 微调两套配置，保持初始化、训练视频、裁剪策略、q 采样、更新次数、种子、失真定义一致。普通分支也使用匹配的原生损失，不能与阶段 A 的单对样本可行性测试比较增益。
3. 起步保留 batch1、两帧、256 裁剪、非零码率损失。训练对中同一裁剪位置应应用于参考、目标和 GT；跳帧/未标注区域要明确处理，不把独立图片伪造成视频。
4. ROI 权重先从预先声明的小范围开始，在验证集检查整体 RGB、GT 鱼区 RGB、真实文件大小与任务结果。权重选择不得依靠本轮保留评价帧的得分。
5. 保存普通与 ROI 权重，各自真实熵编码、独立解码；在同一完整文件预算下比较，包含 I/P/协议/容器及可能的模型更新成本。
6. 另做 SegFormer 的场景适配。采用可信训练标签与原 SUIM 类别保持方案，避免把六类标签直接混成八类真值；先在验证集证明原图鱼类识别可靠。
7. 分割器可靠性与 ROI 增益成立后，再接冻结 SegFormer 的可微任务损失。冻结参数保留输入梯度；训练后增加独立模型/人工 GT 评价，防止只优化同一教师的偏好。
8. 保留 GLC 与强 H.265 对照，补离线 B 帧设置、更多预算和更多独立场景。最后再加入真实限速发送、分包、误码/纠错与重同步。

首个正式 ROI 实验的验收：两套训练都有有效更新；真实码流可独立解码；同预算的目标区域质量/任务结果可复核；测试拍摄组未用于优化；提升与代价都记录。负结果也保留，不凭一个像素指标更高就认定语义通信更好。

## 18. 原始记录与复现

- {link(ROOT / 'audit/phase_b_commands/commands.json', '逐进程命令、退出码与日志索引')}
- {link(ROOT / 'configs/phase_b_pilot.json', '原潜水员预测一致性协议')}
- {link(ROOT / 'configs/phase_b_gt_codecs.json', '新 GT 帧集与固定编解码质量点')}
- {link(ROOT / 'adapters/evaluate_semantics.py', '冻结分割器、GT 与一致性分开的评价实现')}
- {link(ROOT / 'adapters/prepare_annotation_corpus.py', '标注帧配对与无损缓存检查')}
- {link(ROOT / 'adapters/run_phase_b.py', '编解码批次入口')}
- {link(ROOT / 'adapters/probe_roi_backprop.py', 'ROI 实际模型反向通路检查')}
- {link(ROOT / 'audit/requirements-phase-b-lock.txt', '本轮依赖锁定')}

所有目录为新工程下的独立结果。原 GLC、图片联合编码与 SegFormer 最佳权重保持原文件；GLC 复用了原有已验证环境和只读脚本，输出写入新目录。新脚本拒绝覆盖已有实验，复现应更换输出目录名。

本轮新增质量点共 15 组，另有 1 个文件大小匹配的 H.265 样例。传输时间仍是 `file_bytes×8/20000` 理论值；没有实际水声误码、FEC 或传输测速结果。
'''
report=OUT / "阶段B_语义评价与ROI准备_2026-10-13.md"
report.write_text(data_text,encoding="utf-8")
shutil.copy2(report,ROOT / "docs/phase_b_results_20261013.md")
shutil.copy2(ROOT / "runs/phase_b_quality/comparison.json",OUT / "阶段B_真实语义基线数据_2026-10-13.json")
print(json.dumps({"report":str(report),"lines":len(data_text.splitlines()),"bytes":report.stat().st_size,"quality_points":len(quality['rows']),"matched_difference_percent":matched['relative_file_size_difference']*100},ensure_ascii=False,indent=2))
