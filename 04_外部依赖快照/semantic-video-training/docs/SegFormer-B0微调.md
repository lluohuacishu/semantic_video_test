# SegFormer-B0 在 SUIM 上的八类微调

本次从 NVIDIA 发布的 ADE20K 预训练权重开始，将最终分类层由 150 类改为 SUIM 的 8 类。主干与多尺度融合解码器使用原结构和预训练参数，训练时一同更新。这是基于公开模型的水下迁移学习实验。

## 数据与训练设置

继续使用 `data/SUIM/manifest.json` 的固定划分：训练 1187 张、验证 297 张、测试 110 张。未更换数据集、重新划分或把测试图片用于训练。最佳权重按验证集 mIoU 选择；测试集在训练结束后评价。

| 设置 | 本次使用值 |
| --- | --- |
| 模型 | Transformers 5.9.0 的 SegFormer-B0 |
| 初始权重 | `nvidia/segformer-b0-finetuned-ade-512-512` |
| 权重版本 | `489d5cd81a0b59fab9b7ea758d3548ebe99677da` |
| 输入 | 长边缩放到 256，居中补齐为 256×256 |
| 标签 | SUIM 原有 0～7 类；补边为 255，不参与计分 |
| 输出 | 8 类，模型内将类别分数图上采样到输入尺寸 |
| 归一化 | ImageNet RGB 均值和标准差 |
| 批量 | 8 |
| 训练上限 | 60 轮，验证 mIoU 连续 15 轮不提升时停止 |
| 优化器 | AdamW，weight decay 0.0001 |
| 初始学习率 | 主干 0.00006，解码器 0.0006 |
| 学习率变化 | 余弦衰减到初始值的 5% |
| 损失 | 0.7 × 加权交叉熵 + 0.3 × Dice |
| 采样与增强 | 与原优化模型相同的类别平衡采样和 strong 增强 |
| 随机种子 | 20260911 |
| 训练计算 | CUDA 混合精度，梯度裁剪 1.0 |

SUIM 的背景／水体是需要学习的类别 0。这里没有沿用 ADE20K 示例中的标签减一操作，也没有把背景改成忽略标签。数据读取、几何变换和指标计算复用原项目，避免两套处理方式造成比较偏差。

这次与原 LR-ASPP 保持相同的输入尺寸和评价规则，但预训练来源、模型结构与学习率存在差异。因此这是实用的迁移学习方案比较，不能把所有指标变化单独归因于某一个结构设计。测试集在此前实验中也已查看过，不是全新的外部测试数据。

## 运行方法

使用 `.venv-segformer` 独立环境。`setup_segformer.cmd` 安装已锁定的依赖、准备预训练权重。以下命令在项目根目录运行。

```bat
rem 训练；默认会创建带时间戳的新目录
run_segformer_train.cmd

rem 测试已训练的模型
.venv-segformer\Scripts\python.exe scripts\evaluate_segmentation.py --checkpoint runs\suim_segformer_b0_20260915\segmentation_best.pt --split test --device cuda

rem 小目标统计
.venv-segformer\Scripts\python.exe scripts\evaluate_small_objects.py --checkpoint runs\suim_segformer_b0_20260915\segmentation_best.pt --split test --device cuda --output-dir runs\suim_segformer_b0_20260915\small_objects_test

rem 对自己的图片预测
.venv-segformer\Scripts\python.exe scripts\predict_segmentation.py --checkpoint runs\suim_segformer_b0_20260915\segmentation_best.pt --input "D:\images\水下图片.jpg"
```

已有结果目录不会被覆盖。需要再次评价时，换一个 `--output-dir`。训练过程每轮保存 `segmentation_last.pt`；`segmentation_best.pt` 仅在验证 mIoU 改善时更新。该训练入口没有保存优化器恢复状态，不应把从权重再次微调称为精确续训。

双击 `run_segformer_demo.cmd` 可在原有前台中使用本次 SegFormer 八类权重。默认地址是 `http://127.0.0.1:8768/`，可以拖入图片或视频，结果单独保存在 `runs/segformer_demo_console/`。原 `run_demo.cmd` 继续使用之前的 LR-ASPP 权重。

`run_segformer.cmd` 仍然是 ADE20K 150 类的预训练试跑入口；它与这次微调后的八类演示使用不同权重。

## 结果记录

训练和评价输出保存在 `runs/suim_segformer_b0_20260915/`。实际完成 54 轮后触发早停，最佳权重来自第 39 轮。含数据准备和模型加载共约 12.7 分钟；训练阶段 PyTorch 显存分配峰值为 641.1 MiB，预留峰值为 752.0 MiB。这是本次 256×256、batch size 8、混合精度训练的占用，不代表其他分辨率或模型的需求。

下面是相同 256×256 评价条件的结果。LR-ASPP 为原演示模型；FPN 为之前自定义的多尺度融合版本；SegFormer-B0 为本次微调模型。

| 指标 | LR-ASPP | 之前的 FPN | SegFormer-B0 |
| --- | ---: | ---: | ---: |
| 验证集 mIoU | 59.67% | 61.64% | 62.91% |
| 测试集 mIoU | 62.65% | 59.20% | **69.13%** |
| 测试集像素准确率 | 82.73% | 83.13% | **86.56%** |
| 极小区域像素召回（≤0.1%） | 8.59% | **31.26%** | 24.99% |
| 小区域像素召回（0.1%～1%） | 29.35% | **40.30%** | 35.40% |

SegFormer-B0 的测试 mIoU 比原模型提高 **6.48 个百分点**，在整体八类分割上有明显改善。小目标召回高于原 LR-ASPP，但低于之前的 FPN；这次结果不能表述为所有小目标指标都更好。

各类别测试 IoU：

| 类别 | LR-ASPP | SegFormer-B0 |
| --- | ---: | ---: |
| 背景／水体 BW | 87.47% | 89.66% |
| 潜水员 HD | 69.23% | 75.31% |
| 水生植物 PF | 22.96% | 32.81% |
| 沉船与遗迹 WR | 64.20% | 76.54% |
| 机器人与仪器 RO | 67.88% | 67.02% |
| 珊瑚礁与无脊椎动物 RI | 61.53% | 67.76% |
| 鱼与脊椎动物 FV | 64.26% | 74.25% |
| 海床与岩石 SR | 63.69% | 69.73% |

机器人与仪器类略有下降，水生植物类虽然提升但仍较弱。薄尾、细边界以及部分小区域仍有漏分，固定对照图保留了这些情况。

针对之前关心的珊瑚／沉船混淆，单独统计测试集真值为 RI、预测为 WR 的像素，占全部 RI 真值像素的比例：LR-ASPP 为 1.31%，SegFormer-B0 为 1.39%。这个具体错误没有随总体 IoU 一起下降，因此不能宣称已经解决视频里的珊瑚误判沉船问题。

相同 256×256 缓存输入、单图 FP32、预热后 50 次测量：

| 模型 | 前向中位耗时 | P95 耗时 |
| --- | ---: | ---: |
| LR-ASPP | 7.47 ms | 11.16 ms |
| SegFormer-B0 | 10.35 ms | 14.80 ms |

新模型的单次前向增加约 2.88 ms。上述时间包含模型内归一化和输出上采样，不是完整视频处理耗时。运行中的其他应用也会影响笔记本的测量结果。

图片上传及一个 1280×720、2 秒、10 帧的视频片段已经通过前台处理测试。原视频、类别视频和叠加视频均能解码为 10 帧，类别索引和逐帧统计已保存。这个检查验证图片／视频链路可用，不是 30 FPS 实时性测试。

结果文件：

| 路径（相对于本次实验目录） | 用途 |
| --- | --- |
| `segmentation_best.pt` | SUIM 八类模型，第 39 轮 |
| `config.json`、`history.csv`、`summary.json` | 训练设置、逐轮记录、训练汇总 |
| `learning_curve.png` | 训练损失和验证 mIoU 曲线 |
| `evaluation_test/`、`evaluation_validation/` | 全量指标、预测标签、混淆矩阵和固定样本图 |
| `small_objects_test/`、`small_objects_validation/` | 小目标汇总和逐区域 CSV |
| `comparison_with_baseline/before_after.png` | 原图、真值、LR-ASPP、SegFormer 的固定六张样本对照 |
| `benchmark.json` | 同条件模型前向测速 |
| `verification.json`、`demo_jobs.json` | 指标复核和前台图片／视频检查记录 |

目前是单个随机种子的一次微调实验；没有多次独立重复或跨场景外部验证。可以据此说明本项目中取得的改进，不能据此给出普遍优劣或论文级性能结论。

## 评价口径

- mIoU 来自全数据集累计的混淆矩阵，包含背景，忽略补边 255；与原项目一致。
- 小目标以每个前景类别的标注连通区域统计，不是实例检测 AP。极小区域面积不超过有效图像的 0.1%，小区域为 0.1%～1%。
- 小目标像素召回表示这些标注区域内有多少像素被预测成正确类别；需结合每类 IoU 看误检情况。
- 测速在相同缓存输入、相同分辨率、单张图片、FP32 下进行，包含模型内归一化和输出上采样。它不包含视频编解码、文件读写或网络传输。

本次同时修复了小目标统计表的一个索引问题：原脚本在每个批次重新从第一个文件名计数，导致 CSV 中部分区域关联到错误图片。修复不改变聚合的召回计算，只纠正逐区域记录的图片来源。

完成了 24 项相关回归检查，其中包括 3 项 SegFormer 适配检查、12 项原分割检查、1 项小目标跨批次检查和 8 项原前台接口检查。另从保存的 110 张测试标签和 297 张验证标签重新计算混淆矩阵，并逐图核对小目标明细面积；结果均一致。原模型和新模型的测试输入与真值像素也逐张验证一致。

## 参考

- [SegFormer 论文](https://arxiv.org/abs/2105.15203)
- [作者仓库](https://github.com/NVlabs/SegFormer)
- [NVIDIA 的 B0 预训练权重](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- [Transformers SegFormer 文档](https://huggingface.co/docs/transformers/v5.9.0/en/model_doc/segformer)
- [SUIM 数据集与模型](https://github.com/IRVLab/SUIM)
