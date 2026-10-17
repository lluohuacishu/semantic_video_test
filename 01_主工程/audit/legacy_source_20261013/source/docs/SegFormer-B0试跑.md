# SegFormer-B0 预训练试跑

2026-09-15 已在 RTX 4060 Laptop 8GB 上跑通图片推理。本次使用公开权重，没有训练，也没有改变原来的 SUIM 演示模型。

后续的 SUIM 八类训练另见 [SegFormer-B0 微调](SegFormer-B0微调.md)。本文保留最初的 ADE20K 预训练推理记录，两次实验使用不同的分类层和权重。

## 模型与来源

- 论文：[SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)，NeurIPS 2021。
- 作者代码：[NVlabs/SegFormer](https://github.com/NVlabs/SegFormer)。
- 权重：[nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)，固定版本 `489d5cd81a0b59fab9b7ea758d3548ebe99677da`。
- 运行实现：[Hugging Face Transformers 的 SegFormer](https://huggingface.co/docs/transformers/v5.9.0/en/model_doc/segformer)。

作者原仓库依赖旧版 PyTorch/MMCV。本次使用 Transformers 5.9.0 的现成实现，加载 NVIDIA 发布的转换权重，未自行改写主干或解码器。这验证了模型与权重的推理流程，不等于复现了论文训练过程或完整 ADE20K 评测。

权重为约 14.34 MiB 的 `model.safetensors`，下载到 `checkpoints/pretrained/segformer-b0-ade/`。同目录 `provenance.json` 记录来源、版本、文件大小和 SHA-256；下载来源依据作者仓库与 NVIDIA 模型页确认，指纹用于核对文件完整性。推理只读取本地文件，不启用远程自定义代码。

## 环境和操作

独立环境为 `.venv-segformer`，使用 Python 3.13.7、PyTorch 2.12.0+cu126、torchvision 0.27.0、Transformers 5.9.0。GPU 包从 PyTorch 官方索引安装，其他包从 PyPI 安装。固定版本分别在 `requirements-torch.txt` 和 `requirements-segformer.txt`。

本机已经完成安装。需要重新建立时，先安装 Python 3.13，再运行项目根目录的 `setup_segformer.cmd`。它也会准备预训练权重和作者提供的街景示例图片。数据、权重和虚拟环境不随 Git 提交。

在项目目录的终端中运行：

```bat
rem 作者街景示例 + 从 SUIM 验证集固定取出的四张图片
run_segformer.cmd

rem 自己的图片；也可以把图片拖到 run_segformer.cmd 上
run_segformer.cmd "D:\images\水下图片.jpg"

rem 一个目录中的图片，不递归读取子目录
run_segformer.cmd "D:\images"

rem 调整输入尺寸或使用 CPU
run_segformer.cmd "D:\images\水下图片.jpg" --size 256 --device cpu
```

支持 JPG、PNG、BMP、WebP、TIFF 等常见图片格式；当前脚本只处理图片。无参数运行需要现有的 `data/SUIM/manifest.json`。输入会转换成 RGB，并处理 EXIF 方向。

每次运行创建单独的结果目录，终端会打印完整路径。在文件管理器中打开其中的 `index.html`，可查看全部图片、颜色图例和各类像素占比。也可直接查看 `comparison.png`。现有 `run_demo.cmd` 的网页仍是原来的 SUIM 八类演示。

每张图分别保存：

| 文件 | 内容 |
| --- | --- |
| `input.png` | 处理 EXIF 方向后的 RGB 原图 |
| `prediction_index.png` | 0～149 的类别编号，与原图尺寸一致 |
| `prediction_color.png` | 给类别编号着色的显示图 |
| `overlay.png` | 原图与类别颜色叠加 |
| `summary.json` | 输入来源、尺寸、处理耗时、各类别像素数 |

输出目录非空时拒绝覆盖。默认结果目录位于 `runs/segformer_b0_pretrained_*`；离线推理不需要重新下载权重。

## 这次的实测结果

完整报告位于 `runs/segformer_b0_pretrained_20260915_verified/index.html`，数据在同目录的 `summary.json`。测试图片为作者街景示例和按 SUIM 验证清单均匀取样的四张图片，没有按效果挑图。首次报告导出曾遇到集合无法序列化的问题，修复后重新完成五张图片的运行。

| 项目 | 结果 |
| --- | ---: |
| 参数量 | 3,752,694 |
| 计算类型 | 单图、FP32 |
| 网络输入 | 512 × 512 |
| 模型前向中位耗时 | 20.60 ms |
| 模型前向 P95 耗时 | 22.09 ms |
| 含预处理与后处理的单图耗时 | 约 29.5～58.7 ms |
| PyTorch 显存分配峰值 | 515.8 MiB |
| PyTorch 显存预留峰值 | 968.0 MiB |

模型前向耗时在三次预热后测量二十次，并进行 CUDA 同步。它不包含图片读取、预处理、输出上采样和文件保存。单图处理耗时包含预处理、设备复制、前向、logits 上采样、argmax 和结果复制回 CPU，但不包含读盘和保存。两者都不能直接当作完整视频系统的帧率或延迟。

显存是本进程由 PyTorch 统计的图片处理阶段峰值，包含模型，不代表整张显卡总占用，也不代表训练显存需求。

## 如何理解输出

街景中，道路、建筑和天空的主要区域能被分开。水下结果则出现明显类别不匹配：鱼和珊瑚被并入植物、岩石，沉船被预测成山；另一个水下场景的大面积水体被预测成墙。这些错误如实保存在报告里。

这个权重面向 ADE20K 的 150 类，不是当前 SUIM 的八类分类器。它没有当前任务所需的完整水下类别定义，也没有用现有 SUIM 训练数据适配。因此本次不报告 SUIM mIoU，不能据此判断 SegFormer 的水下分割能力一定强于或弱于当前模型。

512×512 输入时，实测四个主干阶段输出为：

| 阶段 | 特征尺寸 | 相对输入的下采样倍数 |
| --- | --- | ---: |
| 1 | 32 通道，128×128 | 4 |
| 2 | 64 通道，64×64 | 8 |
| 3 | 160 通道，32×32 | 16 |
| 4 | 256 通道，16×16 | 32 |

解码器融合这些尺度，产生 150 通道、128×128 的类别分数图，再将分数上采样到原图尺寸后取类别。最深层虽然下采样到 1/32，但解码器还使用 1/4 等较细特征，不能仅凭最深层倍数推断小目标效果。恢复成原图大小也不等于重新找回已经丢失的细节。

要与原模型比较，下一次应保留现有 SUIM 划分，将分类层适配成八类并进行微调；使用一致的标签、输入尺寸和评价口径，同时报告各类 IoU、小目标召回、推理耗时和显存。当前只完成了这之前的公开模型推理验证。

## 已做的检查

- 五张图片输出尺寸、类别编号范围、像素计数与占比都通过检查。
- 图像输出和 HTML 报告的本地链接有效，汇总图已查看。
- 同一街景图片在 CPU 和 CUDA 上的类别图像素一致率为 99.9987%。这是运行一致性检查，不是分割准确率。
- Windows 启动脚本、中文输出路径和缓存权重重复准备均正常。
- 新旧环境的依赖检查通过，原演示模型和训练数据未变更。

检查记录保存在完整结果目录的 `verification.json`。
