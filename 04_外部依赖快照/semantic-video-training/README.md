# 水下视频训练实验

用于学习 PyTorch，开展水下图像重建、八类语义分割和联合语义源编码实验。项目包含 SUIM 数据整理、训练评估、图片／视频分割界面，以及真实图片码流的编解码和 TCP 文件收发。实验记录见 [首次重建训练](docs/SUIM首次训练.md)、[首次语义分割训练](docs/SUIM语义分割训练.md)、[视频逐帧分割记录](docs/视频逐帧分割.md)、[图片压缩与文件传输](docs/图片压缩与文件传输.md)和[联合语义编码](docs/联合语义编码.md)。

普通 CompressAI 压缩保留为对照，其 SegFormer 只是事后评价。新联合模型将发送端预测的水下语义送入编码器，从同一份量化潜变量码流直接解码 RGB 和八类类别图，并以码率、像素和语义损失联合训练。这是理想信道下的任务导向语义源编码原型，尚未加入水声信道或视频时序编码，也不是 GLC / SQ-GAN 的复现。

## 联合语义编码演示

本机已有训练权重时，双击 **run_semantic_codec_demo.cmd**，打开 `http://127.0.0.1:8769/`。选择“联合语义编码”、20／30／40 kbps 等限速后拖入图片；可查看还原图片和“两端类别”。接收端类别来自码流的直接解码，没有对还原图再次分割。模型输入与权重对应，当前固定发送端分割输入为 256。

首轮使用 1187 张训练图、297 张验证图，训练 12 轮。按联合验证损失选择第 6 轮权重，真实 CPU 熵码流的直接语义 mIoU 为 61.36%，原发送端预测为 62.91%；这是 256 补边、忽略 padding 的验证口径。联合文件平均 2566 字节，普通 q3 参考为 2497 字节，尚未获得文件大小优势。720p 图片已完成 20／40 kbps 文件传输；视频语义编码还未实现。

操作、原理、评价口径和限制见 [联合语义编码](docs/联合语义编码.md)。权重位于 `runs/suim_joint_semantic_codec_20261002/semantic_codec_best.pt`，模型与数据不随 Git 上传；第一次克隆需要按文档准备环境并训练。

## 本地演示界面

已有环境和训练权重时，双击 **run_demo.cmd**，浏览器会打开 `http://127.0.0.1:8767/`。等模型就绪后，将图片或视频拖进左侧框即可开始；也可以点击选择文件。演示过程不需要联网或重新训练。

- 图片支持 JPG、PNG、WebP、BMP，视频支持 MP4、AVI、MOV、MKV、WebM、M4V，具体取决于文件中的编码是否可解码。
- 页面提供输入 / 分割结果对照、视频同步播放、类别占比、处理耗时和结果下载。
- 可以调整模型输入尺寸、叠加比例、视频三帧投票、处理帧数上限，以及是否保存逐帧标签。
- 原素材与每次结果单独保存在 `runs/demo_console/`。在最近记录中选中一条，修改参数后可重新处理；旧结果保留。

默认加载 `runs/suim_transfer_20260911/segmentation_best.pt`。模型、数据、虚拟环境和 FFmpeg 都不在 Git 仓库内，首次克隆后需自行准备。视频预览使用本机 FFmpeg 输出浏览器兼容的无声 H.264；可通过 `FFMPEG_PATH` 或 `--ffmpeg` 指定。配置与现场操作见 [演示界面使用说明](docs/演示界面使用说明.md)。

## 图片压缩与限速文件传输

已有图片编码环境时，双击 **run_image_codec_demo.cmd**，打开 `http://127.0.0.1:8769/`。选择质量档位和 20／30／40 kbps 等限速，拖入 JPG、PNG、WebP 或 BMP；页面会依次编码、通过本机 TCP 发送文件、校验接收副本，再还原图片。质量档位与发送速度可分别调整。

- 查看原图／还原图、PSNR、SSIM、实际文件字节数、发送速度和耗时。
- 用“前后类别”对照同一个 SegFormer 在原图和还原图上的预测；预测一致性不是真值准确率。
- 下载 `.svic` 压缩文件、还原 PNG 和传输记录。每次任务保存在 `runs/image_codec_console/`。

采用 CompressAI 官方 `bmshj2018-hyperprior` 预训练模型，接收端只需压缩文件和预先准备好的相同模型。当前编解码统一使用 CPU，语义检查可用 GPU。20 kbps 限制的是发送方向应用层文件及传输头，允许多花时间传完；不代表压缩后视频的平均码率，也不包含 TCP/IP 包头。

本机已完成 720p 图片的独立进程编解码、20／40 kbps 实测，以及固定 6 张 SUIM 验证图的三档质量对照。操作、结果和环境复建见 [图片压缩与文件传输](docs/图片压缩与文件传输.md)。图片链路暂不接收视频；视频仍可在原有模式做逐帧分割。

## 环境与入口

### SegFormer-B0 预训练试跑

`setup_segformer.cmd` 配置独立的 `.venv-segformer` 环境，并从 NVIDIA 的 Hugging Face 模型页面下载权重。双击 `run_segformer.cmd` 可处理作者示例和固定抽取的 SUIM 验证图片，也可以将图片拖到该脚本上。结果包含原图、类别图、叠加图、耗时记录和本地 HTML 报告，保存在 `runs/segformer_b0_pretrained_*`。

这个入口使用 ADE20K 150 类预训练权重，因此不能与 SUIM 八类模型直接比较 mIoU。来源、操作和本机试跑记录见 [SegFormer-B0 试跑](docs/SegFormer-B0试跑.md)。

### SegFormer-B0 的 SUIM 八类微调

`run_segformer_train.cmd` 使用相同的 SUIM 数据划分、256×256 等比缩放与补边，对公开预训练模型进行八类微调。分类层重新初始化，主干和多尺度解码器一同训练。该入口使用 `.venv-segformer`，最佳权重只按验证 mIoU 选择。

本机这次的权重在 `runs/suim_segformer_b0_20260915/segmentation_best.pt`。双击 **run_segformer_demo.cmd** 可打开使用这份权重的图片／视频演示，地址为 `http://127.0.0.1:8768/`。原 `run_demo.cmd` 保留 LR-ASPP 演示。训练方法、评价口径和对照结果见 [SegFormer-B0 微调记录](docs/SegFormer-B0微调.md)。

本次最佳权重来自第 39 轮，110 张测试图片的 mIoU 为 69.13%，原 LR-ASPP 为 62.65%。极小区域召回由 8.59% 提升到 24.99%，仍低于之前 FPN 的 31.26%。比较使用相同测试输入、标签和评价协议，预训练来源与学习率存在差异；这是一次迁移学习实验结果。

### 原有训练环境

环境已配置：Python 3.13.7、PyTorch 2.12.0 + CUDA 12.6、torchvision 0.27.0，以及 NumPy、Pillow、OpenCV、Matplotlib、tqdm、PyYAML、scikit-image。已识别 RTX 4060 Laptop 的 8GB 显存。

- 双击 **run_check.cmd**：检查依赖并运行 8 步 GPU 训练自检，报告写入 runs/environment_check.json。
- 双击 **open_env.cmd**：打开已经激活本项目虚拟环境的命令行，以下命令可以直接在里面执行。
- 双击 **run_demo.cmd**：启动图片 / 视频语义分割演示界面。
- 双击 **run_train.cmd**：使用 data/frames 中的数据训练，默认 256×256、batch size 4、20 轮。
- 双击 **run_suim_train.cmd**：使用 SUIM 固定名单训练，默认 256×256、batch size 8、20 轮，使用 CUDA 和内存图片缓存。
- 双击 **run_segmentation_train.cmd**：训练 SUIM 八类语义分割，默认 256×256、batch size 8、最多 40 轮。
- 双击 **run_segmentation_optimized.cmd**：使用官方预训练特征微调分割模型，默认最多 60 轮，并启用类别平衡采样和较强数据增强。
- **prepare_suim.cmd**：提取、检查和划分 data/SUIM.zip；
- **setup_env.cmd**：需要重建环境时使用，要求已安装 Python 3.13。

依赖来源、固定版本和自检内容见 [环境记录](ENVIRONMENT.md)。VS Code 已设置默认解释器为本目录的 .venv。

## SUIM 数据与训练

SUIM 是带像素级标签的水下图片集，不是连续视频。官方提供 1525 对训练/验证样本和 110 对测试样本。来源见 [SUIM 官方页面](https://irvlab.cs.umn.edu/resources/suim-dataset)；实验室[资源说明](https://irvlab.cs.umn.edu/resources)限定数据用于学术研究。

本次检查发现，原训练数据中有 37 对图片与标签尺寸不同、4 张图片与测试图片像素完全相同。原文件全部保留，这 41 张不参与训练和验证。剩余样本按固定随机种子划分，完全相同的图片放在同一组。

| 用途 | 图片数 | 作用 |
| --- | ---: | --- |
| 训练 | 1187 | 更新模型参数 |
| 验证 | 297 | 选择最佳训练轮次 |
| 测试 | 110 | 训练结束后评估，不参与训练或选模型 |

这是图片层面的划分。精确去重不能排除近似图片和相邻视频帧；数据未提供完整场景编号，不能据此完成严格的跨场景验证。

~~~text
data/SUIM.zip                    保留的原始压缩包
data/SUIM/train_val/              原训练/验证图片与 RGB 标签
data/SUIM/TEST/                   官方测试图片与合并 RGB 标签
data/SUIM/indexed_masks/          另存的 0～7 类别编号 PNG
data/SUIM/manifest.json           固定划分、配对路径、图片指纹、类别表
data/SUIM/preparation_report.json 重复、尺寸问题与颜色噪声记录
data/SUIM/dataset_preview.png     原图与标签对照
~~~

八个类别依次为背景水体、潜水员、水生植物、沉船与遗迹、机器人与仪器、珊瑚礁与无脊椎动物、鱼与脊椎动物、海床与岩石。85 张标签含非标准纯色像素；每个 RGB 通道以 128 为阈值，按最近的八色顶点转换为类别编号。原始 BMP 不变，边界仍有标注噪声。索引 PNG 看起来接近黑色，因为像素值只有 0～7；以后读取它应使用单通道整数，缩放采用最近邻插值。

当前数据已准备好，直接训练即可：

~~~bat
python scripts\train_autoencoder.py --manifest data\SUIM\manifest.json --epochs 20 --batch-size 8 --size 256 --device cuda --cache-images
~~~

图像等比缩放后中心裁剪为 256×256。--cache-images 将裁剪后的 uint8 图片缓存在系统内存中，当前数据的原始缓存约 278MiB，避免每轮重复解码。重建训练只使用图片，不使用分割标签。

需要从原压缩包重新准备时：

~~~bat
python scripts\prepare_suim.py --archive data\SUIM.zip
~~~

脚本检查整个 ZIP 的 CRC，仅提取原图、合并 RGB 标签和 INFO.txt。没有使用包内的旧模型权重、二值标签副本和预计算评估结果。

已有目录默认不覆盖；中断后可加 --reuse-extracted，脚本会再次核对源文件 CRC。若已有模型依赖当前划分，请保持 seed 和 validation-fraction，或使用其他输出目录保存新划分。模型评估会核对数据清单指纹。

## 语义分割

目前有三类可复用的训练路线：TinyUNet 用于观察从零训练的小模型，原始 LR-ASPP + MobileNetV3 用于迁移学习，新增的 `lraspp-hr4` 和 `lraspp-fpn` 用于检查高分辨率与多尺度特征对小目标的影响。已有模型和数据划分均保留。

### 迁移学习版本

已完成一轮优化实验：在相同 110 张测试图、相同输入和标签上，mIoU 从 37.20% 提升到 62.65%，植物类 IoU 从 0.18% 提升到 22.96%。旧模型保留，详细设置、前后对照和剩余问题见 [语义分割优化记录](docs/语义分割优化记录.md)。

先获取官方权重，再训练：

~~~bat
python scripts\prepare_pretrained.py
python scripts\train_segmentation.py --model lraspp --pretrained checkpoints\pretrained\lraspp_mobilenet_v3_large-d234d4ea.pth --balanced-sampling --augmentation strong --epochs 60 --patience 15 --learning-rate 0.001 --backbone-lr-factor 0.1 --batch-size 8 --size 256 --device cuda --cache-images
~~~

也可以直接双击 run_segmentation_optimized.cmd。首次从 torchvision 指定的 [PyTorch 官方下载地址](https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth)获取约 12.5MiB 的权重，缓存到 checkpoints/pretrained；后续复用缓存。来源、大小和完整 SHA-256 记录在同目录 JSON 中。权重通过 weights_only=True 读取；官方 HTTPS 来源是首要依据，指纹只用于完整性核对。

这个模型有 3,219,328 个参数。预训练来源是 COCO 中与 Pascal VOC 对应的类别子集，见 [torchvision 权重说明](https://docs.pytorch.org/vision/0.27/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html)。保留特征提取和共享分割层，两个原分类层替换为八类 SUIM 输出层，不会把原来的 21 类标签直接套用到水下任务。

训练时特征提取层学习率为输出侧学习率的 0.1 倍，各组学习率按相同余弦比例下降。类别采样由 50% 均匀采样与 50% 按类别出现次数平衡的采样混合；只从训练图统计，目标类别占图像至少 0.5% 才计为出现，每轮有放回采样 1187 张。不会为了提高分数修改验证或测试分布。

strong 增强增加图片/标签同步的 ±8° 旋转、最多 20% 放大和 ±2% 平移，亮度/饱和度因子范围扩大到 0.8～1.2。标签保持整数类别，补边仍为 255。输入和评价仍采用 256×256 等比缩放补边，与旧模型使用相同的计分口径。

评估和单图预测入口兼容这些模型，推理时无需预训练下载文件或联网。针对原迁移学习实验：

~~~bat
python scripts\evaluate_segmentation.py --checkpoint runs\suim_transfer_20260911\segmentation_best.pt --device cuda
python scripts\predict_segmentation.py --checkpoint runs\suim_transfer_20260911\segmentation_best.pt --input "path/to/underwater.jpg" --device cuda
python scripts\compare_segmentation_runs.py --baseline runs\suim_segmentation_20260911 --candidate runs\suim_transfer_20260911
~~~

compare_segmentation_runs.py 会检查输入、标签、数据清单、分辨率和评价方式一致，输出逐类前后对照与同一批固定测试图的预测差异。该比较反映组合改进，不是单因素消融；测试集此前已查看，不能把它表述为全新的未知外部数据。

### 小目标与多尺度特征消融

师姐提出的“小目标可能在下采样中丢失”可以用两类实验检查。当前 LR-ASPP 使用 MobileNetV3 的 1/8 低层特征和 1/16 高层特征，训练输入边长为 256；原图中的小目标经过缩放后可能只剩几个像素。

第一组实验只改变评估分辨率，不重新训练模型，用来区分输入缩放的影响：

~~~bat
python scripts\evaluate_segmentation.py --checkpoint runs\suim_transfer_20260911\segmentation_best.pt --split test --size 384 --batch-size 4 --device cuda --output-dir runs\ablation_lraspp_384_test
python scripts\evaluate_small_objects.py --checkpoint runs\suim_transfer_20260911\segmentation_best.pt --split test --size 256 --batch-size 8 --device cuda --output-dir runs\ablation_lraspp_256_small_objects
python scripts\evaluate_small_objects.py --checkpoint runs\suim_transfer_20260911\segmentation_best.pt --split test --size 384 --batch-size 4 --device cuda --output-dir runs\ablation_lraspp_384_small_objects
~~~

目标统计使用标注的 8 邻域连通区域：`tiny` 为目标面积不超过有效图像的 0.1%，`small` 为 0.1%～1%，并报告区域内同类像素召回和召回达到 50% 的目标比例。已有 256→384 结果中，tiny 像素召回由 8.6% 提升到 22.8%，small 由 29.4% 提升到 37.7%，但整体 mIoU 基本没有提升，说明提高输入分辨率有帮助，却不能独立解决类别混淆。

第二组实验保留同一个 MobileNetV3 主干，加入 1/4、1/8、1/16 的逐级融合解码头，模型名为 `lraspp_mobilenet_v3_large_fpn_suim_v1`：

~~~bat
python scripts\train_segmentation.py --model lraspp-fpn --pretrained checkpoints\pretrained\lraspp_mobilenet_v3_large-d234d4ea.pth --balanced-sampling --augmentation strong --epochs 60 --patience 15 --batch-size 8 --size 256 --device cuda --cache-images
python scripts\evaluate_segmentation.py --checkpoint runs\suim_transfer_fpn_20260912\segmentation_best.pt --split test --device cuda --output-dir runs\ablation_fpn_256_test
python scripts\evaluate_small_objects.py --checkpoint runs\suim_transfer_fpn_20260912\segmentation_best.pt --split test --device cuda --output-dir runs\ablation_fpn_256_small_objects
~~~

本机一次训练结果：原 LR-ASPP 测试 mIoU 为 62.65%，三尺度 FPN 版本为 59.20%；但 tiny 像素召回由 8.6% 提升到 31.3%，small 由 29.4% 提升到 40.3%。FPN 版本模型前向中位数约 4.11 ms，原模型约 3.79 ms（RTX 4060、256 输入、预热后单图，不含视频读写）。因此它是针对小目标的候选模型，不应在所有场景直接替换原模型；水生植物和机器人类别仍需继续调整类别损失、采样或裁剪训练。

验证集上的同口径统计为：原模型到 FPN 的验证 mIoU `59.67% -> 61.64%`，tiny 像素召回 `9.3% -> 25.8%`，small 像素召回 `35.0% -> 47.8%`。模型选择仍只看验证集，测试集结果用于最后一次报告。

`lraspp-hr4` 是只把低层分支改为 1/4 的简化对照，`lraspp-fpn` 才是当前推荐继续研究的多尺度版本。`run_segmentation_fpn.cmd` 可直接启动 FPN 训练。小目标报告和训练输出均保存在 `runs/`，不覆盖原来的模型。

### 从零训练的入门版本

使用相同的 1187 / 297 / 110 张训练、验证和测试划分。TinyUNet 是按 U-Net 编码器、解码器和跳跃连接思路写的缩小版本，包含 488,120 个参数，从随机权重训练，没有下载预训练权重。模型输出 8 个类别分数通道，取最高分得到像素类别。

~~~bat
python scripts\train_segmentation.py --manifest data\SUIM\manifest.json --epochs 40 --batch-size 8 --size 256 --device cuda --cache-images
~~~

分割任务的预处理与重建实验不同：按长边等比缩放至 256，居中补边，保留完整视野。标签使用最近邻插值；补边标签设为 255，在损失和指标中忽略。训练时图片和标签一起水平翻转，并仅对图片轻微调整亮度和饱和度。验证和测试不使用随机变换。

损失为 0.7 倍带类别权重的交叉熵，加 0.3 倍 Soft Dice 损失。权重仅由训练集缩放后的有效像素统计，降低类别数量不平衡的影响。优化器为 AdamW，初始学习率 0.001，按余弦曲线下降。默认验证 mIoU 连续 12 轮没有提升时提前停止，可用 --patience 0 禁用。

每次运行保存到新的 runs/suim_segmentation_时间戳：

| 文件 | 内容 |
| --- | --- |
| config.json | 超参数、训练/验证名单、类别权重及数据清单指纹 |
| history.csv、learning_curve.png | 训练损失、验证损失和验证 mIoU |
| segmentation_best.pt | 验证 mIoU 最高轮次的模型 |
| segmentation_last.pt | 最后一个完整轮次的模型 |
| best_validation.json | 最佳轮次的各类指标及混淆矩阵 |
| validation_comparison.png | 验证图的输入、真值、预测和叠加对照 |
| summary.json | 最佳轮次、耗时、样本数、模型大小及张量显存统计 |

这两个权重用于推理，当前没有断点续训。再次训练会新建实验并从头开始，原结果不会覆盖。

对本次最佳模型进行测试：

~~~bat
python scripts\evaluate_segmentation.py --manifest data\SUIM\manifest.json --checkpoint runs\suim_segmentation_20260911\segmentation_best.pt --device cuda
~~~

评估结果保存在模型目录的 evaluation_test，包含汇总、每类 IoU/Dice、逐图指标、混淆矩阵、对照图以及全部 110 张测试图片的预测标签、彩色类别图和叠加图。predictions 和 ground_truth 中的 PNG 保存实际计分的 0～7 类别，补边为 255；inputs 保存对应的缩放补边图，三者尺寸相同。

IoU = 预测与真值的交集 / 两者的并集。先累计所有测试图的混淆矩阵，再对八类 IoU 等权平均得到 mIoU，包含背景；并集为空的类别不参与平均。补边像素不计分。像素准确率也会报告，但不用于选模型，避免背景等大类掩盖小目标识别问题。这里是在缩放后的分辨率计分，不是论文官方协议或原始分辨率评估。

输出目录非空时不会覆盖。重新测试可用 --output-dir 指定新目录；模型与数据清单指纹不一致时会停止。调参使用验证集，不应反复查看测试集来选择超参数。

对自己的单张图片做预测：

~~~bat
python scripts\predict_segmentation.py --checkpoint runs\suim_segmentation_20260911\segmentation_best.pt --input "path/to/underwater.jpg" --device cuda
~~~

支持 Pillow 可读取的图片格式，如 JPG、PNG。输出包括原图、单通道类别编号、彩色类别图、叠加图和预测类别占比。推理先使用缩小的完整图像，再把类别分数上采样回原图尺寸；输出与原图同尺寸不代表原生高清推理。新图片没有标签，脚本不会报告准确率。

## 使用自己的视频

先按原始场景整理视频。例如：

~~~text
data/raw/
  scene01/
    source.avi
  scene02/
    source.mp4
~~~

同一片段的不同码率、分辨率、帧率版本应归入同一个场景目录，不能分别当作训练集和验证集。正式实验应优先使用原始视频。导师资料里现有三个文件名含 reconstructed 的 AVI 是重建结果，适合检查流程，不能直接视为三段独立的原始训练视频。

抽帧：

~~~bat
python scripts\extract_frames.py --input-dir data\raw --output-dir data\frames --stride 5 --max-frames 500
~~~

每 5 帧保存一张 PNG，每个视频最多 500 张；这是按帧间隔抽样，不代表固定的抽样帧率。输出保留目录层级和视频扩展名，例如 data/frames/scene01/source.avi/00000000.png。目标目录已经存在时会停止，避免换参数后混入旧帧。

也可以从其他位置读取单个视频，无需复制源文件：

~~~bat
python scripts\extract_frames.py --input "path/to/sample.mp4" --output-dir data\frames\scene01 --stride 5 --max-frames 500
~~~

## 视频逐帧分割

图片模型可以直接用于视频，但当前采用的是“每帧独立推理”，没有时序网络和跨帧缓存。下面的命令会读取 AVI、MP4 等 OpenCV 能打开的文件，保留原始宽高和 FPS，生成彩色类别视频、原图叠加视频、逐帧耗时表和汇总报告：

~~~bat
python scripts\segment_video.py ^
  --checkpoint runs\suim_transfer_20260911\segmentation_best.pt ^
  --input "path/to/test5_5fps_reconstructed.avi" ^
  --output-dir runs\video_segment_test5_5fps ^
  --device cuda --save-masks
~~~

输出目录必须为空，脚本不会覆盖旧实验。`masks/frame_000000.png` 中保存的是 0～7 的类别编号，适合后续生成重要区域掩码；它看起来接近黑色是正常现象，查看时应使用彩色视频或叠加视频。`summary.json` 会记录输入 FPS、端到端处理 FPS、各阶段耗时、相邻帧类别变化比例和各类别像素占比。端到端处理 FPS 高于输入 FPS 只能说明本机这次离线处理跟得上输入，不能代替真实网络传输延迟。

如果画面颜色有跳变，可以开启因果投票稳定：

~~~bat
python scripts\segment_video.py --checkpoint runs\suim_transfer_20260911\segmentation_best.pt --input "path/to/sample.mp4" --output-dir runs\video_segment_temporal3 --device cuda --temporal-window 3
~~~

`--temporal-window` 只能取 1、3、5。1 表示关闭；3 或 5 会用最近几帧的类别图投票，近期帧权重更高。它可能带来最多一帧到两帧的显示滞后，也不能修复模型本身的误分割。报告同时保存原始预测和保存结果的变化比例，便于判断稳定器是否真的起作用。

例如导师提供的 1280×720、5 FPS 文件在 RTX 4060 Laptop 上处理 50 帧约 3.16 秒，端到端处理速度约 15.84 FPS。相邻输出类别变化比例均值约 18.9%，它只是抖动参考；由于视频没有逐帧真值，不能据此报告视频分割准确率。`--codec MJPG` 可改为 AVI 输出，默认 `mp4v` 输出 MP4。

## 语义感知压缩原型

可以让分割出的重要类别保持原图细节，把背景先降采样再用同一个视频编码器写出，同时生成普通压缩基线：

~~~bat
python scripts\semantic_compress_video.py ^
  --checkpoint runs\suim_transfer_20260911\segmentation_best.pt ^
  --input "path/to/test5_5fps_reconstructed.avi" ^
  --output-dir runs\semantic_compress_test5 ^
  --device cuda --temporal-window 3 --background-scale 0.5 --mask-dilate 2
~~~

输出包含 `baseline.mp4`、`semantic.mp4`、逐帧类别掩码、每帧处理耗时和 `summary.json`。报告按预测掩码分别统计整体、重要区域和背景的 PSNR/SSIM，并记录文件大小变化。`--background-scale 0.25` 会更明显地降低背景空间细节；`--important-classes` 可以改成例如 `1,4,6`，只保护潜水员、机器人和鱼类。

这一步只是验证“重要区域优先”的处理思路。OpenCV 的 `mp4v` 没有稳定的目标码率控制，短视频上语义版本的文件大小可能只减少很少；真正要研究码率收益，应接入固定码率 H.264/HEVC 或把前景、背景和掩码设计成可传输的分层码流。报告中的 PSNR/SSIM 是相对输入视频的工程指标，不能等同于语义任务准确率。

## 视频帧训练

有至少两个独立场景时：

~~~bat
python scripts\train_autoencoder.py --frames-dir data\frames --epochs 20 --batch-size 4 --size 256 --device cuda
~~~

脚本按 data/frames 下的一级场景目录划分训练集和验证集，约留出 20% 的场景用于验证。同一组中的所有帧一起划分。图像先等比缩放，再取中心方形区域；720p/1080p 输入在这一步会变成 256×256 的训练裁剪帧，并非直接进行整幅高清视频训练。

只有一段视频、只想检查训练流程时：

~~~bat
python scripts\train_autoencoder.py --frames-dir data\frames --epochs 3 --batch-size 2 --size 256 --device cuda --allow-single-video
~~~

此模式不会复用训练帧来计算“验证损失”，验证字段留空。重复拟合少量帧只能说明程序在学习，不能证明模型对新场景有效。

CUDA 训练默认使用混合精度。建议从 256×256、batch size 2 或 4 开始；如果显存不足，先降低 batch size。size 必须是 8 的倍数。默认 num_workers=0，便于先排查 Windows 下的数据读取问题。

## 结果

每次训练创建一个新的 runs/autoencoder_时间戳，保存：

| 文件 | 内容 |
| --- | --- |
| config.json | 参数、随机种子、实际训练和验证帧名单 |
| loss.csv、loss.png | 每轮平均重建误差 MSE 及曲线 |
| autoencoder_last.pt | 最近一个完整训练轮次的模型权重 |
| autoencoder_best.pt | 验证 MSE 最低轮次的模型权重，无验证集时不生成 |
| reconstruction.png | 使用最佳验证模型，每行左侧为输入裁剪帧，右侧为重建帧 |
| summary.json | 样本数、误差、耗时、PyTorch 峰值张量显存等 |

MSE 越低表示本实验的像素重建误差越小。它不能代替语义分割准确率、目标识别效果或压缩码率。

权重用于推理，当前没有实现断点续训；再次运行会新建实验并从头训练。

## SUIM 测试与重建图片

训练完成后，指定该次实验的最佳模型。以下路径对应本次已经完成的训练：

~~~bat
python scripts\evaluate_autoencoder.py --manifest data\SUIM\manifest.json --checkpoint runs\suim_autoencoder_20260911\autoencoder_best.pt --device cuda
~~~

默认评估全部 110 张测试图片，保存到模型目录的 evaluation_test：

- reconstructed/：全部重建 PNG。
- reference_crops/：对应的输入裁剪图。
- comparison.png：按名单等间隔选取 6 张，每行左原图、右重建。
- per_image.csv：逐图 MSE、PSNR、SSIM 及文件路径。
- summary.json：平均指标、模型轮次、预处理和指标定义。

本次测试已经完成。输出目录非空时脚本会停止，再次评估可用 --output-dir 指定新目录。模型和数据清单指纹不同也会停止，防止混淆实验。

指标按实际保存的 8-bit RGB PNG 计算，先逐图计算再等权平均。SSIM 使用 scikit-image 的默认 7×7 窗。这与旧 H.264 项目中 FFmpeg 的视频/YUV 指标、论文中的 MS-SSIM 口径不同，不能直接横向比较数值。

本次环境配置附带了一次 8 帧、3 轮的短训练检查，位于 **runs/setup_smoke_20260910132034238/training**。样本取自导师提供的重建视频，用于检查抽帧、训练、绘图和模型保存，没有独立验证集。重建图仍然很粗糙，不是正式训练成果。

## 目录

~~~text
data/raw/                    待准备的原始视频
data/frames/                 按场景分组的帧
data/masks/                  预留语义分割标签
runs/                       每次实验的权重、日志和图片
checkpoints/                预留用于整理选定的模型
scripts/extract_frames.py   抽帧
scripts/train_autoencoder.py  帧重建训练
scripts/prepare_suim.py       SUIM 配对、清洗、标签转换和固定划分
scripts/evaluate_autoencoder.py  独立评估及重建图保存
scripts/segmentation.py       分割数据读取、轻量 U-Net、损失与指标
scripts/train_segmentation.py 分割训练
scripts/evaluate_segmentation.py  分割评估和可视化
scripts/predict_segmentation.py   单张新图片预测
scripts/segment_video.py          视频逐帧分割、速度和抖动统计
scripts/semantic_compress_video.py 重要区域保护的压缩原型
scripts/prepare_pretrained.py 官方预训练权重下载与来源记录
scripts/compare_segmentation_runs.py  同协议结果对照
scripts/benchmark_segmentation.py   同条件 GPU 网络前向耗时比较
environment_check.py        环境和小批次训练检查
requirements.txt            公共依赖的固定版本
requirements-torch.txt      官方 CUDA 包的固定版本和来源
~~~

原始数据、训练产物和虚拟环境已列入 .gitignore。

## GLC-Video 视频文件实验

使用官方 GLC-Image / GLC-Video 预训练网络，补充本地熵编码和 .glcv 容器，实现视频编码、限速 TCP 文件传输与独立进程解码。

本机环境配置后，双击 **run_glc_demo.cmd**，打开 http://127.0.0.1:8770/。在网页拖入 AVI 或 MP4，可以调整质量、分辨率、帧数和传输限速，查看处理进度、同步播放输入与还原片段，并下载码流、还原视频和逐帧数据。支持停止任务、保留原视频重跑与历史记录；结果保存在 runs/glc_console/。操作步骤见 [GLC 网页界面](docs/GLC网页界面.md)。

默认只处理最多 50 帧，宽度不超过 1280、q=63，以 20 kbps 在本机传完后还原。20 kbps 是文件发送速度，与压缩后视频码率分别统计。需要命令行操作时，仍可把视频拖到 run_glc_video.cmd。

官方公开代码只返回估计码量；这里的真实文件接口是本项目补充实现，不是官方码流格式。接收端共享权重，只读取 .glcv，不读取原视频。输出 MP4 是观看副本，不计作传输数据。当前仅验证相同 PyTorch / GPU 环境，属于生成式神经视频源编码，尚未加入本项目的水下分割条件或物理信道。

独立环境、复现命令、实际文件大小和测试限制见 [GLC-Video 说明](docs/GLC-Video.md)。原来的官方前向试跑入口 scripts/glc_video_probe.py 仍保留，用来区分估计码量与真实文件测量。
