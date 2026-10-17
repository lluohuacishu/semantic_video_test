# 环境记录

配置日期：2026-09-10。项目独立使用 `.venv`，不继承系统 Python 的第三方库。

| 组件 | 本机版本 |
| --- | --- |
| Python | 3.13.7，已有的 C:\Python313 |
| PyTorch | 2.12.0+cu126 |
| torchvision | 0.27.0+cu126 |
| GPU | NVIDIA GeForce RTX 4060 Laptop，8 GiB |
| 驱动 | 560.94 |
| PyTorch 内置 CUDA 运行时 | 12.6 |

PyTorch 官方安装入口是 [Start Locally](https://pytorch.org/get-started/locally/)，安装源是 [CUDA 12.6 wheel 索引](https://download.pytorch.org/whl/cu126)。[torchvision 官方兼容表](https://github.com/pytorch/vision#installation)包含 torch 2.12 / torchvision 0.27 / Python 3.13 这一组合。

其他库使用 [PyPI](https://pypi.org/) 发布渠道，版本固定在 `requirements.txt`，包括图像读写、数据处理、绘图和基础指标库。GPU 包另列于 `requirements-torch.txt`，安装时分开指定来源。选择来源依据项目维护者的官方入口和发布渠道；哈希只能辅助校验文件完整性。

`setup_env.cmd` 可以在已安装 Python 3.13 的 Windows 电脑上重建本项目环境：先安装固定的公共依赖，再安装官方 GPU 包，最后执行依赖检查和 GPU 训练自检。脚本使用已安装的 NVIDIA 驱动，无需另外安装完整 CUDA Toolkit。新装 Python 时应从 [Python 官网](https://www.python.org/downloads/windows/)取得。

环境自检使用合成小批次，另用导师提供的视频验证本地读取和短训练。自检结果保存在 `runs/environment_check.json`；这些检查只验证环境和计算流程。

2026-09-11 已接入用户提供的 SUIM.zip，原包保存在 data，处理记录见 data/SUIM/preparation_report.json。后续的重建和分割实验使用固定的数据清单，结果与环境自检分别保存。

分割优化使用 torchvision 0.27 自带的 LR-ASPP + MobileNetV3 模型。官方预训练权重由 scripts/prepare_pretrained.py 从 [download.pytorch.org](https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth)获取，约 12.49MiB，缓存于 checkpoints/pretrained。权重枚举、预训练数据和归一化方式见 [torchvision 官方说明](https://docs.pytorch.org/vision/0.27/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html)。同目录 JSON 保存实际来源、大小和完整指纹。读取权重使用 weights_only=True，原分类层改为 SUIM 的八类输出。模型推理只需微调后的权重，不需要再次联网。

运行 `run_check.cmd` 会重新验证：依赖一致性、CUDA 前向传播和反向传播、优化器参数更新、Torchvision 原生算子、权重保存与加载、中文路径图像读写。可以添加 `--video "完整视频路径"` 验证本地视频首帧。自检失败返回非零退出码。
