# 文件任务与 Orin 优化说明

目标是缩短文件编码、传输和解码等待时间；不要求实时播放帧率。保持相同输入采样、画质要求和压缩设置，分别记录编码完成、实际码流字节数、传输和解码完成时间。无流水线时总等待约为编码时间 + 码流位数/有效链路速率 + 解码时间；分块重叠后直接测完整文件完成时间。

## 上次精度测试的更正与补测

旧的 01—04 轮精度脚本遗漏了 NativeGaussianRans，使用了 Python 熵编码器；不能拿旧结果当作网页默认原生后端的基准。本次已增加明确的 --backend 开关（默认 native），在每个编解码进程保存实际后端名称与运算设置，并保存脚本快照。生产编解码器没有改动。

640×360、12.5 fps、66 帧、q=0、GOP64、参考重置32、8 帧暖机，RTX 4060 Laptop，同模式重复两轮。以下整任务时间包含原有预览和校验，不含首次模型加载和网络传输。

| 模式 | 编码整任务 | 解码整任务 | 编码核心 fps | 解码核心 fps | 平均 PSNR |
|---|---:|---:|---:|---:|---:|
| FP32 + 原生熵编码 | 18.382 s | 11.643 s | 3.784 | 5.822 | 32.28749 dB |
| RGB 前缀 FP16、后缀 FP32 + 原生熵编码 | 18.060 s | 11.037 s | 3.854 | 6.147 | 32.28229 dB |

两个模式的真实压缩载荷均为 13,717 B，SHA-256 完全相同；独立解码的 66 帧 RGB 与参考状态均与各自编码端匹配，解码端不读取源视频。混合模式的核心吞吐增幅约编码 1.8%、解码 5.6%，只做了两轮，桌面应用仍运行且有耗时波动，不能把小百分比当作稳健上限。

这次没有重测 TF32，旧 TF32 结果仍只适用于 Python 熵编码配置。第一次观察到非有限输出的位置为生成器 blocks[22]（第 23 个模块），尚未定位该模块内部的具体非有限值成因。当前混合方案为 blocks[0:22] 使用 FP16 autocast，后续 FP32。

## 如何利用 Orin 硬件

CUDA 核心与 Tensor Core 都由内核实现和 GPU 调度器使用，不是手动设置线程数即可让所有运算单元满负荷。torch.set_num_threads(1) 限制的是 CPU 的 PyTorch 运算线程，不是 GPU 只用了一个核心。当前还没有 SM、Tensor 管线和 DRAM 吞吐硬件计数，不能断言 GPU 算术已满。

1. 保留原生熵编码；测整个文件完成时间，同时采集 GPU 空闲间隙、SM 计算吞吐、Tensor 管线、内存带宽及 CPU 等待。
2. 针对最大热点 VQGAN 输入编码器与 RGB 生成器，测试固定形状的部署引擎、内存布局与算子融合。FP16 只用于通过数值与画质验证的层；概率先验及参考链需要两端一致的运算配置。
3. 先测网络子模块 batch=1/2/4 的吞吐和内存，再测独立 GOP 的 1/2/4 路并发。保持原 GOP 边界，不为增加并行度盲目添加 I 帧。同一 GOP 的参考依赖仍顺序执行，结果按原帧序写出。批处理可能改变浮点运算结果，须验证独立解码；增加并发不保证一定更快。
4. 重叠源文件读取、图像变换、GPU 网络、CPU 原生熵编码、写入与传输。只对无 CPU 中途交互的网络子图考虑 CUDA Graph；不能直接捕获当前 CPU rANS 参与的整个帧函数。
5. 为文件发送模式研究移出编码关键路径的 RGB 预览重建。当前 RGB 不参与下一帧参考状态，但容器校验依赖它，因此需要明确的模式/协议调整和独立解码验证，不能直接删调用。
6. 在 Orin 上确认实际 JetPack、功耗模式、频率、供电和散热。先测 GPU/Tensor Core 路线，再评估可完整部署到 DLA 的子网络，避免频繁 GPU/DLA 切换。DLA 和视频硬件编码器不能无条件执行整套 GLC。

这些部署优化尚未执行，本轮只复核代码、修正测试后端并补测。PC 可筛选算法与实现；Orin 的最佳批量、并发数、频率和真正吞吐必须在盒子上测。

## 依据

- [NVIDIA TensorRT 性能优化](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/performance/optimization.html)：批处理、并发、CUDA Graph 等一般方法；实际部署版本须匹配盒子的 JetPack。
- [NVIDIA TensorRT 8.6.1 开发指南](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/developer-guide/index.html)：DLA 支持限制与 GPU 回退，不能把硬件各单元峰值直接相加为本模型速度。
- [Jetson Orin 功耗与性能管理](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)：nvpmodel、MAXN_SUPER、jetson_clocks 与热约束。
- [Nsight Compute 分析指标](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：计算、Tensor 管线和内存吞吐的分别测量。

原始数据见同目录 summary.json、各模式 encode/run_*、decode/run_* 及 *_runtime.json。脚本快照为 bench_glc_precision.snapshot.py。
