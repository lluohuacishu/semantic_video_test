# GLC 算子级诊断：FP32、GPU 计算与 CPU/GPU 交接

日期：2026-10-25。对象：当前本机纯 GLC，未启用语义改进。

这次已完成真实码流编码、独立进程解码及算子采样。结论是：上游 GLC 支持 FP16；当前本地真实码流实现选择 FP32，并关闭 TF32、启用确定性算法。当前 640×360 负载没有显存容量不足的迹象，最大的耗时集中在 VQGAN 图像网络，尤其是卷积、归一化和逐元素操作。CPU/GPU 交接存在可优化位置，但不能把等待 GPU 的时间全部算成数据传输成本。

## 采集完成情况与闪退

用户确认闪退的是 Codex 应用。首次任务在 17:15:45 左右停止，已保存 66 帧普通编码和首个 I 帧的算子 JSON；当时没有保存完整采样结果或开始解码。检查该时间段的 Windows 日志，没有发现对应的 CUDA OOM、显示驱动重置或 Python 崩溃记录。这只能说明未找到相关记录，不能确定应用闪退的根因，也不能确认它与分析器有无因果关系。

恢复任务复用原来的完整编码基线，重新暖机并补齐编码采样、普通解码和解码采样。恢复版逐个采样帧保存 CSV/JSON，省略额外的格式化表格和 Chrome trace 导出。原始任务文件完整保留在同级 `GLC_operator_profile_20261025_171041` 文件夹。

[validation.json](validation.json) 确认：

- 66 帧全部完成；编码和独立解码的 RGB、参考状态哈希逐帧一致。
- 分析器开启前后，编码码流逐字节相同；解码结果也一致。
- 独立解码器未读取原始视频。
- 码流 SHA-256：`6092d1b0b703d01e452e0c10d9c31429e43b48eaa41bcd2d9696656c39c78370`。
- 本次没有修改生产编解码逻辑、精度、权重或网页服务。

## 环境与口径

RTX 4060 Laptop GPU 8 GB；PyTorch 2.5.1+cu124、CUDA 12.4、cuDNN 9.1；Windows；本地原生 C rANS 后端。实际检查 I/P 两个网络的参数，均为 `torch.float32`。

输入来自 `long108/input_540.mkv`，使用现有 compact 预设：640×360、12.5 fps、66 帧（5.28 秒）、q=0、GOP=64、参考状态重置间隔32。模型常驻，每个进程先运行8帧暖机，包含现有预览与校验，不保存 PNG 序列。编码和解码顺序运行，不争抢同一 GPU。

以下正常速度来自**未开启分析器的基线**。编码复用闪退前的已完成基线，解码在恢复后运行；这是一次诊断，没有多次重复测量的置信区间。

| 指标 | 编码 | 独立解码 |
|---|---:|---:|
| 66帧完成，含预览收尾 | 16.746 s | 10.649 s |
| 上述整任务吞吐 | 3.94 fps | 6.20 fps |
| 编解码逐帧核心处理累计 | 15.838 s | 10.316 s |
| 核心处理平均每帧 | 239.97 ms | 156.30 ms |
| 模型峰值 allocated 显存 | 1638.87 MiB（约1.60 GiB） | 1519.82 MiB（约1.48 GiB） |
| PyTorch 峰值 reserved 显存 | 1774 MiB | 1804 MiB |

编码的输入读取/转换/哈希累计0.171秒，约占含预览总耗时1%；核心处理约占95%。这次的主要问题不在视频读盘。GPU全设备采样包括桌面应用，恢复任务最高6032 MiB；这不是模型单独占用。没有显存溢出，不能据此推广到更高分辨率。

## FP32 是不是强制的

上游 `test_video.py` 第40、67、253行附近提供 `--float16`，会把输入和网络转换为半精度，所以模型本身不是只能用 FP32。

本地 `glc_video_codec.py` 的 `configure()` 使用 `glcv-lab-f32-v1` 运算配置，关闭卷积和矩阵乘的 TF32，启用确定性算法；输入构造也明确调用 `.float()`。这些是当前真实码流一致性方案的选择。浮点网络预测的概率表、量化符号和下一帧参考状态必须在发送端与接收端相容，不能只给一端加 `.half()`。

上游 FP16 推理开关，不等于本地 rANS 实际码流已经完成 FP16 独立解码验证。当前没有测 FP16 的速度、质量或兼容性，不能承诺半精度一定提速多少。FP32也不意味着所有操作都是浮点运算：索引、整数熵编码和最终 RGB uint8 仍采用各自的数据类型。

## 模块级定位

采样帧为零基索引0、1、2、33、64：覆盖I帧、普通P帧及参考状态重置位置。下面给出**第3帧，即索引2的普通P帧**的 CUDA Event 区间。所有209条模块记录见 [模块耗时.csv](模块耗时.csv)。

这些是带分析器时的同一CUDA流上起止区间，可能包含CPU提交空隙，不能当作纯GPU内核忙碌时间。父模块包含子模块，不能重复相加；CPU墙钟与GPU时间也会重叠。

| 顺序与步骤 | 编码 CUDA Event 区间 | 解码 CUDA Event 区间 |
|---|---:|---:|
| 参考上下文 context_generation | 3.78 ms | 6.48 ms |
| 时间先验 temporal_prior_encoder | 0.22 ms | 0.24 ms |
| 权重图 weight_generator | 2.59 ms | 4.28 ms |
| VQGAN 输入图像编码器 | **95.53 ms** | 无此步骤 |
| P帧特征 encoder | 3.11 ms | 无此步骤 |
| temporal_cond + y_temporal_combine | 2.12 ms | 无此步骤 |
| hyper_tokenizer | 4.04 ms | 无此步骤 |
| 码本索引/还原 codebook | 1.57 ms | 1.19 ms |
| hyper_inv_tokenizer | 1.22 ms | 1.96 ms |
| 概率参数 res_prior_param_decoder | 4.55 ms | 6.29 ms |
| 分阶段先验与熵编解码 prior，含子步骤 | 17.30 ms | 20.83 ms |
| P帧特征 decoder | 3.57 ms | 6.73 ms |
| **VQGAN RGB 图像生成器** | **149.22 ms** | **150.92 ms** |
| RGB检查/转换 rgb_array | 1.55 ms | 2.02 ms |
| 参考状态校验 reference_hash | 2.19 ms | 2.09 ms |

该帧整个函数的带分析器耗时为编码368.04 ms、解码249.58 ms，明显慢于正常运行，因此不能拿这组区间直接预测上线速度。VQGAN图像网络在两个方向都占据主要GPU区间，这一定位在其他采样帧也出现。编码端为了本地重建和校验，也执行了完整 RGB 图像生成器。

## 算子级定位

当前 Windows PyTorch 未提供可用的 CUPTI 内核跟踪；`supported_activities()`仅返回CPU。不过显式请求CUDA时，PyTorch 2.5.1会使用旧式CUDA Event回退，仍能得到算子CUDA区间。该行为可在安装环境的 `torch/profiler/profiler.py` 第39—49行核对。

因此本次达到了 **PyTorch/ATen算子级**，没有得到底层cuDNN/CUDA kernel时间线、SM占用率、真实DRAM带宽或Tensor Core使用率。不能由本次数据进一步断言每个卷积到底受算术吞吐还是显存带宽限制。

下表仍以第3帧为例，按算子名称合并不同输入形状，采用分析器的 self CUDA Event 时间：

| 算子 | 编码调用次数 | 编码时间 | 解码调用次数 | 解码时间 |
|---|---:|---:|---:|---:|
| aten::cudnn_convolution | 232 | **123.59 ms** | 141 | **82.28 ms** |
| aten::native_group_norm | 67 | **31.77 ms** | 39 | **19.90 ms** |
| aten::add_ | 238 | 18.09 ms | 147 | 11.79 ms |
| fused_sigmoid_mul | 58 | 16.58 ms | 34 | 9.76 ms |
| aten::add | 125 | 13.73 ms | 79 | 8.52 ms |
| aten::copy_，包含设备内拷贝等 | 75 | 4.10 ms | 49 | 2.55 ms |

回退分析器还给 `fill_`、`empty`、视图操作等分配了Event时间，其中可能包含CPU提交间隙及分析器影响，不能把这些数值都解释为GPU内存操作的实际执行时间。本表用于定位，不作为硬件利用率百分比。

[全部算子.csv](全部算子.csv) 包含10个采样帧的5287条按名称和输入形状分组的记录，包含次数、CPU总时间/self时间、CUDA Event总时间/self时间和输入形状。完整调用位置与模块归属保存在各帧 `profile.json` 的 `operators_by_scope`。

## CPU 与 GPU 到底交接了什么

普通P帧第3帧，在 `codec.frame()` 内捕获到的显式跨设备Tensor调用为：

| 指标 | 编码 | 解码 |
|---|---:|---:|
| 显式CPU↔CUDA复制调用 | 11次 | 9次 |
| 所涉及张量字节之和 | 6,655,104 B | 5,672,064 B |
| 上述Python调用墙钟之和 | 3.17 ms | 3.49 ms |
| GPU布尔检查的CPU阻塞 | **112.52 ms** | **102.40 ms** |
| CPU概率表索引+rANS函数墙钟之和 | 4.21 ms | 3.35 ms |

这里只统计被Python Tensor观察器捕获的调用，不包含 `codec.frame()` 外部的源图上传，也不等于硬件总线计数器。`.numpy()` 对CPU Tensor通常只是视图，没有计作额外跨设备拷贝。调用墙钟也可能包含先前GPU工作的等待。

最容易误读的是 `if not torch.isfinite(tensor).all()`。CPU必须等GPU完成图像生成，才能读取这一个布尔值；对应 `aten::_local_scalar_dense`，编码CPU self时间112.34 ms，而其CUDA Event仅0.11 ms。**这不是传输一个布尔值需要112毫秒，而是前面的GPU任务尚未完成。** 删除这一检查，后面的RGB回传仍会等待，不能因此承诺节省112毫秒。

当前串行路径是：GPU算概率/符号 → CPU rANS → 符号回GPU → 下一阶段先验。P帧两阶段、I帧四阶段，阶段间有真实数据依赖。现有原生rANS已把CPU熵编码本身压到数毫秒量级，它不是本次的主要耗时。

[CPU_GPU交接.csv](CPU_GPU交接.csv) 保存199条观察记录，包括源/目标设备、dtype、shape、字节数、CPU墙钟、源码位置和是否属于显式跨设备复制。

## 可优化位置与优先级

1. **先优化 VQGAN 图像网络。** 卷积、GroupNorm、激活与残差加法是主要热点。可先单独验证最终RGB生成器的FP16或部署引擎方案；它的RGB输出未用作下一帧DPB，但会改变RGB校验和画质，所以必须使用明确的新运算配置并检验质量。涉及输入编码器、概率先验和参考状态的精度变化，需要更完整的真实码流验证。
2. **去掉码本还原里的无用CPU分配/上传。** 上游 `vqgan_arch.py:110` 先在CPU创建one-hot全零矩阵，再复制到GPU。普通P帧约1 MiB，I帧约3.75 MiB。可先尝试直接在GPU分配，进一步用embedding查表替代one-hot矩阵乘；两种方案都要逐帧检查码流/参考状态，不能只凭数学等价认定按位一致。
3. **减少先验阶段的交接与动态索引。** 固定掩码可预计算整数位置，减少 `nonzero`、布尔索引带来的同步。编码端已有的整数符号可尝试保留在GPU供重建使用，只把CPU熵编码需要的副本回传；把同阶段回传合并、复用缓冲区也有价值。概率表索引若移到GPU，要检验临界值舍入是否改变表编号。
4. **再做异步传输与校验/预览流水线。** 可尝试预分配pinned缓冲、CUDA Event与双缓冲，使不依赖当前结果的工作重叠。CPU立即需要回传结果时，`non_blocking=True`不能消除数据依赖；GPU→CPU异步复制完成前读取也会产生错误。参见 [PyTorch官方异步传输说明](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)。
5. **区分部署输出需求。** 编码端目前每帧生成RGB做展示和校验。若实际发送端仅需码流，可研究独立的发送模式，把不参与DPB更新的RGB生成移出发送关键路径；现有容器与校验依赖RGB哈希，不能直接删除后称与当前实现完全相同。该方向可能比单纯减少几次拷贝更有价值。

本次只定位，没有实施这些优化，也没有测量它们的收益。现有证据支持“神经网络执行和串行同步是重点，显存容量不是当前瓶颈”；尚不能精确划分GPU算术与显存带宽瓶颈。Orin的统一内存也不会自动消除这些依赖，或让当前FP32模型自动用上标称INT8 TOPS。

## 文件与复现

- [analysis.json](analysis.json)：基线指标、各采样帧汇总。
- [全部算子.csv](全部算子.csv)、[模块耗时.csv](模块耗时.csv)、[CPU_GPU交接.csv](CPU_GPU交接.csv)、[采样开销.csv](采样开销.csv)。
- `encode/operators/frame_000` 等目录：每帧原始 `operators.csv` 和 `profile.json`。
- `encode/baseline/stream.glcv`：本次实际码流，15,049 B。
- `decode/baseline/receiver/reconstruction_preview.mp4`：独立解码预览。
- [manifest.json](manifest.json)：输入、设置、代码哈希和原始任务位置。
- [validation.json](validation.json)：一致性校验结果。

诊断脚本位于项目 `08_新阶段C_编解码加速与网页展示/profile_glc_operators.py`。在工作区根目录，用GLC环境执行；输出必须使用新文件夹：

```powershell
& '.\01_项目\运行环境\GLC_Python环境\Scripts\python.exe' -u '.\01_项目\代码\semantic_video_test\08_新阶段C_编解码加速与网页展示\profile_glc_operators.py' --output '.\02_实验数据\C_编解码加速与网页展示\新的诊断目录'
```

这会执行暖机、普通运行和五个位置的算子采样。`--resume-from`用于复用上次已完成的编码基线；本次使用了这个参数。`summarize.py`仅汇总现有JSON，不启动模型。
