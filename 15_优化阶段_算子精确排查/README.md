# GLC 编解码真实内核级排查

2026-10-30，沿用`codex/glc-sender-no-rgb`。对象为当前按角色精简版：水下发送端免RGB，岸上接收端保留原完整生成器，FP32、TF32关闭、确定性算法开启、原生C rANS。模型权重及生产默认未修改。

先读[排查报告](算子级精确排查_20261030.md)，数值见[evidence_20261030/analysis.json](evidence_20261030/analysis.json)。本轮直接调用现有CUDA12.4 CUPTI Activity API，取得真实GPU内核、DMA拷贝及CUDA运行时API记录；通过原生C++ NVTX标记关联模型层、ATen算子和输入尺寸。不是将旧CUDA Event区间当作内核忙碌时间。

## 采集与准确性

- 正常基线：每角色三次完整66帧处理，16帧暖机，未开启活动采集或算子观察器。
- 低干扰CUPTI-only：只采实际GPU活动和CUDA API，主机仅有少量阶段标记。
- 原生NVTX：每角色两次，捕获完整CPU算子范围与模型层，供归因使用。NVTX采集会显著扰动部分帧耗时，不用于正常速度结论。
- 采样零基帧0/1/2/3/9/33/64/65，覆盖I/P、三种参考适配索引、参考重置和GOP切换。CUPTI-only的每条GPU内核序列均与两次NVTX的名称、网格/线程块、寄存器和共享内存配置逐项匹配，层/算子标签也一致，才映射其时间。
- 原生NVTX通过`GPU correlation ID → CUDA启动API → 同CPU线程中最内层原生ATen/JIT范围`归因。低干扰记录上的标签是经过完整序列校验的映射，明确与直接NVTX归因区分。
- 补充未初始化内存填充开关的三组配对和两端活动采集；只移除了对应填充内核，剩余计算序列保持一致，码流/参考状态/RGB仍与旧正确版本一致。

## 文件与命令

`cupti_trace.py`只支持本机已检查的64位CUPTI API22/Kernel9 ABI，版本不符会拒绝解释。现有PyTorch只报告CPU profiler活动，但附带的`cupti64_2024.1.0.dll`可直接工作。原生NVTX的`NVTX_INJECTION64_PATH`只在诊断子进程设置，不修改系统环境或驱动；没有安装/升级软件。下载的NVIDIA wheel仅用于只读核对头文件，未安装。

```powershell
$glcPython = 'C:\Users\lenovo\Desktop\works\work\semantic-video-training\.venv-glc\Scripts\python.exe'
& $glcPython -X utf8 '15_优化阶段_算子精确排查/run_profile.py' --output 'C:\新的排查目录'
& $glcPython -X utf8 '15_优化阶段_算子精确排查/run_fill_ablation.py' --campaign 'C:\新的排查目录' --output 'C:\新的填充对照目录'
& $glcPython -X utf8 '15_优化阶段_算子精确排查/analyze_profile.py' --output 'C:\新的排查目录'
```

`profile_worker.py`支持`baseline/activity/nvtx/trace`，其中`trace`是Python dispatch预检/备选模式，正式归因采用原生`nvtx`。`pipeline_timing.py`及`pipeline_details.py`单独排查真实读视频、模型加载、容器、FFmpeg与输入张量准备，不把它们混入暖机后的神经核心基线。

本机原始结果：工作区`02_实验数据/优化_算子精确排查/operator_profile_20261030_01/`。每帧`profile.json`、原生`operators.csv`、Chrome Trace兼容JSON和完整计时保留本地；GitHub只提交小型摘要、低干扰GPU逐层/逐算子/逐输入尺寸CSV及图，不提交大trace、权重或视频。

## 解释规则

时间字段用ns保存不等于纳秒级物理准确度；报告以毫秒/秒展示。CPU墙钟、GPU内核累计、跨流活动并集、CUDA API等待分别统计，不能简单相加。活动间隙只表示本进程未记录到活动，不代表整张GPU空闲；其他桌面程序保留运行。Windows短调用的线程CPU记账较粗，仅作辅助；CUDA驱动忙等也不是有用CPU数学计算。

本次没有测SM占用率、硬件带宽计数器、周期级指令、板卡能耗、Orin或水声链路。填充关闭只是本片/q0的独立实验，默认仍开启；推广前还要验证更多素材/档位和目标盒子。旧10月25日CUDA Event报告保留原口径，不与本轮真实内核时间直接相减。
