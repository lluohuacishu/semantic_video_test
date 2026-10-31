# GLC输入图像编码器推理加速

2026-10-31已接入原网页：[本机实际运行入口](http://127.0.0.1:8773/inference/)，可选择原GLC、局部FP16、TensorRT FP32/FP16，上传或复用视频。默认处理完整视频，显示实际帧数、编码/解码耗时、真实码流大小、kbps、bpp和画质，保留四段同步回放。双击本目录`打开加速效果对比.cmd`启动，详见[网页使用与完整视频验收](网页使用与完整视频验收_20261031.md)。

追加“帧率模式：跟随原视频 / 帧率减半”，可与四种编码方式组合。按用户要求，新增模式只做参数与静态检查，留给用户自行运行；不得自动开始半帧率视频试验。此前82.57秒、2477帧的原帧率整段双模式验收已完成。

先读[实测报告](输入编码器推理加速实测_20261030.md)。已完成40次填充扩大对照、24个单模块任务和48次完整链运行。TensorRT FP16在四组素材上编码核心约2.02–2.09倍加速；FP32约1.25–1.29倍。三组FP32重建像素逐位一致，另一组P帧PSNR仅变化约−0.0009dB；FP16最大P帧下降约0.118dB。16份精简标识码流载荷/参考保持，四次原接收CLI视频输出通过，9项CPU检查通过。所有原始结果已留存，无需重跑已完成矩阵。

2026-10-30，沿用`codex/glc-sender-no-rgb`。对象是原始权重的I/P帧`vqgan.encoder`；保持参考重建、概率模型、熵编码及接收端生成器。无需训练，不修改原训练环境、模型资产或生产默认。

## 对照与验收

1. 扩大未初始化内存填充对照：long108/q0/640×360、SF007/q12/512×288、UV108/q17/512×288、SF025/q32/512×288、UV096/q63/512×288。总547帧；每端每组两次交替配对，独立进程、8帧暖机。原三组逐帧/整流还要匹配历史正确结果；新增两组独立接收器核对参考状态。255帧覆盖四个GOP，不能称为长期部署稳定性验证。
2. 单模块：真实帧0/2/33，I/P分别测试，三轮交替顺序，每帧每模式12次，保留全部观测。控制使用原输入/原权重布局；channels-last转换候选采用等值权重副本。NCHW轴顺序不代表存储连续，补充记录实际stride并单独比较强制NCHW连续输入。CUDA Event与同步墙钟分别记录。包含输入布局转换、图/引擎输出复制；模型/引擎建立另计。
3. 候选包括channels-last、TorchScript冻结图、CUDA Graph、局部TF32、局部FP16，以及TensorRT FP32/FP16。后补显式连续输入、FP16+channels-last和TensorRT独立CUDA流，共12种实现、I/P各一组。除单独填充对照之外都使用相同fill-off条件。算法确定性保持；局部精度开关离开输入编码器后恢复，参考/熵/解码仍为原FP32。
4. 只将有实际价值的候选接入完整编码。两次发送重复采用候选正序/逆序，再由独立进程解码；接收器禁止读取原视频、缓存源帧及GT。比较压缩符号、CDF索引和参考状态。跨框架即使FP32也不能先假定逐位相同。
5. CPU后处理评价整段、I/P分组的聚合RGB MSE/PSNR、码率和已有SFishTrack标注区域。不把无UV标注当作背景，不以单模块潜变量误差代替完整画质。

`candidate_worker.py`在实验流头写入输入编码器实现标识；原`arithmetic`标识对应未改的参考/熵/解码路径。因此完整文件SHA不同也可能仅是头部元数据变化，需另看压缩载荷、符号、参考状态和RGB是否一致。此处仍是GLCSND01实验协议，不是官方GLC标准码流。

`compact_transport.py`将详细溯源移至旁证，头部仅保留精简实现标识；它只重打包，不运行网络、不改变压缩载荷或参考摘要。详细实验头曾为短文件增加约2%–3%的固定开销，精简后TensorRT FP16相对原文件大小变化范围约−0.074%到+0.708%，应与神经网络压缩载荷变化分开说明。

## 环境与运行

本机RTX4060 Laptop 8GB，Windows、PyTorch2.5.1+cu124、原生C rANS。TensorRT10.5.0/cu12、ONNX1.17.0、protobuf5.28.3仅通过`pip --target`安装到工作区`02_实验数据/优化_输入编码器推理/engine_packages_1050`，未升级原环境的包、CUDA或驱动。官方Windows wheel来自PyPI/NVIDIA索引。引擎固定输入形状，TF32显式关闭；TensorRT FP16为允许混合精度，不能宣称每层都使用FP16。

```powershell
$glcPython = 'C:\Users\lenovo\Desktop\works\work\semantic-video-training\.venv-glc\Scripts\python.exe'
$trialDir = 'C:\新的实验目录'
& $glcPython -X utf8 '16_优化阶段_输入编码器推理加速/prepare_experiment.py' --output $trialDir
& $glcPython -X utf8 '16_优化阶段_输入编码器推理加速/run_experiment.py' --output $trialDir --phase fill
& $glcPython -X utf8 '16_优化阶段_输入编码器推理加速/run_candidates.py' --output $trialDir --phase encoder --variants eager channels_last trace graph tf32 amp_fp16 trt_fp32 trt_fp16 contiguous_nchw amp_fp16_cl trt_fp32_stream trt_fp16_stream
& $glcPython -X utf8 '16_优化阶段_输入编码器推理加速/run_candidates.py' --output $trialDir --phase full --variants amp_fp16 trt_fp32_stream trt_fp16_stream
& $glcPython -X utf8 '16_优化阶段_输入编码器推理加速/evaluate_experiment.py' --output $trialDir
```

原始执行目录为工作区`02_实验数据/优化_输入编码器推理/inference_20261030_01`。预检独立留存；正式计划冻结捕获代码和输入摘要，后加控制器与分析脚本另记录最终摘要。失败日志保留，不把失败的候选统计成成功运行。

模型常驻后的每帧收益与冷启动/引擎构建时间分开。PyTorch显存计数不包含TensorRT自行分配的内存，不能据此比较完整部署峰值。桌面其他程序未关闭；没有SM硬件计数器、能耗或Orin测量。Windows/RTX引擎不作为Orin部署文件，目标盒子仍需按其JetPack/TensorRT版本重新构建和验证。

参考：[PyTorch2.5.1 CUDA Graph](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/cuda.rst)、[数值准确性](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/numerical_accuracy.rst)、[TensorRT10.5支持矩阵](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1050/pdf/TensorRT-Support-Matrix-Guide.pdf)、[跨框架FP32差异](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/troubleshooting-reporting.html#reporting-an-accuracy-issue)。
