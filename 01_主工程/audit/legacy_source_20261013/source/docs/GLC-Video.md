# GLC-Video 本机文件编解码实验

本实验使用作者发布的 GLC-Image / GLC-Video 权重，不进行训练。
GLC-Image 负责 I 帧；GLC-Video 利用之前重建的潜特征处理 P 帧。

## 当前实现与用法

官方网络已经能在本机运行。在此基础上，本项目补充了真实文件接口：

原视频 → GLC 编码网络 → 量化潜变量和码本索引 → .glcv → 限速 TCP 传文件 → 独立 GLC 解码进程 → 重建帧。

**GLC 网络和预训练权重来自官方；熵编码器、.glcv 格式、发送与接收入口是本地补充实现。**
官方源码保持原样。这里没有训练新模型，也没有把 H.264 视频或原始 RGB 图藏入 .glcv。
两端需要事先保存同一份模型权重，约 1.17 GB；它们属于共享模型，不计入每段视频的文件大小。
这是生成式神经视频源编码实验，没有接入现有 SUIM 分割模型，不应把它称为已经完成的水下语义通信系统。

把 AVI 或 MP4 拖到 run_glc_video.cmd 上，默认处理前 50 帧、宽度不超过 1280、q=63，以 20 kbps 做本机 TCP 实传，再启动独立接收进程。
结束后打开本地报告。传输速度与视频码率分别记录：例如一个 10 秒的视频文件可以花 30 多秒传完，符合当前的文件传输要求。
源视频保留，不处理音频；输入是名义固定帧率，不保留可变帧率逐帧时间戳。

完整实验命令，在项目根目录运行：

~~~powershell
.\.venv-glc\Scripts\python.exe scripts/glc_video_roundtrip.py --input "C:\path\to\video.mp4" --output runs/glc_my_stream --frames 50 --width 1280 --q 0 63 --kbps 20
~~~

每个质量档位分为 sender/ 和 receiver/。sender.glcv 是编码文件，received.glcv 是 TCP 接收副本；传输后验证两者完全一致。
独立解码器在不含源帧的工作目录中启动，源图编码模块被禁用，记录有没有读取输入图片或视频。
每帧验证参考潜特征校验值和 RGB 像素校验值，外部脚本再逐像素比较发送端重建与接收结果。
这里的“一致”指两端重建一致，不代表与原视频无损一致。

也可以分开执行，解码命令不需要原视频路径：

~~~powershell
.\.venv-glc\Scripts\python.exe scripts/glc_video_codec.py encode --input "C:\path\to\video.mp4" --output runs/glc_manual/clip.glcv --workdir runs/glc_manual/sender --frames 50 --width 1280 --q 63
.\.venv-glc\Scripts\python.exe scripts/glc_video_codec.py decode --input runs/glc_manual/clip.glcv --output runs/glc_manual/receiver
~~~

不覆盖已有实验。若中途失败，未完成的压缩数据保留为 .part，不会作为完整文件交给接收端。
GPU 浮点计算会影响熵模型，因此当前只验证同机、相同 PyTorch / CUDA / GPU 环境，关闭 TF32 并启用确定性运算。
接收端检查环境标识，并逐帧检测参考漂移；还没有实现可跨不同硬件稳定解码的标准格式。

演示时建议先完成编码，现场播放报告里的同步对比视频，并展示 .glcv 文件、传输日志和接收结果。点击视频可以直接播放，重新运行模型则需要等待几分钟。
不要把“观看视频可以正常播放”和“模型能实时编码”混为一谈；当前任务允许离线传完。

## 2026-10-02 实际文件传输结果

最终记录位于 runs/glc_stream_720p_final_20261002/。输入是导师提供的 test5_5fps_reconstructed.avi，1280×720、5 FPS、50 帧，共 10 秒；它本身已经是重建视频，本次属于再次编码。
使用官方预训练权重、float32、复制边缘补边；每档 2 个 I 帧和 48 个 P 帧。未训练或微调。

| 参数 | 完整文件 B | 实际 bpp | 视频码率 kbps | 平均 PSNR dB | 20 kbps 实传 s | 编码处理 s | 解码处理 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q=0 | 40,545 | 0.007039 | 32.44 | 32.91 | 16.27 | 101.66 | 70.07 |
| q=63 | 87,493 | 0.015190 | 69.99 | 34.43 | 35.05 | 101.60 | 70.06 |

这里的文件大小直接读取 .glcv，包含码流、文件头和校验，不是概率估计。容器及校验占约 5 KB；应用层传输头另占 95 B，已计入实传限速。
实测平均发送速率分别为 19.988、19.993 kbps，口径不含 TCP/IP、链路层和反向确认。
PSNR 根据接收端 PNG 与输入 PNG 逐帧计算后取均值，没有经过观看用 MP4 的二次压缩，也不是语义准确率。

编码 / 解码处理时间包含神经网络、熵编码及逐帧校验。加入模型加载、抽帧、PNG 读写等后，编码约 143 秒、解码约 92 秒，仍不含观看视频制作和进程启动。
这台 RTX 4060 Laptop 8 GB 可以完成推理；PyTorch 峰值分配显存约 3.05 GiB（编码）、2.58 GiB（解码），最大预留约 3.69 GiB，不包括其他程序和全部驱动开销。
它能离线传文件，目前不能实时处理 5 FPS 的这段视频。

两档共 100 帧都通过了参考潜特征与 RGB 校验，接收端没有读取源图片或视频，单独进程中的输入编码模块被禁用。
另行逐像素检查全部输入 PNG 与 AVI 解码帧相同，独立复算全部 PSNR；8 个观看 MP4 都已完整读取，均为 50 帧、5 FPS。
11 项熵编码 / 容器测试和 4 项文件传输测试通过；另用 318×179、12 帧的短片验证了多组 I/P 帧、非整齐尺寸和参考重置。

查看结果：

- index.html：两档同步对比和数据表。
- q63/comparison_preview.mp4：左侧输入、右侧独立解码的 q=63 观看副本；q00/ 同理。
- q63/sender.glcv、received.glcv：实际发送与接收文件，两者相同。
- q63/sender/encode.json、receiver/decode.json：逐帧字节数、质量、耗时和环境。
- q63/transfer.json、transfer.csv：TCP 实传结果及发送进度。
- q63/verification.json：独立解码一致性检查。
- output-verification.json：源帧、指标和观看视频的额外核验；verify_outputs.py 可复查本次实验。
- environment-freeze.txt：本次环境版本。

第一次文件实验位于 runs/glc_stream_720p_20261002/，当时入口采用零值补边，虽然往返校验通过，但与官方预处理不一致。已修正为边缘复制并完整重跑；上表只采用 final 目录的数据。
在这段素材上 q=63 更适合展示。q=0 在第 35 帧的人物姿态已有明显偏差，该帧 PSNR 为 28.32 dB，q=63 为 32.22 dB；不能只看平均值就认定运动细节可靠。
两档都存在纹理变化，小字也不保证正确；没有标注真值评测或跨场景结论。

## 本地码流如何构成

- I 帧单独编码；P 帧使用接收端也能恢复的前一帧潜特征。沿用每 32 帧 I 帧、参考重置间隔 32 和长度为 8 的特征适配调度。
- 超先验 z 的码本共有 16384 项，每个索引用 14 bit 保存。I 帧索引数由补边尺寸决定；P 帧保存 16 个索引。
- y 的量化残差按 I 帧四组、P 帧两组依次编码。接收端用相同超先验、历史状态和已解码组预测下一组的均值与尺度。
- 采用 byte-rANS，16 bit CDF，128 档高斯尺度；范围外的整数以 escape 和变长整数保存。概率离散化、帧头及校验会让实际码量高于论文式估计。
- 文件带有尺寸、帧率、质量、模型版本、权重指纹、熵表指纹、逐帧校验和整体校验。实际文件大小包含这些开销，不包含共享权重和 TCP/IP 包头。

P 帧的官方 test() 使用训练用的直通估计表达式取得 z_hat，浮点抵消可能带来微小误差。本地编解码两端统一从离散索引查码本，不依赖发送端的原始 z 浮点值。
这是将推理结果写入可解码文件所需的数值处理；不能保证它与官方前向结果逐像素相同。
接收端与发送端的本地重建则必须完全一致，否则立即报错。

对应文件：scripts/glc_entropy.py、scripts/glc_container.py、scripts/glc_video_codec.py、scripts/glc_video_roundtrip.py。
熵编码递推参考 [Fabian Giesen 的 public-domain byte-rANS](https://github.com/rygorous/ryg_rans/blob/master/rans_byte.h)，在 Python 中实现，未增加本机编译依赖。

基础检查：

~~~powershell
.\.venv-glc\Scripts\python.exe -m unittest discover -s tests -p test_glc_stream.py -v
.\.venv-glc\Scripts\python.exe -m unittest discover -s tests -p test_file_transfer.py -v
~~~

## 2026-10-02 官方前向试跑记录（估计码量）

RTX 4060 Laptop 8 GB 已完成 720p 试跑，采用 float32。
输入为导师提供的 test5_5fps_reconstructed.avi，1280×720、5 FPS、50 帧，共 10 秒。
每档包含 2 个 I 帧、48 个 P 帧；补边后网络处理尺寸为 1280×768，指标只在原画面区域计算。

| 参数 | 估计 bpp | 估计视频码率 kbps | 平均 PSNR dB | 模型前向 FPS | 前向时间 s | 含逐帧读写时间 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q=0 | 0.005812 | 26.78 | 32.89 | 0.597 | 83.80 | 103.32 |
| q=63 | 0.013974 | 64.39 | 34.45 | 0.602 | 83.06 | 102.55 |

两档 PyTorch 峰值分配显存约 3.53 GiB，预留显存约 3.95～3.97 GiB。
这不包含其他程序的显存占用。上述时间不含权重加载、抽取输入片段和 MP4 封装。
在这段素材上，q=63 的画面指标更好，估计码量也更大；不能据此推广到所有视频。

人物的大致轮廓保留下来，但细字、珊瑚纹理有变化，运动帧也会失真。
例如第 35 帧，q=0 的 PSNR 降至 27.95 dB，q=63 为 32.21 dB。
这是生成式重建的观察，不是目标识别准确率；本次没有语义真值评测，也没有定量时序一致性评测。

结果保存在 runs/glc_video_720p_20261002/：

- index.html：本地对比页，可直接用浏览器打开。
- comparison_preview.mp4：左侧输入，中间 q=0，右侧 q=63。
- comparison.png：第 35 帧的并排对比。
- q00/、q63/：各档重建帧、观看视频及估计码量记录。
- summary.json：权重来源、环境、配置、逐帧指标。
- verification.json：输入帧逐像素核对、输出 PNG 和 MP4 完整性检查。
- environment-freeze.txt：本次实际环境版本。

四个观看视频均已完整解码检查，都是 50 帧、5 FPS；两档共 100 张重建 PNG 均为 1280×720。
抽查第 1、33、35、50 帧，以保存后的 PNG 独立复算 PSNR，与浮点记录的差异均小于 0.1 dB。
未生成 GLC .bin 文件，未验证独立码流解码。上述估计码率不能当作实际文件传输能力。

## 源码检查发现的限制

检查的官方提交是 126db25f9c093508cd0c99cee32b53fd60074f9a。
这个版本可以运行网络并重建视频，但**没有实际压缩码流的写入和独立解码接口**。

- src/models/common_model.py 中，get_y_gaussian_bits() 用概率计算估计码量。
- GLC_Image.test() / GLC_Video.test() 返回重建帧和估计 bit 数。
- test_video.py 虽然构造 .bin 路径，但后续只是利用该路径命名重建目录和 JSON，没有向 .bin 写入数据。

因此，glc_video_probe.py 的 bpp / kbps 都明确标为**估计值**。MP4 文件是把重建 PNG 帧另存成 H.264 的观看副本，其文件大小不是 GLC 的码流大小。
该前向记录不能证明独立接收端解码或链路传输能力；真实文件实验使用上面新增的独立入口。
这项源码检查也纠正了此前仅依据脚本参数作出的判断。

## 来源与隔离环境

- [官方仓库](https://github.com/jzyustc/GLC)
- [GLC-Video 官方权重](https://github.com/jzyustc/GLC/releases/tag/GLC-Video)
- [GLC-Image 官方权重](https://github.com/jzyustc/GLC/releases/tag/GLC-Image)
- [论文](https://arxiv.org/abs/2505.16177)

源码位于 tools/GLC/，权重位于 checkpoints/glc/，独立环境位于 .venv-glc/。
这些目录不随 Git 上传。既有 .venv 和 .venv-segformer 不做改动。

依赖采用 Python 3.12、PyTorch 2.5.1 + CUDA 12.4、torchvision 0.20.1。
PyTorch 从 download.pytorch.org 下载，其余必要依赖从 pypi.org 下载。
权重来自上述作者仓库的 Releases；SHA-256 用于确认下载完整性，并不单独构成可信来源证明。
测试入口使用 weights_only=True 读取权重，要求所有推理参数完整匹配。
官方 P 帧权重多出 11 个当前网络不使用的 bit_estimator_z 参数；按官方 strict=False 的加载习惯忽略它们，并把键名记入 summary.json。其他多余参数或缺失参数都会报错。

搭建环境的基本命令（在项目根目录）：

~~~powershell
# 用已安装的 Python 3.12 创建隔离环境
py -3.12 -m venv .venv-glc
.\.venv-glc\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1
.\.venv-glc\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r requirements-glc.txt
git clone https://github.com/jzyustc/GLC.git tools/GLC
git -C tools/GLC checkout 126db25f9c093508cd0c99cee32b53fd60074f9a
~~~

本机复用了 Codex 自带的 Python 3.12.14 来创建虚拟环境，没有另外安装全局 Python。
重建环境时也可以使用自行安装的 Python 3.12。
从官方 Releases 下载 GLC_image.pth.tar 和 GLC_Video.pth.tar，放入 checkpoints/glc/。

## 复查官方前向结果

保留的 scripts/glc_video_probe.py 默认：

- 最多处理前 50 帧，保留源视频名义帧率，不处理音频。
- 宽度最多为 1280，等比例缩小，不把低清视频放大；网络内部补边至 64 的倍数。
- 测试 q=0、q=63；每 32 帧一个 I 帧。
- 输出输入片段、两档重建视频、并排对比视频、逐帧 PNG、JSON 测量记录和对比网页。

也可以指定实验：

~~~powershell
.\.venv-glc\Scripts\python.exe scripts/glc_video_probe.py --input "C:\path\to\video.mp4" --output runs/glc_my_video --frames 50 --width 1280 --q 0 63 --intra-period 32
~~~

首次验证可以用 --frames 2 --width 320 --q 0。
输出目录必须为空，以免覆盖已有实验。

本地入口只负责视频读写、测量和报告，直接调用未修改的官方 model.test()。
没有添加语义分割条件，没有改网络结构、量化或前后帧参考逻辑。
没有加载 LPIPS / DISTS 的辅助网络；这次测量不包含这两项感知指标。

## 如何理解官方前向记录

- **估计 bpp**：官方估计 bit 数总和 / 实际画面宽高 / 帧数，含网络补边带来的码量，不含容器、协议头。
- **估计视频 kbps**：估计 bit 总和 / 片段播放时长 / 1000。不是文件传输限速。
- **PSNR**：比较输入帧与裁去补边后的浮点重建帧，不经过观看用 MP4 的二次编码。
- **模型前向 FPS**：含编码网络、重建网络和首次执行开销，不含权重加载、逐帧读写、评价和 MP4 封装。
- **峰值显存**：PyTorch 分配峰值，另记预留显存；不是整台电脑的显存总占用。

这次使用导师提供的 test5_5fps_reconstructed.avi 时，应注明它本身已经是重建视频。
不能把对该素材的再次编码，等同于从原始高清拍摄文件开始的压缩评测。
单个片段也不能证明模型在水下场景的普遍表现。

## 后续衔接

本机现在也可以双击 `run_glc_demo.cmd`，在网页操作同一套真实文件流程。上传、参数、结果保存和指标说明见 [GLC 网页界面](GLC网页界面.md)。命令行入口保留。

真实文件接口用于验证独立编码、接收和还原。后续应先对不同水下片段比较真实文件大小、重建质量、目标语义保持情况和处理时间。
GLC 本身的生成式先验不等于已加入水下任务监督；语义评价只能先作为诊断。若要主张任务导向语义通信的贡献，还需要语义约束训练、语义条件编码或相应对照实验。
