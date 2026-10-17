# 本机语义通信原型

双击工程根目录的“启动语义通信系统.cmd”，打开 http://127.0.0.1:8776/ 。默认先传目标信息，可切换为视频与目标信息。填入自己的本机视频路径后点击“开始发送并接收”。当前按批次处理，默认前50帧，示例鱼类使用video131真实连续420–469帧。结果保存到runs/semantic_mvp_ui_日期_编号/receiver。

信息模式传类别、位置、置信度、像素面积，接收端展示目标分布；视频模式另传低清背景与高清目标区域，恢复视频。目标框是发送端预测，连接区域数量不等于真实物体数量。鱼类模型泛化仍有限。绿色框完整收到不代表识别正确。

本机环境已安装，不需要下载或训练：发送端复用工程.venv、原SUIM SegFormer以及阶段C固定鱼类适配权重。接收端只需Python、NumPy、Pillow、FFmpeg，不需要GPU或模型。FFmpeg位置可用FFMPEG_PATH环境变量指定。

```powershell
cd C:\Users\lenovo\Desktop\works\work\semantic-video-mlvc
# 默认鱼类轻量模式；运行完打开本地结果页
& .\.venv\Scripts\python.exe semantic_mvp/system.py demo --target fish --mode information --open
# 已有潜水员示例，传视频与目标
& .\.venv\Scripts\python.exe semantic_mvp/system.py demo --target diver --mode video --open
# 自己的视频；新输出目录必须不存在
& .\.venv\Scripts\python.exe semantic_mvp/system.py demo --input 'C:\视频\水下.mp4' --target fish --mode video --frames 50 --kbps 20 --open
# 只从收到的通信包独立解码，不需要原图或模型
& .\.venv\Scripts\python.exe semantic_mvp/system.py decode --input 'C:\接收\message.scv' --output 'C:\接收\恢复结果'
```

两端分开使用：接收端运行 `system.py listen --host 127.0.0.1 --port 8772 --output 新接收目录`；发送端先 `system.py encode --input 视频 --output 新发送目录 --target fish --mode video`，再 `system.py send --input 发送目录/message.scv --host 127.0.0.1 --port 8772 --kbps 20`。改为局域网时显式指定对应地址；当前只实测本机回环，不代表已完成跨机器或水声硬件测试。

通信包为ZIP_STORED格式.scv，包含manifest.json、gzip目标消息及可选H.265码流。全部封装、目标消息和高清区域都计算在完整文件大小中。TCP发送统计还包括发送方向的应用层传输头，不包括反向确认及TCP/IP链路开销。接收端只读取自己的目录，保存逐帧RGB哈希和读取审计；这是程序约束，不是OS沙箱。

每次输出：sender/message.scv、sender/encode.json、transfer.json、end_to_end.json，以及receiver/查看结果.html、视频预览、target_information.json、decode.json、receiver_read_audit.json。发送端私有源图和原始预测掩码不进入通信包。改ROI_CRf、背景比例或片段长度可在命令行操作，目前默认背景0.5/CRF42、ROI CRF32、每32帧一个矩形、外扩16像素。

已经实测：完整鱼类/潜水员视频各50帧，冷接收端独立恢复；鱼类消息2,011B在20kbps下约0.844秒传输。该时间不含识别模型耗时。允许B帧的近似同预算H.265对照与正负结果保留，详见Codex工作区outputs/semantic_communication_mvp。

## 阶段E：可选的自动双区域视频

2026-10-15增加`--roi-layout halves`。每个32帧窗口在画面左右分区内分别组织ROI，最多同时两块，避免相隔很远的框形成一张巨型矩形。模型、阈值和预测框不变；该选项不能补回漏检目标。默认`single`和已有网页启动方式保留。

双击工程根目录“阶段E_双区域视频示例.cmd”，可对既有鱼类连续示例运行新鲜识别、视频编码、20kbps发送、独立接收和结果展示。也可运行：

```powershell
& .\.venv\Scripts\python.exe semantic_mvp/system.py demo --target fish --mode video --frames 50 --kbps 20 --roi-layout halves --roi-crf 31 --open
```

新多区域包标记为`semantic-video-layered-v2`，须使用包含`roi_layout.py`的新接收程序；新接收器仍可读取v1旧包。传输协议和接收依赖仍为既有Python、NumPy、Pillow及FFmpeg，无需新模型或额外安装。接收端拒绝空间重叠、跨分区和超过两层的ROI。

该片段约100KB完整文件的实验结果有取舍：任务召回提高，但小目标仍漏检，整体和汇总鱼区PSNR有代价，因此不将新方式直接改为全局默认。CRF31仅对已测示例接近100KB，其他输入的完整大小不固定。完整报告位于`docs/phase_e_results_20261015.md`，原始证据在`runs/semantic_mvp_stage_e_20261015`。
