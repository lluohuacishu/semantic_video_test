# GLC 水下发送端免 RGB 重建

2026-10-29。当前分支 `codex/glc-sender-no-rgb`，基于已完成两轮蒸馏的 `073c45a`。水下 AI 盒子负责视频编码，船上/岸上接收端有较强 GPU，允许先编码再传输，接收端仍须恢复可观看视频。因此先消除发送端不参与码流或参考链的 RGB 生成，暂停第三轮输出生成器蒸馏。

**[首轮实测已完成](发送端免RGB实测_20261029.md)**：三个设置、18次编码及3次独立解码；228帧保持原模型RGB精确一致，编码核心加速约2.21–2.23倍。40项CPU检查通过。当前仍加载完整权重，显存峰值未下降；Orin及功耗尚未测量。[小型数值证据](evidence_20261029/summary.json)。

## 方法与边界

`NeuralCodec.frame(..., render_rgb=False)` 仅允许编码调用。它保留 VQGAN 输入图像编码器、量化、熵编码、潜变量重建及 `ref_latent/ref_y` 更新；I/P 帧的最终 RGB 生成器均跳过。默认 `render_rgb=True` 的旧路径和严格 RGB 校验继续工作。独立接收端始终运行原始完整 RGB 生成器，不使用学生权重。

新实验容器 `GLCSND01` 使用 `.glcsnd` 扩展名，显式声明 `container_version=3`、`integrity_profile=reference-only-no-sender-rgb-v1`、`sender_rgb_reconstructed=false`。紧凑 z/y 记录与旧 v2 相同；尾部保存有序参考状态 SHA-256 和全文件 SHA-256，不伪造发送端 RGB 摘要。接收端完成参考状态核验后才写成功报告。它计算自己的 RGB 摘要；“发送端 RGB 校验”标记为不可用，不能称为通过。

原生产读取器拒绝新 magic，旧 v1/v2 写入器仍要求 RGB 摘要。本阶段是独立文件实验，还没有接入网页或紧凑 TCP 逐帧协议。现有传输 CRC 为检错，FEC 尚未实现；本格式中的 SHA-256 也不提供纠错。

## 实验入口

- [容器与故障测试](test_sender_container.py)：损坏、截断、结构错误、策略不明、参考状态不同、旧入口兼容保护。
- [试验控制器](run_trial.py)：冻结三个已有输入，三组交替顺序配对，独立接收进程复原。
- [单进程执行器](trial_worker.py)：FP32/native rANS、8 帧暖机、逐帧记录载荷/符号/CDF/参考状态/RGB 摘要及耗时。

运行环境沿用 `08_新阶段C_编解码加速与网页展示/c4_config.py`，不安装软件、不修改官方权重。以仓库为工作目录，PowerShell 执行：

```powershell
$glcPython = 'C:\Users\lenovo\Desktop\works\work\semantic-video-training\.venv-glc\Scripts\python.exe'
& $glcPython -X utf8 -m unittest discover -s '13_优化阶段_发送端免RGB重建' -p 'test_*.py' -v
& $glcPython '13_优化阶段_发送端免RGB重建/run_trial.py' --output 'C:\Users\lenovo\Documents\ChatGPT\语义通信\02_实验数据\优化_发送端免RGB重建\新的试验目录'
```

本机素材来自已有 long108 和 SFish/UV 缓存。各片均越过第65帧 I 帧切换和第34帧参考重置，三个 q 值为0/12/17。后两片的名义 FPS 仅用于帧序列容器，不主张原始拍摄时基。不能由本机结果推导 Orin 时间、功耗或真实水声吞吐。

`publish_evidence.py --output <完整试验目录> --destination <新的公共证据目录>`在CPU上复核保存的载荷字节、文件摘要及运行记录，生成精简数值和全部编码帧计时。需先按`evidence_20261029/verification.json`列出的三条测试命令保存本地测试日志及`verification.json`；执行后的原始结果/数组/码流继续留在工作区，不上传大文件。

## 后续顺序

先根据免 RGB 试验的码流一致性、画质与实际编码耗时判断是否接入发送端，再剥离发送端无用模块和权重。仍慢时才压缩真正的输入图像编码网络；这会影响量化、码率与参考链，需要独立设计和率失真验证，现有 RGB 学生不能复用为编码器。

Orin 上应记录完整文件编码时间、峰值内存、平均/峰值板卡功率、积分能耗、温度、连续运行成功率；同时检验 Orin 编码到岸上 GPU 的参考状态与视频质量。离线处理允许较慢，不能忽略持续功率和散热限制。当前没有连接或测量真实 AI 盒子。
