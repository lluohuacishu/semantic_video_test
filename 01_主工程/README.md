# 水下视频语义通信：MLVC-S 独立验证工程

本工程从原项目隔离，固定 Microsoft MLVC 提交 `64b10d4c0108106e66b7c54d6e5e470200a57abf`。已完成真实编解码、发布者 GT 评价及首轮受控 GT-ROI 微调。阶段 C 三组各更新512步，24个质量点通过独立解码；尚未证明稳定的语义任务收益，并发现单P帧普通微调在完整GOP中明显退化。水声信道实验尚未开展。

## 当前可用入口：最小语义通信系统

双击 [启动语义通信系统.cmd](<启动语义通信系统.cmd>)，打开 http://127.0.0.1:8776/ 。支持鱼类与潜水员，默认只传类别、位置和置信度；可选视频与目标信息。填入自己的视频完整路径即可按批次收发，不需要重新训练。

完整视频与轻量消息均通过真实TCP限速传输、独立接收进程和文件哈希核验。鱼类50帧信息包2,011B，在20kbps下约0.844秒传输；识别耗时另计。视频层暂用稳定的H.265及发送时显式ROI区域。鱼类泛化与传输效率仍待改善，本机原型不代表真实水声链路验证。

- [操作说明](<semantic_mvp/README.md>)
- [最小系统实测说明](<../02_历史资料/C_D报告/semantic_communication_mvp/最小语义通信系统_实测说明_2026-10-14.md>)
- [阶段D正式报告](<docs/phase_d_results_20261014.md>)
- `semantic_mvp/`：发送、接收、限速链路、本机界面及独立质量评价；原MLVC研究适配器保留。

当前优先级是把原型用于实际短视频，改善识别和传输体验；研究训练继续作为升级，不作为系统可用的前提。

## 目录与证据

- `upstream/mlvc`：固定提交源码；仅有一处 Windows C++ const 类型修正，补丁在 `audit/windows_entropy_const.patch`。
- `.venv`：独立 Python 3.12.14、Torch 2.10.0+cu128，未改原项目环境。
- `checkpoints/mlvc-s-psnr-v1.ckpt`：微软官方权重，SHA256 与官方一致。
- `checkpoints/DMCI-6.0-mirror.pth.tar`：公开研究项目镜像；哈希与镜像 LFS 一致，尚未与微软 OneDrive 原件核对。详见 `audit/iframe_mirror_provenance.json`。
- `adapters/mlvc_file.py`：真实编码、独立解码、重建状态验证、统一 RGB 评测。
- `adapters/train_probe.py`：两帧、batch=1、256 裁剪、非零码率损失的最小微调测试。
- `audit/phase_a_commands`：批次启动的命令、退出码、日志。
- `audit/legacy_source_20261013`：原项目源文件及关键报告快照。
- `runs/baseline_720p_dmci_q*`：每个质量点完整文件、发送端记录、独立接收端、指标、预览。
- `runs/comparison_native`：重新评测的 GLC 与 H.265 对照；同源 RGB，编码器内部色彩域不同。
- `docs/plan_20261012.md`：上一轮详细方案，不等于当前完成状态。

## 复现单个完整视频

在本目录 PowerShell 中运行。每次更换新的输出目录，脚本会拒绝覆盖已有结果。

```powershell
& .\.venv\Scripts\python.exe adapters/mlvc_file.py encode `
  --source 'C:\Users\lenovo\Desktop\临时\语义\test5_5fps_reconstructed.avi' `
  --run runs/reproduce_q21 --q-index 21 --intra-period 32 `
  --i-weight checkpoints/DMCI-6.0-mirror.pth.tar

& .\.venv\Scripts\python.exe adapters/mlvc_file.py decode `
  --file runs/reproduce_q21/video.mlvcv --output runs/reproduce_q21/receiver `
  --i-weight checkpoints/DMCI-6.0-mirror.pth.tar --preview

& .\.venv\Scripts\python.exe adapters/mlvc_file.py verify `
  --run runs/reproduce_q21 --decoded runs/reproduce_q21/receiver
```

灰色初始化模式：编码与解码两边同时省略 `--i-weight`。首帧仍真实编码，灰色 YUV=0.5 是双方共享协议，不读取原视频首帧。

独立解码 CLI 没有原视频、原图、发送端缓存或参考状态输入。解码时的 Python 文件读取审计会阻止读取共享权重之外的媒体与状态文件，证据保存在 `receiver_read_audit.json`。目前只验证同一台 GPU、相同 FP32 配置的不同进程，不能声称跨设备解码一致性。

## 微调可行性复现

```powershell
& .\.venv\Scripts\python.exe adapters/train_probe.py `
  --frames runs/baseline_720p_dmci_q0/reference --output runs/reproduce_training `
  --i-weight checkpoints/DMCI-6.0-mirror.pth.tar --steps 20
```

两帧指一帧冻结 I 帧重建作为参考，以及一帧 P 帧参与反向传播。码率权重从第一个更新开始为 1，学习率 1e-5，固定 q=32，FP32；损失为 `bpp + lambda * (6*MSE_Y + MSE_U420 + MSE_V420)/8`，`lambda=768^(32/63)`。这是固定样本的工程测试，不能充当正式训练或泛化结论。

## Windows 熵编码重建

需要 CMake >=3.23、MinGW g++（本机 15.2.0），Python 3.12 头文件/导入库。依赖记录在 `audit/requirements-lock.txt`；Torch 从官方 cu128 wheel 源安装。保持原有 `-Wall -Werror`，将 `PyWrapper.h` 中 `auto name = strrchr(...)` 修正为 `const char* name = strrchr(...)`。

配置和构建本机命令：

```powershell
& 'D:\App\GNU Octave\Octave-10.3.0\mingw64\bin\cmake.exe' `
  -S upstream/mlvc/packages/msrtc_rans -B build/msrtc_rans `
  -G 'MinGW Makefiles' -DCMAKE_CXX_COMPILER=C:/mingw64/bin/g++.exe `
  -DCMAKE_C_COMPILER=C:/mingw64/bin/gcc.exe `
  -DCMAKE_MAKE_PROGRAM=C:/mingw64/bin/mingw32-make.exe `
  -DPython3_EXECUTABLE=C:/Users/lenovo/Desktop/works/work/semantic-video-mlvc/.venv/Scripts/python.exe `
  -DPython3_ROOT_DIR=C:/Users/lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python `
  -DCMAKE_BUILD_TYPE=Release -DINSTALL_ONLY_PYTHON_MODULE=ON -DBUILD_PYTHON_MODULE=ON
& 'D:\App\GNU Octave\Octave-10.3.0\mingw64\bin\cmake.exe' --build build/msrtc_rans -j 4
```

把 `packages/msrtc_rans/python/src/msrtc` 复制到新环境 `Lib/site-packages`，把构建生成的 `_msrtc_rans.cp312-win_amd64.pyd` 放入 `msrtc/rans`，并将 MinGW 的 `libgcc_s_seh-1.dll`、`libstdc++-6.dll`、`libwinpthread-1.dll` 复制到同处。

```powershell
& .\.venv\Scripts\python.exe -m pytest upstream/mlvc/packages/msrtc_rans/python/test/test_msrtc_rans.py -q
```

`video.mlvcv` 是本工程自定义容器，内部帧包采用官方真实熵编码。完整文件大小包含 JSON 协议头、帧长度、哈希、I 帧和 P 帧；预览 MP4、重建 PNG、日志不参与传输码率统计。双方共享模型权重不逐视频发送；如更换或下发微调权重，需另外计算模型更新成本。

## 阶段 B：语义评价与 ROI 准备

详细记录在 [phase_b_results_20261013.md](<docs/phase_b_results_20261013.md>)。

- `data/phase_b_annotation_corpus`：官方 SFISHTRACK 标注帧与二值鱼 GT，3 组共 120 帧；FFV1 参考缓存逐像素无损。
- `configs/phase_b_gt_codecs.json`：冻结模型、真实文件、GT 评价的固定输入清单。
- `runs/phase_b_codecs`：GLC q0、MLVC q0/q21、H.265 CRF44/50 的真实文件及独立解码。
- `runs/phase_b_gt_codec_semantics`：真正鱼类 IoU/F1，与原图预测一致性分开存储。
- `runs/phase_b_rate_match`：只按字节数匹配的单例，文件误差 0.2642%。
- `adapters/precomputed_roi_loss.py`：直接复用官方 ROI 损失，显式 mask，避免辅助分割模型初始化。
- `runs/phase_b_roi_interface` 和 `runs/phase_b_roi_backprop`：4 倍像素梯度权重及真实编码器反向传播证明；没有参数更新。
- `data/recording_groups_review.json`：日期/拍摄组歧义检查，避免把同源片段误作训练/测试独立样本。
- `data/uvmulti_preflight`：独立候选训练数据的官方索引、颜色标签表、采样 mask 和校验记录。

复现冻结模型 GT 评价：

```powershell
& .\.venv\Scripts\python.exe adapters/evaluate_semantics.py `
  --manifest configs/phase_b_gt_codecs.json --output runs/reproduce_gt_semantics
```

主协议教师输入 256。512 是诊断消融，添加 `--size 512` 并使用新输出目录可复现。GT 评价是 fish / non-fish 二值；既有潜水员视频没有人工 GT，只报告预测一致性。

标注序列按名义 5 fps 播放，部分原始图片编号存在缺口，当前结果限于共同帧序列的语义保留，连续视频时序性能另测。全部评价帧仍保留，不用于优化；后续使用独立确认的训练来源开展普通微调与 ROI 同预算对照。

本轮依赖在 `audit/requirements-phase-b-lock.txt`；阶段 A 原锁定文件保留。

## 阶段 C：首轮小规模实测

- [实测报告](<../02_历史资料/C_D报告/semantic_roi_phase_c/阶段C_小规模ROI对照实测_2026-10-13.md>)
- 脚本与配置（本地归档保留）
- `data/uvmulti_downloads/video131`：用户下载并移动的原始小样本，2,303个文件保留。
- `data/phase_c_20261013`：官方索引、真实目录编号、CRC校验、RGB/GT配对和连续帧清单。train24/131/91用于训练，train96整段留出开发，val108用于官方验证。拍摄组独立性未全部确认。
- `runs/phase_c_20261013/training`：M_RD、M_ROI4、M_ROI16相同512次更新、相同样本/裁剪/q/种子，最终权重和实际损失/梯度日志均保留。
- `runs/phase_c_20261013/codecs`：完整文件、字节匹配候选、独立接收端及逐帧RGB/DPB状态校验，24个选中质量点全部通过。
- `runs/phase_c_20261013/fixed_teacher.json`：原八类SegFormer保留结构的解码头适配，512分块、阈值0.9，压缩任务计分前固定。最终计分统一FP32，初次默认精度结果另存。
- `runs/phase_c_20261013/multiframe_probe.json`：4/8帧梯度与显存可行性通过，没有优化器更新，不作为多帧性能证据。

阶段C研究判断：普通微调需修复完整GOP退化，覆盖实际帧适配器和P参考特征，并扩充质量参数采样；普通RD稳定后再做4帧ROI对照。阶段C编码器推理没有ROI掩码输入，属于语义感知损失驱动的微调。当前最低工程目标已通过上面的语义通信原型完成，训练问题保留在升级路线中。

## 阶段 D：最小系统闭环与收发实测

[正式阶段报告](<docs/phase_d_results_20261014.md>)与报告、图表和证据索引（本地归档保留）已保存。

- 固定鱼类/潜水员模型，发送逐帧目标位置、置信度和面积；可选低清背景与高清ROI视频。本阶段没有新增训练。
- 5次独立进程收发核验，涉及两段输入和同源复测。视频模式恢复100个真实视频帧；信息模式恢复105个消息帧记录，不把绘制画布当作原视频重建。
- 轻量鱼类完整流程：50帧提取与打包16.54秒，2,011B通信包、20kbps传输0.844秒，独立接收消息逐项一致。
- 鱼类仅4张已见训练视频GT：近似同预算IoU10.77%对H.265 B4的3.97%，全局PSNR代价2.12dB；不能作为独立泛化证据。潜水员无GT，一致性IoU97.62%低于H.265的98.19%。
- 当前验收结论：最低本机工程闭环通过；鱼类可靠性、视频传输效率、跨设备与实际水声链路仍待验证。
