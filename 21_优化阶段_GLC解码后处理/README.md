# GLC 解码端纹理稳定性短段测试

2026-11-08。用已校验的 GLC q0 / GOP64 / reset32 FP32 解码 RGB 测试纹理稳定性。三段各 6 秒，从同一上传视频的不同镜头选出。片段及矩形观察区在处理前写入 [plan_p0.json](plan_p0.json)。原始序号和数据集来源可从该计划还原。

| 片段 | 原片时段 | 帧数 / FPS | 重点观察 |
| --- | --- | ---: | --- |
| 礁壁 | 2–8 秒 | 180 / 30 | 礁壁上缘纹理、小鱼和气泡 |
| 斑点鱼 | 13–19 秒 | 180 / 30 | 鱼身斑点、尾鳍、下方珊瑚 |
| 移动礁壁 | 39–45 秒 | 90 / 15 | 视角变化、红珊瑚和经过的鱼 |

每段比较原始 GLC、第 11 阶段两帧滤波和两种三帧融合。三帧版对齐前后各一张未经滤波的邻帧，以当前帧为锚点。两侧匹配都可信时才调整高频。保守版要求修正方向一致并选较小的幅度，均值版对两侧高频残差取平均。两版采用相同的 0.45 权重上限、10 个灰阶的 RGB 单通道修正上限和半分辨率 Farnebäck。门控包括前后向误差、低频颜色差、越界和局部光流突变。 [第 11 阶段实现](../11_优化阶段_纹理稳定性/temporal_filter.py)

## 首轮结果

| 片段 | 珊瑚区时序高频 MAE 原始 | 两帧 | 保守三帧 | 均值三帧 | 有效掩码覆盖 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 礁壁 | 8.997 | 8.481（−5.7%） | 8.805（−2.1%） | 8.669（−3.6%） | 68.6% |
| 斑点鱼 | 11.699 | 10.541（−9.9%） | 11.017（−5.8%） | 10.650（−9.0%） | 约 100% |
| 移动礁壁 | 7.400 | 7.044（−4.8%） | 7.276（−1.7%） | 7.215（−2.5%） | 49.3% |

两帧版在三个片段的珊瑚区时序误差均值最低。放宽融合后的三帧版比保守版更接近两帧，斑点鱼片段的逐帧 P95 略低于两帧（14.756 对 14.773）。运动观察区的 RGB 通道平均绝对改变量为保守三帧 0.21–0.50、均值三帧 0.40–0.99、两帧 0.65–1.38。该矩形包含背景，不能用它判断鱼尾是否拖影。

斑点鱼片段的珊瑚区高频幅度与源画面的比值，从原始 GLC 的 0.761 降至两帧的 0.674、保守三帧的 0.717、均值三帧的 0.682。时序误差改善伴随高频幅度下降，纹理、细枝和鱼尾另做观看对照。

在两个 30 FPS 片段，原始 GLC 的 I 帧和参考重置帧附近误差更高。礁壁片段的 I 帧、参考重置帧和其余帧均值分别为 15.80、14.24、8.69，斑点鱼片段分别为 17.89、15.29、11.52。移动礁壁片段三组接近。两份既有码流的 q0 / GOP64 / reset32 和 SHA-256 均用容器读取器核对过。这组比较用于定位，因果关系仍待受控验证。

本机 i9-14900HX，OpenCV 4.10 单线程，OpenCL 关闭。均值三帧这一轮，两帧滤波均值 36.2–40.0 ms/帧，三帧 45.8–51.3 ms/帧，均含光流、门控和融合。三帧版复用相邻帧对的双向光流，需要一帧前瞻。两轮的本机 CPU 耗时波动较大，比较取同轮计时。滤波耗时独立于 I/O、评价和预览。

## 观看反馈与五帧复测

用户在首轮对照中看到第 1 段礁壁和第 3 段红珊瑚有所改善。第 1 段右下的均值三帧观感优于左下的两帧。第 2 段斑点鱼仍难改善，鱼身密集的黑点是这个片段的主要难点。这是单人的非盲观看反馈，记录在指标之外。

复测用相同的 GLC 帧、裁剪和评分掩码。三帧 RGB 与上一轮均值三帧逐位一致。五帧只增加前后第二帧的高频残差。两张远邻都通过对齐门控时，它们最多占修正量的三分之一。当前帧的权重与修正上限沿用三帧设置，远邻不可靠处回退到三帧。

| 片段 | 时序高频 MAE 三帧 → 五帧 | 五帧相对三帧 | CPU 单线程 ms/帧 三帧 → 五帧 |
| --- | ---: | ---: | ---: |
| 礁壁 | 8.6690 → 8.6627 | −0.073% | 33.28 → 58.88 |
| 斑点鱼片段的珊瑚区 | 10.6502 → 10.5911 | −0.555% | 34.29 → 59.98 |
| 移动礁壁 | 7.2148 → 7.2160 | +0.017% | 33.21 → 60.14 |

第 1、3 段珊瑚观察区的五帧与三帧平均绝对像素差分别为 0.041、0.023 个灰阶。五帧每帧多花约 26 ms，并增加一帧前瞻。均值三帧作为当前礁壁候选，两帧保留作对照，斑点鱼保留作运动与细纹的困难样本。第 2 段珊瑚区的分数单独记录，鱼身斑点仍靠对应的局部视频观察。

## 明显晃动的定位与局部修复

复看后，用户确认晃动仍然明显，原三帧与五帧方案按观看验收未通过。第 1 段的三帧偏好仍作为相对观感记录。

用源 RGB 估计相邻帧光流，再对齐源 RGB 和 GLC 的同一珊瑚区域。第 1 段原片 6.40 秒对应零基帧 192。源画面 191–193 的珊瑚纹路基本连续，GLC 第 192 帧切到 I 帧时出现明显的纹理跳变。原始 GLC 全频误差前十的帧中，5 个落在 I 帧或旧版脚本标记的周期 reset 位置，另 5 个为普通 P 帧。周期 reset 的分类在后续内部诊断中细分。第 3 段也观察到参考重置帧和普通帧的误差峰值。

诊断把亮度分为粗尺度（高斯 σ=3 像素）、中尺度（σ=1.2 与 σ=3 之差）、细尺度（原亮度减 σ=1.2）和全频四项，各项 MAE 独立计算。保留三帧融合、仅将运动估计换成源 RGB 光流时，两段的全频误差分别比原三帧再降约 0.29% 和 0.24%。该对照仅说明原三帧设置下的改动有限。I/P 生成器和参考路径的各自贡献仍待拆分。

另测一版局部跳变修复。只从解码 RGB 对齐前后帧，当前帧与两侧都偏离、两侧又相互吻合时，才修正固定珊瑚观察区。以局部差异和运动一致性共同限制修改，并保留大部分粗尺度分量。只在第 1、3 段验证，选区之外的输出逐位等于原始 GLC。

| 片段 | 原始 GLC 全频 MAE | 均值三帧 | 局部修复 | 最大峰值原始 → 局部修复 |
| --- | ---: | ---: | ---: | ---: |
| 礁壁 | 13.931 | 13.603 | 13.678 | 33.30 → 30.50（帧 192） |
| 移动礁壁 | 13.220 | 13.035 | 12.789 | 16.53 → 16.00（帧 673） |

局部修复降低了第 1 段最明显的突跳，但整段均值略高于三帧。第 3 段整段误差相对原始 GLC 下降 3.3%。两段单线程修复约 36–37 ms/帧。当前结果仅支持局部削峰，普通帧的持续晃动仍需进一步诊断。

复现时使用上一轮均值三帧结果，指定新的输出目录。

```powershell
python -B -X utf8 '21_优化阶段_GLC解码后处理/probe_reef.py' --previous $glcTrialMeanOut --output $glcReefProbeOut --repair --ffmpeg $glcFfmpeg
```

本机结果在 `temporal_p0_20261108_01/reef_jump_repair`。每段提供原始、三帧与局部修复的同帧四宫格，以及按源运动对齐的峰值逐帧图。4 个预览核对了帧数、帧率和时长。 [汇总](evidence_20261108/reef_repair/summary.json) · [逐帧 CSV](evidence_20261108/reef_repair/per_frame.csv) · [独立核对](evidence_20261108/reef_repair/verification.json)

## All-In-One-Deflicker 短段试验

首测固定礁壁原片零基帧 152–231，共 80 帧，512×288、30 FPS。第 192 帧 I 帧突跳在片段中部。输入逐张由现有 GLC FP32 无损 RGB 写成 PNG，并回读核对逐位相同。模型处理只读取这些重建帧，源 RGB 用于事后评价。

使用官方源码和发布权重。Windows 独立环境复用现有 CUDA 版 PyTorch，并单独安装图集和图像 I/O 的依赖。按顺序调用官方 RAFT 光流、逐段神经图集拟合、预训练滤波和精修。采用官方配置的 10,001 步图集迭代、100 次映射预训练和 4 倍图集下采样，随机种子固定为 2023。 [官方项目](https://github.com/ChenyangLEI/All-In-One-Deflicker)

以最终编号 PNG 为输出，逐张核验 80 帧的尺寸、RGB 通道和顺序。对照视频由这些 PNG 重新按 30 FPS 封装。对照组为同帧原始 GLC 和均值三帧，三帧在新截取片段的首末帧采用缺邻帧时的原样输出。局部时序指标沿用源画面光流与同一珊瑚观察区，图像指标使用同分辨率源 RGB。

| 方法 | 珊瑚区全频时序 MAE ↓ | 珊瑚区细尺度时序 MAE ↓ | 全图 PSNR ↑ | 珊瑚区 PSNR ↑ | 全图 LPIPS ↓ | 全图 DISTS ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始 GLC | 16.247 | 10.107 | 22.081 dB | 16.462 dB | 0.1246 | 0.1545 |
| 均值三帧 | 15.990 | 9.848 | 22.123 dB | 16.481 dB | 0.1236 | 0.1529 |
| All-In-One-Deflicker | 13.410 | 8.573 | 21.659 dB | 17.134 dB | 0.1279 | 0.1631 |

相对原始 GLC，模型把珊瑚区全频时序误差降了 17.5%，细尺度误差降了 15.2%。6.40 秒的 I 帧误差由 33.30 降到 26.10，紧接着的首个 P 帧由 30.28 降到 23.01。全图 PSNR 下降约 0.42 dB，LPIPS 和 DISTS 升高。珊瑚区 PSNR 则提高约 0.67 dB，细尺度幅度与源的比值由 1.110 降到 0.934。6.40 秒的单帧对照中，模型输出的细纹边缘较软。

本机 4060 Laptop 的模型处理墙钟为 558.2 秒，含双向光流 12.7 秒、图集拟合与渲染 533.5 秒、滤波精修与上游预览导出 6.6 秒，其余为进程启动及落盘。最大 PyTorch 分配量 791 MiB，最大保留量 888 MiB，出现在图集阶段。输入及输出 80 帧均保留 512×288 RGB，三份本机视频都独立核对为 80 帧、30 FPS。 [运行记录](evidence_20261108/deflicker_reef80/run.json) · [评价](evidence_20261108/deflicker_reef80/summary.json) · [独立核对](evidence_20261108/deflicker_reef80/verification.json)

本机结果在工作区 `02_实验数据/优化_GLC后处理/deflicker_reef80_20261108_01`。原分辨率增强视频、完整及局部四宫格、时序曲线位于 `comparison`。四宫格左上为源 RGB，右上为原始 GLC，左下为均值三帧，右下为模型结果。无损结果位于 `trial/deflicker_rgb.npy`。

```powershell
& $deflickerPython -B -X utf8 '21_优化阶段_GLC解码后处理/run_deflicker.py' --upstream $deflickerSource --weights-repo $deflickerWeights --ffmpeg $glcFfmpeg --output $deflickerRun
& $deflickerPython -B -X utf8 '21_优化阶段_GLC解码后处理/compare_deflicker.py' --previous $glcTrialMeanOut --model-run $deflickerRun --ffmpeg $glcFfmpeg --output $deflickerComparison --metrics-root $glcPerceptualCache
```

### 观看反馈与纹理核对

用户观看后表示差距不明显，珊瑚纹理仍持续变化。本段按观看验收未通过记录。为区分纹理幅度减弱与纹路跨帧连续，沿用原来的 79 个源运动对齐帧对和珊瑚掩码，对齐后的细尺度亮度分别做去均值和方差归一化，再计算相关性。源画面的相关性为 0.969，原始 GLC 为 0.577，模型精修后为 0.583。6.40 秒的 I 帧处，原始 GLC 与模型的细纹相关性都约为 0.05–0.06。第 215 帧的普通 P 帧对齐图中仍能看到纹路变化。

另用固定 σ=0.486 像素的高斯滤波逐帧处理原始 GLC，使细纹幅度与模型输出相差约 0.053%。这个空间对照的全频时序 MAE 也从 16.247 降到 14.542。模型结果为 13.410。单帧平滑本身足以降低这项误差，归一化相关性和观看反馈分别记录。

128×72 的图集输出放大后，细纹幅度约为源画面的 18%。它的纹理相关性较高，细节明显较弱。滤波阶段把细纹幅度恢复到接近源画面，相关性仍约为 0.577。精修阶段的相关性变化很小。这里仅对本片的阶段输出作诊断。

对齐图位于本机试验目录 `texture_alignment_audit`。其中第 192 帧和第 215 帧的图依次展示源画面、原始 GLC、空间模糊对照和模型输出。 [核对汇总](evidence_20261108/deflicker_reef80/texture_alignment/summary.json) · [逐帧数据](evidence_20261108/deflicker_reef80/texture_alignment/per_frame.csv)

```powershell
& $deflickerPython -B -X utf8 '21_优化阶段_GLC解码后处理/audit_deflicker_texture.py' --previous $glcTrialMeanOut --pilot $deflickerPilot --output $deflickerTextureAudit
```

## SeedVR2 短段试验

使用同一礁壁的零基帧 152–231（512×288、30 FPS），先以帧 188–196 的 9 帧做运行核验，再处理完整 80 帧。通过 FFV1 封装已有 GLC 解码 RGB，用 OpenCV 和 FFmpeg 分别回读核对像素。源 RGB 只用于评价。

使用 Numz 的低显存独立 CLI 2.5.24（`4490bd1f482e026674543386bb2a4d176da245b9`）、3B EMA Q8_0 和 FP16 VAE 发布权重。保持输入分辨率，使用 SDPA、CPU 卸载、32 层换入换出、LAB 色彩校正和随机种子 2023。短试验为一个 9 帧组，完整片段为 9 帧一组、重叠 3 帧。首次短试验中 OpenCV 无法在中文路径写入 PNG。采用随仓库提供的写盘补丁后，逐张核验输出，并保存同像素的 `seedvr2_rgb.npy`。 [低显存 CLI](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/tree/4490bd1f482e026674543386bb2a4d176da245b9) · [写盘补丁](seedvr2_windows_png.patch)

| 方法 | 珊瑚区全频时序 MAE ↓ | 对齐后细纹跨帧 NCC ↑ | 同帧细纹对源 NCC ↑ | 细纹幅度 / 源 | 全图 PSNR ↑ | 全图 LPIPS ↓ | 全图 DISTS ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始 GLC | 16.247 | 0.577 | 0.064 | 1.110 | 22.081 dB | 0.1246 | 0.1545 |
| All-In-One-Deflicker | 13.410 | 0.583 | 0.065 | 0.934 | 21.659 dB | 0.1279 | 0.1631 |
| Deflicker 等细纹幅度的逐帧模糊 | 14.542 | 0.613 | 0.073 | 0.934 | 22.491 dB | 0.1247 | 0.1565 |
| SeedVR2 3B Q8 | 22.175 | 0.570 | 0.056 | 1.724 | 20.011 dB | 0.2026 | 0.1824 |

跨帧 NCC 沿用源运动对齐和相同的 79 个珊瑚帧对，源画面为 0.969。同帧对源 NCC 衡量珊瑚区对应像素的细纹相关性。SeedVR2 的中尺度跨帧 NCC 从原始 GLC 的 0.810 提升至 0.855，细纹相关性下降，全频时序误差上升 36.5%。第 192 帧的全频误差从 33.30 增至 47.84。6.40 秒的局部同帧对照可见 SeedVR2 珊瑚边缘更硬，沟槽比源画面更深。该配置在这段礁壁上缺少改善细纹稳定性与保真的证据。

RTX 4060 Laptop 上 9 帧和 80 帧的 CLI 墙钟时间分别为 11.7 和 67.5 秒，阶段日志的最大 PyTorch 分配量均为 4.15 GiB，最大采样保留量均为 5.25 GiB。两份本地预览均核验为 80 帧、30 FPS。四宫格左上为源 RGB，右上为原始 GLC，左下为 Deflicker，右下为 SeedVR2。 [完整输入](evidence_20261108/seedvr2_reef80/full_inputs.json) · [运行](evidence_20261108/seedvr2_reef80/full_run.json) · [评价](evidence_20261108/seedvr2_reef80/summary.json) · [独立核对](evidence_20261108/seedvr2_reef80/verification.json)

本机数据位于 `02_实验数据/优化_GLC后处理/seedvr2_reef80_20261108_01`。原速对照、局部对照和细纹曲线位于 `comparison`。复现时在 `$seedPilot/upstream` 固定上述提交并应用写盘补丁，将已核验的两份权重放入 `$seedPilot/models`，使用独立环境及新的输出目录。

```powershell
git -C "$seedPilot/upstream" apply --unidiff-zero (Join-Path (Get-Location) '21_优化阶段_GLC解码后处理/seedvr2_windows_png.patch')
& $seedPython -B -X utf8 '21_优化阶段_GLC解码后处理/run_seedvr2.py' --pilot $seedPilot --deflicker-pilot $deflickerPilot --case full --batch-size 9 --ffmpeg $glcFfmpeg --output $seedRun
& $seedPython -B -X utf8 '21_优化阶段_GLC解码后处理/compare_seedvr2.py' --previous $glcTrialMeanOut --deflicker-run $deflickerRun --seedvr2-run $seedRun --ffmpeg $glcFfmpeg --output $seedComparison --metrics-root $glcPerceptualCache
```

## GLC 内部时序诊断

### GOP 和参考重置

从同一源视频第 0 帧开始编码 240 帧，固定 q0、FP32 eager、512×288、30 FPS 和权重。三条码流各由独立接收进程解码。基线的 240 个压缩帧和 RGB 均与旧结果逐位一致，旧发送日志另有 128 个逐帧参考摘要也全部吻合。三个新码流分别通过整段参考摘要校验。以下指标对原片零基帧 60–239 的珊瑚区计算，码率按各自完整的 8 秒 GLCSND 文件大小计算。

| GOP / reset | 实际码率（kbps） | 179 对全频时序 MAE ↓ | 151 对共同普通 P→P MAE ↓ |
| --- | ---: | ---: | ---: |
| 64 / 32 | 44.708 | 13.931 | 12.862 |
| 72 / 32 | 46.923 | 13.917 | 12.771 |
| 64 / 24 | 47.737 | 14.076 | 12.827 |

I 帧会把 `ref_y` 置空。基线第 193 帧虽命中周期 reset，进入时 `ref_y` 已为空。重分组后，I 帧、I 后首个 P 帧、实际清除 `ref_y` 的 P 帧和普通 P 帧的全频误差分别为 30.65、28.17、21.26、13.24。

GOP 从 64 改到 72 后，原 I 帧 64、128、192 变成普通 P，三处误差均下降。第 192 帧从 33.30 降到 11.14。新 I 帧 72、144、216 的误差均上升，第 216 帧从 24.37 升到 34.35，新 I 帧后的首个 P 帧也都升高。

固定 GOP 64，把 reset 从 32 改到 24 后，五个新增的实际重置位置误差全部升高，两个取消重置的位置全部降低。第 217 帧从 21.52 升到 26.62，第 225 帧从 23.44 降到 14.41。三个 I 帧和三个紧随其后的 P 帧输出逐位相同。第 215 帧在三组里均为普通 P，误差依次为 29.03、29.52、29.03。两个设置能迁移边界突跳，持续晃动在共同普通 P 帧对中仍存在。

### 普通 P 帧的中间张量

从原始码流第 128 帧 I 帧开始独立解码，记录帧 152–231 中的 79 个 P 帧。对经过的 104 帧逐帧核对参考状态和最终 RGB，与核验过的原码流前缀一致。钩子抓取 P 解码器的 `context`、`y_hat`、RGB 生成器输入 `latent` 和裁剪前的浮点 RGB。源 RGB 在独立的分析进程中用于运动对齐。

在原珊瑚区域内把可靠像素和源光流映射到 20×32 特征网格。三个张量按各自通道的标准差归一化，计算对齐后的相邻帧变化，并在相同特征适配器相位内排名。分析覆盖 77 对相邻 P 帧，其中 72 对的前后两帧都为普通 P 帧。

| 第 175 帧与第 215 帧的比较 | 175 | 215 | 215 在同相位 30 对中的排名（变化从大到小） |
| --- | ---: | ---: | ---: |
| `context` 变化 | 0.326 | 0.462 | 6 |
| `y_hat` 变化 | 0.567 | 1.014 | 3 |
| 生成前 `latent` 变化 | 0.063 | 0.307 | 1 |
| 裁剪前浮点 RGB 时序 MAE | 7.95 | 29.05 | 1 |
| 最终 RGB 时序 MAE | 7.96 | 29.03 | 1 |

第 215 帧的异常在送入生成器前的 `latent` 已经显著，`y_hat` 也出现较大变化。其所在的 30 对普通 P 帧中，`latent` 变化和最终 RGB 误差的 Spearman 相关性为 0.931。这是单段礁壁、固定权重和码率档的观察。`context` 和 `y_hat` 的各自贡献以及生成器的放大量仍需单独控制。

本机输出位于 `02_实验数据/优化_GLC后处理/reef_decoder_diagnosis_20261108_02`。原速四宫格及边界图位于 `boundary_evaluation`，上排为源 RGB 和 GOP 64 / reset 32，下排为 GOP 72 / reset 32 和 GOP 64 / reset 24。P 帧曲线位于 `p_evaluation`。 [边界数据](evidence_20261108/reef_decoder_diagnosis/boundary_summary.json) · [P 帧数据](evidence_20261108/reef_decoder_diagnosis/p_summary.json) · [独立核对](evidence_20261108/reef_decoder_diagnosis/verification.json)

复现使用 GLC FP32 环境，以及含 OpenCV、NumPy 和 matplotlib 的 Python 3.12 分析环境。各命令使用新的输出目录。

```powershell
& $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/run_boundary_codec.py' --output $boundaryRun
& $analysisPython -B -X utf8 '21_优化阶段_GLC解码后处理/compare_boundary_codec.py' --run $boundaryRun --output $boundaryEvaluation --ffmpeg $glcFfmpeg
& $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/capture_p_trace.py' --prefix-run "$boundaryRun/g64_r32" --output $pTrace
& $analysisPython -B -X utf8 '21_优化阶段_GLC解码后处理/compare_p_trace.py' --capture $pTrace --output $pEvaluation
```

## 水下观感增强

保留三段已固定的 GLC 无损解码 RGB。礁壁和斑点鱼各为 180 帧、30 FPS，移动礁壁为 90 帧、15 FPS，均为 512×288、6 秒。源 RGB 用于对照和保真评价。模型只处理解码 RGB，使用既有 GLC 码流。

- UVE-Net 使用官方 UVEB 代码 `a9c38b284aa9fca609b443c1857ecd8468ce7a7d` 和 `large_net_g.pth`，严格加载 24 通道架构。以 3 帧窗口、步长 1 推理，相同输出帧按 1:2:1 的窗口内位置权重合成。RGB 在原分辨率推理。 [UVEB 官方仓库](https://github.com/yzbouc/UVEB/tree/a9c38b284aa9fca609b443c1857ecd8468ce7a7d)
- UnDIVE 使用官方代码 `7fac42cb01e5e40cc55470b56c4491162f80d9b9` 和作者压缩包中的 `UnDIVE_100.pth`、`DDPM_100.pth`。使用官方 RGB 推理流程，256×256 分支提取增强参数并应用到原尺寸画面，沿用逐帧 RGB 最小值和最大值映射。此对照直接输入 GLC RGB，省略依赖深度图的后向散射去除。 [UnDIVE 官方仓库](https://github.com/suhas-srinath/undive/tree/7fac42cb01e5e40cc55470b56c4491162f80d9b9)

UVE-Net 让礁壁暖色和蓝水更容易区分，斑点鱼身后的珊瑚减轻了蓝偏。UnDIVE 在样帧中提亮更强，气泡和浅色珊瑚局部发白。其礁壁和鱼段亮度第 95 百分位分别达到 233、231（8 位），UVE-Net 为 172、165。交付版按同像素 70% UVE-Net 和 30% 原始 GLC 混合，三段使用同一比例。

| 片段 | 亮度中位数 原始 → 70% 版 | 珊瑚细纹跨帧 NCC 原始 → 70% 版 | 珊瑚全频时序 MAE 原始 → 70% 版 | 全图 PSNR 原始 → 70% 版 |
| --- | ---: | ---: | ---: | ---: |
| 礁壁 | 87 → 92 | 0.646 → 0.656 | 13.93 → 15.70 | 22.60 → 19.90 dB |
| 斑点鱼 | 70 → 72 | 0.503 → 0.520 | 13.07 → 13.80 | 21.07 → 19.66 dB |
| 移动礁壁 | 56 → 60 | 0.113 → 0.131 | 13.22 → 15.03 | 23.94 → 21.27 dB |

70% 版在三段中提高了珊瑚细纹的归一化相关性，时序差异的幅度也增大。PSNR、LPIPS 和 DISTS 相对源 RGB 均向保真较低的方向变化。斑点鱼观察框内的同帧细纹与原始 GLC 的 NCC 为 0.986，该观察框同时包含背景。颜色偏好、鱼身斑点与运动观感以原速视频核对。

RTX 4060 Laptop 上，UVE-Net 和 UnDIVE 各处理了 450 帧，推理总耗时分别为 100.46 和 11.14 秒。每段有独立增强视频、与原始 GLC 的左右对照，以及源 RGB、GLC、UVE-Net、UnDIVE 的同帧四宫格。斑点鱼另有局部四宫格。独立核验 10 份视频的帧数和帧率，并回读三份独立增强视频。逐帧保真 CSV 使用各片段的原片零基帧号。

本机输出位于 `02_实验数据/优化_GLC后处理/underwater_enhancement_20261108_01/comparison_verified`。各片段的 `enhanced.mp4` 为 70% 版，`before_after.mp4` 左侧为原始 GLC，右侧为增强版，`models.mp4` 为完整模型四宫格。 [汇总](evidence_20261108/underwater_enhancement/summary.json) · [UVE-Net 运行记录](evidence_20261108/underwater_enhancement/uve_run.json) · [UnDIVE 运行记录](evidence_20261108/underwater_enhancement/undive_run.json) · [独立核对](evidence_20261108/underwater_enhancement/verification.json)

复现时克隆上述版本，将 UnDIVE 的两个权重文件解压到作者仓库的 `PretrainedModels`。使用现有 GLC CUDA PyTorch 环境，另把 `torchsummary==1.5.1` 安装到 `$undivePackages`。UnDIVE 的旧 DDPM 权重带有 NumPy 元数据，下面的加载方式仅放行权重和所需的 NumPy 类型。各输出目录使用新路径。

```powershell
& $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/run_underwater_enhancement.py' --method uve --upstream $uveSource --output $uveRun
$env:PYTHONPATH = $undivePackages
$env:TORCH_FORCE_WEIGHTS_ONLY_LOAD = '1'
$loader = @'
import runpy
from pathlib import Path
import sys
import numpy as np
import torch

script = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(script.parent))
with torch.serialization.safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype,
                                       np.dtypes.UInt32DType, np.dtypes.Float64DType]):
    runpy.run_path(str(script), run_name='__main__')
'@
$loader | & $glcPython -B -X utf8 - '21_优化阶段_GLC解码后处理/run_underwater_enhancement.py' --method undive --upstream $undiveSource --output $undiveRun
& $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/compare_underwater_enhancement.py' --uve-run $uveRun --undive-run $undiveRun --output $waterComparison --ffmpeg $glcFfmpeg --metrics-root $glcPerceptualCache
```

## 15 FPS 珊瑚纹理语义流短试验

采用现有 512×288、15 FPS、q0 FP32 基础码流。直接从原码流的 I 帧 576 切出 99 个原始压缩帧，独立接收解码逐帧吻合旧 RGB，取原片 585–674 帧作 6 秒观看对照。前 9 帧用于接收端恢复参考状态。

在第 585 帧手选一块 96×96 珊瑚纹理，连续跟踪源端局部特征，每帧估计从同一锚点到当前视角的仿射变换。传输一个无损 PNG、量化仿射和 6×6 网格的可用区域。接收端在校验 GLC 基础流后，用同一锚点更新可靠区域的中高频亮度，羽化交界。另以第 585 帧的 GLC 自身纹理作对照，沿用相同几何、遮罩和权重。两版遮罩外的 RGB 逐位等于 GLC 基础层。

语义文件实测 18,704 B，包含纹理 17,134 B、压缩跟踪记录 1,084 B、描述 172 B 和容器 314 B。按 6 秒折合 **24.94 kbps**，低于语义流单独的 40 kbps 上限。基础流 22,995 B，按含预解码帧的 6.6 秒折合 27.87 kbps。将两份文件全部计入 6 秒展示片段，合计为 55.60 kbps。

| 比较组 | 运动对齐细纹跨帧 NCC ↑ | 同帧细纹对源 NCC ↑ | 细纹幅度 |
| --- | ---: | ---: | ---: |
| 原始 GLC | 0.418 | 0.086 | 6.69 |
| 复用 GLC 自身纹理 | 0.842 | 0.113 | 4.51 |
| 复用发来的源纹理 | 0.832 | 0.639 | 4.65 |

对 89 对相邻帧在相同的有效复用区域计分。90 帧均有生效像素，平均有效面积约 2,980 像素，约占整帧 2.0%。两种复用均提高局部时序一致性，源纹理在同帧对应原片细纹上更接近，细纹幅度均有所下降。本次仅覆盖一块手选珊瑚，网格遮罩由源图像对齐验证产生。接收端独立读取审计和两份实际文件的字节核验通过。

本机视频和两份流在 `02_实验数据/优化_GLC后处理/coral_reference_sidecar_20261108_01`。跟踪放大和全图四宫格位于 `comparison`，左上为源 RGB，右上为原始 GLC，左下为复用 GLC 纹理，右下为复用收到的源纹理。三份预览均为 90 帧、15 FPS。 [发送记录](evidence_20261108/coral_reference_sidecar/send.json) · [评价](evidence_20261108/coral_reference_sidecar/summary.json) · [核对](evidence_20261108/coral_reference_sidecar/verification.json)

在新建的 `$tileRun` 目录执行以下命令。

```powershell
& $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/coral_reference_sidecar.py' cut --run $tileRun
& $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/coral_reference_sidecar.py' send --run $tileRun
& $glcPython -B -X utf8 '17_优化阶段_解码器推理加速/decoder_worker.py' --stream "$tileRun/base.glcsnd" --output "$tileRun/base_receive" --backend eager --assets $glcRoleAssets
& $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/coral_reference_sidecar.py' receive --run $tileRun
& $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/compare_coral_reference.py' --run $tileRun --ffmpeg $glcFfmpeg
```

### 多块珊瑚扩大覆盖

复用同一 6 秒、15 FPS 基础流，扩大空间覆盖。候选为从不同时间出现的 120×120 局部珊瑚块。每块各自估计并量化局部仿射，发送 JPEG 亮度、可用网格及手标主鱼保护框。选块同时考虑每字节新增面积和各帧覆盖，优先补足覆盖较低的时刻。重叠区域按权重合成，采用较宽的中高频带。接收端独立读取审计通过，主鱼保护框内的 RGB 逐帧逐位等于基础层。

最终选用 9 块，语义文件 29,898 B，折合 **39.864 kbps**。平均有效区域由 2,980 增至 14,318 像素（全帧 **9.7%**），为单块的 4.80 倍。每秒平均覆盖约为 21,503、12,851、15,802、14,224、11,208、10,321 像素。初次多块试验使用 6 块，平均覆盖 8.9%，后两秒较少。最终版将 JPEG 质量从 88 调到 79，增加后段锚点并重新分配覆盖。

| 在最终版共同有效区域比较 | 细纹跨帧 NCC ↑ | 同帧细纹对源 NCC ↑ | 细纹幅度 |
| --- | ---: | ---: | ---: |
| 原始 GLC | 0.366 | 0.073 | 6.10 |
| 上一轮单块源纹理 | 0.418 | 0.174 | 5.70 |
| 多块 GLC 自身纹理 | 0.867 | 0.091 | 4.78 |
| 多块收到的源纹理 | 0.873 | 0.660 | 5.20 |

表中全部方法共用最终版的评价区域和 89 个相邻帧对。有效区外的画面随基础流，细纹幅度在局部仍有下降。四份原速视频在 `02_实验数据/优化_GLC后处理/coral_reference_sidecar_large_20261108_02/comparison`，含左右对照、全图及局部四宫格和独立合成。 [汇总](evidence_20261108/coral_reference_multi/summary.json) · [逐秒覆盖与独立核对](evidence_20261108/coral_reference_multi/verification.json)

```powershell
& $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/multi_coral_reference.py' send --base-run $tileRun --run $multiRun
& $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/multi_coral_reference.py' receive --base-run $tileRun --run $multiRun
& $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/compare_multi_coral.py' --base-run $tileRun --run $multiRun --ffmpeg $glcFfmpeg
```

### 源纹理和接收端缓存混合复用

2026-11-09。沿用同一 6 秒、90 帧、15 FPS 基础流和手标鱼框。取上一版全源包里新增有效面积最大的三块 JPEG，按原字节发送。其余 30 块只发送锚点位置、仿射轨迹和有效网格，从指定 GLC 解码帧取亮度参考。选块以实际 ZIP 字节为成本，优先补足覆盖较少的帧。重叠处按各自的可靠覆盖融合，源参考边缘逐渐过渡到 GLC 缓存。

独立语义文件实测 **29,759 B、39.679 kbps**，三张源 JPEG 合计 8,082 B。接收端以验证过的 GLC 基础层和该语义文件完成 90 帧合成。鱼框内和总遮罩权重为零处的 RGB 与基础层逐位一致。源参考遮罩权重为零处的混合结果与全缓存对照逐位一致。重新回放旧版语义文件后，两个 RGB 对照、遮罩和逐帧评价 CSV 都与既有结果逐位一致。

| 实测覆盖 | 上一版全源 9 块 | 混合 33 块 |
| --- | ---: | ---: |
| 全段平均 | 9.7% | 18.3% |
| 第五秒 | 7.6% | 14.7% |
| 最后一秒 | 7.0% | 10.7% |
| 源纹理全段平均 | 9.7% | 6.1% |

在混合版的有效区域内统一计分，后两秒的细纹跨帧 NCC 为原始 GLC 0.423、上一版 0.575、混合版 0.836。按两版共同覆盖区域比较，跨帧 NCC 为 0.873 → 0.867，同帧细纹对源 NCC 为 0.660 → 0.473。源 JPEG 实际生效的区域中，混合版同帧对源 NCC 为 0.656，全缓存控制为 0.102。增量覆盖主要来自缓存，其保真度随 GLC 锚点本身而变化。

本机对照位于 `02_实验数据/优化_GLC后处理/coral_reference_hybrid_20261109_01/comparison`。 `previous_vs_hybrid.mp4` 左边为上一版，右边为混合版。覆盖图中橙色为源 JPEG 生效区域，绿色为其余有效区域。5 份视频均独立核对为 90 帧、15 FPS。 [实验汇总](evidence_20261109/coral_reference_hybrid/summary.json) · [独立核对](evidence_20261109/coral_reference_hybrid/verification.json)

```powershell
& $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/multi_coral_reference.py' send --base-run $tileRun --run $hybridRun --hybrid-from $multiRun
& $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/multi_coral_reference.py' receive --base-run $tileRun --run $hybridRun
& $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/compare_multi_coral.py' --base-run $tileRun --run $hybridRun --prior-multi-run $multiRun --ffmpeg $glcFfmpeg
```

## 自主缓存与发送端运动对照

2026-11-09。基础流和语义流以各自文件的完整字节数计费，每路上限 40 kbps。本轮另给 C 设置 2,000 B 的几何试验上限。移动礁壁沿用 39–45 秒的开发片段和右侧红珊瑚框。留出镜头取同一源视频的第 832–921 帧，镜头近静止、有鱼群经过，珊瑚框及参数在运行前固定。两段均为 512×288、90 帧、15 FPS。基础流分别为 22,995 B 和 18,956 B，含各自的封装及所需前滚帧。独立接收端解码 RGB 与原存档逐帧一致。

| 方法 | 运动与纹理来源 |
| --- | --- |
| A | 原始 GLC |
| B0 | 对 A 做逐帧高斯平滑。分别匹配 B1、B2 在固定珊瑚框的平均细纹幅度，最大相对差 0.27% |
| B1 | 从 GLC 前一帧的低频图像估计共享仿射运动 |
| B2 | 将最近三帧 GLC 低频结构对齐到上一帧，取中位图估计共享仿射运动 |
| C | 用 B2 的估计流程处理发送端源 RGB，发送每帧共享仿射和有效标记 |

B1、B2 和 C 每 15 帧缓存一张 GLC 原始锚帧，几何无效时重新起锚。三者共用纹理幅度、低频 RGB 差、越界和渐变门控，只替换亮度的中高频残差。门控在接收端计算。C 的文件含基础流哈希、时间轴和 CRC，两段均为 **1,213 B、1.617 kbps**。自主接收进程在只有基础流及其本地解码收据的目录重放，每段六份输出数组均逐字节一致。计分使用源 RGB 独立计算的稠密光流。珊瑚框用于计分和匹配 B0 的平滑强度，缓存和门控均在全画面运行。

| 方法 | 移动礁壁全画面 NCC | 移动礁壁珊瑚 NCC | 移动礁壁珊瑚最后两秒 | 留出镜头珊瑚 NCC |
| --- | ---: | ---: | ---: | ---: |
| 源 RGB | 0.884 | 0.914 | 0.904 | 0.973 |
| A | 0.245 | 0.113 | 0.126 | 0.711 |
| B0，匹配 B2 | 0.257 | 0.121 | 0.135 | 0.720 |
| B1 | 0.285 | 0.131 | 0.129 | 0.738 |
| B2 | 0.276 | 0.135 | 0.129 | 0.738 |
| C | 0.288 | 0.152 | 0.127 | 0.744 |

NCC 为源 RGB 运动对齐后的相邻帧细纹相关性。各方法在完全相同的源纹理、配准掩码上计分。整段固定珊瑚框中，B2 比 B0 高 0.014 和 0.017，但对源细纹的同帧 NCC 仅从 0.059 升到 0.061、从 0.047 升到 0.049。C 在移动礁壁前四秒每秒比 B2 高 0.018–0.031，后两秒分别低 0.001 和 0.002。留出镜头整段只高 0.007。固定珊瑚框的结论支持暂停这套共享仿射缓存的扩张，尤其是镜头移动后段。

| H.265 文件计费点 | 移动礁壁实际 B / kbps | 留出镜头实际 B / kbps |
| --- | ---: | ---: |
| GLC 基础流同字节 | 22,735 / 30.313 | 18,946 / 25.261 |
| 40 kbps | 29,602 / 39.469 | 29,849 / 39.799 |
| 上轮双流 52,754 B | 52,641 / 70.188 | 52,350 / 69.800 |

H.265 使用本机 libx265 medium 默认 B 帧、关键帧间隔上限 64 帧、BT.709 YUV420p，按完整 MP4 字节选 CRF。它是离线字节对照。移动礁壁约 70 kbps 时，H.265 和上轮混合双流在同一珊瑚框的跨帧 NCC 分别是 0.724 和 0.291，同帧对源细纹 NCC 是 0.576 和 0.097。H.265 同时有可见的软化，珊瑚细纹幅度为 2.02，源 RGB 为 3.23，上轮混合为 3.17。高相关性需要结合细纹幅度和原速观感判断。两段来自同一源视频，留出镜头只检验本视频另一个近静止场景。

本机结果目录为 `02_实验数据/优化_GLC后处理/cache_motion_controls_20261109_01`。每段的 `comparison` 含原速缓存对照、珊瑚放大、B0 和 H.265 对照。移动礁壁另有上轮混合双流和近似同字节 H.265 的四宫格。抽查静帧用于检查前景和边缘，整段的拖影及观感以原速复看为准。 [完整汇总](evidence_20261109/cache_motion_controls/summary.json) · [独立核对](evidence_20261109/cache_motion_controls/verification.json)

```powershell
& $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/cache_motion_inputs.py' --output $cacheRun --assets $roleAssets
foreach ($clip in 'moving_reef', 'holdout_reef') {
    foreach ($action in 'receive-b', 'send', 'receive-c') {
        & $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/run_cache_motion.py' $action --clip $clip --run $cacheRun
    }
}
& $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/cache_hevc_baseline.py' --run $cacheRun --ffmpeg $glcFfmpeg
& $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/compare_cache_motion.py' --run $cacheRun --ffmpeg $glcFfmpeg --prior-hybrid $hybridRun
```

## 原片稠密光流的离线上限对照

2026-11-09。沿用移动礁壁第 585–674 帧、15 FPS、512×288 和固定红珊瑚框。用本机已验证的 RAFT-Things 大模型，在同尺寸的无损源 RGB 上以 FP32 迭代 24 次。每帧直接与当前 15 帧周期的源锚帧做双向匹配，避免逐帧累计变换。共处理 84 个非锚帧的 168 个方向，GPU 推理循环耗时 17.6 秒。原片的像素仅参与估计、可靠性检查和计分，两版输出均复用原先的 GLC 亮度纹理。

- **D1** 把 C 的共享仿射换成原片逐像素映射，沿用 C 的 GLC 低频差门控。
- **D2** 使用同一映射，改用原片低频差及 1.5 像素双向一致性阈值判定可复用区域。两版保留 C 的锚帧周期、融合强度、渐变和残差限制。提取共用融合函数后，B1、B2、C 在原来两个片段的 RGB 和遮罩均逐位一致。

在相同的珊瑚区和共同未越界像素上，用原片锚帧映射到当前原帧的灰度 MAE 核验几何精度。全段共享仿射为 12.39，RAFT 为 5.96。最后两秒从 18.39 降到 6.87。该对照在遮挡门控前计算。

| 方法 | 珊瑚细纹跨帧 NCC | 最后两秒 | 同帧对原片细纹 NCC |
| --- | ---: | ---: | ---: |
| 源 RGB | 0.914 | 0.904 | 1.000 |
| A 原始 GLC | 0.113 | 0.126 | 0.059 |
| C 原片共享仿射 | 0.152 | 0.127 | 0.062 |
| D1 原片 RAFT、GLC 门控 | 0.169 | 0.147 | 0.063 |
| D2 原片 RAFT、原片门控 | 0.336 | 0.214 | 0.070 |
| B0 匹配 D2 细纹幅度的模糊 | 0.131 | 0.145 | 0.065 |

逐帧计分沿用上一轮的源 Farnebäck 对齐和固定区域掩码。D2 的细纹幅度 3.034，B0 为 3.031。D2 最后两秒的珊瑚覆盖约 39.6%，平均融合权重 0.147。把几何换成稠密映射有增益，而替换可见性和低频差门控贡献更大。对源细纹的同帧相似度仍低，反映本轮主要收益在时间连续性。此处 RAFT 仍是运动估计，结果是当前 GLC 历史纹理缓存在单个移动镜头的离线参照。

原速复看时，用户看到右下 D2 的其他区域晃动大幅减轻，尤其是画面下方偏蓝的珊瑚，右侧红珊瑚的改善较少。根据这一反馈，用四个原片帧预先圈出下方蓝珊瑚带 `[120, 142, 320, 246)` 再计分。最后两秒，蓝区细纹跨帧 NCC 从 A 的 0.371 升到 D2 的 0.548，匹配细纹幅度的模糊对照为 0.410。D2 的有效覆盖为 78.9%，平均融合权重 0.453，均高于右侧红区。蓝区同帧对原片细纹 NCC 从 0.049 升到 0.053。这个补测取自同一镜头的观看反馈，蓝区框内也含周围背景。 [蓝珊瑚补测](evidence_20261109/source_dense_flow_limit/blue_coral_review.json)

本机 `02_实验数据/优化_GLC后处理/source_dense_flow_limit_20261109_01/comparison` 中有全画面及珊瑚放大的原速四宫格。 [实验汇总](evidence_20261109/source_dense_flow_limit/summary.json) · [光流收据](evidence_20261109/source_dense_flow_limit/flow.json) · [独立核对](evidence_20261109/source_dense_flow_limit/verification.json)

```powershell
& $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/source_raft_flow.py' --upstream $deflickerSource --output $denseRun
& $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/compare_source_dense_flow.py' --cache-run $cacheRun --flow-run $denseRun --ffmpeg $glcFfmpeg
```

## 沉船与鱼群复测

2026-11-09。选取本地展示资料的源视频 `media3.mp4`，截取 2–8 秒。镜头缓慢移动，鱼群和潜水员会经过船体。源视频为 25 FPS 的展示 MP4，抽到 15 FPS、512×288 后得到 90 个不同的 RGB 帧。先在源帧上固定右侧船体框 `[330, 24, 502, 168]`。用同一 FP32 GLC q0、GOP64、reset32 新编码，独立解码验证 90 帧参考状态。基础文件为 28,212 B、37.616 kbps，C 的共享运动文件为 1,213 B、1.617 kbps。

| 方法 | 全画面细纹跨帧 NCC | 船体整段 | 船体最后两秒 |
| --- | ---: | ---: | ---: |
| 源 RGB | 0.619 | 0.651 | 0.666 |
| A 原始 GLC | 0.279 | 0.281 | 0.295 |
| C 原片共享仿射 | 0.306 | 0.336 | 0.348 |
| D1 原片 RAFT、GLC 门控 | 0.304 | 0.343 | 0.350 |
| D2 原片 RAFT、原片门控 | 0.318 | 0.367 | 0.377 |
| B0 匹配 D2 细纹幅度的模糊 | 0.295 | 0.297 | 0.312 |

D1、D2 沿用礁壁实验的 RAFT 权重、24 次 FP32 迭代、15 帧锚帧和融合参数。D2 在最后两秒的船体覆盖为 60.1%，整段同帧对源细纹 NCC 为 0.133，A 为 0.125。船体的时间一致性改善在新素材上复现，幅度小于上一轮的蓝珊瑚区域。C 有 19 个无效运动步并按原规则重新起锚，因此 C 与 D1 的差异还包含锚帧时序。只在双方锚帧相同的 30 个非锚帧上核验几何，源船体灰度对齐 MAE 从仿射的 6.80 降到稠密光流的 5.76。D1、D2 仍是依赖源视频的离线参照。船体框内也有鱼经过，前景观感可在原速全画面对照中核查。

用户原速复看沉船片段，反馈四个版本几乎看不出区别。船体指标上的提升未达到本段观看要求。

本机 `02_实验数据/优化_GLC后处理/source_dense_flow_shipwreck_20261109_01/flow/comparison` 含全画面及船体放大四宫格。右下为 D2。 [输入收据](evidence_20261109/source_dense_flow_shipwreck/inputs.json) · [实验汇总](evidence_20261109/source_dense_flow_shipwreck/summary.json) · [独立核对](evidence_20261109/source_dense_flow_shipwreck/verification.json)

```powershell
& $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/dense_transfer_inputs.py' --input $shipSource --ffmpeg $glcFfmpeg --assets $roleAssets --output $shipRun
& $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/run_dense_transfer.py' flow --run $shipRun --upstream $deflickerSource
& $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/run_dense_transfer.py' compare --run $shipRun --ffmpeg $glcFfmpeg
```

## 用户提供的三个水下片段

2026-11-09。读取 `video1.mp4`、`video2.mp4`、`video3.mp4`，统一使用各文件前 10 秒。三个输入均为 1280×720、30 FPS。固定到 512×288、15 FPS 后，每条均为 150 个不同的 RGB 帧。按原片先定礁壁、斑点鱼右下方珊瑚和画面下方彩色珊瑚三个观察框。抽样帧分别与此前长片的约 0、12.7、54 秒起的镜头对应。本次直接使用用户提供的文件新编码，三条 FP32 GLC 基础流均通过独立参考校验。

| 素材 | 基础流 B / kbps | 原始 GLC | C 共享仿射 | D1 稠密光流 | D2 稠密光流和原片门控 | 等细纹幅度模糊 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| video1 气泡与礁壁 | 31,963 / 25.570 | 0.434 | 0.484 | 0.490 | 0.646 | 0.463 |
| video2 斑点鱼与珊瑚 | 31,849 / 25.479 | 0.302 | 0.522 | 0.547 | 0.641 | 0.321 |
| video3 鱼群与彩色珊瑚 | 30,995 / 24.796 | 0.713 | 0.749 | 0.754 | 0.831 | 0.727 |

表中为各固定背景框整段的细纹跨帧 NCC。最后两秒，A 到 D2 分别是 0.349 到 0.582、0.198 到 0.595、0.682 到 0.818。D2 的背景同帧对源细纹 NCC 仍在 0.028–0.054。C 的三份运动文件均为 1,993 B、1.594 kbps。D1、D2 沿用 24 次 FP32 RAFT 迭代和 15 帧锚点的离线配方。video2 的固定框评价右下珊瑚，8.4 秒的抽查帧里，D2 鱼背的斑点略软。整段晃动、轮廓和拖影用原速左右对照检查。

原速复看时，用户反馈 video1 珊瑚持续扭曲，整体效果一般。video2 的原始 GLC 已经好看，video3 的左右差异不大。补测 video1 固定框的源运动对齐粗尺度亮度误差，A 为 6.22，D2 为 5.98。video2 左栏经回读核对为 25.48 kbps 的 GLC 独立解码。与原片相比，A 的整帧同帧粗尺度亮度 NCC 为 0.989，细尺度为 0.065。鱼体轮廓和尾鳍姿态在抽查帧中已清楚。video2 最后两秒背景细纹对原片的同帧 NCC 从 0.013 变为 0.014，较大的提升集中在跨帧一致性。本机 video2 的同目录另有 `source_vs_glc.mp4`，左边为同尺寸原片，右边为 A。 [原速反馈和补测](evidence_20261109/source_dense_flow_three_clips/visual_review.json)

六个对照均为 150 帧、15 FPS、10 秒。左边是原始 GLC，右边是 D2。每段的全画面 `full_pair.mp4` 和珊瑚放大 `reef_pair.mp4` 在本机 `02_实验数据/优化_GLC后处理/source_dense_flow_three_clips_20261109_01` 各子目录的 `flow/comparison` 中。 [video1](evidence_20261109/source_dense_flow_three_clips/video1/summary.json) · [video2](evidence_20261109/source_dense_flow_three_clips/video2/summary.json) · [video3](evidence_20261109/source_dense_flow_three_clips/video3/summary.json) · [独立核对](evidence_20261109/source_dense_flow_three_clips/verification.json)

```powershell
foreach ($case in 'video1', 'video2', 'video3') {
    & $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/run_dense_batch.py' prepare --case $case --batch $batch --input-dir $media --ffmpeg $glcFfmpeg --assets $roleAssets
    & $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/run_dense_batch.py' flow --case $case --batch $batch --upstream $deflickerSource
    & $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/run_dense_batch.py' compare --case $case --batch $batch --ffmpeg $glcFfmpeg
}
```

## 稠密运动和可靠区的 2×2 消融

2026-11-09。复用用户提供的 video1、video2 各前 10 秒的独立 GLC 解码和同一 RAFT-Things 模型。所有组的 GLC 历史锚帧、15 帧刷新周期、融合公式和主要评价框相同。接收端仅从自己的 GLC 画面估计 RAFT。共用的保护窗口由接收端 GLC 画面手工圈定，只允许修改指定礁面，锁定主鱼、潜水员和水域。每次从锚帧取样时也检查保护边界。两段分别锁定全画面的 80.8% 和 92.1%，所有锁定像素在无损合成 RGB 中逐位等于基础层。

| 组别 | 运动 | 可靠区 | video1 额外字节 / kbps | video2 额外字节 / kbps |
| --- | --- | --- | ---: | ---: |
| A | 接收端 RAFT | 接收端 GLC 低频差 | 0 / 0 | 0 / 0 |
| B | 接收端 RAFT | 发送端原片可靠区 | 13,801 / 11.041 | 2,810 / 2.248 |
| C | 发送端原片 RAFT | 接收端 GLC 低频差 | 32,515 / 26.012 | 10,741 / 8.593 |
| D | 发送端原片 RAFT | 发送端原片可靠区 | 46,316 / 37.053 | 13,551 / 10.841 |
| O | 原片全分辨率 RAFT | 原片全分辨率可靠区 | 离线参照 | 离线参照 |

接收端逐组独立运行，从各自收件目录读取实际封装字节。A 收件目录为空，B、D 的可靠区文件逐字节一致，C、D 的运动文件逐字节一致。源端只编码可编辑礁面附近的 64×36 网格，运动格点量化到 1 像素，可靠度量化为八级，格点经差分和 zlib 压缩，文件绑定基础流哈希、时间轴和 CRC。两段基础流为 25.570 和 25.479 kbps，基础流与辅助流均按各自的 40 kbps 上限计费。O 使用相同的历史 GLC 纹理和保护规则，在本机离线使用未量化的原片信息。

| 固定背景区最后两秒的细纹跨帧 NCC | 原始 GLC | A | B | C | D | O |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| video1 礁壁 | 0.349 | 0.360 | 0.411 | 0.367 | 0.480 | 0.503 |
| video1 上方黄珊瑚 | 0.213 | 0.226 | 0.293 | 0.241 | 0.466 | 0.490 |
| video1 下方青蓝礁面 | 0.433 | 0.457 | 0.533 | 0.461 | 0.577 | 0.623 |
| video2 右下珊瑚 | 0.198 | 0.258 | 0.276 | 0.337 | 0.377 | 0.411 |

video1 上方黄珊瑚的原片锚帧对齐误差从接收端 RAFT 的 14.43 降到实际收到的运动网格的 5.49。只给运动时 C 相比 A 的后两秒 NCC 只高 0.015。在同一发送端可靠区下，D 相比 B 高 0.173。video2 的 C 相比 A 高 0.080，D 比 B 高 0.101。两段中 B 与 D 都有差距，可靠区和运动各自的贡献随场景变化。

video1 上方黄珊瑚最后两秒的粗尺度时序误差，A 为 6.97、D 为 6.11、O 为 6.07。D 的两个最大峰落在零基 64 和 128 帧，约 4.27 秒和 8.53 秒，均为 GLC I 帧。两处 I 帧的 D 平均误差为 11.50，普通 P 帧为 3.81，O 在两处 I 帧仍为 11.53。帧型共现可作为结构跳变的定位线索。video2 保护区包含整段主鱼，抽查 8.4 秒四组鱼身表现一致。保护范围以本轮人工窗口为准。

用户复看四宫格，反馈 A 和 D 肉眼近乎无差别。回读全画面预览并核对 A、D 各自对应的无损输出。最后两秒，两段分别有 92.8% 和 94.2% 的全画面时空像素在 A、D 间逐位相同，最大 RGB 通道差达到 8 灰阶的像素分别占 2.9% 和 3.3%。video1 放大的上方黄珊瑚框里仍有 70.1% 的时空像素完全相同。原速验收记为 D 无明确可见增益。 [观看反馈与 A/D 像素核对](evidence_20261109/dense_gate_ablation/visual_review.json)

本机 `02_实验数据/优化_GLC后处理/dense_gate_ablation_20261109_01` 中，每段有四组全画面和固定区原速四宫格，以及原始 GLC、A、D、O 的离线对照。 [video1](evidence_20261109/dense_gate_ablation/video1/summary.json) · [video2](evidence_20261109/dense_gate_ablation/video2/summary.json) · [独立核对](evidence_20261109/dense_gate_ablation/verification.json)

```powershell
foreach ($case in 'video1', 'video2') {
    & $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/run_gate_ablation.py' fit-receiver --case $case --batch $batch --run $ablationRun --upstream $deflickerSource
    & $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/run_gate_ablation.py' send --case $case --batch $batch --run $ablationRun
    foreach ($variant in 'A', 'B', 'C', 'D') {
        & $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/run_gate_ablation.py' receive --case $case --variant $variant --batch $batch --run $ablationRun
    }
    & $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/run_gate_ablation.py' upper --case $case --batch $batch --run $ablationRun
    & $glcPython -B -X utf8 '21_优化阶段_GLC解码后处理/compare_gate_ablation.py' --case $case --batch $batch --run $ablationRun --ffmpeg $glcFfmpeg
}
```

## 评价与复现

使用同源同尺寸的无损 RGB 计算指标。评估光流及有效掩码只用源 RGB，所有方法在同一掩码上计分。时序指标比较两帧经运动补偿后的灰度高频变化与源画面对应变化的绝对误差。珊瑚区高频幅度、空间误差与运动区 PSNR 一起报告。珊瑚和鱼的观看结论以原速对照为准。

在本机现有 GLC Python 环境中运行，传入配套的 FFmpeg 可执行文件路径和一个新输出目录。

```powershell
python -B -X utf8 '21_优化阶段_GLC解码后处理/run_p0.py' --ffmpeg $glcFfmpeg --output $glcTrialOut
python -B -X utf8 '21_优化阶段_GLC解码后处理/run_p0.py' --ffmpeg $glcFfmpeg --output $glcTrialMeanOut --fusion mean
python -B -X utf8 '21_优化阶段_GLC解码后处理/run_p0.py' --ffmpeg $glcFfmpeg --output $glcTrialFiveOut --compare-five
python -B -m unittest discover -v -s '21_优化阶段_GLC解码后处理'
```

脚本按工作区 `01_项目/GLC后处理/接口与基线.json` 定位无损帧，检查既有数据哈希，并使用 `chunks[].frames` 确定分块有效帧。工作区布局不同时可传入 `--workspace`。每段输出两种增强 RGB、逐帧 CSV、完整四宫格 `compare.mp4` 和两倍局部对照 `zoom.mp4`。上排始终为源画面和原始 GLC。首轮下排为两帧和三帧，复测下排为均值三帧和五帧。18 个预览都用 FFprobe 核对了帧数、帧率和时长。

本机完整输出位于工作区 `02_实验数据/优化_GLC后处理/temporal_p0_20261108_01`，三个运行目录是 `trial`、`trial_mean`、`trial_5`。两种三帧融合共用 GLC 输入和评分掩码，两帧 RGB 结果逐位一致。三段都是同一原片中的开发片段。

- 保守三帧，原执行版本 `ea26a9e`。 [汇总](evidence_20261108/consensus/summary.json) · [逐帧 CSV](evidence_20261108/consensus/per_frame.csv) · [独立核对](evidence_20261108/consensus/verification.json)
- 均值三帧。 [汇总](evidence_20261108/mean/summary.json) · [逐帧 CSV](evidence_20261108/mean/per_frame.csv) · [独立核对](evidence_20261108/mean/verification.json)
- 五帧复测。 [汇总](evidence_20261108/window5/summary.json) · [逐帧 CSV](evidence_20261108/window5/per_frame.csv) · [独立核对](evidence_20261108/window5/verification.json)
