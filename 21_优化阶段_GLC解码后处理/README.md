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
