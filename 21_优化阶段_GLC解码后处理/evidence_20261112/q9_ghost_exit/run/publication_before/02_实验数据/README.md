# 实验数据索引

更新于 2026-11-10。当前工作为 GLC 第 21 阶段纹理稳定性研究。查看 I 帧过渡、GOP 控制、背景复用和既有短段的本机目录。 [当前实验索引](优化_GLC后处理/README.md) · [原速观看](../05_成果与观看/GLC_时序稳定性_20261110/README.md) · [项目交接](../01_项目/代码/semantic_video_test/交接文档.md)

## 历史实验数据与观看入口

2026-10-27 流水线实验见 [流水线优化数据](优化_流水线编解码/README.md)和[首个 16 帧原型](优化_流水线编解码/pipeline_probe_20261027_01/流水线原型结果.md)。

D 轮曾用 ≤40 kbps 和逐片增幅 ≤10% 的试验口径。记录保存在 [D 轮本地索引](D_语义目标增强/README.md)和 [D2 带标签观看页](D_语义目标增强/D2_lowrate_20261022_r1/viewing/index.html)。下方 B/C 同属历史数据。

## B4 四档实际发送文件

| 档位 | 包大小 | 文件 |
|---|---:|---|
| 细节与流畅兼顾 | 78,758 B | [detail_motion](B_GLC通信与压缩/B4_四档最终结果/container_v2_baseline_repack/baseline_v2.glcv) |
| 保留空间细节 | 48,985 B | [detail](B_GLC通信与压缩/B4_四档最终结果/fps_only_540p_12_5/stream.glcv) |
| 保留运动流畅度 | 44,962 B | [motion](B_GLC通信与压缩/B4_四档最终结果/resolution_only_360p_25/stream.glcv) |
| 优先小体积 | 26,397 B | [compact](B_GLC通信与压缩/B4_四档最终结果/joint_360p_12_5/received/stream.glcv) |

程序读取新路径可用[档位文件索引](档位文件索引.json)。历史 pilot_status.json 与 selectable_presets.json 中的旧路径保持原样，兼容入口仍可用。

## 观看视频

三段均为原视频、GLC 基线、对应降级档的同步预览；MP4 只供观看，不是发送包。

- [仅降帧率](B_GLC通信与压缩/B4_四档最终结果/comparisons/fps_only_540p_12_5_comparison.mp4)
- [仅降分辨率](B_GLC通信与压缩/B4_四档最终结果/comparisons/resolution_only_360p_25_comparison.mp4)
- [同时降低](B_GLC通信与压缩/B4_四档最终结果/comparisons/joint_360p_12_5_comparison.mp4)

## C1 加速结果

C1 已完成，原四档码流与重建不变；[阶段报告](../01_项目/代码/semantic_video_test/08_新阶段C_编解码加速与网页展示/C1阶段报告_20261019.md)记录科研约 30%、展示约 37% 的同条件离线提速。

- [C1 证据索引](C_编解码加速与网页展示/README.md)：正式矩阵、补测、门槛、日志与中断现场。
- [compact 展示模式独立解码预览](C_编解码加速与网页展示/C1_final_20261019/decode_display_1_c1/receiver/reconstruction_preview.mp4)。
- [detail 预览](C_编解码加速与网页展示/C1_recovery_20261019/detail_decode/receiver/reconstruction_preview.mp4)、[motion 预览](C_编解码加速与网页展示/C1_recovery_20261019/motion_decode/receiver/reconstruction_preview.mp4)、[detail_motion 预览](C_编解码加速与网页展示/C1_recovery_20261019/detail_motion_decode/receiver/reconstruction_preview.mp4)。

MP4 仍只供观看。C1 不证明实时、语义净收益或真实水声时效；C2 已完成有限验证，C3–C4、C5 本机收尾已经完成，详见当前交接文档。

## 历史实验

C2 已于 2026-10-20 完成有限验证，当前无运行任务。最终组合同期 1.991×，严格 2× 未达到；最新回归 `C2_checks_20261020_r4` 为 114 项通过。第二素材旧 r1 和组合短门槛旧 r1 的失败现场保留，完整新配对已在新目录完成。详见 [C2 阶段报告](../01_项目/代码/semantic_video_test/08_新阶段C_编解码加速与网页展示/C2阶段报告_20261020.md)。

[A 阶段](A_压缩路线选型)、[B 阶段](B_GLC通信与压缩)、[学姐展示材料提取](资料核对/学姐GLC压缩展示)。含“首次运行_保留”的目录是中断或失败证据，不当作最终通过结果，也不要删除。B2 在此通过目录链接访问。
