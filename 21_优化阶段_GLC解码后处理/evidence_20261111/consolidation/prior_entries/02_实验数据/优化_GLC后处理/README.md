# GLC 第 21 阶段本机实验索引

最新运行：`compact_support_20261110_01`。L三段辅助21.04/20.46/15.53 kbps，均低于本体；统一核验与同总字节上限H.265完成。主观反馈不能由PSNR替代，详见运行根`visual_review.json`。

更新于 2026-11-10。各运行目录保留原始路径及完整证据。观看用单独归集的 MP4，计算用运行目录下的无损 RGB 与元数据。 [最新整帧与跨素材](../../05_成果与观看/GLC_整帧与跨素材_20261110/README.md) · [已认可局部 U](../../05_成果与观看/GLC_珊瑚持续结构10秒_20261110/README.md) · [第 21 阶段报告](../../01_项目/代码/semantic_video_test/21_优化阶段_GLC解码后处理/README.md)

| 主题 | 主要目录 |
| --- | --- |
| 最新因果消融 | [固定运动与像素/块门控](fullframe_factorial_20261110_01)，逐位复现 E；DP/DB/GP/GB 的珊瑚 MAE 为 1.819/2.972/1.972/3.084，含拒绝原因、来源权重与预算测算；纯离线诊断 |
| 最新瓶颈诊断 | [现有输出重分析](fullframe_strategy_diagnosis_20261110_02)，含稠密离线／U／E／F 的共同分组、对齐后开关变化和运动近似差；没有生成新候选 |
| 最新整帧修正 C–F | [三素材修正运行](fullframe_reference_refined_20261110_01)，各 150 帧、15 FPS；F 的 video2/3 边界明显减轻，video2 残留可接受，video1 相对旧 U 回归未解决。含实际文件、独立接收、`source_final` 和统一核验 |
| 整帧初版 A/B | [三素材初始运行](fullframe_reference_three_clips_20261110_01)，保留 A 未通过的原速反馈和 B 颜色匹配对照 |
| 已认可局部 U | [实际封装与独立接收](video1_persistent_aux_20261110_01)，基础／辅助 31,963／42,793 B，各 25.57／34.23 kbps；原速反馈确认与 T 接近且比原始更稳，继续作为 video1 回归基线 |
| 10 秒完整结构与换锚 | [M/F/N2/T 对照](video1_persistent_long_20261110_01)，保留旧参考失效、硬换锚与过渡消融，T 获正面原速反馈 |
| 23 帧完整结构与 E/G 旁路 | [C/C2、S＋L、R/R2 与旁路](video1_persistent_structure_20261110_01)，C2 获“分界改善，整体更稳”的反馈 |
| 最新 I 帧保真和显示过渡 | [video1 过渡](video1_boundary_transition_20261110_01)，两组 RGB、权重、四帧窗口、退出帧、对照视频与回读 |
| GOP 位置控制 | [GOP 56 对照](video1_gop_control_20261110_01)，含独立码流、解码与固定区域比较 |
| 普通 P 帧张量与早期 GOP/reset | [早期礁壁诊断](reef_decoder_diagnosis_20261108_02)，`p_capture`、`p_evaluation` 与 `boundary_evaluation` |
| video1 结构带、长锚点 | [S 与 L 对照](structure_memory_video1_20261109_01) |
| 实际辅助运动与可靠区 | [A/B/C/D 与 O](dense_gate_ablation_20261109_01)，video1 和 video2 的文件封装、独立接收及固定区评价 |
| 三素材基线和离线光流 | [三素材运行根](source_dense_flow_three_clips_20261109_01)，各自的 `source.npy`、`base.glcsnd`、`base_receive`、`flow` |
| 其他稠密光流测试 | [移动礁壁](source_dense_flow_limit_20261109_01) · [沉船与鱼群](source_dense_flow_shipwreck_20261109_01) |
| 自主缓存与 H.265 | [共享运动对照](cache_motion_controls_20261109_01)，含零辅助、等细纹幅度模糊与 H.265 三档字节对照 |
| 双流纹理与缓存 | [单源参考](coral_reference_sidecar_20261108_01) · [多源参考 01](coral_reference_sidecar_large_20261108_01) · [多源参考 02](coral_reference_sidecar_large_20261108_02) · [混合 33 块](coral_reference_hybrid_20261109_01) |
| 初期滤波与模型 | [两至五帧和局部修复](temporal_p0_20261108_01) · [Deflicker](deflicker_reef80_20261108_01) · [SeedVR2](seedvr2_reef80_20261108_01) |
| 水下观感 | [UVE-Net、UnDIVE 与交付对照](underwater_enhancement_20261108_01/comparison_verified) |
| 原片检查和保护区 | [三素材](three_source_review_20261109) · [沉船等素材](new_source_review_20261109) · [消融保护区](dense_gate_receiver_annotation_20261109) · [I 帧过渡保护区](video1_boundary_protection_review_20261110) |
| 早期边界缓存 | [早期缓存诊断](reef_boundary_memory_20261108_01) |

源码、证据小表及复现命令集中在实验分支。video1 新一轮可直接复用原有 150 帧源 RGB、两份独立解码与双向原片 RAFT。 [素材和环境定位](../../01_项目/代码/semantic_video_test/05_资料与索引/会话收束_20261110/文件索引.md)
