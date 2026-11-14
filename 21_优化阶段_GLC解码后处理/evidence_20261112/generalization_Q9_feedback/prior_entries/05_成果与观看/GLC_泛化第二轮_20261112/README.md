# GLC 泛化第二轮原速观看

2026-11-12。**没有候选通过全部预设数值筛选；Q6_R25 是更符合适度改善目标的平衡观看候选，尚未验收。** 主观反馈待记录，尚未采用。

最新目标：video2适度改善即可，用户澄清并不是特别难看。请以此为观看基调。

本轮Q6/Q9 × T3/R25/D25，四段共24组新输出，统一参数。R25对门限附近的可疑复用最多降权25%；D25再考虑原GLC低频差异，仍最多25%。Q9有额外字节成本。

**双列左当前Q6_T3、右标题候选**；四宫格左上源、右上原GLC、左下当前、右下候选。每条10秒/15FPS。

| 素材 | 主要双列 | 主要四宫格 | 另一候选双列 |
| --- | --- | --- | --- |
| video1 | [当前 vs Q6_R25](video1_current_vs_Q6_R25.mp4) | [四宫格](video1_source_raw_current_Q6_R25.mp4) | [当前 vs Q9_R25](video1_current_vs_Q9_R25.mp4) |
| video2 | [当前 vs Q6_R25](video2_current_vs_Q6_R25.mp4) | [四宫格](video2_source_raw_current_Q6_R25.mp4) | [当前 vs Q9_R25](video2_current_vs_Q9_R25.mp4) |
| video3 | [当前 vs Q6_R25](video3_current_vs_Q6_R25.mp4) | [四宫格](video3_source_raw_current_Q6_R25.mp4) | [当前 vs Q9_R25](video3_current_vs_Q9_R25.mp4) |
| archive82_first10s | [当前 vs Q6_R25](archive82_first10s_current_vs_Q6_R25.mp4) | [四宫格](archive82_first10s_source_raw_current_Q6_R25.mp4) | [当前 vs Q9_R25](archive82_first10s_current_vs_Q9_R25.mp4) |

固定四段素材共新生成24组，另引用上一轮4组Q12_T3参照。Q6_R25相对当前video2全帧时序误差变化-1.10%，实际总码率30.9904→30.9904kbps。没有统一候选通过全部数值筛选，主观验收尚待记录。

[video2六档完整矩阵](video2_all_six_profiles.mp4)：按行优先Q6_T3、Q6_R25、Q6_D25、Q9_T3、Q9_R25、Q9_D25。

目录也保留另一候选的各片四宫格和四张抽帧图。对照时注意鱼体/背景交界、纹理替换、换锚跳变及video1/3原本稳定的背景；边界无需达到“完全消失”才算有进展。

[完整报告](../../01_项目/代码/semantic_video_test/21_优化阶段_GLC解码后处理/泛化改进第二轮_温和可靠性与中档引导_20261112.md) · [核验](../../02_实验数据/优化_GLC后处理/generalization_reliability_20261112_02/verification.json) · [SHA清单](manifest.json)

历史短段开发实验，未验证独立留出/长视频/实时。MP4只供观看，指标来自无损数组。文件为原结果硬链接，剪辑请先另存副本；未提交/推送。
