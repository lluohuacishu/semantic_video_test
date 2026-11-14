# Q9 残像退出诊断观看（2026-11-12）

本轮候选仅用于观察退出旧纹理的收益与代价，**E32 / EL12 / C32均未采用**。Q9_R25仍是研究参照，正式采用配置未改。用户原先对三段Q9的正面评价保持。

## 优先看这四条

1. [礁石3–5秒放大六宫格](reef_spotted_fish_user_event_crop_all.mp4)：用户4.07秒对应本裁剪播放约1.07秒，重点看两条小鱼离开后的旧位置。
2. [video2鱼尾3–5秒放大六宫格](video2_user_event_crop_all.mp4)：这是本轮选择的检查区间，用户没有指定video2时间。
3. [礁石完整10秒，左Q9_R25 / 右EL12](reef_spotted_fish_Q9_vs_EL12.mp4)。
4. [video2完整10秒，左Q9_R25 / 右EL12](video2_Q9_vs_EL12.mp4)。

六宫格布局（从上到下，每行左→右）：原视频 / 原GLC；Q9_R25 / E32；EL12 / C32。所有视频15 FPS原速播放，裁剪只放大空间，不改变播放速度。右侧EL12有诊断/未采用标签。

## 五片完整六宫格

- [video1：10秒](video1_all_six.mp4)
- [video2：10秒](video2_all_six.mp4)
- [video3：10秒](video3_all_six.mp4)
- [礁石斑点鱼：10秒](reef_spotted_fish_all_six.mp4)
- [海草斑点鱼：10秒](seagrass_spotted_fish_all_six.mp4)

## 静帧辅助

- [礁石4.07秒旧位置细节](reef_exact4p07_footprints.png)
- [礁石3.53 / 4.07 / 4.60秒](reef_spotted_fish_user_event_contact.png)
- [video2鱼尾3.33 / 4.07 / 4.67秒](video2_user_event_contact.png)

本轮已完成15组候选。礁石固定旧位置误差下降，但五片均有时序误差回退；video2尾部改善很小，不能说已经解决。静图不替代原速验收，候选用户反馈仍待记录。

[完整报告](../../01_项目/代码/semantic_video_test/21_优化阶段_GLC解码后处理/Q9残像定位与局部退出实验_20261112.md) · [结果和无损输出目录](../../02_实验数据/优化_GLC后处理/q9_ghost_exit_20261112_04/round_result.json) · [媒体核验](../../02_实验数据/优化_GLC后处理/q9_ghost_exit_20261112_04/previews/verification.json) · [媒体SHA](media_manifest.json)

本目录媒体是运行目录的硬链接；如需编辑请先另存副本，避免改变原实验观看文件。MP4体积不能用作语义通信码率。
