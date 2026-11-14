# Q9 引导结构融合：原速观看（2026-11-13）

本轮优先看video1/2/3的整体可看、当下鱼形与运动。**GS25仅为温和观看候选，未采用。** 它把已传输guide的低频残差加入Q9输出，无新增传输字节；与历史仅原GLC像素的Q9机制不同。

## 优先看三段双列

- [video1：左Q9_R25 / 右GS25](video1_Q9_vs_GS25.mp4)
- [video2：左Q9_R25 / 右GS25](video2_Q9_vs_GS25.mp4)
- [video3：左Q9_R25 / 右GS25](video3_Q9_vs_GS25.mp4)

三段均完整10秒、15 FPS。主要看鱼尾/前景鱼形是否更自然、纹理是否被削弱、运动是否变钝；不能仅凭平均误差下降判断成功。

## 原片与机制对照

- [video1八宫格](video1_all_eight.mp4)
- [video2八宫格](video2_all_eight.mp4)
- [video3八宫格](video3_all_eight.mp4)

每行左→右：原片/原GLC；guide/Q9_R25；CSM50/GB25；GS25/GS50。guide只作诊断参照，CSM50为前轮候选。GB25直接混合更软，GS50修正更强。

- [video1：左Q9 / 右强档GS50](video1_Q9_vs_GS50.mp4)
- [video2：左Q9 / 右强档GS50](video2_Q9_vs_GS50.mp4)
- [video3：左Q9 / 右强档GS50](video3_Q9_vs_GS50.mp4)

## 主体放大

- [video2主鱼和鱼尾](video2_current_fish_crop.mp4)
- [video3前景鱼](video3_current_fish_crop.mp4)

六宫格每行左→右：原片/guide；Q9/GS25；GS50/GB25。宽框包含背景。

## 等细节模糊诊断

- [video1匹配模糊四宫格](video1_matched_blur_control.mp4)
- [video2匹配模糊四宫格](video2_matched_blur_control.mp4)
- [video3匹配模糊四宫格](video3_matched_blur_control.mp4)

布局：原片/Q9；等细节模糊/GS25。BLUR_MATCH只为匹配全片平均细纹理幅度的诊断，不是部署候选。GS25低频误差较低，但细纹理时序误差较高、动态幅度更小。

## 静帧与报告

- [video1全图拼图](video1_whole_scene_contact.png) · [4.07秒模糊对照](video1_matched_blur_frame61.png)
- [video2全图拼图](video2_whole_scene_contact.png) · [4.07秒模糊对照](video2_matched_blur_frame61.png)
- [video3全图拼图](video3_whole_scene_contact.png) · [4.07秒模糊对照](video3_matched_blur_frame61.png)

已看静图，用户原速反馈待记录；未证明鱼消失/残像已修复。礁石、长片本轮未跑，video4等待用户提供。

[完整中文报告](../../01_项目/代码/semantic_video_test/21_优化阶段_GLC解码后处理/Q9已传引导结构融合与纹理对照_20261113.md) · [结构化结果](../../02_实验数据/优化_GLC后处理/q9_received_structure_20261113_01/round_result.json) · [主核验](../../02_实验数据/优化_GLC后处理/q9_received_structure_20261113_01/verification.json) · [模糊核验](../../02_实验数据/优化_GLC后处理/q9_received_structure_20261113_01/blur_control/verification.json)

媒体为硬链接，编辑前请另存副本。观看MP4体积不代表通信码率。正式采用配置保持S8I + CURRENT_I12。
