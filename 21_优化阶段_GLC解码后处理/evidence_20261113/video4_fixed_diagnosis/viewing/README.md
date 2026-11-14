# video4 密集珊瑚：原速观看

完整处理用户提供的6.9秒海龟/珊瑚片段，30→15 FPS后104帧，观看6.933秒；未补到10秒。固定Q9与GS25，没有video4专属调参。总码率38.2788 kbps，GS25无额外传输字节，尚未采用。

- [优先：左Q9 / 右GS25](video4_Q9_vs_GS25.mp4)
- [全画面四宫格：原片 / 原GLC；Q9 / GS25](video4_source_raw_Q9_GS25.mp4)
- [密集珊瑚放大四宫格](video4_coral_crop.mp4)
- [海龟所在宽框放大四宫格](video4_turtle_crop.mp4)
- [原片 / 收到的引导视频](video4_source_vs_guide.mp4)

放大四宫格布局同全画面：上原片/原GLC，下Q9/GS25。宽框包含背景，不是语义分割。

GS25相对Q9珊瑚区低频时序误差-18.23%、RGB误差-8.17%，但动态幅度继续降低。原GLC已经存在珊瑚形态偏差，GS25没有完整恢复；重点看细部形状、海龟鳍部及运动，不仅看稳定程度。用户原速反馈待记录。

[完整报告](../../01_项目/代码/semantic_video_test/21_优化阶段_GLC解码后处理/video4密集珊瑚_Q9与GS25固定参数诊断_20261113.md) · [五列全图静帧](video4_whole_scene_contact.png) · [珊瑚静帧](video4_coral_contact.png) · [海龟静帧](video4_turtle_contact.png) · [核验](../../02_实验数据/优化_GLC后处理/q9_video4_diagnosis_20261113_02/verification.json)

全部对照共享同一新编码基础流。本次短片不等同旧82.6秒长片的同位置重建结果，未验证长片分块/切镜。视频为硬链接，编辑前另存副本；MP4大小不能作为通信码率。
