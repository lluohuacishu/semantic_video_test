# 低码率 L 阶段复盘证据

2026-11-10。只读分析已有 RGB、alpha、像素安全区和实际辅助包；没有生成新优化候选、训练模型或改变原始输入。主报告为 [阶段复盘与下一步优先级](../../阶段复盘与下一步优先级_20261110.md)，下一轮 [冻结计划](../../plan_lowrate_next_20261110.json) 的状态仍为 planned_not_run。

本机运行目录相对工作区为 02_实验数据/优化_GLC后处理/lowrate_retrospective_20261110_01。

| 文件 | 内容 |
| --- | --- |
| summary.json | 主分数复现、空间与时间支持损失、时序分组、共同内部及完整字节构成 |
| temporal_frames.csv | 149 个相邻帧对的误差和沿运动对齐后的权重变化 |
| support_groups.csv | 按 GP/L 持续满权重状态划分的误差 |
| current_frame_causes.csv | 当前帧像素拒绝、4×4 连带拒绝、五帧交集拒绝及开启状态 |
| common_interior.csv | 排除换锚、远离 GP/L 双方边界 >14px 的共同内部 |
| verification.json | 五条已存分数重现、分母和分区完整性 |
| timing_sensitivity.json | 排除已知 I 帧相邻对及周期边界后的敏感性检查 |
| analyze_lowrate_bottlenecks.py | 本次运行的精确源码快照，含原始换行 |
| manifest.json | 上述八份文件的原始 SHA-256 与字节数 |

主分母为 1,499,191 像素帧对。按方法误差和除以固定评价像素数计算 MAE；条件分组只定位残差，不能把 L−GP 净增误差份额解释为某因素造成的总误差比例。GP 与 L 同时改变空间粒度和时间交集，尚欠两个受控对照。共同内部只占主分母 11.68%。

时间分类事先由锚点和五帧组确定。换锚类别包含过渡和退出帧；非换锚仍包含 I/reset。timing_sensitivity.json 从 temporal_frames.csv 重新加总，排除的零基帧对终点写在 JSON 内。周期 32 检查是保守配置敏感性，不断言每个边界均发生实际 reset。源相邻对应仅供评价，未提供给接收端。

从仓库运行 21_优化阶段_GLC解码后处理/analyze_lowrate_bottlenecks.py，传入 --workspace 与新的 --output 目录，可重算 summary 和四份 CSV；脚本拒绝覆盖已存在目录。verification 和 timing_sensitivity 是对 CSV 的独立加总核对，后者公式为 sum(method_sum)/sum(pixels)。复现需本机已归档的大数组、辅助文件以及同阶段依赖代码，不把视频或数组放入 Git。

本目录保留原字节，manifest 可与运行目录及 Git 索引逐项核对。历史实验输出与旧计划保持不变。
