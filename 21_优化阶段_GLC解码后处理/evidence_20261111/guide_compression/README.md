# 固定 T3 的结构引导压码率证据

execution_plan.json 是初始冻结计划；video1/S8I 与 S8C4 的 plan.json 是补整数档和色度档的预先计划。
evaluation_summary.json 汇总全部六档；evaluation 内保存逐帧/逐区域评分，原速视频身份核验在 previews/verification.json。
encoder_audit.json 保留全部编码尝试的整数码率、完整字节、选中状态及第一遍日志头；原始日志、实际码流与大数组在本机运行目录。
每档 send/flow_receive/receive/verification 记录发送、独立接收和不变 T3 重放。
decision.json 与 visual_review.json 区分已认可 A12 和新候选待观看。source_final 保留仓库相对结构、RAFT 源码、环境及哈希。
release_verification.json / viewing_copies.json 核验冻结输入和展示副本；manifest.json 覆盖本目录文件（不包含自身）。
