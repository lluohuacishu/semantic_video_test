# S8I 锚点与接收时间实验小证据

`execution_plan.json` 与 `profile_order.json` 是候选接收前冻结的计划。`evaluation_summary.json`、逐帧 CSV 和原速预览核验描述四档结果；每档接收回执记录输入、实际 S8I 字节数、组件计时和输出哈希。

`flow_outputs.json` 单独给出三份接收端本地光流数组的字节数及 SHA-256。`replay_verification.json` 核验 24 次当前锚点结果与用户已认可 S8I 的 RGB/alpha/shared 数组逐位一致。`decision.json` 保留 S8I 为当前基线，/12 仍待用户原速确认。

`source/` 保存本轮脚本、源码快照索引、运行环境及执行时代码哈希说明。大数组和 RAFT 权重留在本机运行目录；光流与掩码为接收端缓存，不是传输字节。

完整报告：[锚点与接收时间联合实验](../../锚点与接收时间联合实验_20261111.md)。展示副本：[原速观看目录](../../../../../../05_成果与观看/GLC_接收时间与锚点实验_20261111/README.md)。
