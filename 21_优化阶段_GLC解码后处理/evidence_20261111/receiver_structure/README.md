# 接收端零辅助验证证据

execution_plan.json 为候选计算前冻结计划和 56 份输入哈希。
各 video*/RR/receipt.json 是无源接收端回执；SS/SR/RS 仅作离线诊断。
各 evaluation/summary.json 与 CSV 保存同分母无损评分、历史基线复现、覆盖率、运动差异及原速预览核验。
prediction_byte_probe 是保持 K 精确输出的乐观差值字节探针，尚非新传输格式。
source_final 保存精确代码与 SHA；release_verification.json 保存最终核验。
大体积 RGB、光流及视频保留在运行和观看目录，未复制入仓库。
