# C2 原生 byte-rANS 候选（显式启用）

`rans_byte.h` 原文来自 Fabian Giesen / rygorous 的 public-domain 实现：
https://github.com/rygorous/ryg_rans/blob/c9d162d996fd600315af9ae8eb89d832576cb32d/rans_byte.h

- 固定上游提交：`c9d162d996fd600315af9ae8eb89d832576cb32d`。
- 原文件 SHA-256：`66e9521b4485c1b36ee7527f1a6cd7c9fe7f8495f81420602fbdc26b7273bea6`。
- 保留原始注释和 public-domain 声明，不修改上游文件。
- 本地 `glc_rans.c` 使用其 `RansEncInit/Put/Flush` 和 `RansDecGet/AdvanceStep`；接收端自行执行有边界检查的归一化，避免对坏输入调用不检查末尾的读取函数。
- 本地头、CDF、escape、归一化常数、字节序及 `ENTROPY_ID` 不变。CDF 和浮点 scale 索引仍由原 Python/NumPy 生成。
- 用现有 MinGW GCC 显式构建到全新实验目录；不自动安装、升级环境或改动参考入口。DLL 和构建记录绑定源码及哈希，DLL 不进入 Git。
- 本适配不使用编码端 reciprocal 优化、查表解码大表或 GPU 编译；先验证最小改动。

2026-10-19 C1 展示路径的 66 帧短诊断实测熵相关占两端约 21.8%，据此选择候选 A。该比例不能替代完整 125 帧性能配对或其他三档结论。
