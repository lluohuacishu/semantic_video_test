# GLC 纹理稳定性优化

本轮针对生成纹理闪动，先比较原GLC、固定更高码率和轻量因果时域后处理；纠错留后。原生产网页与模型未替换。

- [首轮结果与限制](纹理稳定性首轮结果_20261027.md)
- [小型实验数据](evidence_20261027/summary.json)
- [滤波器](temporal_filter.py)、[实验入口](run_trial.py)、[评价与视频](analyze_trial.py)、[本地查看器](serve_viewer.py)

本机对比页：http://127.0.0.1:8780/ 。桌面入口：`临时/材料/纹理稳定性对比.cmd`。

在本目录检查：`python -m unittest -v test_temporal_filter test_viewer_server`。需要现有NumPy/OpenCV；HTTP检查可独立用`python -S -m unittest -v test_viewer_server`。

复现须先准备现有GLC环境与计划中的历史素材：`python run_trial.py --output <新目录>`，完成后执行`python analyze_trial.py --run <该目录>`。这不是自带模型/视频的一键部署包。结果已完成，不自动重复实验或扩大参数搜索。
