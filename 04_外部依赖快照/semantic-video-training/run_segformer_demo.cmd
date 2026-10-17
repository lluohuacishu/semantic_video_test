@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
if not exist ".venv-segformer\Scripts\python.exe" goto missing
if not exist "runs\suim_segformer_b0_20260915\segmentation_best.pt" goto missing
echo 正在加载 SegFormer-B0 的 SUIM 八类权重。
".venv-segformer\Scripts\python.exe" demo_server.py --checkpoint runs\suim_segformer_b0_20260915\segmentation_best.pt --data-dir runs\segformer_demo_console --port 8768 --open %*
set "segformer_demo_result=%errorlevel%"
if not "%segformer_demo_result%"=="0" if not defined SEMANTIC_NO_PAUSE pause
exit /b %segformer_demo_result%

:missing
echo 找不到 SegFormer 环境或训练权重，请查看 docs\SegFormer-B0微调.md。
if not defined SEMANTIC_NO_PAUSE pause
exit /b 1
