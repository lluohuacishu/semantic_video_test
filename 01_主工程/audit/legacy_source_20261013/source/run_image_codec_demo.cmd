@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
if not exist ".venv-segformer\Scripts\python.exe" goto missing
if not exist "checkpoints\pretrained\image-codec\provenance.json" goto missing
echo 正在启动图片压缩与 TCP 文件传输演示。
".venv-segformer\Scripts\python.exe" demo_server.py --checkpoint runs\suim_segformer_b0_20260915\segmentation_best.pt --data-dir runs\image_codec_console --port 8769 --default-mode image_codec --open %*
set "image_codec_result=%errorlevel%"
if not "%image_codec_result%"=="0" if not defined SEMANTIC_NO_PAUSE pause
exit /b %image_codec_result%
:missing
echo 找不到图片编码环境或权重，请先运行 setup_image_codec.cmd。
if not defined SEMANTIC_NO_PAUSE pause
exit /b 1
