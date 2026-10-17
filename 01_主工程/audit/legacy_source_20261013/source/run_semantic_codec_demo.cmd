@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
if not exist ".venv-segformer\Scripts\python.exe" goto missing
if not exist "runs\suim_joint_semantic_codec_20261002\semantic_codec_best.pt" goto missing
echo 正在启动联合语义编码演示：从码流直接解码图片与类别。
".venv-segformer\Scripts\python.exe" demo_server.py --checkpoint runs\suim_segformer_b0_20260915\segmentation_best.pt --semantic-checkpoint runs\suim_joint_semantic_codec_20261002\semantic_codec_best.pt --data-dir runs\image_codec_console --port 8769 --default-mode semantic_codec --open %*
set "semantic_codec_result=%errorlevel%"
if not "%semantic_codec_result%"=="0" if not defined SEMANTIC_NO_PAUSE pause
exit /b %semantic_codec_result%
:missing
echo 缺少本机环境或联合训练权重。请保留 runs 下训练文件，具体见 docs 中的联合语义编码说明。
if not defined SEMANTIC_NO_PAUSE pause
exit /b 1
