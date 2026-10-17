@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
if not exist ".venv-segformer\Scripts\python.exe" goto missing
".venv-segformer\Scripts\python.exe" scripts\setup_image_codec.py %*
set "image_codec_setup_result=%errorlevel%"
if not "%image_codec_setup_result%"=="0" if not defined SEMANTIC_NO_PAUSE pause
exit /b %image_codec_setup_result%
:missing
echo 请先按项目说明准备 .venv-segformer 环境。
if not defined SEMANTIC_NO_PAUSE pause
exit /b 1
