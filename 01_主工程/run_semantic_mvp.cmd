@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" goto missing
echo 正在运行鱼类语义通信原型：发送端识别、分层编码、20kbps传输、独立接收。
if "%~1"=="" (
  ".venv\Scripts\python.exe" semantic_mvp\system.py demo --target fish --open
) else (
  ".venv\Scripts\python.exe" semantic_mvp\system.py demo --input "%~1" --target fish --open
)
set "semantic_mvp_exit=%errorlevel%"
if not "%semantic_mvp_exit%"=="0" pause
exit /b %semantic_mvp_exit%
:missing
echo 找不到本机Python环境，请保留工程目录中的.venv。
pause
exit /b 1
