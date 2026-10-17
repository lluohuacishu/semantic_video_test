@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
echo 正在运行轻量语义通信：只传目标类别、位置和置信度。
if "%~1"=="" (
  ".venv\Scripts\python.exe" semantic_mvp\system.py demo --target fish --mode information --open
) else (
  ".venv\Scripts\python.exe" semantic_mvp\system.py demo --input "%~1" --target fish --mode information --open
)
set "semantic_information_exit=%errorlevel%"
if not "%semantic_information_exit%"=="0" pause
exit /b %semantic_information_exit%
