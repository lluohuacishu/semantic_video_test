@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
echo 本机语义通信界面：http://127.0.0.1:8776/
echo 使用期间请保留此窗口，关闭窗口即可停止界面服务。
".venv\Scripts\python.exe" semantic_mvp\console.py --port 8776 --open
if errorlevel 1 pause
