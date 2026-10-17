@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
if not exist ".venv\Scripts\python.exe" goto missing
echo 正在启动水下语义分割演示界面，请稍等模型加载。
echo 关闭本窗口即可停止服务。无需重新训练，也无需联网。
".venv\Scripts\python.exe" "demo_server.py" --open %*
set "DEMO_EXIT=%ERRORLEVEL%"
if "%DEMO_EXIT%"=="0" exit /b 0
echo 启动失败，请查看上面的提示。
if not defined SEMANTIC_NO_PAUSE pause
exit /b %DEMO_EXIT%
:missing
echo 找不到 .venv\Scripts\python.exe，请保持项目文件夹完整。
echo 如果是新下载的项目，请先按 README 配置环境和训练权重。
if not defined SEMANTIC_NO_PAUSE pause
exit /b 1
