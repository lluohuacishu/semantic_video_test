@echo off
chcp 65001 >nul
setlocal
set "C7_PROJECT=%~1"
if not defined C7_PROJECT set /p "C7_PROJECT=请输入 C5 包内 semantic_video_test 文件夹的完整路径（可拖入文件夹）: "
set "C7_PROJECT=%C7_PROJECT:"=%"
if not exist "%C7_PROJECT%\runtime\.venv\Scripts\python.exe" (
  echo 未找到 C5 已安装的 Python 环境，请确认指向 semantic_video_test 文件夹。
  pause
  exit /b 1
)
"%C7_PROJECT%\runtime\.venv\Scripts\python.exe" -X utf8 -B "%~dp0c7_decode_probe.py" --project "%C7_PROJECT%"
set "C7_EXIT=%ERRORLEVEL%"
echo.
echo 诊断结束，返回码 %C7_EXIT%。请把本目录 results 下最新的整个子文件夹发回来。
pause
exit /b %C7_EXIT%
