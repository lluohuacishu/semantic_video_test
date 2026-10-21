@echo off
chcp 65001 >nul
setlocal
echo 请先关闭原 C5 发送端、接收端和诊断窗口。安装不会启动模型。
set "C7_PROJECT=%~1"
if not defined C7_PROJECT set /p "C7_PROJECT=请输入 C5 包内 semantic_video_test 文件夹完整路径: "
set "C7_PROJECT=%C7_PROJECT:"=%"
if not exist "%C7_PROJECT%\runtime\.venv\Scripts\python.exe" (
  echo 未找到原接收包安装的 Python 环境，请确认文件夹。
  pause
  exit /b 1
)
"%C7_PROJECT%\runtime\.venv\Scripts\python.exe" -X utf8 -B "%~dp0c7_apply_preview_patch.py" --project "%C7_PROJECT%"
set "C7_EXIT=%ERRORLEVEL%"
pause
exit /b %C7_EXIT%
