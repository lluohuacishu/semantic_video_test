@echo off
chcp 65001 >nul
setlocal
set "C7_PROJECT=%~1"
if not defined C7_PROJECT set /p "C7_PROJECT=请输入 C5 包内 semantic_video_test 文件夹的完整路径: "
set "C7_PROJECT=%C7_PROJECT:"=%"
if not exist "%C7_PROJECT%\runtime\.venv\Scripts\python.exe" (
  echo 未找到 C5 已安装的 Python 环境。
  pause
  exit /b 2
)
set /p "C7_LEFT=请输入 4060 的 results 子文件夹完整路径: "
set /p "C7_RIGHT=请输入 4070 的 results 子文件夹完整路径: "
set "C7_LEFT=%C7_LEFT:"=%"
set "C7_RIGHT=%C7_RIGHT:"=%"
set "C7_OUTPUT=%~dp0comparison_%RANDOM%_%RANDOM%.json"
"%C7_PROJECT%\runtime\.venv\Scripts\python.exe" -X utf8 -B "%~dp0c7_compare_results.py" --left "%C7_LEFT%" --right "%C7_RIGHT%" --output "%C7_OUTPUT%"
set "C7_EXIT=%ERRORLEVEL%"
echo.
echo 返回码 %C7_EXIT%。0=已完成数值比较，1=缺少成对原始数据，2=输入被拒绝。
echo 比较完成不代表跨卡兼容通过。结果路径: %C7_OUTPUT%
pause
exit /b %C7_EXIT%
