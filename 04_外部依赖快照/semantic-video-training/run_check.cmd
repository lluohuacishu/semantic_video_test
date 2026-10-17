@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
".venv\Scripts\python.exe" environment_check.py %*
set "check_result=%errorlevel%"
if not defined SEMANTIC_NO_PAUSE pause
exit /b %check_result%

:missing
echo Missing .venv. Run setup_env.cmd first.
:failed
if not defined SEMANTIC_NO_PAUSE pause
exit /b 1
