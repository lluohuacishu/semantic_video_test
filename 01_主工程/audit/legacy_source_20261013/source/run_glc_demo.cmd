@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist ".venv-glc\Scripts\python.exe" (
  echo GLC environment missing. Read docs\GLC-Video.md for setup.
  pause
  exit /b 1
)
echo Starting GLC-Video at http://127.0.0.1:8770/
".venv-glc\Scripts\python.exe" -u glc_demo_server.py --open %*
set "glc_web_result=%errorlevel%"
if not "%glc_web_result%"=="0" if not defined GLC_NO_PAUSE pause
exit /b %glc_web_result%
