@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PIP_CONFIG_FILE=NUL
set PIP_INDEX_URL=
set PIP_EXTRA_INDEX_URL=
set PIP_FIND_LINKS=
set PIP_TRUSTED_HOST=
set PIP_NO_INDEX=
if not exist ".venv\Scripts\python.exe" (
    py -3.13 -m venv .venv
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install --only-binary=:all: --index-url https://pypi.org/simple -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --no-deps -r requirements-torch.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
".venv\Scripts\python.exe" environment_check.py
if errorlevel 1 goto failed
echo Environment is ready.
if not defined SEMANTIC_NO_PAUSE pause
exit /b 0

:failed
echo Setup failed. Please check the message above.
if not defined SEMANTIC_NO_PAUSE pause
exit /b 1
