@echo off
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
if not exist ".venv\Scripts\activate.bat" (
    echo Missing .venv. Run setup_env.cmd first.
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
echo Environment: %VIRTUAL_ENV%
echo Read README.md for extraction and training commands.
cmd /k
