@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv. Run setup_env.cmd first.
    if not defined SEMANTIC_NO_PAUSE pause
    exit /b 1
)
".venv\Scripts\python.exe" scripts\train_autoencoder.py --manifest data\SUIM\manifest.json --device cuda --batch-size 8 --cache-images %*
set "train_result=%errorlevel%"
if not defined SEMANTIC_NO_PAUSE pause
exit /b %train_result%
