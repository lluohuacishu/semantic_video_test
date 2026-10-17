@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
if not exist ".venv-segformer\Scripts\python.exe" (
    echo Missing .venv-segformer. Run setup_segformer.cmd first.
    if not defined SEMANTIC_NO_PAUSE pause
    exit /b 1
)
".venv-segformer\Scripts\python.exe" scripts\predict_segformer.py %*
set "segformer_result=%errorlevel%"
if not defined SEMANTIC_NO_PAUSE pause
exit /b %segformer_result%
