@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PIP_CONFIG_FILE=NUL
if not exist ".venv-segformer\Scripts\python.exe" (
    py -3.13 -m venv .venv-segformer
    if errorlevel 1 goto failed
)
".venv-segformer\Scripts\python.exe" -m pip --isolated install --only-binary=:all: -r requirements-torch.txt
if errorlevel 1 goto failed
".venv-segformer\Scripts\python.exe" -m pip --isolated install --only-binary=:all: --index-url https://pypi.org/simple -r requirements-segformer.txt
if errorlevel 1 goto failed
".venv-segformer\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
".venv-segformer\Scripts\python.exe" scripts\prepare_segformer.py
if errorlevel 1 goto failed
echo SegFormer environment and pretrained model are ready.
if not defined SEMANTIC_NO_PAUSE pause
exit /b 0

:failed
echo Setup failed. Please check the message above.
if not defined SEMANTIC_NO_PAUSE pause
exit /b 1
