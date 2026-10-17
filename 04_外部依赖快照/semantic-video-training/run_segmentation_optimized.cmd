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
if /I "%~1"=="--help" goto train
if /I "%~1"=="-h" goto train
".venv\Scripts\python.exe" scripts\prepare_pretrained.py
if errorlevel 1 (
    if not defined SEMANTIC_NO_PAUSE pause
    exit /b 1
)
:train
".venv\Scripts\python.exe" scripts\train_segmentation.py --model lraspp --pretrained checkpoints\pretrained\lraspp_mobilenet_v3_large-d234d4ea.pth --balanced-sampling --augmentation strong --epochs 60 --patience 15 --batch-size 8 --device cuda --cache-images %*
set "optimized_result=%errorlevel%"
if not defined SEMANTIC_NO_PAUSE pause
exit /b %optimized_result%
