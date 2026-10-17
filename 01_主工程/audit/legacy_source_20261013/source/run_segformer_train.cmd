@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
if not exist ".venv-segformer\Scripts\python.exe" (
    echo Missing .venv-segformer. Run setup_segformer.cmd first.
    if not defined SEMANTIC_NO_PAUSE pause
    exit /b 1
)
".venv-segformer\Scripts\python.exe" scripts\train_segmentation.py --model segformer-b0 --pretrained checkpoints\pretrained\segformer-b0-ade --size 256 --batch-size 8 --epochs 60 --patience 15 --learning-rate 0.0006 --backbone-lr-factor 0.1 --balanced-sampling --augmentation strong --device cuda --cache-images %*
set "segformer_train_result=%errorlevel%"
if not defined SEMANTIC_NO_PAUSE pause
exit /b %segformer_train_result%
