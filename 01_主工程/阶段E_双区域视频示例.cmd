@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" "semantic_mvp\system.py" demo --target fish --mode video --frames 50 --kbps 20 --roi-layout halves --roi-crf 31 --open
pause
