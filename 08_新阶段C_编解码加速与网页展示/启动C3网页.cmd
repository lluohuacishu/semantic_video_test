@echo off
chcp 65001 >nul
cd /d "%~dp0.."
"C:\Users\lenovo\Desktop\works\work\semantic-video-training\.venv-glc\Scripts\python.exe" -B -u "%~dp0c3_server.py" --open
if errorlevel 1 pause
