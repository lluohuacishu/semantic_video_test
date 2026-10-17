@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-glc\Scripts\python.exe" (
  echo GLC environment missing. Read docs\GLC-Video.md for setup.
  pause
  exit /b 1
)
if "%~1"=="" (
  echo Drag an AVI or MP4 video onto this launcher.
  echo Encodes a real GLC-based stream, transfers it at 20 kbps, and decodes it.
  echo Uses official networks with a local experimental file format.
  pause
  exit /b 1
)
set "GLC_RUN=runs\glc_stream_%RANDOM%_%RANDOM%"
".venv-glc\Scripts\python.exe" -u "scripts\glc_video_roundtrip.py" --input "%~f1" --output "%GLC_RUN%" --frames 50 --width 1280 --q 63 --kbps 20
if errorlevel 1 (
  echo GLC test failed. Read the error above.
  pause
  exit /b 1
)
start "" "%~dp0%GLC_RUN%\index.html"
pause
