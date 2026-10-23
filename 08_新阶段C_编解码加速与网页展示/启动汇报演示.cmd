@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch_c5.ps1" -Role demo %*
if errorlevel 1 pause
