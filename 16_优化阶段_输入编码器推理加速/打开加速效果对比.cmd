@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0打开加速效果对比.ps1" %*
if errorlevel 1 pause
