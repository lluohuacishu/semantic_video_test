@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch_c5.ps1" -Role receiver %*
if errorlevel 1 pause
