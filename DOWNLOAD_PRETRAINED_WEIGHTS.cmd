@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0portable\download_pretrained_weights.ps1"
if errorlevel 1 goto failed

echo.
echo [OK] Official YOLOv5 v7.0 detection weights are ready.
pause
exit /b 0

:failed
echo.
echo [ERROR] Weight download failed. Keep this window open.
pause
exit /b 1

