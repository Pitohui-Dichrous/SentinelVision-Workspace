@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0portable\save_version.ps1"
if errorlevel 1 goto failed

echo.
echo [OK] Version save finished.
pause
exit /b 0

:failed
echo.
echo [ERROR] Version save failed. Keep this window open.
pause
exit /b 1

