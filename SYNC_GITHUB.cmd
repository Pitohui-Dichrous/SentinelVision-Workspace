@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0portable\sync_github.ps1"
if errorlevel 1 goto failed

echo.
echo [OK] GitHub synchronization finished.
pause
exit /b 0

:failed
echo.
echo [ERROR] GitHub synchronization failed. Keep this window open.
pause
exit /b 1

