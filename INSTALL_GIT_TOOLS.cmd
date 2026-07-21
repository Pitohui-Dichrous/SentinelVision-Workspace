@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0portable\install_git_tools.ps1"
if errorlevel 1 goto failed

echo.
echo [OK] Portable Git tools are ready.
pause
exit /b 0

:failed
echo.
echo [ERROR] Git tool installation failed. Keep this window open.
pause
exit /b 1

