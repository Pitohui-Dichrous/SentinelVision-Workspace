@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0portable\enable_git.ps1"
if errorlevel 1 goto failed

echo.
echo [OK] This workspace is now a Git repository.
echo Next: double-click SYNC_GITHUB.cmd once to connect GitHub.
pause
exit /b 0

:failed
echo.
echo [ERROR] Git initialization failed. Keep this window open.
pause
exit /b 1

