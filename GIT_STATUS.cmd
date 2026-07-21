@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0portable\git_status.ps1"
set "RESULT=%ERRORLEVEL%"
pause
exit /b %RESULT%

