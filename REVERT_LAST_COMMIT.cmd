@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
if errorlevel 1 goto invalid_project_path

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0portable\revert_last_commit.ps1"
set "ACTION_EXIT=%ERRORLEVEL%"
if not "%ACTION_EXIT%"=="0" goto failed

echo.
echo [OK] Revert operation finished.
pause
exit /b 0

:failed
echo.
echo [STOPPED] Revert did not complete. Review the message above.
pause
exit /b %ACTION_EXIT%

:invalid_project_path
echo [ERROR] The workspace directory cannot be opened.
pause
exit /b 1
