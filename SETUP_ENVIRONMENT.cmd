@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
if errorlevel 1 goto invalid_project_path

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0portable\setup_environment.ps1"
set "SETUP_EXIT=%ERRORLEVEL%"
if not "%SETUP_EXIT%"=="0" goto setup_failed

echo.
echo [DONE] The offline portable runtime is ready. Run START_HERE.cmd.
pause
exit /b 0

:setup_failed
echo.
echo [ERROR] Offline runtime verification or repair did not finish.
echo Keep the error text above for diagnosis.
pause
exit /b %SETUP_EXIT%

:invalid_project_path
echo [ERROR] The project directory cannot be opened.
pause
exit /b 1
