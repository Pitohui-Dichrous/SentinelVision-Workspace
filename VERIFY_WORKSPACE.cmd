@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
if errorlevel 1 goto invalid_project_path

call "%~dp0portable\runtime_env.cmd"
if errorlevel 1 goto missing_runtime
"%ENV_PYTHON%" -I -B -X utf8 portable\verify_workspace.py
set "VERIFY_EXIT=%ERRORLEVEL%"
pause
exit /b %VERIFY_EXIT%

:missing_runtime
echo [ERROR] The bundled RUNTIME is missing or cannot be used.
echo Run SETUP_ENVIRONMENT.cmd to repair it offline.
pause
exit /b 1

:invalid_project_path
echo [ERROR] The workspace directory cannot be opened.
pause
exit /b 1
