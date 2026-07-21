@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
if errorlevel 1 goto invalid_project_path

call "%~dp0portable\runtime_env.cmd"
if errorlevel 1 goto missing_environment

"%ENV_PYTHON%" -I -B -X utf8 portable_check.py
if errorlevel 1 goto check_failed

"%ENV_PYTHON%" -I -B -X utf8 launcher_GPT.py
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" pause
exit /b %APP_EXIT%

:check_failed
echo.
echo [ERROR] Environment check failed. The application was not started.
pause
exit /b 1

:missing_environment
echo [ERROR] The bundled RUNTIME is missing or cannot be used.
echo Run SETUP_ENVIRONMENT.cmd to repair it offline.
pause
exit /b 1

:invalid_project_path
echo [ERROR] The project directory cannot be opened.
pause
exit /b 1
