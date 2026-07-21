@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
if errorlevel 1 goto invalid_project_path

call "%~dp0portable\runtime_env.cmd"
if errorlevel 1 goto missing_environment

"%ENV_PYTHON%" -I -B -X utf8 safe_train.py %*
set "TRAIN_EXIT=%ERRORLEVEL%"
if not "%TRAIN_EXIT%"=="0" goto training_failed
exit /b 0

:training_failed
echo.
echo [STOPPED] Training did not start or did not complete.
pause
exit /b %TRAIN_EXIT%

:missing_environment
echo [ERROR] The bundled RUNTIME is missing or cannot be used.
echo Run SETUP_ENVIRONMENT.cmd to repair it offline.
pause
exit /b 1

:invalid_project_path
echo [ERROR] The project directory cannot be opened.
pause
exit /b 1
