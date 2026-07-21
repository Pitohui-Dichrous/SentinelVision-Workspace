@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
if errorlevel 1 goto invalid_project_path

call "%~dp0portable\runtime_env.cmd"
if errorlevel 1 goto repair_runtime

echo ============================================================
echo SentinelVision Workspace
echo Project: %~dp0
echo ============================================================

:run_check
"%ENV_PYTHON%" -I -B -X utf8 portable_check.py
if errorlevel 1 goto check_failed

"%ENV_PYTHON%" -I -B -X utf8 workspace_manager.py
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" goto app_failed
exit /b 0

:repair_runtime
echo.
echo The bundled runtime is missing. Starting offline repair...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0portable\setup_environment.ps1"
if errorlevel 1 goto setup_failed
call "%~dp0portable\runtime_env.cmd"
if errorlevel 1 goto setup_failed
goto run_check

:setup_failed
echo.
echo [ERROR] The bundled runtime repair failed.
echo Keep this window open and provide the error text for diagnosis.
pause
exit /b 1

:check_failed
echo.
echo [ERROR] Workspace check failed. The manager was not started.
pause
exit /b 1

:app_failed
echo.
echo [ERROR] Workspace manager exited with code %APP_EXIT%.
pause
exit /b %APP_EXIT%

:invalid_project_path
echo [ERROR] The workspace directory cannot be opened.
pause
exit /b 1
