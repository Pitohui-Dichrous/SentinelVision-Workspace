@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
if errorlevel 1 goto invalid_project_path

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0portable\commit_with_message.ps1"
if errorlevel 1 goto failed

echo.
echo [OK] 已完成带备注的版本保存。
pause
exit /b 0

:failed
echo.
echo [ERROR] 版本保存失败；请保留此窗口中的错误信息。
pause
exit /b 1

:invalid_project_path
echo.
echo [ERROR] 无法打开工作库目录。
pause
exit /b 1
