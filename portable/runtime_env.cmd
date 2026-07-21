@echo off
for %%I in ("%~dp0..") do set "SENTINEL_ROOT=%%~fI\"

set "ENV_PYTHON=%SENTINEL_ROOT%RUNTIME\python\python.exe"
set "PYTHONHOME="
set "PYTHONPATH="
set "VIRTUAL_ENV="
set "CONDA_PREFIX="
set "CONDA_DEFAULT_ENV="
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"
set "YOLOV5_AUTOINSTALL=false"
set "SENTINEL_OFFLINE=1"
set "GIT_PYTHON_REFRESH=quiet"

set "SENTINEL_CACHE=%SENTINEL_ROOT%.runtime\cache\%COMPUTERNAME%"
set "SENTINEL_TEMP=%SENTINEL_ROOT%.runtime\temp\%COMPUTERNAME%"
set "YOLOV5_CONFIG_DIR=%SENTINEL_CACHE%\yolov5"
set "TORCH_HOME=%SENTINEL_CACHE%\torch"
set "TORCH_EXTENSIONS_DIR=%SENTINEL_CACHE%\torch_extensions"
set "MPLCONFIGDIR=%SENTINEL_CACHE%\matplotlib"
set "CUDA_CACHE_PATH=%SENTINEL_CACHE%\cuda"
set "TEMP=%SENTINEL_TEMP%"
set "TMP=%SENTINEL_TEMP%"

set "QT_PLUGIN_PATH=%SENTINEL_ROOT%RUNTIME\python\Lib\site-packages\PySide6\plugins"
set "QT_QPA_PLATFORM_PLUGIN_PATH=%QT_PLUGIN_PATH%\platforms"
set "PATH=%SENTINEL_ROOT%RUNTIME\python;%SENTINEL_ROOT%RUNTIME\python\DLLs;%SENTINEL_ROOT%RUNTIME\python\Lib\site-packages\torch\lib;%PATH%"

if not exist "%ENV_PYTHON%" exit /b 1
if not exist "%SENTINEL_CACHE%" md "%SENTINEL_CACHE%" >nul 2>nul
if not exist "%SENTINEL_TEMP%" md "%SENTINEL_TEMP%" >nul 2>nul
if not exist "%SENTINEL_CACHE%" exit /b 2
if not exist "%SENTINEL_TEMP%" exit /b 2
exit /b 0
