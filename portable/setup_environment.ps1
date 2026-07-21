[CmdletBinding()]
param(
    [switch]$ForceRebuild,
    [switch]$VerifyBundleOnly
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Split-Path -Parent $PSScriptRoot).TrimEnd('\')
$RuntimeRoot = Join-Path $ProjectRoot "RUNTIME"
$PortablePythonRoot = Join-Path $RuntimeRoot "python"
$PortablePython = Join-Path $PortablePythonRoot "python.exe"
$StagingRoot = Join-Path $RuntimeRoot "_python_staging"
$BootstrapRoot = Join-Path $RuntimeRoot "bootstrap"
$PythonArchive = Join-Path $BootstrapRoot "python-3.11.9-embed-amd64.zip"
$GetPipScript = Join-Path $BootstrapRoot "get-pip.py"
$RequirementsLock = Join-Path $BootstrapRoot "requirements-lock.txt"
$RuntimeManifest = Join-Path $BootstrapRoot "runtime_manifest.json"
$Wheelhouse = Join-Path $RuntimeRoot "wheelhouse"
$RuntimeCache = Join-Path $ProjectRoot ".runtime\cache\$env:COMPUTERNAME"
$BootstrapTemp = Join-Path $ProjectRoot ".runtime\bootstrap_temp\$env:COMPUTERNAME"
$script:ReplacedRuntime = $null

function Assert-SafeRuntimePath {
    param([Parameter(Mandatory = $true)][string]$Path)
    $FullPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
    $ExpectedPrefix = [System.IO.Path]::GetFullPath($RuntimeRoot).TrimEnd('\') + '\'
    if (-not $FullPath.StartsWith($ExpectedPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "拒绝操作 RUNTIME 目录之外的路径: $FullPath"
    }
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    $PreviousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Executable @Arguments
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousPreference
    }
    if ($ExitCode -ne 0) {
        throw "命令执行失败（退出码 $ExitCode）: $Executable $Arguments"
    }
}

function Set-PortableEnvironment {
    param([Parameter(Mandatory = $true)][string]$PythonRoot)
    $env:PIP_CONFIG_FILE = "NUL"
    $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
    $env:PIP_NO_INPUT = "1"
    $env:PIP_NO_INDEX = "1"
    $env:PYTHONNOUSERSITE = "1"
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONDONTWRITEBYTECODE = "1"
    $env:PYTHONHOME = $null
    $env:PYTHONPATH = $null
    $env:VIRTUAL_ENV = $null
    $env:CONDA_PREFIX = $null
    $env:CONDA_DEFAULT_ENV = $null
    $env:YOLOV5_AUTOINSTALL = "false"
    $env:SENTINEL_OFFLINE = "1"
    $env:GIT_PYTHON_REFRESH = "quiet"
    $env:YOLOV5_CONFIG_DIR = Join-Path $RuntimeCache "yolov5"
    $env:TORCH_HOME = Join-Path $RuntimeCache "torch"
    $env:TORCH_EXTENSIONS_DIR = Join-Path $RuntimeCache "torch_extensions"
    $env:MPLCONFIGDIR = Join-Path $RuntimeCache "matplotlib"
    $env:CUDA_CACHE_PATH = Join-Path $RuntimeCache "cuda"
    $env:TEMP = $BootstrapTemp
    $env:TMP = $BootstrapTemp
    $TorchLib = Join-Path $PythonRoot "Lib\site-packages\torch\lib"
    $env:PATH = "$PythonRoot;$PythonRoot\DLLs;$TorchLib;$env:PATH"
}

function Test-PortableRuntime {
    param([Parameter(Mandatory = $true)][string]$PythonPath)
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        return $false
    }
    $PythonRoot = Split-Path -Parent $PythonPath
    Set-PortableEnvironment -PythonRoot $PythonRoot
    try {
        $PreviousPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $PythonPath -I -B -X utf8 -c "import importlib.util,struct,sys,torch,torchvision,cv2,PySide6,numpy,yaml,pandas,matplotlib,PIL,scipy,seaborn,tensorboard,tqdm,thop,requests,psutil; assert importlib.util.find_spec('pkg_resources'); assert sys.version_info[:3] == (3,11,9); assert struct.calcsize('P') == 8; assert sys.prefix == sys.base_prefix; assert 'RUNTIME' in sys.prefix; assert torch.__version__ == '2.2.1+cu118'; assert torchvision.__version__ == '0.17.1+cu118'"
            $ExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $PreviousPreference
        }
        return ($ExitCode -eq 0)
    }
    catch {
        return $false
    }
}

function Test-RecoveryBundle {
    if (-not (Test-Path -LiteralPath $RuntimeManifest -PathType Leaf)) {
        throw "缺少离线恢复清单: $RuntimeManifest"
    }
    $Manifest = Get-Content -LiteralPath $RuntimeManifest -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($Manifest.schema_version -ne 1 -or $Manifest.python_version -ne "3.11.9") {
        throw "离线恢复清单版本不受支持。"
    }
    $Index = 0
    foreach ($Entry in $Manifest.files) {
        $Index++
        $RelativePath = ([string]$Entry.path).Replace('/', '\')
        $FilePath = Join-Path $RuntimeRoot $RelativePath
        if (-not (Test-Path -LiteralPath $FilePath -PathType Leaf)) {
            throw "离线恢复文件缺失: $RelativePath"
        }
        $Item = Get-Item -LiteralPath $FilePath
        if ($Item.Length -ne [int64]$Entry.size) {
            throw "离线恢复文件大小异常: $RelativePath"
        }
        $ActualHash = (Get-FileHash -LiteralPath $FilePath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ActualHash -ne [string]$Entry.sha256) {
            throw "离线恢复文件指纹异常: $RelativePath"
        }
        if (($Index % 20) -eq 0) {
            Write-Host "已校验离线包 $Index / $($Manifest.files.Count)"
        }
    }
    Write-Host "离线恢复包完整：$($Manifest.files.Count) 个文件。" -ForegroundColor Green
}

function New-PortableRuntime {
    Assert-SafeRuntimePath -Path $StagingRoot
    if (Test-Path -LiteralPath $StagingRoot) {
        Remove-Item -LiteralPath $StagingRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $StagingRoot | Out-Null

    Write-Host "正在解压项目自带的 Python 3.11.9..." -ForegroundColor Yellow
    Expand-Archive -LiteralPath $PythonArchive -DestinationPath $StagingRoot -Force
    $PathFile = Join-Path $StagingRoot "python311._pth"
    $PathText = "python311.zip`r`n.`r`nLib`r`nLib\site-packages`r`n..\..`r`nimport site`r`n"
    [System.IO.File]::WriteAllText($PathFile, $PathText, [System.Text.Encoding]::ASCII)

    $StagingPython = Join-Path $StagingRoot "python.exe"
    Set-PortableEnvironment -PythonRoot $StagingRoot
    Write-Host "正在从移动硬盘离线安装 pip..." -ForegroundColor Yellow
    Invoke-Checked -Executable $StagingPython -Arguments @(
        $GetPipScript,
        "--no-index", "--find-links", $Wheelhouse,
        "pip==24.3.1", "setuptools==75.3.0", "wheel==0.45.1"
    )

    Write-Host "正在从移动硬盘离线安装 CUDA、界面和训练依赖..." -ForegroundColor Yellow
    Invoke-Checked -Executable $StagingPython -Arguments @(
        "-m", "pip", "--isolated", "--disable-pip-version-check", "--no-input",
        "install", "--no-index", "--find-links", $Wheelhouse,
        "--no-warn-script-location", "-r", $RequirementsLock
    )

    # Keep the Microsoft C++ runtime app-local.  This lets PyTorch and Qt run
    # on a standard Windows PC that does not have a separate VC++ redistributable.
    $QtRuntimeRoot = Join-Path $StagingRoot "Lib\site-packages\PySide6"
    foreach ($RuntimeDll in @(
        "concrt140.dll", "msvcp140.dll", "msvcp140_1.dll",
        "msvcp140_2.dll", "msvcp140_codecvt_ids.dll"
    )) {
        Copy-Item -LiteralPath (Join-Path $QtRuntimeRoot $RuntimeDll) `
            -Destination (Join-Path $StagingRoot $RuntimeDll) -Force
    }

    # pip console launchers contain the temporary build prefix in their
    # embedded shebang.  The application always uses `python.exe -m ...`, so
    # remove those non-portable launchers before promotion.
    $ScriptLaunchers = Join-Path $StagingRoot "Scripts"
    if (Test-Path -LiteralPath $ScriptLaunchers) {
        Assert-SafeRuntimePath -Path $ScriptLaunchers
        Remove-Item -LiteralPath $ScriptLaunchers -Recurse -Force
    }
    Get-ChildItem -LiteralPath $StagingRoot -Recurse -File -Filter "*.pyc" -ErrorAction SilentlyContinue | ForEach-Object {
        if (-not $_.FullName.StartsWith($StagingRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "字节码清理路径超出临时运行时: $($_.FullName)"
        }
        Remove-Item -LiteralPath $_.FullName -Force
    }
    Get-ChildItem -LiteralPath $StagingRoot -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
        Sort-Object { $_.FullName.Length } -Descending | ForEach-Object {
            if ($_.FullName.StartsWith($StagingRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
                Remove-Item -LiteralPath $_.FullName -Recurse -Force
            }
        }

    if (-not (Test-PortableRuntime -PythonPath $StagingPython)) {
        throw "离线重建出的运行时未通过依赖检查。"
    }

    if (Test-Path -LiteralPath $PortablePythonRoot) {
        Assert-SafeRuntimePath -Path $PortablePythonRoot
        $BrokenName = "_python_replaced_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        $BrokenPath = Join-Path $RuntimeRoot $BrokenName
        Assert-SafeRuntimePath -Path $BrokenPath
        Move-Item -LiteralPath $PortablePythonRoot -Destination $BrokenPath
        $script:ReplacedRuntime = $BrokenPath
        Write-Host "旧运行时已保留在: $BrokenPath" -ForegroundColor DarkYellow
    }
    Move-Item -LiteralPath $StagingRoot -Destination $PortablePythonRoot
}

Write-Host "SentinelVision 便携运行时检查" -ForegroundColor Cyan
Write-Host "项目目录: $ProjectRoot"
Write-Host "内置 Python: $PortablePython"

$ProjectDriveName = [System.IO.Path]::GetPathRoot($ProjectRoot).TrimEnd('\').TrimEnd(':')
$ProjectDrive = Get-PSDrive -Name $ProjectDriveName -ErrorAction SilentlyContinue
if ($null -ne $ProjectDrive -and $ProjectDrive.Free -lt 15GB) {
    throw "移动硬盘剩余空间不足 15 GB，无法安全检查或修复运行时。"
}
New-Item -ItemType Directory -Force -Path $RuntimeCache, $BootstrapTemp | Out-Null

if ($VerifyBundleOnly) {
    Test-RecoveryBundle
    exit 0
}

$Healthy = Test-PortableRuntime -PythonPath $PortablePython
if ($Healthy -and -not $ForceRebuild) {
    Write-Host "内置运行时完整，不需要安装。" -ForegroundColor Green
}
else {
    if ($ForceRebuild) {
        Write-Host "已要求强制离线重建运行时。" -ForegroundColor Yellow
    }
    else {
        Write-Host "内置运行时缺失或损坏，开始离线修复。" -ForegroundColor Yellow
    }
    Test-RecoveryBundle
    New-PortableRuntime
}

if (-not (Test-PortableRuntime -PythonPath $PortablePython)) {
    throw "内置运行时最终检查失败。"
}

Write-Host "正在执行工作库自检..." -ForegroundColor Yellow
Set-PortableEnvironment -PythonRoot $PortablePythonRoot
Push-Location $ProjectRoot
try {
    Invoke-Checked -Executable $PortablePython -Arguments @("-I", "-B", "-X", "utf8", "portable_check.py")
}
finally {
    Pop-Location
}
if ($null -ne $script:ReplacedRuntime -and (Test-Path -LiteralPath $script:ReplacedRuntime)) {
    Assert-SafeRuntimePath -Path $script:ReplacedRuntime
    Remove-Item -LiteralPath $script:ReplacedRuntime -Recurse -Force
    Write-Host "新运行时通过自检，旧副本已清理。" -ForegroundColor Green
}
Write-Host "便携运行时已就绪；目标电脑无需安装 Python 或联网。" -ForegroundColor Green
