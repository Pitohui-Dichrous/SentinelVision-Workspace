param(
    [Parameter(Mandatory = $true)][string]$Destination
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$source = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
if (-not (Test-Path -LiteralPath $Destination -PathType Container)) {
    throw "Target workspace not found: $Destination"
}
$target = (Resolve-Path -LiteralPath $Destination).Path
if ($target -eq $source) {
    throw "Source and destination workspaces must be different."
}
if (-not (Test-Path -LiteralPath (Join-Path $target "SentinelVision4.py") -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $target "RESULTS") -PathType Container) -or
    -not (Test-Path -LiteralPath (Join-Path $target "models") -PathType Container)) {
    throw "Refusing a destination that is not an existing SentinelVision workspace: $target"
}

$files = @(
    ".gitignore", ".gitattributes", "AGENTS.md", "GIT_GUIDE_CN.md",
    "INSTALL_GIT_TOOLS.cmd", "ENABLE_GIT.cmd", "SAVE_VERSION.cmd",
    "SYNC_GITHUB.cmd", "GIT_STATUS.cmd", "DOWNLOAD_PRETRAINED_WEIGHTS.cmd",
    "project_paths.py", "pretrained_weights.py", "safe_train.py",
    "portable_check.py", "ui_theme.py", "workspace_manager.py",
    "ui_motion.py", "ui_font.py", "ui_artwork.py",
    "assets\ui\check.svg", "assets\ui\chevron-down.svg", "assets\ui\chevron-up.svg",
    "SentinelVision4.py", "START_HERE_CN.md", "PORTABLE_GUIDE_CN.md",
    "portable\git_common.ps1", "portable\install_git_tools.ps1",
    "portable\enable_git.ps1", "portable\save_version.ps1",
    "portable\git_status.ps1", "portable\sync_github.ps1",
    "portable\download_pretrained_weights.ps1", "portable\verify_workspace.py",
    "portable\apply_to_e_drive.ps1"
)

foreach ($relative in $files) {
    $from = Join-Path $source $relative
    if (-not (Test-Path -LiteralPath $from -PathType Leaf)) {
        throw "Required source file missing: $relative"
    }
    $to = Join-Path $target $relative
    $parent = Split-Path -Parent $to
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    Copy-Item -LiteralPath $from -Destination $to -Force
    Write-Host "[OK] $relative"
}

$weightReadme = Join-Path $source "PRETRAINED_WEIGHTS\README_WEIGHTS_CN.md"
if (Test-Path -LiteralPath $weightReadme -PathType Leaf) {
    $weightTarget = Join-Path $target "PRETRAINED_WEIGHTS"
    New-Item -ItemType Directory -Path $weightTarget -Force | Out-Null
    Copy-Item -LiteralPath $weightReadme -Destination $weightTarget -Force
}

Write-Host "[OK] E-drive source upgrade copied. Datasets, RESULTS and runtimes were untouched."
