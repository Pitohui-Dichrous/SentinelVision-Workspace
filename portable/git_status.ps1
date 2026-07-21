Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "git_common.ps1")

$root = Get-ProjectRoot
$git = Find-GitExecutable
if ($null -eq $git) {
    Write-Host "Git tools are not installed. Run INSTALL_GIT_TOOLS.cmd."
    exit 1
}
if (-not (Test-GitRepository -Git $git)) {
    Write-Host "This workspace is not initialized. Run ENABLE_GIT.cmd."
    exit 1
}

Write-Host "=== Workspace status ==="
& $git -C $root status --short --branch
Write-Host ""
Write-Host "=== Recent versions ==="
& $git -C $root log --date=local --pretty=format:"%h  %ad  %s" -n 12
Write-Host ""
