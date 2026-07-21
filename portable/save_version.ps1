param(
    [string]$Message = "",
    [switch]$NoPush
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "git_common.ps1")

$root = Get-ProjectRoot
$tools = Ensure-GitTools
$git = $tools.Git
$gh = $tools.Gh

if (-not (Test-GitRepository -Git $git)) {
    & (Join-Path $PSScriptRoot "enable_git.ps1")
    if ($LASTEXITCODE -ne 0) { throw "Git initialization failed." }
}

Invoke-Checked -Executable $git -Arguments @("-C", $root, "add", "-A")
Assert-SafeStagedFiles -Git $git
& $git -C $root diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] No source changes need to be saved."
    return
}
if ($LASTEXITCODE -ne 1) { throw "Unable to inspect staged changes." }

if ([string]::IsNullOrWhiteSpace($Message)) {
    $Message = "Workspace save " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
}
Invoke-Checked -Executable $git -Arguments @("-C", $root, "commit", "-m", $Message)
Write-Host "[OK] Local version committed: $Message"

if (-not $NoPush) {
    if (Test-NativeCommand -Executable $gh -Arguments @("auth", "status", "--hostname", "github.com")) {
        Ensure-OriginRemote -Git $git
        & $git -C $root push origin main
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] GitHub is synchronized."
        }
        else {
            Write-Warning "The local version is safe, but GitHub push needs SYNC_GITHUB.cmd."
        }
    }
    else {
        Write-Host "[INFO] Local version is safe. Run SYNC_GITHUB.cmd for first-time GitHub login."
    }
}
